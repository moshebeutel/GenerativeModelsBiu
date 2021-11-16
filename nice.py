"""NICE model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.transforms import Transform, SigmoidTransform, AffineTransform
from torch.distributions import Uniform, TransformedDistribution
import numpy as np

"""Additive coupling layer.
"""


class AdditiveCoupling(nn.Module):
    def __init__(self, in_out_dim, mid_dim, hidden, mask_config):
        """Initialize an additive coupling layer.
        Args:
            in_out_dim: input/output dimensions.
            mid_dim: number of units in a hidden layer.
            hidden: number of hidden layers.
            mask_config: 1 if transform odd units, 0 if transform even units.
        """
        super(AdditiveCoupling, self).__init__()
        self._in_out_dim = in_out_dim
        self._mid_dim = mid_dim
        self._hidden = hidden
        self._mask_config = mask_config

        # layers architecture
        self.input_layer = nn.Linear(in_out_dim // 2, mid_dim)
        self.hidden_layers = nn.ModuleList([nn.Linear(mid_dim, mid_dim)\
             for i in range(hidden)])
        self.output_layer = nn.Linear(mid_dim, in_out_dim // 2)

    def forward(self, x, log_det_J, reverse=False):
        """Forward pass.
        Args:
            x: input tensor.
            log_det_J: log determinant of the Jacobian
            reverse: True in inference mode, False in sampling mode.
        Returns:
            transformed tensor and updated log-determinant of Jacobian.
        """

        
        shifted_entries, fixed_entries = (x[:, 1::2], x[:, 0::2]) if self._mask_config \
            else (x[:, 0::2], x[:, 1::2])

        y = F.relu(self.input_layer(fixed_entries))
        for i in range(len(self.hidden_layers)):
            y = F.relu(self.hidden_layers[i](y))
        shift = self.output_layer(y)

        shifted_entries = shifted_entries - shift if reverse else shifted_entries + shift

        # this ugly tmp is needed so that the grads on x will aggregate
        tmp = torch.ones_like(x)
        tmp[:, 1::2], tmp[:, 0::2] = (shifted_entries, fixed_entries) if self._mask_config \
            else (fixed_entries, shifted_entries)
        x = tmp

        return x, log_det_J  


class AffineCoupling(nn.Module):
    def __init__(self, in_out_dim, mid_dim, hidden, mask_config):
        """Initialize an affine coupling layer.
        Args:
            in_out_dim: input/output dimensions.
            mid_dim: number of units in a hidden layer.
            hidden: number of hidden layers.
            mask_config: 1 if transform odd units, 0 if transform even units.
        """
        super(AffineCoupling, self).__init__()
        
        self._in_out_dim = in_out_dim
        self._mid_dim = mid_dim
        self._hidden = hidden
        self._mask_config = mask_config
        self.eps = 1e-4

        # shift network architecture
        self.shift_input_layer = nn.Linear(in_out_dim // 2, mid_dim)
        shift_layer_list = [nn.Linear(mid_dim, mid_dim) for i in range(hidden)]
        self.shift_hidden_layers = nn.ModuleList(shift_layer_list)
        self.shift_output_layer = nn.Linear(mid_dim, in_out_dim // 2)

        # scale network architecture
        self.scale_input_layer = nn.Linear(in_out_dim // 2, mid_dim)
        scale_layer_list = [nn.Linear(mid_dim, mid_dim) for i in range(hidden)]
        self.scale_hidden_layers = nn.ModuleList(scale_layer_list)
        self.scale_output_layer = nn.Linear(mid_dim, in_out_dim // 2)
        

    def forward(self, x, log_det_J, reverse=False):
        """Forward pass.
        Args:
            x: input tensor.
            log_det_J: log determinant of the Jacobian
            reverse: True in inference mode, False in sampling mode.
        Returns:
            transformed tensor and updated log-determinant of Jacobian.
        """
    
        shifted_entries, fixed_entries = (x[:, 1::2], x[:, 0::2]) if self._mask_config \
            else (x[:, 0::2], x[:, 1::2])

        shift = F.relu(self.shift_input_layer(fixed_entries))
        for hidden in self.shift_hidden_layers:
            shift = F.relu(hidden(shift))
        shift = self.shift_output_layer(shift)

        scale = F.relu(self.scale_input_layer(fixed_entries))
        for hidden in self.scale_hidden_layers:
            scale = F.relu(hidden(scale))
        scale = torch.sigmoid(self.scale_output_layer(scale))

        shifted_entries = (shifted_entries - shift) / scale if reverse \
            else shifted_entries * scale + shift
        
        # this ugly tmp is needed so that the grads on x will aggregate
        tmp = torch.ones_like(x)
        tmp[:, 1::2], tmp[:, 0::2] = (shifted_entries, fixed_entries) if self._mask_config \
            else (fixed_entries, shifted_entries)
        x = tmp
    
        log_det_J = torch.sum(torch.log(scale))
        return x, log_det_J


"""Log-scaling layer.
"""


class Scaling(nn.Module):
    def __init__(self, dim):
        """Initialize a (log-)scaling layer.
        Args:
            dim: input/output dimensions.
        """
        super(Scaling, self).__init__()
        self.scale = nn.Parameter(
            torch.zeros((1, dim)), requires_grad=True)
        self.eps = 1e-5

    def forward(self, x, reverse=False):
        """Forward pass.
        Args:
            x: input tensor.
            reverse: True in inference mode, False in sampling mode.
        Returns:
            transformed tensor and log-determinant of Jacobian.
        """
        scale = torch.exp(self.scale) + self.eps
        
        log_det_J = torch.sum(self.scale) + self.eps
        if reverse:
            scale = torch.exp(-self.scale) + self.eps
        x *= scale
        return x, log_det_J


"""Standard logistic distribution.
"""
logistic = TransformedDistribution(Uniform(0, 1), [SigmoidTransform().inv, AffineTransform(loc=0., scale=1.)])

"""NICE main model.
"""


class NICE(nn.Module):
    def __init__(self, prior, coupling, coupling_type,
                 in_out_dim, mid_dim, hidden, device):
        """Initialize a NICE.
        Args:
            coupling_type: 'additive' or 'adaptive'
            coupling: number of coupling layers.
            in_out_dim: input/output dimensions.
            mid_dim: number of units in a hidden layer.
            hidden: number of hidden layers.
            device: run on cpu or gpu
        """
        super(NICE, self).__init__()
        self.device = device
        if prior == 'gaussian':
            self.prior = torch.distributions.Normal(
                torch.tensor(0.).to(device), torch.tensor(1.).to(device))
        elif prior == 'logistic':
            self.prior = logistic
        else:
            raise ValueError('Prior not implemented.')
        self.in_out_dim = in_out_dim
        self.coupling = coupling
        self.coupling_type = coupling_type

        coupling_layers = []
        assert coupling_type in ['additive', 'affine']
        for i in range(coupling):
            layer = AdditiveCoupling(in_out_dim, mid_dim, hidden, mask_config=i % 2) if coupling_type == 'additive' \
                else AffineCoupling(in_out_dim, mid_dim, hidden, mask_config=i % 2)
            coupling_layers.append(layer)

        self.coupling_module_list = nn.ModuleList(coupling_layers)
        self.scaling = Scaling(in_out_dim)

    def f_inverse(self, z):
        """Transformation g: Z -> X (inverse of f).
        Args:
            z: tensor in latent space Z.
        Returns:
            transformed tensor in data space X.
        """
        x, _ = self.scaling(z, reverse=True)
        for coupling_layer in reversed(self.coupling_module_list):
            x, _ = coupling_layer(x, 0, reverse=True)
        return x

    def f(self, x):
        """Transformation f: X -> Z (inverse of g).
        Args:
            x: tensor in data space X.
        Returns:
            transformed tensor in latent space Z and log determinant Jacobian
        """
        
        # log_det_J = 0
        log_det_J = torch.zeros(x.shape[0]).to(x.device)
        for coupling_layer in self.coupling_module_list:
            x, ldj = coupling_layer(x, log_det_J)
            log_det_J += ldj
        x, ldj = self.scaling(x)
        log_det_J += ldj
        return x, log_det_J

    def log_prob(self, x):
        """Computes data log-likelihood.
        (See Section 3.3 in the NICE paper.)
        Args:
            x: input minibatch.
        Returns:
            log-likelihood of input.
        """
        z, log_det_J = self.f(x)
        log_det_J -= np.log(256) * self.in_out_dim  # log det for rescaling from [0.256] (after dequantization) to [0,1]
        # log_ll = torch.sum(self.prior.log_prob(z), dim=1)
        # The following back and forth to and from gpu is due to an annoying exception I could not mange to solve otherwise :(
        log_ll = torch.sum(self.prior.log_prob(z.to('cpu')), dim=1).to(x.device)
        return log_ll + log_det_J

    def sample(self, size):
        """Generates samples.
        Args:
            size: number of samples to generate.
        Returns:
            samples from the data space X.
        """
        z = self.prior.sample((size, self.in_out_dim)).to(self.device)
        
        return self.f_inverse(z)

    def forward(self, x):
        """Forward pass.
        Args:
            x: input minibatch.
        Returns:
            log-likelihood of input.
        """
        return self.log_prob(x)
