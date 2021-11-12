"""Training procedure for NICE.
"""

import argparse
import torch
from torch._C import device
import torchvision
from torchvision import transforms
from collections import defaultdict
from tqdm import trange, tqdm
import matplotlib.pyplot as plt
import nice
import datetime

def train(flow, trainloader, optimizer, epoch, device):
    flow.train()  # set to training mode
    running_loss = 0
    batch_num = 1
    for inputs, _ in trainloader:
        batch_num += 1
        inputs = inputs.view(inputs.shape[0], inputs.shape[1] * inputs.shape[2] * inputs.shape[3])  # change  shape from BxCxHxW to Bx(C*H*W)
        inputs = inputs.to(device)
        # TODO Fill in
        optimizer.zero_grad()
        loss = -flow(inputs).mean()
        running_loss += float(loss)
        loss.backward()
        optimizer.step()
    return running_loss / batch_num


def test(flow, testloader, filename, epoch, sample_shape, device):
    flow.eval()  # set to inference mode
    running_loss = 0
    with torch.no_grad():
        num_samples = 100
        samples = flow.sample(num_samples).cpu()
        samples = samples.view(num_samples, sample_shape[0] , sample_shape[1] , sample_shape[2])
        torchvision.utils.save_image(torchvision.utils.make_grid(samples),
                                     './samples/' + filename + 'epoch%d.png' % epoch)
        # TODO full in
        for n_batches, data in enumerate(testloader, 1):
            inputs, _ = data
            inputs = inputs.view(inputs.shape[0], inputs.shape[1] * inputs.shape[2] * inputs.shape[3])  # change  shape from BxCxHxW to Bx(C*H*W)
            inputs = inputs.to(device)
            loss = -flow(inputs).mean()
            running_loss += float(loss)
    return running_loss / n_batches


def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = 'cpu'
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (1.,)),
        transforms.Lambda(lambda x: x + torch.zeros_like(x).uniform_(0., 1. / 256.))  # dequantization
    ])

    if args.dataset == 'mnist':
        trainset = torchvision.datasets.MNIST(root='./data/MNIST',
                                              train=True, download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset,
                                                  batch_size=args.batch_size, shuffle=True, num_workers=2)
        testset = torchvision.datasets.MNIST(root='./data/MNIST',
                                             train=False, download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset,
                                                 batch_size=args.batch_size, shuffle=False, num_workers=2)
    elif args.dataset == 'fashion-mnist':
        trainset = torchvision.datasets.FashionMNIST(root='~/torch/data/FashionMNIST',
                                                     train=True, download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset,
                                                  batch_size=args.batch_size, shuffle=True, num_workers=2)
        testset = torchvision.datasets.FashionMNIST(root='./data/FashionMNIST',
                                                    train=False, download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset,
                                                 batch_size=args.batch_size, shuffle=False, num_workers=2)
    else:
        raise ValueError('Dataset not implemented')

    z = trainset.data[0].size()
    model_save_filename = '%s_' % args.dataset \
                          + 'batch%d_' % args.batch_size \
                          + 'coupling%d_' % args.coupling \
                          + 'coupling_type%s_' % args.coupling_type \
                          + 'mid%d_' % args.mid_dim \
                          + 'hidden%d_' % args.hidden \
                          + '.pt'
    full_dim = trainset.data[0].view(-1).size(dim=0)
    flow = nice.NICE(
        prior=args.prior,
        coupling=args.coupling,
        coupling_type=args.coupling_type,
        in_out_dim=full_dim,
        mid_dim=args.mid_dim,
        hidden=args.hidden,
        device=device).to(device)

    optimizer = torch.optim.Adam(
        flow.parameters(), lr=args.lr)

    # TODO fill in
    train_losses, test_losses = [], []
    last_train_loss = 10000000
    for epoch in range(args.epochs):
        epoch_loss = train(flow, trainloader, optimizer, epoch, device)
        train_losses.append(epoch_loss)
        if last_train_loss - epoch_loss < 0.01:
            break
        last_train_loss = epoch_loss
        epoch_test_loss = test(flow, testloader=testloader, epoch=epoch, sample_shape=(1,28,28),
                               filename=f"{args.dataset}_sampled", device=device)
        test_losses.append(epoch_loss)
        print(f'epoch num {epoch}: train loss - {epoch_loss}  test loss - {epoch_test_loss}')

    fig = plt.figure()
    fig.gca().plot(list(range(len(train_losses))), train_losses, list(range(len(test_losses))), test_losses)
    plt.savefig(f'./samples/train_plot {datetime.datetime.now()}.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('')
    parser.add_argument('--dataset',
                        help='dataset to be modeled.',
                        type=str,
                        default='mnist')
    parser.add_argument('--prior',
                        help='latent distribution.',
                        type=str,
                        default='logistic')
    parser.add_argument('--batch_size',
                        help='number of images in a mini-batch.',
                        type=int,
                        default=128)
    parser.add_argument('--epochs',
                        help='maximum number of iterations.',
                        type=int,
                        default=50)
    parser.add_argument('--sample_size',
                        help='number of images to generate.',
                        type=int,
                        default=64)
    parser.add_argument('--coupling-type',
                        help='.',
                        type=str,
                        default='adaptive')
    parser.add_argument('--coupling',
                        help='.',
                        # type=int,
                        default=4)
    parser.add_argument('--mid-dim',
                        help='.',
                        type=int,
                        default=1000)
    parser.add_argument('--hidden',
                        help='.',
                        type=int,
                        default=5)
    parser.add_argument('--lr',
                        help='initial learning rate.',
                        type=float,
                        default=1e-3)

    args = parser.parse_args()
    main(args)