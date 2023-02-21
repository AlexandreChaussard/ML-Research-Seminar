"""Train Real NVP on CIFAR-10.
Train script adapted from: https://github.com/kuangliu/pytorch-cifar/
"""
import os
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
import src.utils.realnvp as util

from src.model.normalizing_flow.realnvp import RealNVP
from src.model.normalizing_flow.realnvp.models.real_nvp.real_nvp_loss import RealNVPLoss
from tqdm import tqdm


def main(n_epochs, batch_size, learning_rate, weight_decay, max_grad_norm, n_samples):
    device = 'cpu'
    start_epoch = 0

    # Note: No normalization applied, since RealNVP expects inputs in (0, 1).
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor()
    ])

    trainset = torchvision.datasets.CIFAR10(root='data', train=True, download=True, transform=transform_train)
    trainloader = data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=1)

    testset = torchvision.datasets.CIFAR10(root='data', train=False, download=True, transform=transform_test)
    testloader = data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=1)

    # Model
    print('Building model..')
    net = RealNVP(num_scales=2, in_channels=3, mid_channels=64, num_blocks=8)
    net = net.to(device)

    loss_fn = RealNVPLoss()
    param_groups = util.get_param_groups(net, weight_decay, norm_suffix='weight_g')
    optimizer = optim.Adam(param_groups, lr=learning_rate)

    for epoch in range(start_epoch, start_epoch + n_epochs):
        train(epoch, net, trainloader, device, optimizer, loss_fn, max_grad_norm)
        main_test(epoch, net, testloader, device, loss_fn, n_samples)


def train(epoch, net, trainloader, device, optimizer, loss_fn, max_grad_norm):
    print('\nEpoch: %d' % epoch)
    net.train()
    total_loss = 0
    for x, _ in trainloader:
        x = x.to(device)
        optimizer.zero_grad()
        z, sldj = net(x, reverse=False)
        loss = loss_fn(z, sldj)
        total_loss += loss.item()
        loss.backward()
        util.clip_grad_norm(optimizer, max_grad_norm)
        optimizer.step()

    print("    * Loss:", total_loss)


def sample(net, batch_size, device):
    """Sample from RealNVP model.
    Args:
        net (torch.nn.DataParallel): The RealNVP model wrapped in DataParallel.
        batch_size (int): Number of samples to generate.
        device (torch.device): Device to use.
    """
    z = torch.randn((batch_size, 3, 32, 32), dtype=torch.float32, device=device)
    x, _ = net(z, reverse=True)
    x = torch.sigmoid(x)

    return x


def main_test(epoch, net, testloader, device, loss_fn, num_samples):
    global best_loss
    net.eval()
    loss_meter = util.AverageMeter()
    with torch.no_grad():
        with tqdm(total=len(testloader.dataset)) as progress_bar:
            for x, _ in testloader:
                x = x.to(device)
                z, sldj = net(x, reverse=False)
                loss = loss_fn(z, sldj)
                loss_meter.update(loss.item(), x.size(0))
                progress_bar.set_postfix(loss=loss_meter.avg,
                                         bpd=util.bits_per_dim(x, loss_meter.avg))
                progress_bar.update(x.size(0))

    # Save checkpoint
    if loss_meter.avg < best_loss:
        print('Saving...')
        state = {
            'net': net.state_dict(),
            'test_loss': loss_meter.avg,
            'epoch': epoch,
        }
        os.makedirs('ckpts', exist_ok=True)
        torch.save(state, 'ckpts/best.pth.tar')
        best_loss = loss_meter.avg

    # Save samples and data
    images = sample(net, num_samples, device)
    os.makedirs('samples', exist_ok=True)
    images_concat = torchvision.utils.make_grid(images, nrow=int(num_samples ** 0.5), padding=2, pad_value=255)
    torchvision.utils.save_image(images_concat, 'samples/epoch_{}.png'.format(epoch))


main(
    n_epochs=200,
    batch_size=60,
    learning_rate=0.1,
    weight_decay=10e-5,
    max_grad_norm=10000,
    n_samples=5
)