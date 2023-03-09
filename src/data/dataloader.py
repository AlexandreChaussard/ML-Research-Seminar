"""
Data loader for all datasets
"""

from torchvision import datasets, transforms
import torch


def fetch_mnist_loader(
        n_samples_train=1000,
        n_samples_test=512,
        batch_size=256,
        path_to_data="."
) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, tuple]:
    mnist_trainset = datasets.MNIST(f'{path_to_data}', train=True, download=True, transform=transforms.ToTensor())
    mnist_testset = datasets.MNIST(f'{path_to_data}', train=False, download=True, transform=transforms.ToTensor())

    # create data loader with said dataset size
    mnist_trainset_reduced = torch.utils.data.random_split(
        mnist_trainset,
        [n_samples_train, len(mnist_trainset) - n_samples_train]
    )[0]
    mnist_train_loader = torch.utils.data.DataLoader(mnist_trainset_reduced, batch_size=batch_size, shuffle=True,
                                                     drop_last=True)

    # download test dataset
    mnist_testset_reduced = torch.utils.data.random_split(
        mnist_testset,
        [n_samples_test, len(mnist_testset) - n_samples_test]
    )[0]
    mnist_test_loader = torch.utils.data.DataLoader(mnist_testset_reduced, batch_size=batch_size, shuffle=True,
                                                    drop_last=True)

    img_shape = (1, 28, 28)

    return mnist_train_loader, mnist_test_loader, img_shape


def fetch_cifar_loader(
        n_samples_train=1000,
        n_samples_test=512,
        batch_size=256,
        path_to_data="."
) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, tuple]:
    trainset = datasets.CIFAR10(f'{path_to_data}', train=True, download=True, transform=transforms.ToTensor())
    testset = datasets.CIFAR10(f'{path_to_data}', train=False, download=True, transform=transforms.ToTensor())

    # create data loader with said dataset size
    trainset_reduced = torch.utils.data.random_split(
        trainset,
        [n_samples_train, len(trainset) - n_samples_train]
    )[0]
    train_loader = torch.utils.data.DataLoader(trainset_reduced, batch_size=batch_size, shuffle=True,
                                                     drop_last=True)

    # download test dataset
    testset_reduced = torch.utils.data.random_split(
        testset,
        [n_samples_test, len(testset) - n_samples_test]
    )[0]
    test_loader = torch.utils.data.DataLoader(testset_reduced, batch_size=batch_size, shuffle=True,
                                                    drop_last=True)

    img_shape = (3, 32, 32)

    return train_loader, test_loader, img_shape


def fetch_celeba_loader(
        n_samples_train=1000,
        n_samples_test=512,
        batch_size=256,
        path_to_data="."
) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, tuple]:
    def CelebACrop(images):
        return transforms.functional.crop(images, 40, 15, 148, 148)

    transform = transforms.Compose(
        [CelebACrop,
         transforms.Resize(64),
         transforms.RandomHorizontalFlip(p=0.5),
         transforms.ToTensor()]
    )

    trainset = datasets.CelebA(f'{path_to_data}', split="train", download=True, transform=transform)
    testset = datasets.CelebA(f'{path_to_data}', split="test", download=True, transform=transform)

    # create data loader with said dataset size
    trainset_reduced = torch.utils.data.random_split(
        trainset,
        [n_samples_train, len(trainset) - n_samples_train]
    )[0]
    train_loader = torch.utils.data.DataLoader(trainset_reduced, batch_size=batch_size, shuffle=True,
                                               drop_last=True)

    # download test dataset
    testset_reduced = torch.utils.data.random_split(
        testset,
        [n_samples_test, len(testset) - n_samples_test]
    )[0]
    test_loader = torch.utils.data.DataLoader(testset_reduced, batch_size=batch_size, shuffle=True,
                                              drop_last=True)

    img_shape = (3, 64, 64)

    return train_loader, test_loader, img_shape