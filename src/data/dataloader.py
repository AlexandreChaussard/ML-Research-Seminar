"""
Data loader for all datasets
"""

from torchvision import datasets, transforms
import torch
import ssl
ssl._create_default_https_context = ssl._create_unverified_context


def fetch_mnist_loader(
        n_samples_train=1000,
        n_samples_test=512,
        batch_size=256,
        path_to_data="."
):
    transform_pipeline = transforms.Compose(
        [
            transforms.ToTensor(),
            # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    mnist_trainset = datasets.MNIST(f'{path_to_data}', train=True, download=True, transform=transform_pipeline)
    mnist_testset = datasets.MNIST(f'{path_to_data}', train=False, download=True, transform=transform_pipeline)

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
        n_samples_train=40000,
        n_samples_test=512,
        batch_size=256,
        path_to_data="."
):
    transform_pipeline = transforms.Compose(
        [
            transforms.ToTensor(),
            # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    mnist_trainset = datasets.CIFAR10(f'{path_to_data}', train=True, download=True, transform=transform_pipeline)
    mnist_testset = datasets.CIFAR10(f'{path_to_data}', train=False, download=True, transform=transform_pipeline)

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

    img_shape = (3, 32, 32)

    return mnist_train_loader, mnist_test_loader, img_shape


