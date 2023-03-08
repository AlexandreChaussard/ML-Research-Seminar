import torch.optim as optim
import torch
import torch.distributions as distributions

from src.data.dataloader import fetch_cifar_loader
from src.model.normalizing_flow.realnvp import RealNVP, Hyperparameters, train_realnvp, generate_data
from src.utils.viz import display_images
from src.utils.utils import save_model

# Load the MNIST dataset

data_train_loader, data_test_loader, (n_channels, n_rows, n_cols) = fetch_cifar_loader(
    n_samples_train=2000,
    n_samples_test=2000,
    batch_size=50,
    path_to_data="../../src/data/"
)

# Prior distribution
prior = distributions.Normal(  # isotropic standard normal distribution
        torch.tensor(0.), torch.tensor(1.)
    )

hps = Hyperparameters(
    base_dim=n_rows,
    res_blocks=8,
    bottleneck=0,
    skip=1,
    weight_norm=1,
    coupling_bn=1,
    affine=1
)

# Create the model
model = RealNVP(
    dataset_name='cifar10',
    input_size=n_rows,
    n_channels=n_channels,
    prior=prior,
    hps=hps
)

# Define the optimizer of the model
optimizer = optim.Adamax(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-7)

# Train the model
n_epoch = 10
model = train_realnvp(model, optimizer, data_train_loader, n_epoch=n_epoch)

# Generate new samples
generated_imgs = generate_data(model, n_data=5)
# Display the results
display_images(generated_imgs)

# Saving the model
save_model(model, f"realnvp_{n_epoch}_cifar10")
