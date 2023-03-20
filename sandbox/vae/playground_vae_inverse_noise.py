import torch.optim as optim

from src.data.dataloader import fetch_mnist_loader, fetch_cifar_loader, fetch_celeba_loader
from src.model.vae.vae import VAE, train_vae_inverse_noise, restore_noisy_data
from src.utils.viz import display_restoration_process

# Load the MNIST dataset

data_train_loader, data_test_loader, (n_channels, n_rows, n_cols) = fetch_celeba_loader(
    n_samples_train=1000,
    n_samples_test=1000,
    batch_size=256,
    path_to_data="../../src/data/"
)

# Set the noise parameters
noise_mean = 0.1
noise_std = 0.1

# Create the model
model = VAE(
    hidden_sizes_encoder=[512, 256],
    hidden_sizes_decoder=[256, 512],
    z_dim=200,
    n_rows=n_rows,
    n_cols=n_cols,
    n_channels=n_channels
)

# Define the optimizer of the model
optimizer = optim.Adam(model.parameters(), lr=10e-4)

# Train the model
n_epoch = 100
model = train_vae_inverse_noise(
    model,
    optimizer,
    data_train_loader,
    n_epoch=n_epoch,
    noise_mean=noise_mean,
    noise_std=noise_std
)

# Try restoring data from the test set with the same noise applied as for the training set
target_data_list, noisy_data_list, restored_data_list = restore_noisy_data(
    model,
    data_test_loader,
    noise_mean=noise_mean,
    noise_std=noise_std
)

# Display the results
display_restoration_process(
    target_data_list,
    noisy_data_list,
    restored_data_list,
    max_samples=5,
    permutation_shape=(1, 2, 0)
)
