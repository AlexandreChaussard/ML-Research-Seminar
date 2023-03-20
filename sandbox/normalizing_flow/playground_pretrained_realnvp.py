import torch.optim as optim
import torch
import torch.distributions as distributions

from src.data.dataloader import fetch_cifar_loader
from src.model.normalizing_flow.realnvp import train_realnvp, generate_data
from src.utils.viz import display_images
from src.utils.utils import save_model, load_model

# Load the MNIST dataset

data_train_loader, data_test_loader, (n_channels, n_rows, n_cols) = fetch_cifar_loader(
    n_samples_train=2000,
    n_samples_test=2000,
    batch_size=50,
    path_to_data="../../src/data/"
)

# Create the model
original_model = 235
model = load_model(f"realnvp_{original_model}_cifar10")

# Define the optimizer of the model
optimizer = optim.Adamax(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-7)

# Train the model
n_epoch = 100
# model = train_realnvp(model, optimizer, data_train_loader, n_epoch=n_epoch)

# Generate new samples
generated_imgs = generate_data(model, n_data=5)
# Display the results
display_images(generated_imgs)

# Saving the model
save_model(model, f"realnvp_{original_model+n_epoch}_cifar10")
