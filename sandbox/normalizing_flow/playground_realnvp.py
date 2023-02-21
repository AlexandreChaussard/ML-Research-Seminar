import torch.optim as optim

from src.data.dataloader import fetch_cifar_loader
from src.model.normalizing_flow.realnvp import RealNVP, train_realnvp, generate_data
from src.utils.viz import display_images

# Load the MNIST dataset

data_train_loader, data_test_loader, (n_channels, n_rows, n_cols) = fetch_cifar_loader(
    n_samples_train=1000,
    n_samples_test=512,
    batch_size=256,
    path_to_data="../../src/data/"
)

# Create the model
model = RealNVP(
    num_scales=2,
    in_channels=n_channels,
    mid_channels=64,
    num_blocks=8
)

# Define the optimizer of the model
optimizer = optim.Adam(model.parameters(), lr=10e-2)

# Train the model
n_epoch = 100
model = train_realnvp(model, optimizer, data_train_loader, n_epoch=n_epoch, grad_clip_max=1000)

# Generate new samples
generated_imgs = generate_data(model, 5, *(n_channels, n_rows, n_cols))
# Display the results
display_images(generated_imgs)
