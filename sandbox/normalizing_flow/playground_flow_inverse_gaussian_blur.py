import torch.optim as optim
import torch.nn.functional as F

from src.data.dataloader import fetch_mnist_loader
from src.model.normalizing_flow.classic.flow import FlowModel, train_flow_inverse_blur, restore_blur_data
from src.utils.viz import display_restoration_process

# Load the MNIST dataset

mnist_train_loader, mnist_test_loader, (n_channels, n_rows, n_cols) = fetch_mnist_loader(
    n_samples_train=1000,
    n_samples_test=1000,
    batch_size=256,
    path_to_data="../../src/data/"
)

# Define the encoding dimension
z_dim = 20

# Set the blur parameters
kernel_size = 7
sigma = 5

# Create the model

model = FlowModel(
    flows=['PlanarFlow'] * 4,
    hidden_sizes_encoder=[512, 256],
    hidden_sizes_decoder=[256, 512],
    z_dim=z_dim,
    n_channels=n_channels,
    n_rows=n_rows,
    n_cols=n_cols,
    activation=F.elu
)

# Define the optimizer of the model
optimizer = optim.Adam(model.parameters(), lr=10e-4)

# Train the model
n_epoch = 500
model = train_flow_inverse_blur(
    model,
    optimizer,
    mnist_train_loader,
    n_epoch=n_epoch,
    kernel_size=kernel_size,
    sigma=sigma
)

# Try restoring data from the test set with the same noise applied as for the training set
target_data_list, noisy_data_list, restored_data_list = restore_blur_data(
    model,
    mnist_test_loader,
    kernel_size=kernel_size,
    sigma=sigma
)

# Display the results
display_restoration_process(
    target_data_list,
    noisy_data_list,
    restored_data_list,
    max_samples=5
)
