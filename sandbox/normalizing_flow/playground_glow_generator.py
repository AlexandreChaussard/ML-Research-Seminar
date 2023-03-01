import torch.optim as optim

from src.data.dataloader import fetch_mnist_loader
from src.model.normalizing_flow.glow.model import Glow, train_glow, generate_data
from src.utils.viz import display_images

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# Load the MNIST dataset
data_train_loader, data_test_loader, (n_channels, n_rows, n_cols) = fetch_mnist_loader(
    n_samples_train=1000,
    n_samples_test=512,
    batch_size=256,
    path_to_data="../../src/data/"
)

# Create the model
n_flows = 32
n_blocks = 3

model = Glow(
    in_channel=n_channels,
    n_flow=n_flows,
    n_block=n_blocks,
    affine=True,
    conv_lu=True
)

# Define the optimizer of the model
optimizer = optim.Adam(model.parameters(), lr=10e-2)

# Train the model
n_epoch = 50
model = train_glow(model, optimizer, data_train_loader, n_epoch=n_epoch)

# Generate new samples
generated_imgs = generate_data(
    model,
    n_data=5,
    input_size=n_channels * n_rows * n_cols,
    n_blocks=n_blocks
)

# Display the results
display_images(generated_imgs)
