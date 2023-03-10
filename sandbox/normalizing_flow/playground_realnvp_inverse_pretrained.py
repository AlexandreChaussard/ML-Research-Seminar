import torch.optim as optim
import torch
import torch.distributions as distributions

from src.data.dataloader import fetch_cifar_loader
from src.model.normalizing_flow.realnvp import train_inverse_realnvp, restore_data
from src.utils.viz import display_restoration_process
import src.utils.utils as utils

# Load the MNIST dataset

data_train_loader, data_test_loader, (n_channels, n_rows, n_cols) = fetch_cifar_loader(
    n_samples_train=1000,
    n_samples_test=50,
    batch_size=50,
    path_to_data="../../src/data/"
)

model = utils.load_model("realnvp_inverse_inpainting_pretrained135_2_cifar10")

# Alteration
alteration_function = utils.pytorch_add_square
alteration_args = [4]

# Try restoring data from the test set with the same noise applied as for the training set
target_data_list, noisy_data_list, restored_data_list = restore_data(
    model,
    data_test_loader,
    alteration_function=alteration_function,
    alteration_args=alteration_args
)

# Display the results
display_restoration_process(
    target_data_list,
    noisy_data_list,
    restored_data_list,
    max_samples=5,
    permutation_shape=(1, 2, 0)
)