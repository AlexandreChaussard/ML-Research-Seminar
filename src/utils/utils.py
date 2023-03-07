import math
import torch
import torchvision.transforms as transforms
import torch.distributions as distributions
import torch.nn.functional as F
import numpy as np

def sampling(mu, log_var):
    # this function samples a Gaussian distribution,
    # with average (mu) and standard deviation specified (using log_var)
    std = torch.exp(0.5 * log_var)
    eps = torch.randn_like(std)
    return eps.mul(std).add_(mu)  # return z sample


def pytorch_to_numpy(x):
    return x.detach().numpy()


def pytorch_noise(x, noise_mean, noise_std):
    return x + (noise_mean + noise_std * torch.randn(x.shape))


def pytorch_add_square(x, square_size):
    tensor = x.clone()
    img_width = x.shape[3]
    img_height = x.shape[2]
    tensor[:, :, img_height // 2 - square_size:img_height // 2 + square_size,
    img_width // 2 - square_size:img_width // 2 + square_size] = 0
    return tensor


def pytorch_gaussian_blur(x, kernel_size, sigma):
    return transforms.GaussianBlur(kernel_size=kernel_size, sigma=sigma)(x)

def compute_same_pad(kernel_size, stride):
    if isinstance(kernel_size, int):
        kernel_size = [kernel_size]

    if isinstance(stride, int):
        stride = [stride]

    assert len(stride) == len(
        kernel_size
    ), "Pass kernel size and stride both as int, or both as equal length iterable"

    return [((k - 1) * s + 1) // 2 for k, s in zip(kernel_size, stride)]


def uniform_binning_correction(x, n_bits=8):
    """Replaces x^i with q^i(x) = U(x, x + 1.0 / 256.0).
    Args:
        x: 4-D Tensor of shape (NCHW)
        n_bits: optional.
    Returns:
        x: x ~ U(x, x + 1.0 / 256)
        objective: Equivalent to -q(x)*log(q(x)).
    """
    b, c, h, w = x.size()
    n_bins = 2 ** n_bits
    chw = c * h * w
    x += torch.zeros_like(x).uniform_(0, 1.0 / n_bins)

    objective = -math.log(n_bins) * chw * torch.ones(b, device=x.device)
    return x, objective


def split_feature(tensor, type="split"):
    """
    type = ["split", "cross"]
    """
    C = tensor.size(1)
    if type == "split":
        return tensor[:, : C // 2, ...], tensor[:, C // 2 :, ...]
    elif type == "cross":
        return tensor[:, 0::2, ...], tensor[:, 1::2, ...]


def logit_transform(x, constraint=0.9, reverse=False):
    '''Transforms data from [0, 1] into unbounded space.
    Restricts data into [0.05, 0.95].
    Calculates logit(alpha+(1-alpha)*x).
    Args:
        x: input tensor.
        constraint: data constraint before logit.
        reverse: True if transform data back to [0, 1].
    Returns:
        transformed tensor and log-determinant of Jacobian from the transform.
        (if reverse=True, no log-determinant is returned.)
    '''
    if reverse:
        x = 1. / (torch.exp(-x) + 1.)  # [0.05, 0.95]
        x *= 2.  # [0.1, 1.9]
        x -= 1.  # [-0.9, 0.9]
        x /= constraint  # [-1, 1]
        x += 1.  # [0, 2]
        x /= 2.  # [0, 1]
        return x, 0
    else:
        [B, C, H, W] = list(x.size())

        # dequantization
        noise = distributions.Uniform(0., 1.).sample((B, C, H, W))
        x = (x * 255. + noise) / 256.

        # restrict data
        x *= 2.  # [0, 2]
        x -= 1.  # [-1, 1]
        x *= constraint  # [-0.9, 0.9]
        x += 1.  # [0.1, 1.9]
        x /= 2.  # [0.05, 0.95]

        # logit data
        logit_x = torch.log(x) - torch.log(1. - x)

        # log-determinant of Jacobian from the transform
        pre_logit_scale = torch.tensor(
            np.log(constraint) - np.log(1. - constraint))
        log_diag_J = F.softplus(logit_x) + F.softplus(-logit_x) \
                     - F.softplus(-pre_logit_scale)

        return logit_x, torch.sum(log_diag_J, dim=(1, 2, 3))