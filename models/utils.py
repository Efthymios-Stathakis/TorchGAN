import torch
from torch import nn

def get_noise(n_samples, z_dim, device):
    """
    Generates a tensor of random noise for the given number of samples and dimension.

    Args:
        n_samples (int): The number of samples to generate noise for.
        z_dim (int): The dimension of the noise vector.
        device (torch.device): The device to create the tensor on.

    Returns:
        torch.Tensor: A tensor of shape (n_samples, z_dim) containing random noise.
    """
    return torch.randn((n_samples, z_dim), device=device)


def combine_tensors(x, y):
    """
    Concatenates two tensors along the specified dimension.

    Args:
        x (torch.Tensor): The first tensor to concatenate.
        y (torch.Tensor): The second tensor to concatenate.
        dim (int, optional): The dimension along which to concatenate. Defaults to 1.

    Returns:
        torch.Tensor: The concatenated tensor.
    """
    return torch.concat((x,y), dim=1)


def weights_init(m):
    """
    Initializes the weights of a module using a normal distribution 
    for convolutional and batch normalization layers.

    Args:
        m (nn.Module): The module to initialize weights for.
    """
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        torch.nn.init.constant_(m.bias, 0)