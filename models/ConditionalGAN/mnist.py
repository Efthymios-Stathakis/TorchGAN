from torch import nn
from ..generator_utils import GenConvTransposeBlock
from ..discriminator_utils import DiscConvBlock

class Generator(nn.Module):
    """
    A generator model for the Conditional GAN architecture on MNIST dataset.
    
    This class defines the generator network architecture for the 
    Conditional GAN model on MNIST dataset. It takes a noise vector 
    as input and generates an image.
    
    Attributes:
        z_dim (int): The dimensionality of the noise vector.
        im_chan (int): The number of channels in the output image.
        hidden_dim (int): The dimensionality of the hidden layers in the generator.
    """
    def __init__(self, z_dim, im_chan=1, hidden_dim=64):
        """
        Initializes the generator model.
        
        Args:
            z_dim (int, optional): The dimensionality of the noise vector.
            im_chan (int, optional): The number of channels in the output image. Defaults to 1.
            hidden_dim (int, optional): The dimensionality of the hidden layers in the 
                                         generator. Defaults to 64.
        """
        super().__init__()
        self.z_dim = z_dim
        self.gen = nn.Sequential(
            GenConvTransposeBlock(z_dim, hidden_dim * 4),
            GenConvTransposeBlock(hidden_dim * 4, hidden_dim * 2, kernel_size=4, stride=1),
            GenConvTransposeBlock(hidden_dim * 2, hidden_dim * 1),
            GenConvTransposeBlock(hidden_dim * 1, im_chan,  kernel_size=4, final=True)
        )

    def unsqueeze_noise(self, x):
        """
        Unsqueezes the noise vector to match the expected input shape of the generator.
        
        Args:
            x (torch.Tensor): The noise vector.
        
        Returns:
            torch.Tensor: The unsqueezed noise vector.
        """
        return x.view(len(x), self.z_dim, 1, 1)
    
    def forward(self, x):
        """
        Forward pass through the generator network.
        
        Args:
            x (torch.Tensor): The noise vector.
        
        Returns:
            torch.Tensor: The generated image.
        """
        x = self.unsqueeze_noise(x)
        return self.gen(x)
    

class Discriminator(nn.Module):
    """
    A discriminator model for the Conditional GAN architecture on MNIST dataset.
    
    This class defines the discriminator network architecture for the Conditional GAN 
    model on MNIST dataset. It takes an image as input and outputs a probability that the image is real.
    
    Attributes:
        im_chan (int): The number of channels in the input image.
        hidden_dim (int): The dimensionality of the hidden layers in the discriminator.
    """
    def __init__(self, im_chan=1, hidden_dim=32):
        """
        Initializes the discriminator model.
        
        Args:
            im_chan (int, optional): The number of channels in the input image. Defaults to 1.
            hidden_dim (int, optional): The dimensionality of the hidden layers in the discriminator. Defaults to 32.
        """
        super().__init__()
        self.disc = nn.Sequential(
            DiscConvBlock(im_chan,        hidden_dim * 1),
            DiscConvBlock(hidden_dim * 1, hidden_dim * 2),
            DiscConvBlock(hidden_dim * 2, 1, final=True),
        )

    def forward(self, x):
        """
        Forward pass through the discriminator network.
        
        Args:
            x (torch.Tensor): The input image.
        
        Returns:
            torch.Tensor: The probability that the image is real.
        """
        return self.disc(x).view(-1, 1)