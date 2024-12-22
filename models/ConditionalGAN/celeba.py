from torch import nn
from torch.utils.data import Dataset
from ..generator_utils import GenConvTransposeBlock
from ..discriminator_utils import DiscConvBlock

class Generator(nn.Module):
    """
    A generator model for the Conditional GAN architecture.
    
    This class defines the generator network architecture for the Conditional GAN model. It takes a noise vector as input and generates an image.
    
    Attributes:
        z_dim (int): The dimensionality of the noise vector.
        im_chan (int): The number of channels in the output image.
        hidden_dim (int): The dimensionality of the hidden layers in the generator.
    """
    def __init__(self, z_dim=256, im_chan=3, hidden_dim=64):
        """
        Initializes the generator model.
        
        Args:
            z_dim (int, optional): The dimensionality of the noise vector. Defaults to 256.
            im_chan (int, optional): The number of channels in the output image. Defaults to 3.
            hidden_dim (int, optional): The dimensionality of the hidden layers in the generator. Defaults to 64.
        """
        super().__init__()
        self.z_dim = z_dim
        self.gen = nn.Sequential(
            GenConvTransposeBlock(z_dim,           hidden_dim * 8, kernel_size=4,  stride=1, padding = 0),
            GenConvTransposeBlock(hidden_dim * 8,  hidden_dim * 4, kernel_size=4, stride=2, padding = 1),
            GenConvTransposeBlock(hidden_dim * 4,  hidden_dim * 2, kernel_size=4, stride=2, padding = 1),
            GenConvTransposeBlock(hidden_dim * 2,  hidden_dim * 1, kernel_size=4, stride=2, padding = 1),
            GenConvTransposeBlock(hidden_dim * 1,  im_chan,        kernel_size=4, stride=2, padding = 1, final=True)
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
    A discriminator model for the Conditional GAN architecture.
    
    This class defines the discriminator network architecture for the Conditional GAN model. It takes an image as input and outputs a probability that the image is real.
    
    Attributes:
        im_chan (int): The number of channels in the input image.
        hidden_dim (int): The dimensionality of the hidden layers in the discriminator.
    """
    def __init__(self, im_chan=3, hidden_dim=64):
        """
        Initializes the discriminator model.
        
        Args:
            im_chan (int, optional): The number of channels in the input image. Defaults to 3.
            hidden_dim (int, optional): The dimensionality of the hidden layers in the discriminator. Defaults to 64.
        """
        super().__init__()
        self.disc = nn.Sequential(
            DiscConvBlock(im_chan,        hidden_dim * 1, kernel_size=4, stride=2, padding=1),
            DiscConvBlock(hidden_dim * 1, hidden_dim * 2, kernel_size=4, stride=2, padding=1),
            DiscConvBlock(hidden_dim * 2, hidden_dim * 2, kernel_size=4, stride=2, padding=1), 
            DiscConvBlock(hidden_dim * 2, hidden_dim * 2, kernel_size=4, stride=2, padding=1),
            DiscConvBlock(hidden_dim * 2, 1, kernel_size=4, stride=1, final=True)
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


class CelebADataset(Dataset):
    """
    A dataset class for the CelebA dataset.
    
    This class loads and preprocesses the CelebA dataset for training the Conditional GAN model.
    
    Attributes:
        image_folder (torchvision.datasets.ImageFolder): The folder containing the CelebA images.
        attributes_df (pd.DataFrame): The dataframe containing the attributes for each image.
        transform (callable, optional): Optional transform to be applied on each sample.
    """
    def __init__(self, image_folder, attributes_df, transform=None):
        """
        Initializes the CelebADataset.
        
        Args:
            image_folder (torchvision.datasets.ImageFolder): The folder containing the CelebA images.
            attributes_df (pd.DataFrame): The dataframe containing the attributes for each image.
            transform (callable, optional): Optional transform to be applied on each sample. Defaults to None.
        """
        self.image_folder = image_folder
        self.attributes_df = attributes_df
        self.transform = transform

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        
        Returns:
            int: The total number of samples.
        """
        return len(self.attributes_df)

    def __getitem__(self, idx):
        """
        Loads and preprocesses a sample from the dataset.
        
        Args:
            idx (int): The index of the sample to load.
        
        Returns:
            tuple: A tuple containing the preprocessed image and its attributes.
        """
        img, _ = self.image_folder[idx]  # Get image from ImageFolder
        attributes = self.attributes_df.iloc[idx].values.astype(float)  # Get attributes
        
        if self.transform: img = self.transform(img)

        return img, attributes