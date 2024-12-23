from ..utils import crop, combine_tensors
from torch import nn

class ContractingBlock(nn.Module):
    """
    A building block for the contracting path of a U-Net.
    
    This module consists of two convolutional layers followed by a ReLU activation function.
    Optionally, it can include a max pooling layer for downsampling.
    
    Attributes:
        contract (bool): If True, the module includes a max pooling layer for downsampling.
        conv1 (nn.Conv2d): The first convolutional layer.
        conv2 (nn.Conv2d): The second convolutional layer.
        activation (nn.ReLU): The ReLU activation function.
        maxp (nn.MaxPool2d): The max pooling layer for downsampling.
    """
    def __init__(self, in_channels, out_channels):
        """
        Initializes the ContractingBlock module.
        
        Args:
            in_channels (int): The number of input channels.
            out_channels (int): The number of output channels.
            contract (bool, optional): If True, the module includes a max pooling layer for downsampling. Defaults to True.
        """
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels , out_channels=out_channels, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3)
        self.activation = nn.ReLU()

    def forward(self, x):
        """
        Forward pass through the ContractingBlock module.
        
        Args:
            x (torch.Tensor): The input tensor.
        
        Returns:
            torch.Tensor: The output tensor.
        """
        x = self.conv1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.activation(x)
        return x
    

class UpsamplingBlock(nn.Module):
    """
    A building block for the expanding path of a U-Net.
    
    This module consists of a transposed convolutional layer for upsampling.
    
    Attributes:
        upconv (nn.ConvTranspose2d): The transposed convolutional layer for upsampling.
    """
    def __init__(self, in_channels, out_channels):
        """
        Initializes the UpsamplingBlock module.
        
        Args:
            in_channels (int): The number of input channels.
            out_channels (int): The number of output channels.
        """
        super().__init__()
        self.upconv = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=2, stride=2)

    def forward(self, x):
        """
        Forward pass through the UpsamplingBlock module.
        
        Args:
            x (torch.Tensor): The input tensor.
        
        Returns:
            torch.Tensor: The output tensor after upsampling.
        """
        return self.upconv(x)
    
class ExpandingBlock(nn.Module):
    """
    A building block for the expanding path of a U-Net.
    
    This module consists of two convolutional layers followed by a ReLU activation function.
    It also includes a tensor concatenation operation for combining the upsampled feature map
    with the cropped feature map from the contracting path.
    
    Attributes:
        exp_block (nn.Sequential): A sequence of convolutional layers and ReLU activations.
    """
    def __init__(self, in_channels, out_channels):
        """
        Initializes the ExpandingBlock module.
        
        Args:
            in_channels (int): The number of input channels.
            out_channels (int): The number of output channels.
        """
        super().__init__()
        self.exp_block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,  out_channels=out_channels, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3),
            nn.ReLU()
        )

    def forward(self, x, crop_con):
        """
        Forward pass through the ExpandingBlock module.
        
        Args:
            x (torch.Tensor): The input tensor.
            crop_con (torch.Tensor): The cropped feature map from the contracting path.
        
        Returns:
            torch.Tensor: The output tensor after concatenation and convolutional operations.
        """
        x = combine_tensors(x, crop(crop_con, x.shape[2:]))
        return self.exp_block(x)
class UNet(nn.Module):
    """
    A U-Net model for image segmentation tasks.
    
    This model consists of a contracting path and an expanding path. The contracting path
    is responsible for feature extraction and downsampling, while the expanding path
    upsamples and refines the features for the final output. The model uses skip connections
    to preserve spatial information.
    
    Attributes:
        contracting_block1 (ContractingBlock): The first contracting block.
        contracting_block2 (ContractingBlock): The second contracting block.
        contracting_block3 (ContractingBlock): The third contracting block.
        contracting_block4 (ContractingBlock): The fourth contracting block.
        contracting_block5 (ContractingBlock): The fifth contracting block.
        expanding_block5 (ExpandingBlock): The fifth expanding block.
        expanding_block4 (ExpandingBlock): The fourth expanding block.
        expanding_block3 (ExpandingBlock): The third expanding block.
        expanding_block2 (ExpandingBlock): The second expanding block.
        upsampling_block5 (UpsamplingBlock): The fifth upsampling block.
        upsampling_block4 (UpsamplingBlock): The fourth upsampling block.
        upsampling_block3 (UpsamplingBlock): The third upsampling block.
        upsampling_block2 (UpsamplingBlock): The second upsampling block.
        output (nn.Conv2d): The final output layer.
    """
    def __init__(self, input_dim, label_dim, hidden_size):
        """
        Initializes the UNet model.
        
        Args:
            input_dim (int): The number of input channels.
            label_dim (int): The number of output channels.
            hidden_size (int): The base number of hidden channels.
        """
        super(UNet, self).__init__()
        self.contracting_block1 = ContractingBlock(input_dim,       hidden_size)
        self.contracting_block2 = ContractingBlock(hidden_size * 1, hidden_size * 2)
        self.contracting_block3 = ContractingBlock(hidden_size * 2, hidden_size * 4)
        self.contracting_block4 = ContractingBlock(hidden_size * 4, hidden_size * 8)
        self.contracting_block5 = ContractingBlock(hidden_size * 8, hidden_size * 16)
        
        self.expanding_block5 = ExpandingBlock(hidden_size * 16, hidden_size * 8)
        self.expanding_block4 = ExpandingBlock(hidden_size * 8, hidden_size * 4)
        self.expanding_block3 = ExpandingBlock(hidden_size * 4, hidden_size * 2)
        self.expanding_block2 = ExpandingBlock(hidden_size * 2, hidden_size * 1)
        
        self.upsampling_block5 = UpsamplingBlock(hidden_size * 16, hidden_size * 8)
        self.upsampling_block4 = UpsamplingBlock(hidden_size * 8, hidden_size * 4)
        self.upsampling_block3 = UpsamplingBlock(hidden_size * 4, hidden_size * 2)
        self.upsampling_block2 = UpsamplingBlock(hidden_size * 2, hidden_size * 1)
        
        self.output = nn.Conv2d(hidden_size, label_dim, kernel_size=1)
        
    def forward(self, x):
        """
        Forward pass through the UNet model.
        
        Args:
            x (torch.Tensor): The input tensor.
        
        Returns:
            torch.Tensor: The output tensor after passing through the UNet.
        """
        cb1  = self.contracting_block1(x)
        cb1_ = nn.MaxPool2d(kernel_size=2, stride=2)(cb1)
        cb2  = self.contracting_block2(cb1_)
        cb2_ = nn.MaxPool2d(kernel_size=2, stride=2)(cb2)
        cb3  = self.contracting_block3(cb2_)
        cb3_ = nn.MaxPool2d(kernel_size=2, stride=2)(cb3)
        cb4  = self.contracting_block4(cb3_)
        cb4_ = nn.MaxPool2d(kernel_size=2, stride=2)(cb4)
        cb5 = self.contracting_block5(cb4_)
        
        # Expanding block
        eb5  = self.upsampling_block5(cb5)
        eb4  = self.expanding_block5(eb5, cb4)
        eb4_ = self.upsampling_block4(eb4)
        eb3  = self.expanding_block4(eb4_, cb3)
        eb3_ = self.upsampling_block3(eb3)
        eb2  = self.expanding_block3(eb3_, cb2)
        eb2_ = self.upsampling_block2(eb2)
        eb1  = self.expanding_block2(eb2_, cb1)
        
        output = self.output(eb1)
        
        return output
