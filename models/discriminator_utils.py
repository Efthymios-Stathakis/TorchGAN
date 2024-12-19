from torch import nn

class DiscConvBlock(nn.Module):
    """Discriminator convolutional block used in DCGAN.
    
    This block consists of a convolutional layer followed by batch normalization 
    and LeakyReLU activation. For the final block, only convolution is applied.
    
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels 
        kernel_size (int, optional): Size of convolving kernel. Defaults to 4.
        stride (int, optional): Stride of convolution. Defaults to 2.
        padding (int, optional): Padding added to input. Defaults to 0.
        final (bool, optional): If True, only applies convolution without normalization
            and activation. Defaults to False.
    """
    
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=0, final=False):
        super().__init__()
        if not final:
            # Standard conv block with normalization and activation
            self.block = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, 
                          out_channels=out_channels, 
                          kernel_size=kernel_size,
                          stride=stride,
                          padding=padding),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(negative_slope=0.2, inplace=True)
            )
        else:
            # Final block with just convolution
            self.block = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, 
                          out_channels=out_channels, 
                          kernel_size= kernel_size, 
                          stride=stride,
                          padding=padding),
            )

    def forward(self, x):
        """Forward pass of the block.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Output after convolution (and optional normalization/activation)
        """
        return self.block(x)