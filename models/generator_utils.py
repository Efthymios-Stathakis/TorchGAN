from torch import nn

class GenConvTransposeBlock(nn.Module):
    """A transposed convolution block used in the DCGAN generator.
    
    This block performs upsampling using transposed convolution, followed by batch normalization 
    and activation. For the final block, only transposed convolution and Tanh activation are used.
    
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels 
        kernel_size (int, optional): Size of the convolving kernel. Default: 3
        stride (int, optional): Stride of the convolution. Default: 2
        padding (int, optional): Padding added to input. Default: 0
        final (bool, optional): If True, uses Tanh activation without batch norm. Default: False
    """
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=0, final=False):
        super().__init__()
        if not final:
            self.block = nn.Sequential(
                nn.ConvTranspose2d(in_channels=in_channels, 
                                   out_channels=out_channels, 
                                   kernel_size=kernel_size,
                                   stride=stride,
                                   padding=padding),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        else:
            self.block = nn.Sequential(
                nn.ConvTranspose2d(in_channels=in_channels, 
                                   out_channels=out_channels, 
                                   kernel_size= kernel_size, 
                                   stride=stride,
                                   padding=padding),
                nn.Tanh()
            )

    def forward(self, x):
        """Forward pass of the block.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Output after transposed convolution and activation
        """
        return self.block(x)