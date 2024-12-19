from torch import nn
from ..generator_utils import GenConvTransposeBlock
from ..discriminator_utils import DiscConvBlock

class Generator(nn.Module):

    def __init__(self, z_dim, im_chan=1, hidden_dim=64):
        super().__init__()
        self.z_dim = z_dim
        self.gen = nn.Sequential(
            GenConvTransposeBlock(z_dim, hidden_dim * 4),
            GenConvTransposeBlock(hidden_dim * 4, hidden_dim * 2, kernel_size=4, stride=1),
            GenConvTransposeBlock(hidden_dim * 2, hidden_dim * 1),
            GenConvTransposeBlock(hidden_dim * 1, im_chan,  kernel_size=4, final=True)
        )

    def unsqueeze_noise(self, x):
        return x.view(len(x), self.z_dim, 1, 1)
    
    def forward(self, x):
        x = self.unsqueeze_noise(x)
        return self.gen(x)
    

class Discriminator(nn.Module):

    def __init__(self, im_chan, hidden_dim=32):
        super().__init__()
        self.disc = nn.Sequential(
            DiscConvBlock(im_chan, hidden_dim * 1),
            DiscConvBlock(hidden_dim * 1, hidden_dim * 2),
            DiscConvBlock(hidden_dim * 2, 1, final=True),
        )

    def forward(self, x):
        return self.disc(x).view(-1, 1)