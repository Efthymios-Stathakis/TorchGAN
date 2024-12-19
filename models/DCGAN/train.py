import os, sys, torch
from torch import nn
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging
import argparse

from ..vis_utils import show_img_batch
from ..utils import get_noise, weights_init
from .mnist import (Generator as MnistGenerator, 
                    Discriminator as MnistDiscriminator)

# Set up logging
logging.basicConfig(level=logging.INFO)

def train(n_epochs, 
          z_dim=128,
          display_step=5000, 
          save_step=10000, 
          device="mps",
          dataset="mnist"):
    
    cur_step = 0
    generator_losses = []
    discriminator_losses = []

    for epoch in range(n_epochs):

        for real, _ in tqdm(mnist_dl):
            cur_batch_size = len(real)
            real = real.to(device)

            ## Update discriminator ##
            disc_opt.zero_grad()
            
            # Get noise corresponding to the current batch_size
            noise = get_noise(cur_batch_size, z_dim, device=device)
                    
            fake = gen(noise)
            disc_fake = disc(fake.detach())
            disc_fake_loss = criterion(disc_fake, torch.zeros_like(disc_fake, device=device))
            disc_real = disc(real)
            disc_real_loss = criterion(disc_real, torch.ones_like(disc_fake, device=device))
            disc_loss = (disc_real_loss + disc_fake_loss) / 2

            disc_loss.backward(retain_graph=True)
            disc_opt.step()

            discriminator_losses += [disc_loss.item()]
            
            ### Update generator ###
            gen_opt.zero_grad()
            noise = get_noise(cur_batch_size, z_dim, device=device)
                    
            fake = gen(noise)
            gen_fake = disc(fake)
            gen_fake_loss = criterion(gen_fake, torch.ones_like(gen_fake, device=device))
            gen_fake_loss.backward()
            gen_opt.step()        
            
            # Keep track of the average generator loss
            generator_losses += [gen_fake_loss.item()]

            ### Visualization code ###
            if cur_step % display_step == 0 and cur_step > 0:
                mean_gen_loss = sum(generator_losses[-display_step:])/display_step
                mean_disc_loss = sum(discriminator_losses[-display_step:])/display_step
                logging.info(f"Epoch {epoch}, step {cur_step}: Generator loss: {mean_gen_loss:.2f}, discriminator loss: {mean_disc_loss:.2f}")
                show_img_batch((real + 1) / 2)
                show_img_batch((fake + 1) / 2)
            if cur_step % save_step == 0 and cur_step > 0:
                assets_path = os.path.join(os.path.dirname(__file__), "../../assets/DCGAN")
                torch.save({
                    'generator': gen.state_dict(),
                    'discriminator': disc.state_dict(),
                    'gen_opt': gen_opt.state_dict(),
                    'disc_opt': disc_opt.state_dict(),
                    'generator_losses': generator_losses,
                    'discriminator_losses': discriminator_losses
                }, f"{assets_path}/{dataset}_epoch_{epoch}_step_{cur_step}.pth")
                logging.info(f"Models, optimizers, and mean losses saved at step {cur_step}")
            cur_step += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--z_dim',      type=int,   default=128,    help='Dimension of the noise vector')
    parser.add_argument('--batch_size', type=int,   default=128,    help='Batch size for training')
    parser.add_argument('--hidden_dim', type=int,   default=64,     help='Hidden dimension for the models')
    parser.add_argument('--lr',         type=float, default=0.0002, help='Learning rate for the optimizers')
    parser.add_argument('--beta_1',     type=float, default=0.5,    help='Beta 1 for Adam optimizer')
    parser.add_argument('--beta_2',     type=float, default=0.999,  help='Beta 2 for Adam optimizer')
    parser.add_argument('--dataset',    type=str ,  default="mnist",help='Beta 2 for Adam optimizer')

    args = parser.parse_args()
    
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    if args.dataset == "mnist":
        # Initialize the dataset and dataloader
        mnist_transforms = transforms.Compose([transforms.ToTensor(), 
                                            transforms.Normalize((0.5,), (0.5,))])
        mnist_dt = MNIST(root="../Data/", download=False, transform=mnist_transforms)
        mnist_dl = DataLoader(mnist_dt, batch_size=args.batch_size, shuffle=True)

        # Initialize the generator and discriminator
        gen  = MnistGenerator(z_dim=args.z_dim, hidden_dim=args.hidden_dim).to(device)
        disc = MnistDiscriminator(im_chan=1, hidden_dim=args.hidden_dim).to(device)

    # elif args.dataset == "celeba":
    #     # Initialize the dataset and dataloader
    #     celeba_transforms = transforms.Compose([transforms.ToTensor(), 
    #                                         transforms.Normalize((0.5,), (0.5,))])
    #     celeba_dt = MNIST(root="../Data/", download=False, transform=celeba_transforms)
    #     celeba_dl = DataLoader(celeba_dt, batch_size=args.batch_size, shuffle=True)

    #     # Initialize the generator and discriminator
    #     gen  = CelebaGenerator(z_dim=args.z_dim, hidden_dim=args.hidden_dim).to(device)
    #     disc = CelebaDiscriminator(im_chan=1, hidden_dim=args.hidden_dim).to(device)
    
    # Initialize the optimizers for generator and discriminator
    gen_opt  = torch.optim.Adam(gen.parameters(),  lr=args.lr, betas=(args.beta_1, args.beta_2))
    disc_opt = torch.optim.Adam(disc.parameters(), lr=args.lr, betas=(args.beta_1, args.beta_2))

    gen = gen.apply(weights_init)
    disc = disc.apply(weights_init)

    criterion = nn.BCEWithLogitsLoss()

    # Call the train function
    train(n_epochs=50, 
          z_dim=args.z_dim,
          device=device, 
          dataset="mnist" 
    )