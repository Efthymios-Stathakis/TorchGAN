import os
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as tf
from torchvision.datasets import MNIST
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging
import argparse

from ..vis_utils import show_img_batch
from ..utils import get_noise, weights_init, combine_tensors
from .mnist import (Generator as MnistGenerator, 
                    Discriminator as MnistDiscriminator)
from .celeba import (Generator as CelebaGenerator, 
                    Discriminator as CelebaDiscriminator,
                    CelebADataset)

# Set up logging
logging.basicConfig(level=logging.INFO)


def train(n_epochs, 
          z_dim=128,
          hidden_dim=64,
          batch_size=128,
          display_step=5000, 
          save_step=10000, 
          device="mps",
          dataset="mnist",
          opt_params={}):
    
    data_path = os.path.join(os.path.dirname(__file__), "../../datasets/")
    if dataset == "mnist":
        
        # Initialize the dataset and dataloader
        mnist_transforms = transforms.Compose([transforms.ToTensor(), 
                                               transforms.Normalize((0.5,), (0.5,))])
        torch_dt = MNIST(root=data_path, download=False, transform=mnist_transforms)
        torch_dl = DataLoader(torch_dt, batch_size=batch_size, shuffle=True)

        # Initialize the generator and discriminator
        size = (1, 28, 28)
        num_classes = 10
        
        gen  = MnistGenerator(z_dim=z_dim+num_classes  , hidden_dim=hidden_dim).to(device)
        disc = MnistDiscriminator(im_chan=1+num_classes, hidden_dim=hidden_dim).to(device)
        

    elif dataset == "celeba":
        
        # Read labels
        df_labels = pd.read_csv(f"{data_path}/celeba_align/list_attr_celeba.csv", 
                         delim_whitespace=True, index_col=0)
        df_labels = df_labels.replace(-1, 0)

        celeba_transforms = transforms.Compose([transforms.ToTensor(),
                                         transforms.CenterCrop((178, 178)),
                                         transforms.Resize((64, 64))])

        celeba = ImageFolder(f"{data_path}/celeba", transform=celeba_transforms)
        torch_dt = CelebADataset(celeba, df_labels)
        torch_dl = DataLoader(torch_dt, batch_size=batch_size, shuffle=True)
        
        # Initialize the generator and discriminator
        size = (3, 64, 64)
        num_classes = 40

        gen  = CelebaGenerator(z_dim=z_dim+num_classes  , im_chan=3, hidden_dim=hidden_dim).to(device)
        disc = CelebaDiscriminator(im_chan=3+num_classes, hidden_dim=hidden_dim).to(device)
        
    
    # Initialize the optimizers for generator and discriminator
    criterion = nn.BCEWithLogitsLoss()
    gen_opt  = torch.optim.Adam(gen.parameters(),  **opt_params)
    disc_opt = torch.optim.Adam(disc.parameters(), **opt_params)

    gen = gen.apply(weights_init)
    disc = disc.apply(weights_init)

    cur_step = 0
    generator_losses = []
    discriminator_losses = []

    for epoch in range(n_epochs):

        for real, labels in tqdm(torch_dl):
            cur_batch_size = len(real)
            real = real.to(device)

            ## Update discriminator ##
            disc_opt.zero_grad()

            if dataset == "mnist":
                one_hot_vec = tf.one_hot(labels.to(device), num_classes=num_classes) # Num_classes 
            elif dataset == "celeba":
                one_hot_vec = labels.float().to(device)
            one_hot_img = one_hot_vec[:,:,None,None].expand(-1, -1, *size[1:]) # (28, 28) for mnist, (64, 64) for Celeba

            # Get noise corresponding to the current batch_size
            noise = get_noise(cur_batch_size, z_dim, device=device)

            fake = gen(combine_tensors(noise, one_hot_vec))
            fake_comb = combine_tensors(fake, one_hot_img)
            disc_fake = disc(fake_comb.detach())
            disc_fake_loss = criterion(disc_fake, torch.zeros_like(disc_fake, device=device))          
            real_comb = combine_tensors(real, one_hot_img)
            disc_real = disc(real_comb)
            disc_real_loss = criterion(disc_real, torch.ones_like(disc_real, device=device))
            disc_loss = (disc_real_loss + disc_fake_loss) / 2

            disc_loss.backward(retain_graph=True)
            disc_opt.step()

            discriminator_losses += [disc_loss.item()]
            
            ### Update generator ###
            gen_opt.zero_grad()
            noise = get_noise(cur_batch_size, z_dim, device=device)
                    
            fake = gen(combine_tensors(noise, one_hot_vec))
            fake_comb = combine_tensors(fake, one_hot_img)
            gen_fake = disc(fake_comb)
            gen_fake_loss = criterion(gen_fake, torch.ones_like(gen_fake, device=device))
            gen_fake_loss.backward()
            gen_opt.step()  
            
            # Keep track of the average generator loss
            generator_losses += [gen_fake_loss.item()]

            ### Visualization code ###
            if cur_step % display_step == 0 and cur_step > 0:
                mean_gen_loss  = sum(generator_losses[-display_step:])/display_step
                mean_disc_loss = sum(discriminator_losses[-display_step:])/display_step
                logging.info(f"Epoch {epoch}, step {cur_step}: Generator loss: {mean_gen_loss:.2f}, discriminator loss: {mean_disc_loss:.2f}")
                show_img_batch((real + 1) / 2, size=size, save_path=f"./models/ConditionalGAN/figures/{dataset}_epoch_{epoch}_step_{cur_step}_real.jpg")
                show_img_batch((fake + 1) / 2, size=size, save_path=f"./models/ConditionalGAN/figures/{dataset}_epoch_{epoch}_step_{cur_step}_fake.jpg")
            if cur_step % save_step == 0 and cur_step > 0:
                assets_path = os.path.join(os.path.dirname(__file__), "../../assets/ConditionalGAN")
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
    parser.add_argument('--n_epochs',     type=int,   default=50,      help='Number of epochs to train')
    parser.add_argument('--z_dim',        type=int,   default=128,     help='Dimension of the noise vector')
    parser.add_argument('--batch_size',   type=int,   default=128,     help='Batch size for training')
    parser.add_argument('--hidden_dim',   type=int,   default=64,      help='Hidden dimension for the models')
    parser.add_argument('--lr',           type=float, default=0.0002,  help='Learning rate for the optimizers')
    parser.add_argument('--beta_1',       type=float, default=0.5,     help='Beta 1 for Adam optimizer')
    parser.add_argument('--beta_2',       type=float, default=0.999,   help='Beta 2 for Adam optimizer')
    parser.add_argument('--dataset',      type=str ,  default="mnist", help='Dataset to load')
    parser.add_argument('--display_step', type=int ,  default=5000,    help='How often to get real and fake samples')

    args = parser.parse_args()
    
    device = "mps" if torch.backends.mps.is_available() else "cpu"    
    if torch.cuda.is_available(): device = "cuda"

    cur_file = os.path.join(os.path.dirname(__file__))
    model = cur_file.split("/")[-1]
    logging.info(f"Training a {model} for the {args.dataset} dataset, on {device}.")

    # Call the train function
    train(
        n_epochs=args.n_epochs, 
        z_dim=args.z_dim,
        hidden_dim=args.hidden_dim,
        batch_size=args.batch_size,
        display_step=args.display_step,
        device=device, 
        dataset=args.dataset,
        opt_params= {
            "lr": args.lr, 
            "betas": (args.beta_1, args.beta_2)
            }    
    )