# Generator Class

A Deep Convolutional GAN (DCGAN) generator that transforms random noise into images.

## Parameters
- `z_dim` (int, default=10): Dimension of the noise vector input
- `im_chan` (int, default=1): Number of channels in output images (1 for MNIST grayscale)
- `hidden_dim` (int, default=64): Base dimension for hidden layers

## Architecture
The generator consists of 4 sequential blocks:
1. Input noise → hidden_dim * 4
2. hidden_dim * 4 → hidden_dim * 2 (kernel_size=4, stride=1)
3. hidden_dim * 2 → hidden_dim
4. hidden_dim → im_chan (final layer with Tanh activation, kernel_size=4) 

## Key Methods
### make_gen_block()
Creates a generator block with:
- Transposed convolution
- Batch normalization (except final layer)
- ReLU activation (Tanh for final layer)

### unsqueeze_noise()
Reshapes noise tensor for processing:
- Input: (n_samples, z_dim)
- Output: (n_samples, z_dim, 1, 1)

### forward()
Performs forward pass:
1. Reshapes input noise
2. Passes through generator blocks
3. Returns generated images

## Helper Function
### get_noise()
Generates random noise vectors:
- Creates tensor of shape (n_samples, z_dim)
- Fills with values from normal distribution
- Returns tensor on specified device

# Discriminator Class

A Deep Convolutional GAN (DCGAN) discriminator that classifies images as real or fake.

## Parameters
- `im_chan` (int, default=1): Number of channels in input images (1 for MNIST grayscale)
- `hidden_dim` (int, default=16): Base dimension for hidden layers

## Architecture
The discriminator consists of 3 sequential blocks:
1. im_chan → hidden_dim
2. hidden_dim → hidden_dim * 2
3. hidden_dim * 2 → 1 (final layer)

## Key Methods
### make_disc_block()
Creates a discriminator block with:
- Convolutional layer
- Batch normalization (except final layer)
- LeakyReLU activation (slope=0.2, except final layer)

### forward()
Performs forward pass:
1. Takes image tensor as input
2. Passes through discriminator blocks
3. Returns flattened predictions (real/fake)