# WGAN-GP Implementation

## Generator Class
A Deep Convolutional WGAN generator that transforms noise into images.

### Parameters
- `z_dim` (int, default=10): Dimension of the noise vector input
- `im_chan` (int, default=1): Number of channels in output images  
- `hidden_dim` (int, default=64): Base dimension for hidden layers

### Architecture
Sequential blocks:
1. z_dim → hidden_dim * 4
2. hidden_dim * 4 → hidden_dim * 2 (kernel_size=4, stride=1)
3. hidden_dim * 2 → hidden_dim
4. hidden_dim → im_chan ( kernel_size=4, final layer with Tanh)

### make_gen_block(kernel_size=3, stride=2)
Creates generator block with:
- Transposed convolution
- BatchNorm2d (except final layer)
- ReLU activation (Tanh for final layer)

## Critic Class
A Deep Convolutional WGAN critic that scores images.

### Parameters
- `im_chan` (int, default=1): Number of input image channels
- `hidden_dim` (int, default=64): Base dimension for hidden layers

### Architecture
Sequential blocks:
1. im_chan → hidden_dim
2. hidden_dim → hidden_dim * 2
3. hidden_dim * 2 → 1 (final layer)

### make_crit_block(kernel_size=4, stride=2)
Creates critic block with:
- Convolution
- BatchNorm2d (except final layer)
- LeakyReLU(0.2) activation (none for final layer)


## Core WGAN-GP Functions

### get_gradient(crit, real, fake, epsilon)
Computes gradient for gradient penalty calculation:
1. Creates mixed images by interpolating between real and fake images using epsilon that is a vector of length equal to the batch size
2. Calculates critic scores for mixed images
3. Returns gradient of scores with respect to mixed images

### gradient_penalty(gradient)
Calculates the gradient penalty to enforce the Lipschitz constraint:
1. Flattens the gradient per image so that the `2D` tensor is `batch_size x (n_c x h x w)`
2. Computes $L_2$ norm of each gradient
3. Calculates penalty as: $\lambda * (\|\nabla C(x)\|_2 - 1)^2$ where $\lambda$ is penalty weight (default=10)

### get_gen_loss(crit_fake_pred)
Computes generator loss:
- Returns negative mean of critic's scores on fake images
- Generator aims to maximize critic's scores (minimize negative scores)

### get_crit_loss(crit_fake_pred, crit_real_pred, gp, c_lambda)
Computes critic loss with Wasserstein distance and gradient penalty:
1. Calculates Wasserstein distance: mean(fake_scores) - mean(real_scores)
2. Adds gradient penalty weighted by c_lambda
3. Returns total loss that critic aims to minimize