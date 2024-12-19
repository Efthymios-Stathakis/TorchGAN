The Generator class is a neural network designed to generate synthetic images:

- **Input**: Takes a combined vector of noise and class information (input_dim)
- **Architecture**:
  - Params: `input_dim=10, im_chan=1, hidden_dim=64`
  - Uses sequential convolutional transpose layers
  - Has 4 generator blocks with dimensions:
    1. input_dim → hidden_dim * 4
    2. hidden_dim * 4 → hidden_dim * 2, kernel_size=4, stride=1
    3. hidden_dim * 2 → hidden_dim
    4. hidden_dim → im_chan (output channels), kernel_size=4
  - Each block (except final) contains:
    - Transposed convolution
    - Batch normalization
    - ReLU activation
    - Defaults `kernel_size=3, stride=2`
  - Final block uses Tanh activation and no Batch normalization
- **Output**: Generated images with shape (batch_size, im_chan, height, width)

The Discriminator class evaluates whether images are real or fake:

- **Input**: Takes images with class information as additional channels
- **Architecture**: 
  - Params: `im_chan=1, hidden_dim=64`
  - Uses sequential convolutional layers
  - Has 3 discriminator blocks:
    1. im_chan → hidden_dim
    2. hidden_dim → hidden_dim * 2
    3. hidden_dim * 2 → 1 (output)
  - Each block (except final) contains:
    - Convolution
    - Batch normalization  
    - LeakyReLU activation (default=0.2)
    - Defaults `kernel_size=4, stride=2`
  - Final block uses only Convolution
- **Output**: Classification scores for real/fake images

get_input_dimensions():
- Purpose: Calculates input dimensions for generator and discriminator
- Parameters:
  - z_dim: noise vector dimension
  - mnist_shape: image shape (channels, height, width)
  - n_classes: number of classes
- Returns:
  - generator_input_dim: z_dim + n_classes
  - discriminator_im_chan: mnist_shape[0] + n_classes

combine_vectors():
- Purpose: Concatenates noise and class vectors
- Parameters:
  - x: First vector (noise)
  - y: Second vector (class one-hot)
- Returns: Combined vector as float tensor

- criterion: BCEWithLogitsLoss
- n_epochs: 200
- z_dim: 64 (noise dimension)
- display_step: 2500 (visualization frequency)
- batch_size: 128
- learning_rate: 0.0002
- device: MPS if available, else CPU
- dataset: MNIST with normalization transforms