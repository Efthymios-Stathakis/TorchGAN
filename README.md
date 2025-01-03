# GAN Models Project

This project implements various Generative Adversarial Networks (GANs) and related architectures for image generation and segmentation tasks. Below is an overview of the different models, utilities, and instructions on how to execute them.

## Models

1. **DCGAN - Deep Convolutional GAN** [paper](https://arxiv.org/abs/1511.06434):
   - A GAN architecture that uses deep convolutional networks for both the generator and discriminator. It is designed for generating high-quality images.

2. **ACGAN - Auxiliary Classifier GAN** [paper](https://arxiv.org/abs/1610.09585):
   - An extension of the GAN that includes an auxiliary classifier in the discriminator. This allows the model to generate images conditioned on class labels.

3. **Conditional GAN** ([paper](https://arxiv.org/abs/1411.1784)):
   - A GAN that generates images based on specific conditions, such as class labels or attributes. It can be used for tasks like image-to-image translation.

4. **U-Net** ([paper](https://arxiv.org/abs/1505.04597v1)):
   - A convolutional network architecture primarily used for image segmentation tasks. It features a contracting path to capture context and a symmetric expanding path for precise localization.

5. **Pix2Pix** ([paper](https://arxiv.org/abs/1611.07004)):
   - A conditional GAN architecture designed for image-to-image translation tasks. It learns a mapping from input images to output images, making it suitable for tasks like converting photographs to sketches.

## Utilities

The project includes utility modules that provide base classes for the generator and discriminator components of the GANs:

- **Generator Utilities**:
  - `GenConvTransposeBlock`: A transposed convolution block used in the generator, which performs upsampling followed by batch normalization and activation.

- **Discriminator Utilities**:
  - `DiscConvBlock`: A convolutional block used in the discriminator, which consists of a convolutional layer followed by batch normalization and LeakyReLU activation.

## Training a Model

To train the individual models, you can use the provided `train_GAN.sh` script. Here’s how to run it, with default parameters:

1. **Make the script executable**:
   ```bash
   chmod +x train_GAN.sh
   ```

2. **Run the script with the desired parameters**:
   ```bash
   ./train_GAN.sh [n_epochs] [z_dim] [batch_size] [hidden_dim] [lr] [beta_1] [beta_2] [model_type] [dataset] [display_step]
   ```

   - **Parameters**:
     - `n_epochs`: Number of epochs to train (default: 100).
     - `z_dim`: Dimension of the noise vector (default: 128).
     - `batch_size`: Batch size for training (default: 128).
     - `hidden_dim`: Hidden dimension for the models (default: 64).
     - `lr`: Learning rate for the optimizers (default: 0.0002).
     - `beta_1`: Beta 1 for Adam optimizer (default: 0.5).
     - `beta_2`: Beta 2 for Adam optimizer (default: 0.999).
     - `model_type`: Type of model to train (options: `DCGAN`, `ACGAN`, `ConditionalGAN`).
     - `dataset`: Dataset to load (options: `mnist`, `celeba`). 
     - `display_step`: How often to display training progress (default: 2500).

The parameters can be specified directly in the script file or defined in the terminal, as shown in the example below.

### Example Command
To train a Conditional GAN on the CelebA dataset for 100 epochs:

```bash
./train_GAN.sh 100 128 128 64 0.0002 0.5 0.999 ConditionalGAN celeba 2500
```

