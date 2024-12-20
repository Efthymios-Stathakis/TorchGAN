from torchvision.utils import make_grid
import matplotlib.pyplot as plt

def show_img_batch(img_batch, n_images=32, nrow=8, size=(1,28,28), save_path=None):
    img_batch = img_batch.detach().cpu().view(-1, *size)
    img_grid = make_grid(img_batch[:n_images], nrow=nrow)
    if save_path:
        plt.ioff()
        plt.imshow(img_grid.permute(1,2,0).squeeze())
        plt.savefig(save_path)
    else:
        plt.imshow(img_grid.permute(1,2,0).squeeze())
        plt.show()