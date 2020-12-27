import numpy as np
import imageio

import torchvision.transforms as transforms


def generate_gif(images):
    # save the generated images as GIF file
    to_pil_image = transforms.ToPILImage()
    imgs = [np.array(to_pil_image(img)) for img in images]
    imageio.mimsave("outputs/generator_images.gif", imgs)
