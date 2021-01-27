from torchvision.utils import save_image

from ..config import conf
from .holder import modelHolder
from .train import create_noise


def generation_procedure(c: modelHolder):
    # Create the noise vectors
    noise = create_noise(
        sample_size=conf["predict"]["nevents"], nz=conf["model"]["gan"]["nz"]
    )
    # create the final fake image for the epoch
    generated_img = c.generator(noise).cpu().detach()
    # shape : sample_size * x * y *z

    save_image(
        generated_img[0, :, :, 7], f"wd/{conf.tag}/gen_img{c.metrics['epoch']}.png"
    )
