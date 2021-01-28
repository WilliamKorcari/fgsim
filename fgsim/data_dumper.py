# Uproot4 is yet to implement writing...
from collections import OrderedDict

import awkward as ak
import numpy as np
import uproot3 as uproot

from .config import conf

# import imageio
# import numpy as np
# import torchvision.transforms as transforms


# def generate_gif(images):
#     # save the generated images as GIF file
#     to_pil_image = transforms.ToPILImage()
#     imgs = [np.array(to_pil_image(img)) for img in images]
#     imageio.mimsave("outputs/generator_images.gif", imgs)


def dump_generated_events(arr: ak.Array):
    ak0_array = ak.to_awkward0(arr)
    fn = f"wd/{conf['tag']}/output.root"
    branchD = OrderedDict()
    branchD["eventNumber"] = int
    branchD["energy"] = np.float32
    for v in conf["mapper"]["xyz"]:
        branchD[v] = np.float32
    with uproot.recreate(fn) as file:
        file["tree1"] = uproot.newtree(branchD)
        for branch in branchD:
            file["tree1"].extend(
                {branch: ak0_array[branch], "n": ak0_array[branch].counts}
            )
