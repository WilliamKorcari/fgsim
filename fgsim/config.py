import os
from omegaconf import OmegaConf

from .cli import args


def get_device():
    import torch

    return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


device = get_device()

fn = f"wd/{args.tag}/config.yaml"
if not os.path.isfile(fn):
    fn = 'fgsim/default.yaml'

with open(fn, "r") as fp:
    fileconf = OmegaConf.load(fp)

conf = OmegaConf.merge(vars(args), fileconf)

