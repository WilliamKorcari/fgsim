from omegaconf import OmegaConf

from .cli import args


def get_device():
    import torch

    return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


device = get_device()


with open(f"wd/{args.tag}/config.yaml", "r") as fp:
    conf = OmegaConf.load(fp)
