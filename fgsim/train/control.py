import os

import torch
import torch.optim as optim

from ..config import conf, device
from ..data_loader import eventarr, posD
from ..geo.mapper import Geomapper
from .model import Discriminator, Generator


class traincac:
    def __init__(self) -> None:
        mapper = Geomapper(posD)
        self.train_data = mapper.map_events(eventarr)

        self.pathD = {
            n: "wd/{conf.tag}/{n}.torch" for n in ["generator", "discriminator"]
        }

        self.load_model()

        print("##### GENERATOR #####")
        print(self.generator)
        print("######################")

        print("\n##### DISCRIMINATOR #####")
        print(self.discriminator)
        print("######################")

        # optimizers
        self.optim_g = optim.Adam(self.generator.parameters(), lr=conf.model.gan.lr)
        self.optim_d = optim.Adam(self.discriminator.parameters(), lr=conf.model.gan.lr)
        # loss function
        self.criterion = torch.nn.BCELoss()

    def load_model(self):
        self.discriminator = Discriminator().to(device)
        if os.path.isfile(self.pathD["discriminator"]):
            self.discriminator.load_state_dict(torch.load(self.pathD["discriminator"]))

        self.generator = Generator(conf.model.gan.nz).to(device)
        if os.path.isfile(self.pathD["generator"]):
            self.generator.load_state_dict(torch.load(self.pathD["generator"]))

    def save_model(self):
        torch.save(self.discriminator.state_dict(), self.pathD["discriminator"])
        torch.save(self.generator.state_dict(), self.pathD["generator"])

    def run_training(self):
        from .train import training_procedure

        self.generator, self.discriminator, images = training_procedure(self)

        print("DONE TRAINING")
        torch.save(self.generator.state_dict(), "output/generator.pth")

        from ..data_dumper import generate_gif

        generate_gif(images)
