from functools import reduce

import torch.nn as nn

from ..config import conf

imgpixels = reduce(lambda a, b: a * b, conf["mapper"]["shape"][1:4])


class Generator(nn.Module):
    def __init__(self, nz):
        super(Generator, self).__init__()
        self.nz = conf["model"]["gan"]["nz"]
        self.main = nn.Sequential(
            nn.Linear(self.nz, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 100),
            nn.LeakyReLU(0.2),
            nn.Linear(100, imgpixels),
            nn.Tanh(),
        )

    def forward(self, x):
        # change the shape of the output to the shape of the calorimeter image
        # the first dimension = number of events is inferred by the -1 value
        return self.main(x).view(-1, *conf["mapper"]["shape"][1:]).float()
        # return(self.main(x).view(-1,*conf["mapper"]["shape"][1:]))


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.n_input = imgpixels
        self.main = nn.Sequential(
            nn.Linear(self.n_input, 100),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(100, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # flatten the image
        # pass the tensor through the discrimnator
        return self.main(x.view(-1, imgpixels).float())
