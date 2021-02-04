import gc

import torch
# import torch.autograd.profiler as profiler
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..config import conf, device
from ..plot import plotlosses
from ..utils.memory import memGB, memReport
from .holder import modelHolder


# to create real labels (1s)
def label_real(size):
    data = torch.ones(size, 1)
    return data.to(device)


# to create fake labels (0s)
def label_fake(size):
    data = torch.zeros(size, 1)
    return data.to(device)


# function to create the noise vector
def create_noise(sample_size, nz):
    return torch.randn(sample_size, nz).to(device)


# function to train the discriminator network
def train_discriminator(discriminator, optimizer, criterion, data_real, data_fake):
    b_size = data_real.size(0)
    real_label = label_real(b_size)
    fake_label = label_fake(b_size)

    optimizer.zero_grad()

    output_real = discriminator(data_real)
    loss_real = criterion(output_real, real_label)

    output_fake = discriminator(data_fake)
    loss_fake = criterion(output_fake, fake_label)

    loss_real.backward()
    loss_fake.backward()
    optimizer.step()

    return loss_real + loss_fake


# function to train the generator network
def train_generator(generator, discriminator, optimizer, criterion, data_fake):
    b_size = data_fake.size(0)
    real_label = label_real(b_size)

    optimizer.zero_grad()

    output = discriminator(data_fake)
    loss = criterion(output, real_label)

    loss.backward()
    optimizer.step()

    return loss


def training_procedure(c: modelHolder):
    # Make the configuration locally available
    batch_size, n_epochs, k, nz, sample_size = (
        conf.model.gan[x] for x in ["batch_size", "n_epochs", "k", "nz", "sample_size"]
    )

    train_loader = DataLoader(c.train_data, batch_size=2, shuffle=True)

    # Initialize the training
    c.generator.train()
    c.discriminator.train()
    print(f"Starting with epoch {c.metrics['epoch'] + 1}")

    # Iterate over the Epochs
    for c.metrics["epoch"] in range(c.metrics["epoch"] + 1, n_epochs):
        loss_g = 0.0
        loss_d = 0.0
        n_iter = int(len(c.train_data) / batch_size)

        # Iterate over the batches
        for bi, data_real in tqdm(
            enumerate(train_loader),
            total=n_iter,
        ):
            # with profiler.profile(profile_memory=True, record_shapes=True) as prof:
            b_size = len(data_real)
            data_real = data_real.to(device)

            # Train the discriminator for k number of steps
            for _ in range(k):
                data_fake = c.generator(create_noise(b_size, nz)).detach()
                # train the discriminator network
                loss_d += train_discriminator(
                    c.discriminator,
                    c.optim_d,
                    c.criterion,
                    data_real,
                    data_fake,
                ).item()

            # Train the Generator
            data_fake = c.generator(create_noise(b_size, nz))
            # train the generator network
            loss_g += train_generator(
                c.generator, c.discriminator, c.optim_g, c.criterion, data_fake
            ).item()

            if bi % 3 == 0:
                gc.collect()
            # print(prof.key_averages().table(sort_by="cpu_memory_usage", row_limit=10))

        # save the generated torch tensor models to disk (img 1, 7)
        if c.metrics["epoch"] % 10 == 0:
            c.save_model()

        epoch_loss_g = loss_g / bi  # total generator loss for the epoch
        epoch_loss_d = loss_d / bi  # total discriminator loss for the epoch
        c.metrics["losses_g"].append(epoch_loss_g)
        c.metrics["losses_d"].append(epoch_loss_d)

        memory_used = round(memGB(), 2)
        c.metrics["memory"].append(memory_used)
        if conf["loglevel"] >= 2:
            memReport()
            print(f"memory used: {c.metrics['memory']}")

        print(
            f"Epoch { c.metrics['epoch'] }/{n_epochs}: "
            + f"Generator loss: {epoch_loss_g:.8f}, "
            + f"Discriminator loss: {epoch_loss_d:.8f}"
        )

        # Crash if there is a memory bug
        if memory_used > 25:
            from sys import exit

            exit(-1)
    plotlosses(c.metrics["losses_g"], c.metrics["losses_d"])
