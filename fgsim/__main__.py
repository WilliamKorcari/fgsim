"""Main module."""
import torch
import torch.optim as optim
import numpy as np
import pretty_errors

pretty_errors.configure(
    separator_character="*",
    filename_display=pretty_errors.FILENAME_EXTENDED,
    line_number_first=True,
    display_link=True,
    lines_before=5,
    lines_after=2,
    line_color=pretty_errors.RED + "> " + pretty_errors.default_config.line_color,
    code_color="  " + pretty_errors.default_config.line_color,
    truncate_code=True,
    display_locals=True,
)


from .cli import args

if args.command == "geo":
    from .geo import mapper

    foo = mapper.geomapper("data/test.toml")


if args.command == "train":
    from .config import device,nz
    from .model import Generator, Discriminator
    from .data_loader import train_data

    generator = Generator(nz).to(device)
    discriminator = Discriminator().to(device)

    print("##### GENERATOR #####")
    print(generator)
    print("######################")

    print("\n##### DISCRIMINATOR #####")
    print(discriminator)
    print("######################")

    # optimizers
    optim_g = optim.Adam(generator.parameters(), lr=0.0002)
    optim_d = optim.Adam(discriminator.parameters(), lr=0.0002)

    # loss function
    criterion = torch.nn.BCELoss()

    from .train import training_procedure

    generator, discriminator, images = training_procedure(
        generator, discriminator, optim_g, optim_d, criterion, train_data
    )

    print("DONE TRAINING")
    torch.save(generator.state_dict(), "output/generator.pth")

    from .data_dumper import generate_gif

    generate_gif(images)
