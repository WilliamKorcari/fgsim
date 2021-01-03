# learning parameters
batch_size = 512
epochs = 200
sample_size = 64  # fixed sample size
nz = 128  # latent vector size
k = 1  # number of steps to apply to the discriminator


def get_device():
    import torch

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


device = get_device()
