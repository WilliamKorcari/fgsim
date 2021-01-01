import torchvision.transforms as transforms
import torchvision.datasets as datasets

from .config import *

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ]
)


train_data = datasets.FashionMNIST(
    root="input/data", train=True, download=True, transform=transform
)
