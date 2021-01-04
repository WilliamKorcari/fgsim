import torchvision.datasets as datasets
import torchvision.transforms as transforms

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ]
)


train_data = datasets.FashionMNIST(
    root="input/data", train=True, download=True, transform=transform
)
