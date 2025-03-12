# file to define how the data are imported as DataSets and DataLoaders
import torchvision.transforms.v2 as transforms
import torch.utils.data as data
import torch
from torchvision import datasets
from pathlib import Path


def get_MNIST_data(
    valid_fraction: float,
    random_seed: int,
    n_workers: int,
    batch_n: int,
    download_data: bool,
):

    # define the image transformation
    image_transform = transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float, scale=True),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
    )

    # define the data directory and getting the datasets from the directory
    data_directory = Path(__file__).parent.parent.parent.resolve() / "torch_data"
    trainset = datasets.MNIST(
        root=data_directory, train=True, download=download_data, transform=image_transform
    )
    testset = datasets.MNIST(
        root=data_directory, train=False, download=download_data, transform=image_transform
    )

    # creating the test data DataLoader
    testloader = torch.utils.data.DataLoader(
        dataset=testset, num_workers=n_workers, batch_size=batch_n, persistent_workers=True
    )

    # splitting off the validation set from the larger training set
    valid_set_size = int(len(trainset) * valid_fraction)
    train_set_size = len(trainset) - valid_set_size
    seed = torch.Generator().manual_seed(random_seed)
    train_set, valid_set = data.random_split(
        trainset, [train_set_size, valid_set_size], generator=seed
    )

    # defining the training data and validation data DataLoader
    trainloader = torch.utils.data.DataLoader(
        dataset=train_set,
        num_workers=n_workers,
        batch_size=batch_n,
        shuffle=True,
        persistent_workers=True,
    )

    validloader = torch.utils.data.DataLoader(
        dataset=valid_set,
        num_workers=n_workers,
        batch_size=batch_n,
        persistent_workers=True,
    )

    return train_set, valid_set, testset, trainloader, validloader, testloader
