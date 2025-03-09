# standard imports
import torch
from torchvision import datasets
from pathlib import Path
import torchvision.transforms.v2 as transforms
import torch.nn as nn
import torch.optim as optim
from torch.profiler import profile, record_function, ProfilerActivity

# adding all the modules and submodules to the path
import sys

sys.path.insert(0, "C:/Users/Gil/Documents/Repositories/Python/CS_6140/Project")

# defining the device
dev = "cuda"

# # importing the correct package defined functions
from src.network.create_network import create_network
from src.network.train_test_network import (
    train_network,
    test_network,
    train_test_network_loop,
)

testing_network = create_network(
    current_device=dev,
    number_input_features=784,
    number_output_features=10,
    input_dropout_probability=0.2,
    hidden_dropout_probability=0.5,
    output_dropout_probability=0.1,
    hidden_layer_nodes_1=500,
    hidden_layer_nodes_2=100,
    hidden_layer_nodes_3=50,
    relu=True,
)

# portions of code that need to be modularized in the network subpackage

# transform, data processing, and score definition (one file)
image_transform = transforms.Compose(
    [
        transforms.ToImage(),
        transforms.ToDtype(torch.float, scale=True),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ]
)

data_directory = Path(__file__).parent.parent.resolve() / "torch_data"

trainset = datasets.MNIST(
    root=data_directory, train=True, download=True, transform=image_transform
)

trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=64, shuffle=True)

testset = datasets.MNIST(
    root=data_directory, train=False, download=True, transform=image_transform
)

testloader = torch.utils.data.DataLoader(dataset=testset, batch_size=64, shuffle=True)

criterion = nn.CrossEntropyLoss()

# optimizer initialization (one file, needs separation due to hyperparameter inputs)
optimizer_ffn = optim.Adam(params=testing_network.parameters())

# training (already modular) (profiling is not)
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    with record_function("model_training"):
        epoch_loss, epoch_accuracy, average_train_loss = train_network(
            trainloader, testing_network, criterion, optimizer_ffn, dev, print_out=False
        )

print(prof.key_averages().table(sort_by="cuda_time_total"))

# # testing (already modular)

# test_loss, test_accuracy, average_test_loss = test_network(
#     testloader, testing_network, criterion, dev
# )

# print(test_loss)
# print(test_accuracy)
# print(average_test_loss)

# # full train/test network loop testing
# (
#     train_loss_list,
#     train_accuracy_list,
#     avg_train_loss_list,
#     test_loss_list,
#     test_accuracy_list,
#     avg_test_loss_list,
# ) = train_test_network_loop(
#     num_epochs=3,
#     train_loader=trainloader,
#     test_loader=testloader,
#     model=testing_network,
#     loss_fn=criterion,
#     optimizer=optimizer_ffn,
#     device=dev,
#     writer=None,
#     param_file=None,
#     model_name="Manual Testing Network",
#     print_out=True,
#     log_tb=False,
#     save_params=False,
# )
