# standard imports
import torch
from torchvision import datasets
import torch.utils.data as data
from pathlib import Path
from torchvision.transforms import ToTensor
import torchvision.transforms.v2 as transforms
import torch.nn as nn
import torch.optim as optim
import lightning as L
import time
from multiprocessing import Process, freeze_support
from torch.profiler import profile, ProfilerActivity
from lightning.pytorch.profilers import (
    SimpleProfiler,
    AdvancedProfiler,
    PyTorchProfiler,
)
from lightning.pytorch.callbacks import ModelSummary
from torch.utils.tensorboard import SummaryWriter

# reducing some of the precision to hopefully increase performance
torch.set_float32_matmul_precision("high")

# adding all the modules and submodules to the path
import sys

sys.path.insert(0, "C:/Users/Gil/Documents/Repositories/Python/CS_6140/Project")

# defining the device
dev = "cuda"

# # importing the correct package defined functions
from src.network.create_network import create_ff_network
from src.network.train_test_network import (
    train_network,
    test_network,
    train_test_network_loop,
)
from src.network.create_network_lightning import create_ff_classifier

import logging
import warnings

warnings.filterwarnings("ignore")  # Disable data logger warnings
logging.getLogger("lightning").setLevel(logging.ERROR)  # Disable GPU/TPU prints

testing_network = create_ff_network(
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
params_directory = Path(__file__).parent.parent.resolve() / "param_files"
tb_directory = Path(__file__).parent.parent.resolve() / "tb_files"

trainset = datasets.MNIST(
    root=data_directory, train=True, download=True, transform=image_transform
)

testset = datasets.MNIST(
    root=data_directory, train=False, download=True, transform=image_transform
)

testloader = torch.utils.data.DataLoader(dataset=testset, num_workers=15, batch_size=64)

# use 20% of training data for validation
train_set_size = int(len(trainset) * 0.8)
valid_set_size = len(trainset) - train_set_size

# split the train set into two
seed = torch.Generator().manual_seed(42)
train_set, valid_set = data.random_split(
    trainset, [train_set_size, valid_set_size], generator=seed
)
trainloader = torch.utils.data.DataLoader(
    dataset=train_set,
    num_workers=15,
    batch_size=64,
    shuffle=True,
    persistent_workers=True,
)
validloader = torch.utils.data.DataLoader(
    dataset=valid_set, num_workers=15, batch_size=64, persistent_workers=True
)

# criterion = nn.CrossEntropyLoss()

# optimizer initialization (one file, needs separation due to hyperparameter inputs)
# optimizer_ffn = optim.Adam(params=testing_network.parameters())

# run = "run_3"

# test_writer = SummaryWriter(tb_directory / run)

# # full train/test network loop testing (modular)
# if __name__ == "__main__":
#     freeze_support()
#     print("starting training and testing")
#     (
#         train_loss_list,
#         train_accuracy_list,
#         avg_train_loss_list,
#         train_duration_list,
#         cumulative_train_duration_list,
#         test_loss_list,
#         test_accuracy_list,
#         avg_test_loss_list,
#         test_duration_list,
#         cumulative_test_duration_list
#     ) = train_test_network_loop(
#         num_epochs=20,
#         train_loader=trainloader,
#         test_loader=validloader,
#         model=testing_network,
#         loss_fn=criterion,
#         optimizer=optimizer_ffn,
#         device=dev,
#         writer=test_writer,
#         param_file= params_directory / (run + ".pth"),
#         model_name="Manual Testing Network",
#         print_out=False,
#         log_tb=True,
#         save_params=True,
#     )
#     print("ending training and testing")
#     total_params = sum(p.numel() for p in testing_network.parameters() if p.requires_grad)
#     print(f'Total number of parameters: {total_params}')
#     print("Training durations, seconds:", train_duration_list)
#     print("Test durations, seconds:", test_duration_list)

# # profiler testing (needs modularization)
# with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, profile_memory=True) as prof:
#     test_loss, test_accuracy, average_test_loss = test_network(dataloader=testloader,
#                                                                model=testing_network,
#                                                                loss_fn=criterion,
#                                                                device=dev)

# for fEventAvg in prof.key_averages():
#     if fEventAvg.key != "[memory]":
#         print("Total CPU Time for", fEventAvg.key, ":", fEventAvg.cpu_time_total, sep=" ")
#         print("Total CUDA Time for", fEventAvg.key, ":", fEventAvg.device_time_total, sep=" ")
#         print("Total CPU Memory for", fEventAvg.key, ":", fEventAvg.cpu_memory_usage, sep=" ")
#         print("Total CUDA Memory for", fEventAvg.key, ":", fEventAvg.device_memory_usage, sep=" ")

# print(prof.key_averages().table())

# found method to get memory and time used during inference, so likely will try for a three-objective optimization
# overhead is high, so may not be effective for training

# # testing Lightning framework

test_lighting_module = create_ff_classifier(
    number_input_features=784,
    number_output_features=10,
    input_dropout_probability=0.2,
    hidden_dropout_probability=0.5,
    output_dropout_probability=0.1,
    hidden_layer_nodes_1=500,
    hidden_layer_nodes_2=100,
    hidden_layer_nodes_3=50,
    activation=nn.ReLU(),
    loss=nn.CrossEntropyLoss(),
    learning_rate=1e-3,
    beta1=0.9,
    beta2=0.999,
    w_decay=0,
)

# testing_lightning_network = create_ff_pl_network(
#     loss=criterion,
#     number_input_features=784,
#     number_output_features=10,
#     input_dropout_probability=0.2,
#     hidden_dropout_probability=0.5,
#     output_dropout_probability=0.1,
#     hidden_layer_nodes_1=500,
#     hidden_layer_nodes_2=100,
#     hidden_layer_nodes_3=50,
#     relu=True,
#     learning_rate=0.001,
#     beta1=0.9,
#     beta2=0.999,
#     w_decay=0,
# )

# testing_lightning_network_2 = create_ff_pl_network_2(
#     784, 10, 0.2, 0.5, 0.1, 500, 100, 50, True, criterion, 1e-3, 0.9, 0.999, 0
# )

simple_profiler = SimpleProfiler(filename="fit_profiling_output")
trainer = L.Trainer(
    max_epochs=20, enable_progress_bar=True, logger=True, profiler=simple_profiler
)

if __name__ == "__main__":
    freeze_support()

    import logging

    logging.getLogger("lightning").setLevel(logging.ERROR)

    # configure logging at the root level of Lightning
    logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)

    # configure logging on module level, redirect to file
    logger = logging.getLogger("lightning.pytorch.core")
    logger.addHandler(logging.FileHandler("core.log"))

    trainer.fit(
        model=test_lighting_module,
        train_dataloaders=trainloader,
        val_dataloaders=validloader,
    )

    trainer.test(model=test_lighting_module, dataloaders=testloader)
