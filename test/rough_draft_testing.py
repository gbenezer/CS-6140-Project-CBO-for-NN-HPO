# standard imports

# adding all the modules and submodules to the path
import sys
sys.path.insert(0, "C:/Users/Gil/Documents/Repositories/Python/CS_6140/Project")

# # importing the correct package defined functions
from src.network.create_network import create_network

test_network = create_network(current_device="cuda",
                              number_input_features=784,
                              number_output_features=10,
                              input_dropout_probability=0.2,
                              hidden_dropout_probability=0.5,
                              output_dropout_probability=0.1,
                              hidden_layer_nodes_1=500,
                              hidden_layer_nodes_2=100,
                              hidden_layer_nodes_3=50,
                              relu=True)

print(test_network)

# data_directory = Path(__file__).parent.parent.resolve() / "torch_data"

# trainset = datasets.MNIST(
#     root=data_directory, train=True, download=True, transform=ToTensor()
# )

# trainset = datasets.MNIST(
#     root=data_directory, train=False, download=True, transform=ToTensor()
# )