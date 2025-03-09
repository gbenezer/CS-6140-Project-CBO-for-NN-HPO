# Functions to output custom neural networks using PyTorch
# current restrictions are:
# - 3 layers (cannot vary due to hierarchical nature of hyperparameter)
# - uniform activation function across all layers
# - only two supported activation functions (Sigmoid and Relu)
# - feedforward

# TODO: finish functionality

# import statements
from torch import nn

def create_ff_network(
    current_device: str,
    number_input_features: int,
    number_output_features: int,
    input_dropout_probability: float,
    hidden_dropout_probability: float,
    output_dropout_probability: float,
    hidden_layer_nodes_1: int,
    hidden_layer_nodes_2: int,
    hidden_layer_nodes_3: int,
    relu: bool
) -> nn.Module:
    
    # TODO: implement better activation function logic
    
    if relu:
        activation = nn.ReLU()
    else:
        activation = nn.Sigmoid()

    class customNN(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.flatten = nn.Flatten()
            self.input_dropout = nn.Dropout(
                p=input_dropout_probability
            )
            self.stack = nn.Sequential(
                nn.Linear(in_features=number_input_features, out_features=hidden_layer_nodes_1),
                activation,
                nn.Dropout(p=hidden_dropout_probability),
                nn.Linear(in_features=hidden_layer_nodes_1, out_features=hidden_layer_nodes_2),
                activation,
                nn.Dropout(p=hidden_dropout_probability),
                nn.Linear(in_features=hidden_layer_nodes_2, out_features=hidden_layer_nodes_3),
                activation,
                nn.Dropout(p=output_dropout_probability),
                nn.Linear(in_features=hidden_layer_nodes_3, out_features=number_output_features)
            )
            
        def forward(self, x):
            x = self.flatten(x) # flatten input to 1D vector
            x = self.input_dropout(x) # dropout random amount of input
            x = self.stack(x)
            return x
    
    output_model = customNN().to(device=current_device)
    return output_model