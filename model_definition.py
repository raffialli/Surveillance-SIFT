import torch.nn as nn

class NN(nn.Module):
    """
    Simple neural network with two fully connected layers.

    Parameters:
    - input_size (int): Number of input features.
    - num_classes (int): Number of output classes.

    The network consists of two linear layers. The first layer (fc1) 
    maps the input to a 50-dimensional space, and the second layer (fc2)
    maps these 50 dimensions to the number of output classes.
    """

    def __init__(self, input_size, num_classes):
        """
        Initializes the neural network layers.
        """
        super(NN, self).__init__()
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        """
        Defines the forward pass of the network.
        
        Args:
        - x (torch.Tensor): Input tensor.

        Returns:
        - torch.Tensor: Output tensor of the network.
        """
        x = self.fc1(x)
        x = self.fc2(x)
        return x
