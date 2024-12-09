import torch
import torch.nn as nn
import math
import torch.nn.functional as F

def positional_encoding(x, degree=1):
    """
    Apply positional encoding to the input tensor x.

    :param x: Input tensor of shape (batch_size, 2 | 3).
    :param degree: Degree of positional encoding.
    :return: Positional encoded tensor.
    """
    if degree < 1:
        return x

    pe = [x]
    for d in range(degree):
        for fn in [torch.sin, torch.cos]:
            pe.append(fn(2.0**d * math.pi * x))
    return torch.cat(pe, dim=-1)

class SemanticPredictor(nn.Module):
    def __init__(
        self, input_dim=3, camera_dim=12, output_dim=8, hidden_dim=256, num_layers=8, degree=1
    ):
        """
        Initialize the MLP with positional encoding.

        :param input_dim: Dimension of the input, default is 3 for a 3D point.
        :param camera_dim: Dimension of the camera extrinsics (9 for rotation + 3 for translation).
        :param output_dim: Dimension of the output, which is the dimension of learned semantic features.
        :param hidden_dim: Number of hidden units in each layer.
        :param num_layers: Number of fully-connected layers.
        :param degree: Degree of positional encoding.
        """
        super(SemanticPredictor, self).__init__()

        self.degree = degree
        self.input_dim = input_dim * (
            2 * degree + 1
        ) + camera_dim # adjust the input dimension for different positional encoding
        print(f"self.input_dim: {self.input_dim}")
        layers = []
        layers.append(nn.Linear(self.input_dim, hidden_dim))
        layers.append(nn.ReLU())

        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(hidden_dim, output_dim))

        self.model = nn.Sequential(*layers)

    def forward(self, x, camera_features):
        """
        Forward pass of the MLP.

        :param x: Input tensor of shape (batch_size, 3).
        :param camera_features: Camera features of shape (camera_dim).
        :return: Output tensor of shape (batch_size, output_dim) with probabilities.
        """
        x = positional_encoding(x, self.degree)
        # repeat camera features for each point
        camera_features = camera_features.unsqueeze(0).repeat(x.shape[0], 1).float()
        x = torch.cat([x, camera_features], dim=-1)
        output = self.model(x)
        return output