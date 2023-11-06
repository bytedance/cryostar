from typing import Union, List

import torch
from torch import nn
# from utils.layers import Linear  # Why this?
from torch.nn import Linear


class ResLinear(nn.Module):

    def __init__(self, in_chs, out_chs):
        super().__init__()
        self.linear = Linear(in_chs, out_chs)

    def forward(self, x):
        return self.linear(x) + x


class MLP(nn.Module):

    def __init__(self, in_dims: List[int], out_dims: List[int]):
        super().__init__()
        self.in_dims = in_dims
        self.out_dims = out_dims

        layers = []
        for (i, o) in zip(in_dims, out_dims):
            layers.append(ResLinear(i, o) if i == o else Linear(i, o))
            layers.append(nn.ReLU(inplace=True))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class Encoder(nn.Module):

    def __init__(self, in_dim: int, hidden_dim: Union[int, List[int]], out_dim: int, num_hidden_layers=3):
        super().__init__()
        self.in_dim = in_dim
        if isinstance(hidden_dim, int):
            self.hidden_dim = (hidden_dim, ) * num_hidden_layers
        elif isinstance(hidden_dim, (list, tuple)):
            assert len(hidden_dim) == num_hidden_layers
            self.hidden_dim = hidden_dim
        else:
            raise NotImplementedError
        self.out_dim = out_dim
        self.num_hidden_layers = num_hidden_layers

        self.input_layer = nn.Sequential(
            ResLinear(in_dim, self.hidden_dim[0]) if in_dim == self.hidden_dim[0] else Linear(
                in_dim, self.hidden_dim[0]), nn.ReLU(inplace=True))
        self.mlp = MLP(self.hidden_dim[:-1], self.hidden_dim[1:])

        self.output_layer = Linear(hidden_dim, out_dim)

    def forward(self, x):
        return self.output_layer(self.mlp(self.input_layer(x)))


class VAEEncoder(nn.Module):

    def __init__(self, in_dim: int, hidden_dim: Union[int, List[int]], out_dim: int, num_hidden_layers=3):
        super().__init__()
        self.in_dim = in_dim
        if isinstance(hidden_dim, int):
            self.hidden_dim = (hidden_dim, ) * num_hidden_layers
        elif isinstance(hidden_dim, (list, tuple)):
            assert len(hidden_dim) == num_hidden_layers
            self.hidden_dim = hidden_dim
        else:
            raise NotImplementedError
        self.out_dim = out_dim
        self.num_hidden_layers = num_hidden_layers

        self.input_layer = nn.Sequential(
            ResLinear(in_dim, self.hidden_dim[0]) if in_dim == self.hidden_dim[0] else Linear(
                in_dim, self.hidden_dim[0]), nn.ReLU(inplace=True))
        self.mlp = MLP(self.hidden_dim[:-1], self.hidden_dim[1:])

        self.mean_layer = Linear(self.hidden_dim[-1], out_dim)
        self.var_layer = Linear(self.hidden_dim[-1], out_dim)

    def forward(self, x):
        x = self.mlp(self.input_layer(x))
        mean = self.mean_layer(x)
        log_var = self.var_layer(x)
        return mean, log_var


class Decoder(nn.Module):

    def __init__(self, in_dim: int, hidden_dim: Union[int, List[int]], out_dim: int, num_hidden_layers=3):
        super().__init__()
        self.in_dim = in_dim
        if isinstance(hidden_dim, int):
            self.hidden_dim = (hidden_dim, ) * num_hidden_layers
        elif isinstance(hidden_dim, (list, tuple)):
            assert len(hidden_dim) == num_hidden_layers
            self.hidden_dim = hidden_dim
        else:
            raise NotImplementedError
        self.out_dim = out_dim
        self.num_hidden_layers = num_hidden_layers

        self.input_layer = nn.Sequential(
            ResLinear(in_dim, self.hidden_dim[0]) if in_dim == self.hidden_dim[0] else Linear(
                in_dim, self.hidden_dim[0]), nn.ReLU(inplace=True))
        self.mlp = MLP(self.hidden_dim[:-1], self.hidden_dim[1:])

        self.out_layer = Linear(self.hidden_dim[-1], out_dim)

    def forward(self, x):
        x = self.mlp(self.input_layer(x))
        return self.out_layer(x)


def reparameterize(mu, log_var):
    std = torch.exp(0.5 * log_var)
    eps = torch.randn_like(std)
    return mu + eps * std
