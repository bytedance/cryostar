import math
import functools

import numpy as np

import einops
import torch
from torch import nn


def init_weights_requ(m):
    if type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            nn.init.kaiming_normal_(m.weight, a=0.0, nonlinearity='relu', mode='fan_in')
        # if hasattr(m, 'bias'):
        #     nn.init.uniform_(m.bias, -.5,.5)


def init_weights_normal(m):
    if type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            nn.init.kaiming_normal_(m.weight, a=0.0, nonlinearity='relu', mode='fan_out')
        if hasattr(m, 'bias'):
            nn.init.uniform_(m.bias, -1, 1)
            # m.bias.data.fill_(0.)


def init_weights_selu(m):
    if type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            nn.init.normal_(m.weight, std=1 / math.sqrt(num_input))
        # if hasattr(m, 'bias'):
        #     m.bias.data.fill_(0.)


def init_weights_elu(m):
    if type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            nn.init.normal_(m.weight, std=math.sqrt(1.5505188080679277) / math.sqrt(num_input))
        # if hasattr(m, 'bias'):
        #     m.bias.data.fill_(0.)


def init_weights_xavier(m):
    if type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            nn.init.xavier_normal_(m.weight)
        if hasattr(m, 'bias'):
            m.bias.data.fill_(0.)


def sine_init(m, w0=30):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            m.weight.uniform_(-np.sqrt(6 / num_input) / w0, np.sqrt(6 / num_input) / w0)


def first_layer_sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            m.weight.uniform_(-1 / num_input, 1 / num_input)


class FirstSine(nn.Module):

    def __init__(self, w0=20):
        """
        Initialization of the first sine nonlinearity.

        Parameters
        ----------
        w0: float
        """
        super().__init__()
        self.w0 = torch.tensor(w0)

    def forward(self, input):
        return torch.sin(self.w0 * input)


class Sine(nn.Module):

    def __init__(self, w0=20.0):
        """
        Initialization of sine nonlinearity.

        Parameters
        ----------
        w0: float
        """
        super().__init__()
        self.w0 = torch.tensor(w0)

    def forward(self, input):
        return torch.sin(self.w0 * input)


class Swish(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input * torch.sigmoid(input)


def layer_factory(layer_type):
    layer_dict = \
        {'relu': (nn.ReLU(inplace=True), init_weights_normal),
         'sigmoid': (nn.Sigmoid(), init_weights_xavier),
         'sine': (Sine(), sine_init),
         'gelu': (nn.GELU(), init_weights_selu),
         'swish': (Swish(), init_weights_selu),
         }
    return layer_dict[layer_type]


class ResLayer(nn.Module):

    def __init__(self, layer, nl) -> None:
        super().__init__()
        self.layer = layer
        self.nl = nl

    def forward(self, x):
        return x + self.nl(self.layer(x))


class PositionalEncoding(nn.Module):

    def __init__(self, pe_dim=6, D=128, pe_type="geo", include_input=True):
        """
        Initilization of a positional encoder.

        Parameters
        ----------
        num_encoding_functions: int
        include_input: bool
        normalize: bool
        input_dim: int
        gaussian_pe: bool
        gaussian_std: float
        """
        super().__init__()
        self.pe_dim = pe_dim
        self.include_input = include_input
        self.pe_type = pe_type
        self.D = D

        if self.pe_type == "gau1":
            self.gaussian_weights = nn.Parameter(torch.randn(3 * self.pe_dim, 3) * D / 4, requires_grad=False)
        elif self.pe_type == "gau2":
            # FTPositionalDecoder (https://github.com/zhonge/cryodrgn/blob/master/cryodrgn/models.py)
            # __init__():
            #   rand_freqs = randn * 0.5
            # random_fourier_encoding():
            #   freqs = rand_freqs * coords * D/2
            # decode()/eval_volume():
            #   extent < 0.5 -> coords are in (-0.5, 0.5), while in cryostar coords are in (-1, 1)
            self.gaussian_weights = nn.Parameter(torch.randn(3 * self.pe_dim, 3) * D / 8, requires_grad=False)
        elif self.pe_type == "geo1":
            # frequency: (1, D), wavelength: (2pi/D, 2pi)
            f = D
            self.frequency_bands = nn.Parameter(f * (1. / f)**(torch.arange(self.pe_dim) / (self.pe_dim - 1)),
                                                requires_grad=False)
        elif self.pe_type == "geo2":
            # frequency: (1, D*pi)
            f = D * np.pi
            self.frequency_bands = nn.Parameter(f * (1. / f)**(torch.arange(self.pe_dim) / (self.pe_dim - 1)),
                                                requires_grad=False)
        elif self.pe_type == "no":
            pass
        else:
            raise NotImplemented

    def __repr__(self):
        return str(self.__class__.__name__) + f"({self.pe_type}, num={self.pe_dim})"

    def out_dim(self):
        if self.pe_type == "no":
            return 3
        else:
            ret = 3 * 2 * self.pe_dim
            if self.include_input:
                ret += 3
            return ret

    def forward(self, tensor) -> torch.Tensor:
        with torch.autocast("cuda", enabled=False):
            assert tensor.dtype == torch.float32
            if self.pe_type == "no":
                return tensor

            encoding = [tensor] if self.include_input else []
            if "gau" in self.pe_type:
                x = torch.matmul(tensor, self.gaussian_weights.T)
                encoding.append(torch.cos(x))
                encoding.append(torch.sin(x))
            elif "geo" in self.pe_type:
                bsz, num_coords, _ = tensor.shape
                x = self.frequency_bands[None, None, None, :] * tensor[:, :, :, None]
                x = x.reshape(bsz, num_coords, -1)
                encoding.append(torch.cos(x))
                encoding.append(torch.sin(x))

            ret = torch.cat(encoding, dim=-1)

        return ret


class FCBlock(nn.Module):

    def __init__(self,
                 in_features,
                 features,
                 out_features,
                 nonlinearity='gelu',
                 last_nonlinearity=None,
                 batch_norm=False,
                 use_residual=False):
        """
        Initialization of a fully connected network.

        Parameters
        ----------
        in_features: int
        features: list
        out_features: int
        nonlinearity: str
        last_nonlinearity: str
        batch_norm: bool
        """
        super().__init__()

        # Create hidden features list
        self.hidden_features = [int(in_features)]
        if features != []:
            self.hidden_features.extend(features)
        self.hidden_features.append(int(out_features))

        self.use_residual = use_residual
        self.net = []
        for i in range(len(self.hidden_features) - 1):
            hidden = False
            if i < len(self.hidden_features) - 2:
                if nonlinearity is not None:
                    nl = layer_factory(nonlinearity)[0]
                    init = layer_factory(nonlinearity)[1]
                hidden = True
            else:
                if last_nonlinearity is not None:
                    nl = layer_factory(last_nonlinearity)[0]
                    init = layer_factory(last_nonlinearity)[1]

            layer = nn.Linear(self.hidden_features[i], self.hidden_features[i + 1])

            if (hidden and (nonlinearity is not None)) or ((not hidden) and (last_nonlinearity is not None)):
                init(layer)
                if self.use_residual and layer.in_features == layer.out_features:
                    self.net.append(ResLayer(layer, nl))
                else:
                    self.net.append(layer)
                    self.net.append(nl)
            else:
                # init_weights_normal(layer)
                self.net.append(layer)
            if hidden:
                if batch_norm:
                    self.net.append(nn.BatchNorm1d(num_features=self.hidden_features[i + 1]))

        self.net = nn.Sequential(*self.net)

    def forward(self, x):
        output = self.net(x)
        return output


class SIREN(nn.Module):

    def __init__(self, in_features, out_features, num_hidden_layers, hidden_features, outermost_linear=False, w0=30.0):
        """
        Initialization of a SIREN.

        Parameters
        ----------
        in_features: int
        out_features: int
        num_hidden_layers: int
        hidden_features: int
        outermost_linear: bool
        w0: float
        """
        super(SIREN, self).__init__()

        nl = Sine(w0)
        first_nl = FirstSine(w0)
        self.weight_init = functools.partial(sine_init, w0=w0)
        self.first_layer_init = first_layer_sine_init

        self.net = []
        self.net.append(nn.Sequential(nn.Linear(in_features, hidden_features), first_nl))

        for i in range(num_hidden_layers):
            self.net.append(nn.Sequential(nn.Linear(hidden_features, hidden_features), nl))

        if outermost_linear:
            self.net.append(nn.Sequential(nn.Linear(hidden_features, out_features), ))
        else:
            self.net.append(nn.Sequential(nn.Linear(hidden_features, out_features), nl))

        self.net = nn.Sequential(*self.net)
        if self.weight_init is not None:
            self.net.apply(self.weight_init)

        if self.first_layer_init is not None:
            self.net[0].apply(self.first_layer_init)

    def forward(self, x):
        output = self.net(x)
        return output


class FourierNet(nn.Module):

    def __init__(self, net_type, z_dim, pe_dim, pe_type, D, layer=3, hidden_dim=256, force_symmetry=False):
        super().__init__()

        self.pe = PositionalEncoding(D=D, pe_dim=pe_dim, pe_type=pe_type)
        in_features = self.pe.out_dim() + z_dim

        self.force_symmetry = force_symmetry
        if force_symmetry:
            self.symmetrizer = Symmetrizer()

        self.net_type = net_type
        if net_type == "cryoai":
            out_features = 2
            self.net_modulant = SIREN(in_features, out_features, layer, hidden_dim, outermost_linear=True, w0=40)
            self.net_enveloppe = SIREN(in_features, out_features, layer, hidden_dim, outermost_linear=True, w0=30)
        elif net_type in ("cryodrgn", "cryodrgn-fourier"):
            if net_type == "cryodrgn-fourier":
                out_features = 2
            else:
                out_features = 1
            self.net = FCBlock(in_features=in_features,
                               features=[hidden_dim] * (layer + 1),
                               out_features=out_features,
                               nonlinearity='gelu',
                               last_nonlinearity=None,
                               use_residual=True)
        else:
            raise NotImplementedError
        self.out_features = out_features  # 2: fourier, 1: hartley

    def forward(self, z, coords):
        """
            z:      None | (bsz, z_dim)
            coords: (bsz, num, 3)
        """
        assert len(coords.shape) == 3

        if self.force_symmetry:
            self.symmetrizer.initialize(coords)
            coords = self.symmetrizer.symmetrize_input(coords)

        coords = self.pe(coords)

        if z is not None:
            num_coords = coords.shape[1]
            z_expand = einops.repeat(z, "bsz z_dim -> bsz num_coords z_dim", num_coords=num_coords)
            net_input = torch.cat([z_expand, coords], dim=2)
        else:
            net_input = coords

        if self.net_type == "cryoai":
            output = torch.exp(self.net_enveloppe(net_input)) * self.net_modulant(net_input)
        elif self.net_type in ("cryodrgn", "cryodrgn-fourier"):
            output = self.net(net_input)

        if self.force_symmetry:
            output = self.symmetrizer.antisymmetrize_output(output)

        return output


class Symmetrizer():

    def __init__(self):
        """
        Initialization of a Symmetrizer, to enforce symmetry in Fourier space.
        """
        self.half_space_indicator = None
        self.DC_indicator = None

    def initialize(self, coords):
        self.half_space_indicator = which_half_space(coords)
        self.DC_indicator = where_DC(coords)

    def symmetrize_input(self, coords):
        # Place the "negative" coords in the "positive" half space
        coords[self.half_space_indicator] = -coords[self.half_space_indicator]
        return coords

    def antisymmetrize_output(self, output):
        # 1. flip the imaginary part on the "negative" half space
        # 2. force the DC component's imaginary part to be zero
        batch_sz = output.shape[0]
        N = output.shape[1]
        output = output.reshape(batch_sz, N, -1, 2)
        # output.shape = Batch, N, channels, 2
        channels = output.shape[2]
        half_space = self.half_space_indicator.reshape(batch_sz, N, 1, 1).repeat(1, 1, channels, 2)
        DC = self.DC_indicator.reshape(batch_sz, N, 1, 1).repeat(1, 1, channels, 2)
        output_sym = torch.where(
            half_space,
            # (real + i imag) -> (real - i imag)
            torch.cat((output[..., 0].unsqueeze(-1), -output[..., 1].unsqueeze(-1)), dim=-1),
            output)
        output_sym_DC = torch.where(
            DC,
            # (real + i imag) -> (real + 0)
            torch.cat((output_sym[..., 0].unsqueeze(-1), torch.zeros_like(output_sym[..., 0].unsqueeze(-1))), dim=-1),
            output_sym)
        output_sym_DC = output_sym_DC.reshape(batch_sz, N, -1)
        return output_sym_DC


def which_half_space(coords, eps=1e-6):
    """
        A plane case:
            - On the 5x5 grid, choose the left-part (the center vertical line not included).
            - On the center vertical line, choose the upper-part (the origin not included)

        Inputs: 
            (5x5, 3)
        Returns:
            tensor([[ True,  True,  True, False, False],
                    [ True,  True,  True, False, False],
                    [ True,  True, False, False, False],
                    [ True,  True, False, False, False],
                    [ True,  True, False, False, False]])
    """
    x = coords[..., 0]
    y = coords[..., 1]
    z = coords[..., 2]

    slab_xyz = (x < -eps)
    slab_yz = torch.logical_and(torch.logical_and(x > -eps, x < eps), y < -eps)
    slab_z = torch.logical_and(
        torch.logical_and(torch.logical_and(x > -eps, x < eps), torch.logical_and(y > -eps, y < eps)), z < -eps)

    return torch.logical_or(slab_xyz, torch.logical_or(slab_yz, slab_z))


def where_DC(coords, eps=1e-6):
    """
        All coordinates must be between -1 and 1. The DC term is located at (0, 0, ...)

        A plane case:

        Inputs: 
            (4x4, 3)
        Returns:
            tensor([[False, False, False, False],
                    [False, False, False, False],
                    [False, False,  True, False],
                    [False, False, False, False]])
    """
    x = coords[..., 0]
    y = coords[..., 1]
    z = coords[..., 2]

    slab_x = torch.logical_and(x > -eps, x < eps)
    slab_y = torch.logical_and(y > -eps, y < eps)
    slab_z = torch.logical_and(z > -eps, z < eps)

    return torch.logical_and(slab_x, torch.logical_and(slab_y, slab_z))


def conv3x3(in_planes, out_planes, stride=1, bias=True):
    """
    3x3 convolution with padding.

    Parameters
    ----------
    in_planes: int
    out_planes: int
    stride: int
    bias: bool

    Returns
    -------
    out: torch.nn.Module
    """
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=bias)


class DoubleConvBlock(nn.Module):
    def __init__(self, in_size, out_size, batch_norm, triple=False):
        """
        Initialization of a double convolutional block.

        Parameters
        ----------
        in_size: int
        out_size: int
        batch_norm: bool
        triple: bool
        """
        super(DoubleConvBlock, self).__init__()
        self.batch_norm = batch_norm
        self.triple = triple

        self.conv1 = conv3x3(in_size, out_size)
        self.conv2 = conv3x3(out_size, out_size)
        if triple:
            self.conv3 = conv3x3(out_size, out_size)

        self.relu = nn.ReLU(inplace=True)

        if batch_norm:
            self.bn1 = nn.BatchNorm2d(out_size)
            self.bn2 = nn.BatchNorm2d(out_size)
            if triple:
                self.bn3 = nn.BatchNorm2d(out_size)

    def forward(self, x):
        out = self.conv1(x)
        if self.batch_norm:
            out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        if self.batch_norm:
            out = self.bn2(out)

        if self.triple:
            out = self.relu(out)

            out = self.conv3(out)
            if self.batch_norm:
                out = self.bn3(out)

        out = self.relu(out)

        return out


class WrappedCNNEncoderVGG16(nn.Module):
    def __init__(self, in_channels, side_len) -> None:
        super().__init__()
        from einops.layers.torch import Rearrange
        self.side_len = side_len
        self.net = torch.nn.Sequential(
            Rearrange("bsz (c ny nx) -> bsz c ny nx", c=in_channels, ny=side_len, nx=side_len),
            CNNEncoderVGG16(in_channels),
            torch.nn.Flatten(1)
        )

    def forward(self, x):
        return self.net(x)

    def get_out_dim(self):
        return self.forward(torch.rand(1, 1 * self.side_len * self.side_len)).shape[1]


class CNNEncoderVGG16(nn.Module):
    def __init__(self, in_channels=3, batch_norm=True, high_res=False):
        super(CNNEncoderVGG16, self).__init__()

        self.in_channels = in_channels
        if high_res:
            self.feature_channels = [64, 128, 256, 256, 1024, 2048]
        else:
            self.feature_channels = [64, 128, 256, 256, 256]

        self.net = []

        # VGG16 first 3 layers
        prev_channels = self.in_channels
        next_channels = self.feature_channels[0]
        self.net.append(
            DoubleConvBlock(prev_channels, next_channels, batch_norm=batch_norm)
        )
        self.net.append(
            nn.MaxPool2d(kernel_size=2)
        )
        prev_channels = next_channels
        next_channels = self.feature_channels[1]
        self.net.append(
            DoubleConvBlock(prev_channels, next_channels, batch_norm=batch_norm)
        )
        self.net.append(
            nn.MaxPool2d(kernel_size=2)
        )
        prev_channels = next_channels
        next_channels = self.feature_channels[2]
        self.net.append(
            DoubleConvBlock(prev_channels, next_channels, batch_norm=batch_norm, triple=True)
        )
        self.net.append(
            nn.MaxPool2d(kernel_size=2)
        )

        # Rest of encoder
        prev_channels = next_channels
        next_channels = self.feature_channels[3]
        self.net.append(
            DoubleConvBlock(prev_channels, next_channels, batch_norm=batch_norm)
        )
        self.net.append(
            nn.AvgPool2d(kernel_size=2)
        )
        prev_channels = next_channels
        next_channels = self.feature_channels[4]
        self.net.append(
            DoubleConvBlock(prev_channels, next_channels, batch_norm=batch_norm)
        )
        self.net.append(
            nn.AvgPool2d(kernel_size=2)
        )
        if high_res:
            prev_channels = next_channels
            next_channels = self.feature_channels[5]
            self.net.append(
                DoubleConvBlock(prev_channels, next_channels, batch_norm=batch_norm)
            )
            self.net.append(
                nn.AvgPool2d(kernel_size=2)
            )
        self.net.append(
            nn.MaxPool2d(kernel_size=2)
        )

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        out = self.net(input)
        return out
