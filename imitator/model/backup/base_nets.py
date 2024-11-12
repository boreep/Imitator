#!/usr/bin/env python3


import math
import numpy as np
from typing import Optional, Union, Tuple, List, Dict


import torch
import torch.nn as nn
import torch.nn.functional as F

import imitator.utils.tensor_utils as TensorUtils

import ssl

ssl._create_default_https_context = ssl._create_unverified_context


class Reshape(nn.Module):
    """
    Module that reshapes a tensor.
    """

    def __init__(self, shape: Union[int, Tuple[int, ...]]) -> None:
        super(Reshape, self).__init__()
        self._shape = shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(self._shape)


class Permute(nn.Module):
    """
    Module that permutes a tensor.
    """

    def __init__(self, dims: Union[List[int], Tuple[int, ...]]) -> None:
        super(Permute, self).__init__()
        self._dims = dims

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.permute(self._dims)


class Normalize(nn.Module):
    """
    Module that Normalize a tensor with mean and std.
    """

    def __init__(
        self,
        mean: Union[float, List[float], np.ndarray, torch.Tensor],
        std: Union[float, List[float], np.ndarray, torch.Tensor],
    ) -> None:
        super(Normalize, self).__init__()
        # requires_grad = False
        self._mean = TensorUtils.to_float(TensorUtils.to_tensor(mean))  # scalar or [3]
        self._std = TensorUtils.to_float(TensorUtils.to_tensor(std))  # scalar or [3]

        self.register_buffer("mean", self._mean)
        self.register_buffer("std", self._std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # if len(x.shape) == 4:  # [B, C, H, W]
        #     return (x - self.mean[None, :, None, None]) / self.std[None, :, None, None]
        return (x - self.mean) / self.std


class Unnormalize(nn.Module):
    """
    Module that Unnormalize a tensor with mean and std.
    """

    def __init__(
        self,
        mean: Union[float, List[float], np.ndarray, torch.Tensor],
        std: Union[float, List[float], np.ndarray, torch.Tensor],
    ) -> None:
        super(Unnormalize, self).__init__()
        # requires_grad = False
        self._mean = TensorUtils.to_float(TensorUtils.to_tensor(mean))
        self._std = TensorUtils.to_float(TensorUtils.to_tensor(std))

        self.register_buffer("mean", self._mean)
        self.register_buffer("std", self._std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.std + self.mean


class Unsqueeze(nn.Module):
    """
    Module that unsqueezes a tensor.
    """

    def __init__(self, dim: int) -> None:
        super(Unsqueeze, self).__init__()
        self._dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.unsqueeze(dim=self._dim)


class Squeeze(nn.Module):
    """
    Module that squeezes a tensor.
    """

    def __init__(self, dim: int) -> None:
        super(Squeeze, self).__init__()
        self._dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.squeeze(dim=self._dim)


class SoftPositionEmbed(nn.Module):
    """
    Module that adds soft positional embeddings to a tensor.
    """

    def __init__(
        self, hidden_dim: int, resolution: Union[Tuple[int, int], List[int]]
    ) -> None:
        super(SoftPositionEmbed, self).__init__()
        self._hidden_dim = hidden_dim
        self._resolution = resolution
        self._embedding = nn.Linear(4, hidden_dim)
        self._grid = self.build_grid(resolution)  # device?

        self.register_buffer("grid", self._grid)

    def build_grid(self, resolution: Union[Tuple[int, int], List[int]]) -> torch.Tensor:
        ranges = [np.linspace(0.0, 1.0, num=res) for res in resolution]
        grid = np.meshgrid(*ranges, sparse=False, indexing="ij")
        grid = np.stack(grid, axis=-1)
        grid = np.reshape(grid, [resolution[0], resolution[1], -1])
        grid = np.expand_dims(grid, axis=0)
        grid = grid.astype(np.float32)
        return TensorUtils.to_tensor(np.concatenate([grid, 1.0 - grid], axis=-1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        grid = self._embedding(self.grid)
        return x + grid
        # grid = self._embedding(self._grid)
        # return x + grid


class SlotAttention(nn.Module):
    """
    Module that performs slot attention.
    ref : https://github.com/lucidrains/slot-attention
    """

    def __init__(
        self,
        num_slots: int = 7,
        dim: int = 64,
        num_iters: int = 3,
        eps: float = 1e-8,
        hidden_dim: int = 128,
    ) -> None:
        super(SlotAttention, self).__init__()
        self._num_slots = num_slots
        self._num_iters = num_iters
        self._eps = eps
        self._scale = dim**-0.5
        hidden_dim = max(dim, hidden_dim)

        self.slots_mu = nn.Parameter(torch.randn(1, 1, dim))
        self.slots_log_sigma = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.xavier_uniform_(self.slots_log_sigma)

        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)

        self.gru = nn.GRUCell(dim, dim)
        self.mlp = MLP(
            input_dim=dim,
            output_dim=dim,
            layer_dims=[hidden_dim],
            activation=nn.ReLU,
        )

        self.norm_input = nn.LayerNorm(dim)
        self.norm_slots = nn.LayerNorm(dim)
        self.norm_mlp = nn.LayerNorm(dim)

    # def step(self, slots, k, v):
    #     q = self.to_q(self.norm_slots(slots))
    #     k = k * self._scale
    #     attn = F.softmax(torch.einsum("bkd,bqd->bkq", k, q), dim=-1)
    #     attn = attn / torch.sum(attn + self._eps, dim=-2, keepdim=True)
    #     updates = torch.einsum("bvq,bvd->bqd", attn, v)
    #     slots = self.gru(updates, slots)
    #     slots = slots + self.mlp(self.norm_mlp(slots))
    #     return slots

    # def iterate(self, f, x):
    #     for _ in range(self._num_iters):
    #         x = f(x)
    #     return x

    # def forward(self, inputs, slots):
    #     inputs = self.norm_input(inputs)
    #     k, v = self.to_k(inputs), self.to_v(inputs)
    #     slots = self.iterate(lambda x: self.step(x, k, v), slots)
    #     slots = self.step(slots.detach(), k, v)
    #     return slots

    def forward(self, inputs, num_slots=None):
        b, _, d, device, dtype = *inputs.shape, inputs.device, inputs.dtype
        n_s = num_slots if num_slots is not None else self._num_slots
        mu = self.slots_mu.expand(b, n_s, -1)
        sigma = self.slots_log_sigma.exp().expand(b, n_s, -1)
        slots = mu + sigma * torch.randn(mu.shape, device=device, dtype=dtype)
        inputs = self.norm_input(inputs)
        k, v = self.to_k(inputs), self.to_v(inputs)
        for _ in range(self._num_iters):
            slots_prev = slots
            slots = self.norm_slots(slots)
            q = self.to_q(slots)
            dots = torch.einsum("bid,bjd->bij", q, k) * self._scale
            attn = dots.softmax(dim=1) + self._eps
            attn = attn / attn.sum(dim=-1, keepdim=True)
            updates = torch.einsum("bjd,bij->bid", v, attn)
            slots = self.gru(updates.reshape(-1, d), slots_prev.reshape(-1, d))
            # slots = slots.reshape(b, -1, d)
            slots = slots.view(b, -1, d)
            slots = slots + self.mlp(self.norm_mlp(slots))

        return slots


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        layer_dims: List[int],
        layer: nn.Module = nn.Linear,
        layer_kwargs: Optional[dict] = None,
        activation: nn.Module = nn.ReLU,
        dropouts: Optional[List[float]] = None,
        normalization=None,
        output_activation: Optional[nn.Module] = None,
    ) -> None:
        super(MLP, self).__init__()
        if dropouts is not None:
            assert len(dropouts) == len(layer_dims)
        layers = []
        dim = input_dim
        layer_kwargs = layer_kwargs if layer_kwargs is not None else dict()
        for i, l in enumerate(layer_dims):
            layers.append(layer(dim, l, **layer_kwargs))
            if normalization is not None:
                layers.append(normalization(l))
            layers.append(activation())
            if dropouts is not None and dropouts[i] > 0.0:
                layers.append(nn.Dropout(dropouts[i]))
            dim = l
        layers.append(layer(dim, output_dim))
        if output_activation is not None:
            layers.append(output_activation())
        self._layer = layer
        self._nets = layers
        self._model = nn.Sequential(*layers)

        self._layer_dims = layer_dims
        self._input_dim = input_dim
        self._output_dim = output_dim
        self._dropouts = dropouts
        self._act = activation
        self._output_act = output_activation

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self._model(inputs)


class RNN(nn.Module):
    def __init__(
        self,
        rnn_input_dim: int,
        rnn_hidden_dim: int,
        rnn_num_layers: int,
        rnn_type: str,
        rnn_kwargs: Optional[Dict] = None,
        per_step_net: Optional[nn.Module] = None,
    ) -> None:
        super(RNN, self).__init__()

        assert rnn_type in ["LSTM", "GRU"]
        assert per_step_net is None or isinstance(per_step_net, nn.Module)

        rnn_cls = nn.LSTM if rnn_type == "LSTM" else nn.GRU
        rnn_kwargs = rnn_kwargs if rnn_kwargs is not None else dict()

        self._rnn_input_dim = rnn_input_dim
        self._rnn_hidden_dim = rnn_hidden_dim
        self._rnn_num_layers = rnn_num_layers
        self._rnn_type = rnn_type
        self._per_step_net = per_step_net
        self._is_bidirectional = rnn_kwargs.get("bidirectional", False)
        self._num_directions = 2 if self._is_bidirectional else 1

        self.nets = rnn_cls(
            input_size=self._rnn_input_dim,
            hidden_size=self._rnn_hidden_dim,
            num_layers=self._rnn_num_layers,
            batch_first=True,
            **rnn_kwargs,
        )

    @property
    def rnn_type(self) -> str:
        return self._rnn_type

    def get_rnn_init_state(
        self, batch_size: int, device: torch.device
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        h_0 = torch.zeros(
            self._rnn_num_layers * self._num_directions,
            batch_size,
            self._rnn_hidden_dim,
            device=device,
        )
        if self._rnn_type == "LSTM":
            c_0 = torch.zeros(
                self._rnn_num_layers * self._num_directions,
                batch_size,
                self._rnn_hidden_dim,
                device=device,
            )
            return (h_0, c_0)
        else:
            return h_0

    def forward(
        self,
        inputs: torch.Tensor,
        rnn_state: Optional[torch.Tensor] = None,
        return_rnn_state: bool = False,
    ):
        # def time_distributed(inputs, net):
        #     """
        #     function that applies a network to a time distributed input
        #     inputs : (batch_size, seq_len, ...)
        #     outputs : (batch_size, seq_len, ...)
        #     """
        #     batch_size, seq_len, = inputs.shape[:2]
        #     # inputs = inputs.reshape(-1, inputs.shape[-1])
        #     outputs = net(inputs)
        #     # outputs = outputs.reshape(batch_size, seq_len, -1)
        #     return outputs

        assert inputs.ndim == 3  # (batch_size, seq_len, input_dim)
        batch_size, _, _ = inputs.shape
        if rnn_state is None:
            rnn_state = self.get_rnn_init_state(batch_size, inputs.device)

        outputs, rnn_state = self.nets(inputs, rnn_state)
        if self._per_step_net is not None:
            outputs = self._per_step_net(outputs)
            # outputs = time_distributed(outputs, self._per_step_net)
        if return_rnn_state:
            return outputs, rnn_state
        else:
            return outputs

    def forward_step(self, inputs: torch.Tensor, rnn_state: torch.Tensor):
        """
        return rnn outputs and rnn state for the next step
        inputs : (batch_size, input_dim)
        """
        assert inputs.ndim == 2
        inputs = TensorUtils.to_sequence(inputs)  # (batch_size, 1, input_dim)
        outputs, rnn_state = self.forward(inputs, rnn_state, return_rnn_state=True)
        return (
            outputs[:, 0, :],
            rnn_state,
        )  # (batch_size, output_dim), (batch_size, hidden_dim)


class CNN(nn.Module):
    """
    Base 2D Convolutional neural network.
    inputs like (batch_size, channels, height, width)
    """

    def __init__(
        self,
        input_channel: int,
        channels: List[int],
        kernel_sizes: List[int],
        strides: List[int],
        paddings: List[int],
        layer: nn.Module = nn.Conv2d,
        layer_kwargs: Optional[dict] = None,
        activation: nn.Module = nn.ReLU,
        dropouts: Optional[List[float]] = None,
        normalization=None,
        output_activation: Optional[nn.Module] = None,
    ) -> None:
        super(CNN, self).__init__()

        assert len(channels) == len(kernel_sizes) == len(strides) == len(paddings)
        layers = []
        layer_kwargs = layer_kwargs if layer_kwargs is not None else dict()

        for i in range(len(channels)):
            if i == 0:
                in_channels = input_channel
            else:
                in_channels = channels[i - 1]
            out_channels = channels[i]
            layers.append(
                layer(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_sizes[i],
                    stride=strides[i],
                    padding=paddings[i],
                    **layer_kwargs,
                )
            )
            if (
                normalization is not None and i != len(channels) - 1
            ):  # not the last layer
                layers.append(normalization(out_channels))
            if i != len(channels) - 1:  # not the last layer
                layers.append(activation())
            if dropouts is not None:
                layers.append(nn.Dropout(dropouts[i]))

        if output_activation is not None:
            layers.append(output_activation())

        self.net = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)


class CoordConv(nn.Conv2d, nn.Module):
    def __init__(
        self,
        input_channel: int = 3,
        output_channel: int = 64,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        coord_encoding: str = "position",
    ) -> None:
        self.coord_encoding = coord_encoding
        if coord_encoding == "position":
            input_channel += 2
            self._position_enc = None
        else:
            raise NotImplementedError

        nn.Conv2d.__init__(
            self,
            in_channels=input_channel,
            out_channels=output_channel,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        b, c, h, w = inputs.shape
        if self.coord_encoding == "position":
            if self._position_enc is None:
                pos_y, pos_x = torch.meshgrid(torch.arange(h), torch.arange(w))
                pos_y = pos_y.float().to(inputs.device) / float(h)
                pos_x = pos_x.float().to(inputs.device) / float(w)
                self._position_enc = torch.stack((pos_y, pos_x)).unsqueeze(0)
            pos_enc = self._position_enc.expand(b, -1, -1, -1)
            inputs = torch.cat((inputs, pos_enc), dim=1)
        return super(CoordConv, self).forward(inputs)


class SpatialSoftmax(nn.Module):
    """
    Spatial Softmax Layer.

    Based on Deep Spatial Autoencoders for Visuomotor Learning by Finn et al.
    https://rll.berkeley.edu/dsae/dsae.pdf
    """

    def __init__(
        self,
        input_shape: List[int],
        num_kp: int = 32,
        temperature: float = 1.0,
        learnable_temperature: bool = False,
        output_variance: bool = False,
        noise_std: float = 0.0,
    ) -> None:
        super(SpatialSoftmax, self).__init__()
        assert len(input_shape) == 3
        self._in_c, self._in_h, self._in_w = input_shape  # (C, H, W)

        if num_kp is not None:
            self.nets = torch.nn.Conv2d(self._in_c, num_kp, kernel_size=1)
            self._num_kp = num_kp
        else:
            self.nets = None
            self._num_kp = self._in_c
        self.learnable_temperature = learnable_temperature
        self.output_variance = output_variance
        self.noise_std = noise_std

        if self.learnable_temperature:
            # temperature will be learned
            temperature = torch.nn.Parameter(
                torch.ones(1) * temperature, requires_grad=True
            )
            self.register_parameter("temperature", temperature)
        else:
            # temperature held constant after initialization
            temperature = torch.nn.Parameter(
                torch.ones(1) * temperature, requires_grad=False
            )
            self.register_buffer("temperature", temperature)

        pos_x, pos_y = np.meshgrid(
            np.linspace(-1.0, 1.0, self._in_w), np.linspace(-1.0, 1.0, self._in_h)
        )
        pos_x = torch.from_numpy(pos_x.reshape(1, self._in_h * self._in_w)).float()
        pos_y = torch.from_numpy(pos_y.reshape(1, self._in_h * self._in_w)).float()
        self.register_buffer("pos_x", pos_x)
        self.register_buffer("pos_y", pos_y)

        self.kps = None

        self.output_dim = [self._num_kp, 2]

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through spatial softmax layer. For each keypoint, a 2D spatial
        probability distribution is created using a softmax, where the support is the
        pixel locations. This distribution is used to compute the expected value of
        the pixel location, which becomes a keypoint of dimension 2. K such keypoints
        are created.

        Returns:
            out (torch.Tensor or tuple): mean keypoints of shape [B, K, 2], and possibly
                keypoint variance of shape [B, K, 2, 2] corresponding to the covariance
                under the 2D spatial softmax distribution
        """

        assert feature.shape[1] == self._in_c
        assert feature.shape[2] == self._in_h
        assert feature.shape[3] == self._in_w
        if self.nets is not None:
            feature = self.nets(feature)

        # [B, K, H, W] -> [B * K, H * W] where K is number of keypoints
        feature = feature.reshape(-1, self._in_h * self._in_w)
        # 2d softmax normalization
        attention = F.softmax(feature / self.temperature, dim=-1)
        # [1, H * W] x [B * K, H * W] -> [B * K, 1] for spatial coordinate mean in x and y dimensions
        expected_x = torch.sum(self.pos_x * attention, dim=1, keepdim=True)
        expected_y = torch.sum(self.pos_y * attention, dim=1, keepdim=True)
        # stack to [B * K, 2]
        expected_xy = torch.cat([expected_x, expected_y], 1)
        # reshape to [B, K, 2]
        feature_keypoints = expected_xy.view(-1, self._num_kp, 2)

        if self.training:
            noise = torch.randn_like(feature_keypoints) * self.noise_std
            feature_keypoints += noise

        if self.output_variance:
            # treat attention as a distribution, and compute second-order statistics to return
            expected_xx = torch.sum(
                self.pos_x * self.pos_x * attention, dim=1, keepdim=True
            )
            expected_yy = torch.sum(
                self.pos_y * self.pos_y * attention, dim=1, keepdim=True
            )
            expected_xy = torch.sum(
                self.pos_x * self.pos_y * attention, dim=1, keepdim=True
            )
            var_x = expected_xx - expected_x * expected_x
            var_y = expected_yy - expected_y * expected_y
            var_xy = expected_xy - expected_x * expected_y
            # stack to [B * K, 4] and then reshape to [B, K, 2, 2] where last 2 dims are covariance matrix
            feature_covar = torch.cat([var_x, var_xy, var_xy, var_y], 1).reshape(
                -1, self._num_kp, 2, 2
            )
            feature_keypoints = (feature_keypoints, feature_covar)

        if isinstance(feature_keypoints, tuple):
            self.kps = (feature_keypoints[0].detach(), feature_keypoints[1].detach())
        else:
            self.kps = feature_keypoints.detach()
        return feature_keypoints  # [B, K, 2] or ([B, K, 2], [B, K, 2, 2])


class GEGLU(nn.Module):
    """
    References:
        Shazeer et al., "GLU Variants Improve Transformer," 2020.
        https://arxiv.org/abs/2002.05202
    Implementation: https://github.com/pfnet-research/deep-table/blob/237c8be8a405349ce6ab78075234c60d9bfe60b7/deep_table/nn/layers/activation.py
    """

    def geglu(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[-1] % 2 == 0
        a, b = x.chunk(2, dim=-1)
        return a * F.gelu(b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.geglu(x)


class PositionalEncoding(nn.Module):
    """
    Taken from https://pytorch.org/tutorials/beginner/transformer_tutorial.html.
    """

    def __init__(self, embed_dim: int) -> None:
        """
        Standard sinusoidal positional encoding scheme in transformers.

        Positional encoding of the k'th position in the sequence is given by:
            p(k, 2i) = sin(k/n^(i/d))
            p(k, 2i+1) = sin(k/n^(i/d))

        n: set to 10K in original Transformer paper
        d: the embedding dimension
        i: positions along the projected embedding space (ranges from 0 to d/2)

        Args:
            embed_dim: The number of dimensions to project the timesteps into.
        """
        super(PositionalEncoding, self).__init__()
        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input timestep of shape BxT
        """
        position = x

        # computing 1/n^(i/d) in log space and then exponentiating and fixing the shape
        div_term = (
            torch.exp(
                torch.arange(0, self.embed_dim, 2, device=x.device)
                * (-math.log(10000.0) / self.embed_dim)
            )
            .unsqueeze(0)
            .unsqueeze(0)
            .repeat(x.shape[0], x.shape[1], 1)
        )
        pe = torch.zeros((x.shape[0], x.shape[1], self.embed_dim), device=x.device)
        pe[:, :, 0::2] = torch.sin(position.unsqueeze(-1) * div_term)
        pe[:, :, 1::2] = torch.cos(position.unsqueeze(-1) * div_term)
        return pe.detach()


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        context_length: int,
        attn_dropout: float = 0.1,
        output_dropout: float = 0.1,
    ) -> None:
        super(CausalSelfAttention, self).__init__()

        assert (
            embed_dim % num_heads == 0
        ), "num_heads: {} does not divide embed_dim: {} exactly".format(
            num_heads, embed_dim
        )

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.context_length = context_length
        self.attn_dropout = attn_dropout
        self.output_dropout = output_dropout
        self.nets = nn.ModuleDict()

        # projection layers for key, query, value, across all attention heads
        self.nets["qkv"] = nn.Linear(self.embed_dim, 3 * self.embed_dim, bias=False)

        # dropout layers
        self.nets["attn_dropout"] = nn.Dropout(self.attn_dropout)
        self.nets["output_dropout"] = nn.Dropout(self.output_dropout)

        # output layer
        self.nets["output"] = nn.Linear(self.embed_dim, self.embed_dim)

        # causal mask (ensures attention is only over previous inputs) - just a lower triangular matrix of 1s
        mask = torch.tril(torch.ones(context_length, context_length)).view(
            1, 1, context_length, context_length
        )
        self.register_buffer("mask", mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through Self-Attention block.
        Input should be shape (B, T, D) where B is batch size, T is seq length (@self.context_length), and
        D is input dimension (@self.embed_dim).
        """

        # enforce shape consistency
        assert len(x.shape) == 3
        B, T, D = x.shape
        assert (
            T <= self.context_length
        ), "self-attention module can only handle sequences up to {} in length but got length {}".format(
            self.context_length, T
        )
        assert D == self.embed_dim
        NH = self.num_heads  # number of attention heads
        DH = D // NH  # embed dimension for each attention head

        # compute key, query, and value vectors for each member of sequence, and split across attention heads
        qkv = self.nets["qkv"](x)
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        k = k.view(B, T, NH, DH).transpose(1, 2)  # [B, NH, T, DH]
        q = q.view(B, T, NH, DH).transpose(1, 2)  # [B, NH, T, DH]
        v = v.view(B, T, NH, DH).transpose(1, 2)  # [B, NH, T, DH]

        # causal self-attention mechanism

        # batched matrix multiplication between queries and keys to get all pair-wise dot-products.
        # We broadcast across batch and attention heads and get pair-wise dot-products between all pairs of timesteps
        # [B, NH, T, DH] x [B, NH, DH, T] -> [B, NH, T, T]
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        # use mask to replace entries in dot products with negative inf to ensure they don't contribute to softmax,
        # then take softmax over last dimension to end up with attention score for each member of sequence.
        # Note the use of [:T, :T] -  this makes it so we can handle sequences less than @self.context_length in length.
        att = att.masked_fill(self.mask[..., :T, :T] == 0, float("-inf"))
        att = F.softmax(
            att, dim=-1
        )  # shape [B, NH, T, T], last dimension has score over all T for each sequence member

        # dropout on attention
        att = self.nets["attn_dropout"](att)

        # take weighted sum of value vectors over whole sequence according to attention, with batched matrix multiplication
        # [B, NH, T, T] x [B, NH, T, DH] -> [B, NH, T, DH]
        y = att @ v
        # reshape [B, NH, T, DH] -> [B, T, NH, DH] -> [B, T, NH * DH] = [B, T, D]
        y = y.transpose(1, 2).contiguous().view(B, T, D)

        # pass through output layer + dropout
        y = self.nets["output"](y)
        y = self.nets["output_dropout"](y)
        return y


class SelfAttentionBlock(nn.Module):
    """
    A single Transformer Block, that can be chained together repeatedly.
    It consists of a @CausalSelfAttention module and a small MLP, along with
    layer normalization and residual connections on each input.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        context_length: int,
        attn_dropout: float = 0.1,
        output_dropout: float = 0.1,
        activation: nn.Module = nn.GELU(),
    ) -> None:
        """
        Args:
            embed_dim (int): dimension of embeddings to use for keys, queries, and values
                used in self-attention

            num_heads (int): number of attention heads - must divide @embed_dim evenly. Self-attention is
                computed over this many partitions of the embedding dimension separately.

            context_length (int): expected length of input sequences

            attn_dropout (float): dropout probability for attention outputs

            output_dropout (float): dropout probability for final outputs

            activation (str): string denoting the activation function to use in each transformer block
        """
        super(SelfAttentionBlock, self).__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.context_length = context_length
        self.attn_dropout = attn_dropout
        self.output_dropout = output_dropout
        self.nets = nn.ModuleDict()

        # self-attention block
        self.nets["attention"] = CausalSelfAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            context_length=context_length,
            attn_dropout=attn_dropout,
            output_dropout=output_dropout,
        )

        if type(activation) == GEGLU:
            mult = 2
        else:
            mult = 1

        # small 2-layer MLP
        self.nets["mlp"] = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim * mult),
            activation,
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(output_dropout),
        )

        # layer normalization for inputs to self-attention module and MLP
        self.nets["ln1"] = nn.LayerNorm(embed_dim)
        self.nets["ln2"] = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass - chain self-attention + MLP blocks, with residual connections and layer norms.
        """
        x = x + self.nets["attention"](self.nets["ln1"](x))
        x = x + self.nets["mlp"](self.nets["ln2"](x))
        return x


class GPT(nn.Module):
    """the GPT model, with a context size of block_size"""

    def __init__(
        self,
        embed_dim: int,
        context_length: int,
        attn_dropout: float = 0.1,
        block_output_dropout: float = 0.1,
        num_layers: int = 6,
        num_heads: int = 8,
        activation="gelu",
    ) -> None:
        super(GPT, self).__init__()

        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.context_length = context_length
        self.attn_dropout = attn_dropout
        self.block_output_dropout = block_output_dropout

        if activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "geglu":
            self.activation = GEGLU()

        # create networks
        self._create_networks()

        # initialize weights
        self.apply(self._init_weights)

        print(
            "Created {} model with number of parameters: {}".format(
                self.__class__.__name__, sum(p.numel() for p in self.parameters())
            )
        )

    def _create_networks(self):
        """
        Helper function to create networks.
        """
        self.nets = nn.ModuleDict()

        # transformer - cascaded transformer blocks
        self.nets["transformer"] = nn.Sequential(
            *[
                SelfAttentionBlock(
                    embed_dim=self.embed_dim,
                    num_heads=self.num_heads,
                    context_length=self.context_length,
                    attn_dropout=self.attn_dropout,
                    output_dropout=self.block_output_dropout,
                    activation=self.activation,
                )
                for _ in range(self.num_layers)
            ]
        )

        # decoder head
        self.nets["output_ln"] = nn.LayerNorm(self.embed_dim)

    def _init_weights(self, module):
        """
        Weight initializer.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        assert inputs.shape[1:] == (self.context_length, self.embed_dim), inputs.shape
        x = self.nets["transformer"](inputs)
        transformer_output = self.nets["output_ln"](x)
        return transformer_output
