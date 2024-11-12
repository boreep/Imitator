#!/usr/bin/env python3

import numpy as np
import math
from collections import OrderedDict

from typing import (
    List,
    Dict,
    Tuple,
    Union,
    Optional,
    Any,
)

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import models as vision_models
from torchvision import transforms

import imitator.utils.tensor_utils as TensorUtils
from imitator.models.base_nets import (
    Reshape,
    Permute,
    SoftPositionEmbed,
    SlotAttention,
    CoordConv,
    SpatialSoftmax,
    MLP,
    CNN,
)


def calculate_conv_output_size(
    input_size: List[int],
    kernel_sizes: List[int],
    strides: List[int],
    paddings: List[int],
) -> List[int]:
    assert len(kernel_sizes) == len(strides) == len(paddings)
    output_size = list(input_size)
    for i in range(len(kernel_sizes)):
        output_size[0] = (
            output_size[0] + 2 * paddings[i] - kernel_sizes[i]
        ) // strides[i] + 1
        output_size[1] = (
            output_size[1] + 2 * paddings[i] - kernel_sizes[i]
        ) // strides[i] + 1
    return output_size


def calculate_deconv_output_size(
    input_size: List[int],
    kernel_sizes: List[int],
    strides: List[int],
    paddings: List[int],
    output_paddings: List[int],
) -> List[int]:
    assert len(kernel_sizes) == len(strides) == len(paddings) == len(output_paddings)
    output_size = list(input_size)
    for i in range(len(kernel_sizes)):
        output_size[0] = (
            (output_size[0] - 1) * strides[i]
            - 2 * paddings[i]
            + kernel_sizes[i]
            + output_paddings[i]
        )
        output_size[1] = (
            (output_size[1] - 1) * strides[i]
            - 2 * paddings[i]
            + kernel_sizes[i]
            + output_paddings[i]
        )
    return output_size


class VisionModule(nn.Module):
    def __init__(self) -> None:
        super(VisionModule, self).__init__()
        self.training = True
        self.data_augmentation = False

    def eval(self) -> "VisionModule":
        self.training = False
        return super(VisionModule, self).eval()

    def train(self, mode: bool = True) -> "VisionModule":
        self.training = True
        return super(VisionModule, self).train(mode)

    def augmentation(self, inputs: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    """
    inputs like uint8 (B, C, H, W) or (B, C, H, W) or (C, H, W) torch.Tensor
    """

    def preprocess(self, inputs: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False


class ConvEncoder(VisionModule):
    def __init__(
        self,
        input_size: List[int] = [224, 224],
        input_channel: int = 3,
        channels: List[int] = [8, 16, 32, 64, 128, 256],
        kernel_sizes: List[int] = [3, 3, 3, 3, 3, 3],
        strides: List[int] = [2, 2, 2, 2, 2, 2],
        paddings: List[int] = [1, 1, 1, 1, 1, 1],
        latent_dim: int = 16,
        mean_var: bool = False,
        activation: nn.Module = nn.ReLU,
        dropouts: Optional[List[float]] = None,
        normalization=None,
        output_activation: Optional[nn.Module] = None,
        **kwargs,
    ) -> None:
        super(ConvEncoder, self).__init__()
        output_conv_size = calculate_conv_output_size(
            input_size=input_size,  # TODO: input size
            kernel_sizes=kernel_sizes,
            strides=strides,
            paddings=paddings,
        )

        self.mean_var = mean_var
        self.output_activation = (
            output_activation if output_activation is not None else lambda x: x
        )

        self.nets = nn.ModuleDict()
        self.nets["conv"] = CNN(
            input_channel=input_channel,
            channels=channels,
            kernel_sizes=kernel_sizes,
            strides=strides,
            paddings=paddings,
            layer=nn.Conv2d,
            activation=activation,
            dropouts=dropouts,
            normalization=normalization,
            output_activation=None,
        )
        self.nets["reshape"] = Reshape(
            (-1, channels[-1] * output_conv_size[0] * output_conv_size[1])
        )

        if mean_var:
            self.nets["mlp_mu"] = MLP(
                input_dim=channels[-1] * output_conv_size[0] * output_conv_size[1],
                output_dim=latent_dim,
                layer_dims=[latent_dim * 4, latent_dim * 2],
                activation=activation,
                dropouts=None,
                normalization=(
                    nn.BatchNorm1d if normalization is not None else normalization
                ),
            )
            self.nets["mlp_logvar"] = MLP(
                input_dim=channels[-1] * output_conv_size[0] * output_conv_size[1],
                output_dim=latent_dim,
                layer_dims=[latent_dim * 4, latent_dim * 2],
                activation=activation,
                dropouts=None,
                normalization=(
                    nn.BatchNorm1d if normalization is not None else normalization
                ),
            )
        else:
            self.nets["mlp"] = MLP(
                input_dim=channels[-1] * output_conv_size[0] * output_conv_size[1],
                output_dim=latent_dim,
                layer_dims=[latent_dim * 4, latent_dim * 2],
                activation=activation,
                dropouts=None,
                normalization=(
                    nn.BatchNorm1d if normalization is not None else normalization
                ),
            )

    def reparametrize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.nets["conv"](x)
        x = self.nets["reshape"](x)
        if self.mean_var:
            mu = self.output_activation(self.nets["mlp_mu"](x))
            logvar = self.output_activation(self.nets["mlp_logvar"](x))
            z = self.reparametrize(mu, logvar)
            return z, mu, logvar
        else:
            z = self.output_activation(self.nets["mlp"](x))
            return z


class ConvDecoder(VisionModule):
    def __init__(
        self,
        input_conv_size: List[int] = [4, 4],
        output_channel: int = 3,
        channels: List[int] = [256, 128, 64, 32, 16, 8],
        kernel_sizes: List[int] = [3, 4, 4, 4, 4, 4],
        strides: List[int] = [2, 2, 2, 2, 2, 2],
        paddings: List[int] = [1, 1, 1, 1, 1, 1],
        latent_dim: int = 16,
        activation: nn.Module = nn.ReLU,
        dropouts: Optional[List[float]] = None,
        normalization=None,
        output_activation: Optional[nn.Module] = nn.Sigmoid,
        **kwargs,
    ) -> None:
        super(ConvDecoder, self).__init__()

        self.output_activation = (
            output_activation if output_activation is not None else lambda x: x
        )

        self.nets = nn.ModuleDict()
        self.nets["mlp"] = MLP(
            input_dim=latent_dim,
            output_dim=channels[0] * input_conv_size[0] * input_conv_size[1],
            layer_dims=[latent_dim * 2, latent_dim * 4],
            activation=activation,
            dropouts=None,
            normalization=(
                nn.BatchNorm1d if normalization is not None else normalization
            ),
        )
        self.nets["reshape"] = Reshape(
            (-1, channels[0], input_conv_size[0], input_conv_size[1])
        )
        self.nets["deconv"] = CNN(
            input_channel=channels[0],
            channels=channels[1:] + [output_channel],
            kernel_sizes=kernel_sizes,
            strides=strides,
            paddings=paddings,
            layer=nn.ConvTranspose2d,
            activation=activation,
            dropouts=dropouts,
            normalization=normalization,
            output_activation=output_activation,
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.nets["mlp"](z)
        x = self.nets["reshape"](x)
        x = self.nets["deconv"](x)
        return x


class AutoEncoder(VisionModule):
    """
    AutoEncoder for image compression using class Conv for Encoder and Decoder
    """

    def __init__(
        self,
        input_size: List[int] = [224, 224],
        input_channel: int = 3,
        channels: List[int] = [8, 16, 32, 64, 128, 256],
        encoder_kernel_sizes: List[int] = [3, 3, 3, 3, 3, 3],
        decoder_kernel_sizes: List[int] = [3, 4, 4, 4, 4, 4],
        strides: List[int] = [2, 2, 2, 2, 2, 2],
        paddings: List[int] = [1, 1, 1, 1, 1, 1],
        latent_dim: int = 16,
        activation: nn.Module = nn.ReLU,
        dropouts: Optional[List[float]] = None,
        normalization=nn.BatchNorm2d,
        output_activation: Optional[nn.Module] = nn.Sigmoid,
        **kwargs,
    ) -> None:
        super(AutoEncoder, self).__init__()

        assert (
            len(channels)
            == len(encoder_kernel_sizes)
            == len(strides)
            == len(paddings)
            == len(decoder_kernel_sizes)
        )

        output_conv_size = calculate_conv_output_size(
            input_size=input_size,  # TODO: input size
            kernel_sizes=encoder_kernel_sizes,
            strides=strides,
            paddings=paddings,
        )

        self.criterion = nn.MSELoss()

        self.nets = nn.ModuleDict()
        self.nets["encoder"] = ConvEncoder(
            input_size=input_size,
            input_channel=input_channel,
            channels=channels,
            kernel_sizes=encoder_kernel_sizes,
            strides=strides,
            paddings=paddings,
            latent_dim=latent_dim,
            mean_var=False,
            activation=activation,
            dropouts=dropouts,
            normalization=normalization,
            output_activation=None,
        )

        self.nets["decoder"] = ConvDecoder(
            input_conv_size=output_conv_size,
            output_channel=input_channel,
            channels=list(reversed(channels)),
            kernel_sizes=decoder_kernel_sizes,
            strides=list(reversed(strides)),
            paddings=list(reversed(paddings)),
            latent_dim=latent_dim,
            activation=activation,
            dropouts=dropouts,
            normalization=normalization,
            output_activation=output_activation,
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.nets["encoder"](x)
        x = self.nets["decoder"](z)
        return x, z

    def forward_train(
        self, x: torch.Tensor, ground_truth: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        outputs = OrderedDict()
        z = self.nets["encoder"](x)
        reconstruction = self.nets["decoder"](z)
        reconstruction_loss = self.criterion(reconstruction, ground_truth)
        outputs["reconstruction"] = reconstruction
        outputs["z"] = z
        outputs["reconstruction_loss"] = reconstruction_loss
        outputs["loss"] = reconstruction_loss
        return outputs


class VariationalAutoEncoder(VisionModule):
    def __init__(
        self,
        input_size: List[int] = [224, 224],
        input_channel: int = 3,
        channels: List[int] = [8, 16, 32, 64, 128, 256],
        encoder_kernel_sizes: List[int] = [3, 3, 3, 3, 3, 3],
        decoder_kernel_sizes: List[int] = [3, 4, 4, 4, 4, 4],
        strides: List[int] = [2, 2, 2, 2, 2, 2],
        paddings: List[int] = [1, 1, 1, 1, 1, 1],
        latent_dim: int = 16,
        activation: nn.Module = nn.ReLU,
        dropouts: Optional[List[float]] = None,
        normalization=nn.BatchNorm2d,
        output_activation: Optional[nn.Module] = nn.Sigmoid,
        kld_weight: float = 1e-1,
        **kwargs,
    ) -> None:
        super(VariationalAutoEncoder, self).__init__()
        assert (
            len(channels)
            == len(encoder_kernel_sizes)
            == len(strides)
            == len(paddings)
            == len(decoder_kernel_sizes)
        )

        output_conv_size = calculate_conv_output_size(
            input_size=input_size,  # TODO: input size
            kernel_sizes=encoder_kernel_sizes,
            strides=strides,
            paddings=paddings,
        )

        self.kld_weight = kld_weight
        self.criterion = nn.MSELoss()

        self.nets = nn.ModuleDict()
        self.nets["encoder"] = ConvEncoder(
            input_size=input_size,
            input_channel=input_channel,
            channels=channels,
            kernel_sizes=encoder_kernel_sizes,
            strides=strides,
            paddings=paddings,
            latent_dim=latent_dim,
            mean_var=True,
            activation=activation,
            dropouts=dropouts,
            normalization=normalization,
            output_activation=None,
        )

        self.nets["decoder"] = ConvDecoder(
            input_conv_size=output_conv_size,
            output_channel=input_channel,
            channels=list(reversed(channels)),
            kernel_sizes=decoder_kernel_sizes,
            strides=list(reversed(strides)),
            paddings=list(reversed(paddings)),
            latent_dim=latent_dim,
            activation=activation,
            dropouts=dropouts,
            normalization=normalization,
            output_activation=output_activation,
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z, mu, logvar = self.nets["encoder"](x)
        x = self.nets["decoder"](z)
        return x, z

    def forward_train(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        outputs = OrderedDict()
        x = batch["obs"]
        ground_truth = batch["ground_truth"]
        z, mu, logvar = self.nets["encoder"](x)
        reconstruction = self.nets["decoder"](z)
        reconstruction_loss = self.criterion(reconstruction, ground_truth)
        kld_loss = self.compute_kld(mu, logvar)
        outputs["reconstruction"] = reconstruction
        outputs["z"] = z
        outputs["mu"] = mu
        outputs["logvar"] = logvar
        outputs["reconstruction_loss"] = reconstruction_loss
        outputs["kld_loss"] = kld_loss
        outputs["loss"] = reconstruction_loss + kld_loss
        return outputs

    def compute_kld(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        # kld_weight = 1e2 * mu.size(1) / (224 * 224 * 3 * batch_size)  # TODO
        kld_loss = (
            torch.mean(
                -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1),
                dim=0,
            )
            * self.kld_weight
        )
        return kld_loss


class SlotAttentionEncoder(VisionModule):
    """
    Slot Attention Encoder
    """

    def __init__(
        self,
        input_size: List[int] = [224, 224],
        input_channel: int = 3,
        channels: List[int] = [64, 64, 64, 64, 64, 64],
        kernel_sizes: List[int] = [5, 5, 5, 5, 5, 5],
        strides: List[int] = [1, 1, 1, 1, 1, 1],
        paddings: List[int] = [2, 2, 2, 2, 2, 2],
        num_iters: int = 3,
        eps: float = 1e-8,
        hidden_dim: int = 64,
        mlp_hidden_dim: int = 128,
        activation: nn.Module = nn.ReLU,
        dropouts: Optional[List[float]] = None,
        normalization=None,
        output_activation: Optional[nn.Module] = nn.ReLU,
        num_slots: int = 7,
        **kwargs,
    ) -> None:
        super(SlotAttentionEncoder, self).__init__()

        self.nets = nn.ModuleDict()

        self.nets["conv"] = CNN(
            input_channel=input_channel,
            channels=channels,
            kernel_sizes=kernel_sizes,
            strides=strides,
            paddings=paddings,
            activation=activation,
            dropouts=dropouts,
            normalization=normalization,
            output_activation=output_activation,
        )

        self.nets["slot_attention"] = SlotAttention(
            num_slots=num_slots,
            dim=hidden_dim,
            num_iters=num_iters,
            eps=eps,
            hidden_dim=mlp_hidden_dim,
        )

        self.nets["pos_encoder"] = SoftPositionEmbed(
            hidden_dim=hidden_dim, resolution=input_size
        )

        self.nets["layer_norm"] = nn.LayerNorm(hidden_dim)
        self.nets["permute"] = Permute([0, 2, 3, 1])
        # self.nets["reshape"] = Reshape([-1, inputs_size[0] * inputs_size[1], hid_dim])

        self.nets["mlp_encoder"] = MLP(
            input_dim=hidden_dim,
            output_dim=hidden_dim,
            layer_dims=[hidden_dim],
            activation=activation,
            dropouts=dropouts,
            normalization=None,
            output_activation=None,
        )

    def forward(
        self, x: torch.Tensor, vectorized_slot_feature: bool = False
    ) -> torch.Tensor:
        x = self.nets["conv"](x)  # [B, 64, 128, 128]
        x = self.nets["permute"](x)  # [B, 128, 128, 64]
        x = self.nets["pos_encoder"](x)  # [B, 128, 128, 64]
        x = x.view(x.shape[0], -1, x.shape[-1])  # [B, 128*128, 64]
        x = self.nets["layer_norm"](x)
        x = self.nets["mlp_encoder"](x)
        if vectorized_slot_feature:
            x = self.nets["slot_attention"](x).view(x.shape[0], -1)
        else:
            x = self.nets["slot_attention"](x)
        return x


class SlotDecoder(nn.Module):
    def __init__(self, hid_dim, resolution):
        super().__init__()
        self.decoder_initial_size = (8, 8)
        self.decoder_pos = SoftPositionEmbed(hid_dim, self.decoder_initial_size)

        self.conv1 = nn.ConvTranspose2d(hid_dim, hid_dim, 4, stride=(2, 2), padding=2)
        self.conv2 = nn.ConvTranspose2d(hid_dim, hid_dim, 4, stride=(2, 2), padding=1)
        self.conv3 = nn.ConvTranspose2d(hid_dim, hid_dim, 4, stride=(2, 2), padding=1)
        self.conv4 = nn.ConvTranspose2d(hid_dim, hid_dim, 4, stride=(2, 2), padding=1)
        self.conv5 = nn.ConvTranspose2d(hid_dim, 4, 4, stride=(2, 2), padding=1)

        self.resolution = resolution

    def forward(self, x):
        x = self.decoder_pos(x)  # [-1, 8, 8, slot_dim]
        x = x.permute(0, 3, 1, 2)  # [-1, slot_dim, 8, 8]
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.conv5(x)
        x = x.permute(0, 2, 3, 1)  # [B, 128, 128, 4]
        return x


class SlotAttentionDecoder(VisionModule):
    """
    Slot Attention Encoder
    """

    def __init__(
        self,
        input_size: List[int] = [128, 128],
        input_channel: int = 3,
        channels: List[int] = [8, 16, 32, 64, 128, 256],
        kernel_sizes: List[int] = [3, 3, 3, 3, 3, 3],
        strides: List[int] = [2, 2, 2, 2, 2, 2],
        paddings: List[int] = [1, 1, 1, 1, 1, 1],
        latent_dim: int = 16,
        mean_var: bool = False,
        activation: nn.Module = nn.ReLU,
        dropouts: Optional[List[float]] = None,
        normalization=None,
        output_activation: Optional[nn.Module] = None,
        num_slots: int = 7,
        **kwargs,
    ) -> None:
        super(SlotAttentionDecoder, self).__init__()
        hid_dim = 64

        self.nets = nn.ModuleDict()

        self.decoder_cnn = SlotDecoder(hid_dim, input_size)

        self.nets["layer_norm"] = nn.LayerNorm(hid_dim)

        self.nets["mlp_encoder"] = MLP(
            input_dim=hid_dim,
            output_dim=hid_dim,
            layer_dims=[hid_dim],
            activation=activation,
            dropouts=None,
            normalization=None,
            output_activation=None,
        )

    def forward(self, slots: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor]]:
        batch_size = slots.shape[0]
        # slots [B, num_slots, slot_dim]
        # slots = slots.reshape((-1, slots.shape[-1])).unsqueeze(1).unsqueeze(2)
        slots = slots.view((-1, slots.shape[-1])).unsqueeze(1).unsqueeze(2)
        slots = slots.repeat((1, 8, 8, 1))

        # `slots` has shape: [batch_size*num_slots, width_init, height_init, slot_size].
        x = self.decoder_cnn(slots)
        # `x` has shape: [batch_size*num_slots, width, height, num_channels+1].

        # Undo combination of slot and batch dimension; split alpha masks.
        recons, masks = x.view(
            batch_size, -1, x.shape[1], x.shape[2], x.shape[3]
        ).split([3, 1], dim=-1)
        # `recons` has shape: [batch_size, num_slots, width, height, num_channels].
        # `masks` has shape: [batch_size, num_slots, width, height, 1].

        # Normalize alpha masks over slots.
        masks = nn.Softmax(dim=1)(masks)
        recon_combined = torch.sum(recons * masks, dim=1)  # Recombine image.
        recon_combined = recon_combined.permute(0, 3, 1, 2)
        # `recon_combined` has shape: [batch_size, width, height, num_channels].

        return recon_combined, recons, masks, slots


class SlotAttentionAutoEncoder(VisionModule):
    """
    Slot Attention AutoEncoder
    """

    def __init__(
        self,
        input_size: List[int] = [128, 128],  # [224, 224],
        input_channel: int = 3,
        channels: List[int] = [64, 64, 64, 64],
        encoder_kernel_sizes: List[int] = [5, 5, 5, 5],
        decoder_kernel_sizes: List[int] = [3, 4, 4, 4],
        strides: List[int] = [1, 1, 1, 1],
        paddings: List[int] = [2, 2, 2, 2],
        latent_dim: int = 16,
        activation: nn.Module = nn.ReLU,
        dropouts: Optional[List[float]] = None,
        normalization=nn.BatchNorm2d,
        output_activation: Optional[nn.Module] = nn.Sigmoid,
        **kwargs,
    ) -> None:
        super(SlotAttentionAutoEncoder, self).__init__()

        self.nets = nn.ModuleDict()
        self.nets["encoder"] = SlotAttentionEncoder()
        self.nets["decoder"] = SlotAttentionDecoder()

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor]]:
        x = self.nets["encoder"](x)
        x = self.nets["decoder"](x)
        return x


class Resnet(VisionModule):
    def __init__(
        self,
        input_size: List[int] = [224, 224],
        input_channel: int = 3,
        resnet_type: str = "resnet18",  # resnet18, resnet34, resnet50, resnet101, resnet152
        input_coord_conv: bool = False,
        pretrained: bool = False,
        pool: Optional[str] = "SpatialSoftmax",  # TODO seperate it from resnet
        latent_dim: Optional[int] = None,
        pool_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        super(Resnet, self).__init__()

        assert input_channel == 3, "input_channel should be 3"

        output_shape = [math.ceil(input_size[0] / 32), math.ceil(input_size[1] / 32)]

        RESNET_TYPES = ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"]
        RESNET_OUTPUT_DIM = {
            "resnet18": [512] + output_shape,
            "resnet34": [512] + output_shape,
            "resnet50": [2048] + output_shape,
            "resnet101": [2048] + output_shape,
            "resnet152": [2048] + output_shape,
        }

        assert (
            resnet_type in RESNET_TYPES
        ), f"resnet_type should be one of {RESNET_TYPES}"

        RESNET_WEIGHTS = {
            "resnet18": vision_models.ResNet18_Weights.DEFAULT,
            "resnet34": vision_models.ResNet34_Weights.DEFAULT,
            "resnet50": vision_models.ResNet50_Weights.DEFAULT,
            "resnet101": vision_models.ResNet101_Weights.DEFAULT,
            "resnet152": vision_models.ResNet152_Weights.DEFAULT,
        }
        if pretrained:
            weights = RESNET_WEIGHTS[resnet_type]
            self.transform = RESNET_WEIGHTS[resnet_type].transforms()
        else:
            weights = None

        self.pretrained = pretrained
        self.nets = nn.ModuleDict()
        resnet = getattr(vision_models, resnet_type)(weights=weights)

        # for coord conv
        if input_coord_conv:
            resnet.conv1 = CoordConv(
                input_channel,
                64,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
            )
        elif input_channel != 3:
            resnet.conv1 = nn.Conv2d(
                input_channel,
                64,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
            )

        # for pool
        if pool is not None:
            pool_kwargs.update({"input_shape": RESNET_OUTPUT_DIM[resnet_type]})
            self.pool = eval(pool)(**pool_kwargs)
        else:
            self.pool = None

        resnet_conv = list(resnet.children())[
            :-2
        ]  # [B] + RESNET_OUTPUT_DIM[resnet_type]

        encoder_list = []
        encoder_list.extend(resnet_conv)
        if self.pool is not None:
            encoder_list.append(self.pool)
        encoder_list.append(nn.Flatten(start_dim=1, end_dim=-1))
        if latent_dim is not None:
            if self.pool is not None:
                encoder_list.append(
                    nn.Linear(int(np.prod(self.pool.output_dim)), latent_dim)
                )
            else:
                encoder_list.append(
                    nn.Linear(int(np.prod(RESNET_OUTPUT_DIM[resnet_type])), latent_dim)
                )
        self.nets["encoder"] = nn.Sequential(*encoder_list)

        self.output_dim = (
            latent_dim
            if latent_dim is not None
            else np.prod(RESNET_OUTPUT_DIM[resnet_type])
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.pretrained:
            x = self.preprocess(x)
        x = self.nets["encoder"](x)
        return x

    def preprocess(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.transform(inputs)


class R3M(VisionModule):
    def __init__(
        self,
        input_size: List[int] = [224, 224],
        input_channel: int = 3,
        r3m_type: str = "resnet18",
        pretrained: bool = True,
        **kwargs,
    ) -> None:
        super(R3M, self).__init__()

        try:
            from r3m import load_r3m
        except ImportError:
            print(
                "WARNING: could not load r3m library! Please follow https://github.com/facebookresearch/r3m to install R3M"
            )

        R3M_TYPES = ["resnet18", "resnet34", "resnet50"]
        R3M_OUTPUT_DIM = {
            "resnet18": 512,
            "resnet34": 512,
            "resnet50": 2048,
        }
        assert input_channel == 3  # R3M only support input image with channel size 3
        assert r3m_type in R3M_TYPES, f"resnet_type should be one of {R3M_TYPES}"

        self.nets = nn.ModuleDict()
        self.nets["encoder"] = load_r3m(r3m_type)

        # self.nets["encoder"] = nn.Sequential(
        #     *list(load_r3m(r3m_type).module.convnet.children())
        # )
        self._input_channel = input_channel
        self._r3m_type = r3m_type
        self._freeze = pretrained

        self.output_dim = R3M_OUTPUT_DIM[r3m_type]

        self.preprocess = transforms.Compose(
            [
                transforms.Resize((224, 224)),
            ]
        )

        # self.preprocess = transforms.Compose(
        #     [
        #         transforms.Resize(256),
        #         transforms.CenterCrop(224),
        #         # transforms.ToPILImage(),
        #     ]
        # )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x expected to be [B, C, H, W] with C=3 and torch tensor of uint8 [0, 255]
        x = TensorUtils.to_float(x)
        x = self.preprocess(x)
        x = self.nets["encoder"](x)
        return x

    # def process_obs(self, obs: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
    #     """
    #     Images like (B, T, H, W, C) or (B, H, W, C) or (H, W, C) torch tensor or numpy ndarray of uint8.
    #     Processing obs into a form that can be fed into the encoder like (B*T, C, H, W) or (B, C, H, W) torch tensor of float32.
    #     """
    #     assert len(obs.shape) == 5 or len(obs.shape) == 4 or len(obs.shape) == 3
    #     obs = self.preprocess(obs)
    #     obs = TensorUtils.to_float(TensorUtils.to_tensor(obs))  # to torch float tensor
    #     obs = TensorUtils.to_device(obs, "cuda:0")  # to cuda
    #     # to Batched 4D tensor
    #     obs = obs.view(-1, obs.shape[-3], obs.shape[-2], obs.shape[-1])
    #     # to BHWC to BCHW and contigious
    #     obs = TensorUtils.contiguous(obs.permute(0, 3, 1, 2))
    #     # normalize
    #     # obs = self.normalizer(obs)
    #     # obs = (
    #     #     obs - self.mean
    #     # ) / self.std  # to [0, 1] of [B, C, H, W] torch float tensor
    #     return obs


class CLIP(VisionModule):
    def __init__(
        self,
        input_size: List[int] = [224, 224],
        input_channel: int = 3,
        clip_type: str = "ViT-B/32",
        pretrained: bool = True,
        **kwargs,
    ) -> None:
        super(CLIP, self).__init__()

        try:
            import clip
        except ImportError:
            print(
                "WARNING: could not load r3m library! Please follow https://github.com/openai/CLIP to install clip"
            )

        CLIP_TYPES = ["resnet18", "resnet34", "resnet50"]
        CLIP_OUTPUT_DIM = {
            "resnet18": 512,
            "resnet34": 512,
            "resnet50": 2048,
        }
        assert input_channel == 3  # CLIP only support input image with channel size 3
        assert clip_type in CLIP_TYPES, f"clip_type should be one of {CLIP_TYPES}"

        self.nets = nn.ModuleDict()
        self.nets["encoder"], self.preprocess = clip.load(clip_type)


class MVP(VisionModule):
    def __init__(
        self,
        input_channel=3,
        mvp_model_class="vitb-mae-egosoup",
        freeze=True,
        **kwargs,
    ):
        super(MVP, self).__init__()

        try:
            import mvp
        except ImportError:
            print(
                "WARNING: could not load mvp library! Please follow https://github.com/ir413/mvp to install MVP."
            )

        self.nets = mvp.load(mvp_model_class)
        if freeze:
            self.nets.freeze()

        assert input_channel == 3  # MVP only support input image with channel size 3
        assert mvp_model_class in [
            "vits-mae-hoi",
            "vits-mae-in",
            "vits-sup-in",
            "vitb-mae-egosoup",
            "vitl-256-mae-egosoup",
        ]  # make sure the selected r3m model do exist

        self._input_channel = input_channel
        self._freeze = freeze
        self._mvp_model_class = mvp_model_class

        if "256" in mvp_model_class:
            input_img_size = 256
        else:
            input_img_size = 224

        self.transform = transforms.Compose(
            [
                transforms.Resize((input_img_size, input_img_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def preprocess(self, inputs: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        preprocess inputs to fit the pretrained model
        """
        assert inputs.ndim == 4
        assert inputs.shape[1] == self._input_channel
        assert inputs.dtype in [np.uint8, np.float32, torch.uint8, torch.float32]
        inputs = TensorUtils.to_tensor(inputs)
        inputs = self.transform(inputs)
        return inputs


if __name__ == "__main__":
    test_input = torch.randn(5, 3, 224, 224)
    pool_kwargs = dict(
        num_kp=32,
        temperature=1.0,
        learnable_temperature=False,
        output_variance=False,
        noise_std=0.0,
    )
    resnet_encoder = Resnet(
        pool="SpatialSoftmax", pool_kwargs=pool_kwargs, latent_dim=64
    )

    test_output = resnet_encoder(test_input)
    print(test_output.shape)
