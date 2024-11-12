from abc import ABC, abstractmethod
import numpy as np
from copy import deepcopy
from collections import OrderedDict

import yaml

import torch
import torch.nn as nn

import imitator.utils.tensor_utils as TensorUtils
from imitator.models.base_nets import *
from imitator.models.obs_nets import AutoEncoder, VariationalAutoEncoder, Resnet

from torchvision import transforms as T

from typing import Union, List, Tuple, Dict


# fucntion that get mean and std from max and min
def get_normalize_params(min_val, max_val):
    min_array = np.array(min_val).astype(np.float32)
    max_array = np.array(max_val).astype(np.float32)
    mean = (min_array + max_array) / 2.0
    std = (max_array - min_array) / 2.0
    return mean, std


def get_obs_modality_from_config(obs_key, config):
    obs_modality = config["obs"][obs_key]["modality"]
    return obs_modality


def obs_to_modality_dict(config):
    obs_modality_dict = OrderedDict()
    for obs_key in config.obs.keys():
        obs_modality_dict[obs_key] = get_obs_modality_from_config(obs_key, config)
    return obs_modality_dict


def concatenate_image(
    image1: Union[np.ndarray, torch.Tensor], image2: Union[np.ndarray, torch.Tensor]
) -> np.ndarray:
    assert image1.ndim == image2.ndim == 3
    if isinstance(image1, torch.Tensor):
        image1 = TensorUtils.to_numpy(image1)
    if isinstance(image2, torch.Tensor):
        image2 = TensorUtils.to_numpy(image2)
    if image1.dtype != np.uint8:
        image1 = (image1 * 255).astype(np.uint8)
    if image2.dtype != np.uint8:
        image2 = (image2 * 255).astype(np.uint8)

    assert image1.shape == image2.shape
    image = np.concatenate([image1, image2], axis=1)
    return image


class AddGaussianNoise(object):
    """
    Input is (B, C, H, W) or  (C, H, W) [0, 1] float tensor
    """

    def __init__(self, mean=0.0, std=1.0, p=0.5):
        self.std = std
        self.mean = mean
        self.p = p

    def __call__(self, tensor):
        if torch.rand(1) < self.p:
            # tensor = tensor + torch.randn(tensor.size()) * self.std + self.mean
            tensor = (
                tensor
                + torch.randn(tensor.size()).to(tensor.device) * self.std
                + self.mean
            )
            # # clip to [0, 1]
            tensor = torch.clamp(tensor, 0.0, 1.0)
            return tensor

        return tensor

    def __repr__(self):
        return self.__class__.__name__ + "(mean={0}, std={1}, p={2})".format(
            self.mean, self.std, self.p
        )


class RGBShifter(object):
    """
    Input is (B, C, H, W) or  (C, H, W) [0, 1] float tensor
    """

    def __init__(self, r_shift_limit=0.2, g_shift_limit=0.2, b_shift_limit=0.2, p=0.5):
        self.r_shift_limit = r_shift_limit
        self.g_shift_limit = g_shift_limit
        self.b_shift_limit = b_shift_limit
        self.p = p

    def __call__(self, tensor):
        # tensor : [B, C, H, W] or [C, H, W]
        if torch.rand(1) < self.p:
            r_shift = torch.rand(1) * self.r_shift_limit * 2 - self.r_shift_limit  #
            g_shift = torch.rand(1) * self.g_shift_limit * 2 - self.g_shift_limit
            b_shift = torch.rand(1) * self.b_shift_limit * 2 - self.b_shift_limit
            if tensor.ndim == 4:
                tensor[:, 0, :, :] += r_shift.to(tensor.device)
                tensor[:, 1, :, :] += g_shift.to(tensor.device)
                tensor[:, 2, :, :] += b_shift.to(tensor.device)
            elif tensor.ndim == 3:
                tensor[0, :, :] += r_shift.to(tensor.device)
                tensor[1, :, :] += g_shift.to(tensor.device)
                tensor[2, :, :] += b_shift.to(tensor.device)
            else:
                raise NotImplementedError
            # clip to [0, 1]
            tensor = torch.clamp(tensor, 0.0, 1.0)
        return tensor

    def __repr__(self):
        return (
            self.__class__.__name__
            + "(r_shift_limit={0}, g_shift_limit={1}, b_shift_limit={2}, p={3})".format(
                self.r_shift_limit, self.g_shift_limit, self.b_shift_limit, self.p
            )
        )


class Modality(ABC, nn.Module):
    def __init__(
        self,
        name: str,
        shape: Union[int, List[int], Tuple[int]],
        mean: Union[float, List[float], np.ndarray, torch.Tensor] = 0.0,
        std: Union[float, List[float], np.ndarray, torch.Tensor] = 1.0,
    ) -> None:
        super(Modality, self).__init__()
        self.name = name
        self.shape = shape
        self.dim = 1 if isinstance(shape, int) else len(shape)
        self.mean = mean
        self.std = std
        self.set_scaler(mean, std)

    def set_scaler(
        self,
        mean: Union[float, List[float], np.ndarray, torch.Tensor],
        std: Union[float, List[float], np.ndarray, torch.Tensor],
    ) -> None:
        self.normalizer = Normalize(mean, std)
        self.unnormalizer = Unnormalize(mean, std)

    @abstractmethod
    def _default_process_obs(
        self, obs: Union[np.ndarray, torch.Tensor]
    ) -> torch.Tensor:
        return self.normalizer(obs)

    @abstractmethod
    def _default_unprocess_obs(self, obs: torch.Tensor) -> np.ndarray:
        return self.unnormalizer(obs)

    def set_obs_processor(self, processor=None):
        self._custom_obs_processor = processor

    def set_obs_unprocessor(self, unprocessor=None):
        self._custom_obs_unprocessor = unprocessor

    def process_obs(self, obs: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        if hasattr(self, "_custom_obs_processor"):
            return self._custom_obs_processor(obs)
        else:
            return self._default_process_obs(obs)

    def unprocess_obs(self, obs: torch.Tensor) -> np.ndarray:
        if hasattr(self, "_custom_obs_unprocessor"):
            return self._custom_obs_unprocessor(obs)
        else:
            return self._default_unprocess_obs(obs)


class ImageModality(Modality):
    def __init__(
        self,
        name: str,
        shape: Union[List[int], Tuple[int]] = [224, 224, 3],
        mean: Union[float, List[float], np.ndarray, torch.Tensor] = 0.0,
        std: Union[float, List[float], np.ndarray, torch.Tensor] = 255.0,
    ) -> None:
        super(ImageModality, self).__init__(name, shape, mean, std)

        assert self.dim == 3
        self.height = shape[0]
        self.width = shape[1]
        self.num_channels = shape[2]

    def _default_process_obs(
        self, obs: Union[np.ndarray, torch.Tensor]
    ) -> torch.Tensor:
        """
        Images like (B, T, H, W, C) or (B, H, W, C) or (H, W, C) torch tensor or numpy ndarray of uint8.
        Processing obs into a form that can be fed into the encoder like (B*T, C, H, W) or (B, C, H, W) torch tensor of float32.
        """
        assert len(obs.shape) == 5 or len(obs.shape) == 4 or len(obs.shape) == 3
        obs = TensorUtils.to_float(TensorUtils.to_tensor(obs))  # to torch float tensor
        obs = TensorUtils.to_device(obs, "cuda:0")  # to cuda
        # to Batched 4D tensor
        obs = obs.view(-1, obs.shape[-3], obs.shape[-2], obs.shape[-1])
        # to BHWC to BCHW and contigious
        obs = TensorUtils.contiguous(obs.permute(0, 3, 1, 2))
        # normalize
        obs = self.normalizer(obs)
        # obs = (
        #     obs - self.mean
        # ) / self.std  # to [0, 1] of [B, C, H, W] torch float tensor
        return obs

    def _default_unprocess_obs(self, processed_obs: torch.Tensor) -> np.ndarray:
        """
        Images like (B, C, H, W) torch tensor.
        Unprocessing obs into a form that can be fed into the decoder like (B, H, W, C) numpy ndarray of uint8.
        """
        assert len(processed_obs.shape) == 4
        # to [0, 255] of [B, C, H, W] torch float tensor
        # unprocessed_obs = TensorUtils.to_numpy(processed_obs * 255.0)
        unprocessed_obs = self.unnormalizer(processed_obs)
        unprocessed_obs = TensorUtils.to_numpy(unprocessed_obs)
        # to BCHW to BHWC
        unprocessed_obs = unprocessed_obs.transpose(0, 2, 3, 1)
        # to numpy ndarray of uint8
        unprocessed_obs = unprocessed_obs.astype(np.uint8)
        return unprocessed_obs


class FloatVectorModality(Modality):
    def __init__(
        self,
        name: str,
        shape: Union[int, List[int], Tuple[int]],
        mean: Union[float, List[float], np.ndarray, torch.Tensor] = 0.0,
        std: Union[float, List[float], np.ndarray, torch.Tensor] = 1.0,
    ) -> None:
        super(FloatVectorModality, self).__init__(name, shape, mean, std)
        assert self.dim == 1

    def _default_process_obs(
        self, obs: Union[np.ndarray, torch.Tensor]
    ) -> torch.Tensor:
        """
        Vector like (B, T, D) or (B, D) or (D) torch tensor or numpy ndarray of float.
        Processing obs into a form that can be fed into the encoder like (B*T, D) or (B, D) torch tensor of float32.
        """
        assert len(obs.shape) == 3 or len(obs.shape) == 2 or len(obs.shape) == 1
        obs = TensorUtils.to_float(TensorUtils.to_tensor(obs))  # to torch float tensor
        obs = TensorUtils.to_device(obs, "cuda:0")  # to cuda
        obs = TensorUtils.contiguous(obs)  # to contigious
        obs = obs.view(-1, obs.shape[-1])
        # obs = (obs - self.mean) / self.std  # normalize
        obs = self.normalizer(obs)
        return obs

    def _default_unprocess_obs(self, processed_obs: torch.Tensor) -> np.ndarray:
        """
        Vector like (B, D) or (B, T, D) torch tensor
        Unprocessing obs into a form that can be fed into the decoder like (B, D) or (B, T, D) numpy ndarray of float32.
        """
        assert len(processed_obs.shape) == 2 or len(processed_obs.shape) == 3
        # unprocessed_obs = processed_obs * self.std + self.mean
        unprocessed_obs = self.unnormalizer(processed_obs)
        unprocessed_obs = TensorUtils.to_numpy(unprocessed_obs)
        return unprocessed_obs


class ModalityEncoderBase(nn.Module):
    # pass
    def __init__(
        self, obs_name: str, modality: Union[ImageModality, FloatVectorModality]
    ) -> None:
        super(ModalityEncoderBase, self).__init__()
        self.obs_name = obs_name
        self.modality = modality


class ImageModalityEncoder(ModalityEncoderBase):
    def __init__(self, cfg: Dict, obs_name: str) -> None:
        self.cfg = cfg

        self.normalize = cfg.normalize
        self.data_augmentation = cfg.data_augmentation
        self.input_dim = cfg.obs_encoder.input_dim
        self.output_dim = cfg.obs_encoder.output_dim
        self.layer_dims = cfg.obs_encoder.layer_dims
        self.trainable = cfg.obs_encoder.trainable
        self.has_decoder = cfg.obs_encoder.has_decoder
        self.encoder_model = cfg.obs_encoder.model
        self.freeze = cfg.obs_encoder.freeze
        self.model_kwargs = cfg.obs_encoder.model_kwargs
        self.activation = eval("nn." + cfg.get("activation", "ReLU"))

        # normalization spec
        # original : [0, 255]
        # if [0, 1] : mean = 0.0, std = 1.0
        # if [-1, 1] : mean = 255.0 / 2.0, std = 255.0 / 2.0
        # if normalize with MEAN and STD after [0, 1] : mean = 255 * MEAN, std = 255 * STD
        if self.normalize:
            # [0, 255] -> [0, 1]
            mean, std = 0.0, 255.0
        else:
            # do not normalize
            mean = 0.0
            std = 1.0

        # TODO
        super(ImageModalityEncoder, self).__init__(
            obs_name=obs_name,
            modality=ImageModality(
                name=obs_name, shape=self.input_dim, mean=mean, std=std
            ),
        )

        if self.encoder_model in ["AutoEncoder", "VariationalAutoEncoder"]:  # TODO
            self.model = eval(self.encoder_model)(
                input_size=cfg.obs_encoder.input_dim[:2],
                input_channel=cfg.obs_encoder.input_dim[2],
                latent_dim=cfg.obs_encoder.output_dim,
            )
        else:
            self.model = eval(self.encoder_model)(**self.model_kwargs)

        # TODO
        # test = self.model.process_obs
        # self.modality.set_obs_processor(test)

        if not self.trainable:
            if self.encoder_model in ["AutoEncoder", "VariationalAutoEncoder"]:
                self.model.load_state_dict(
                    torch.load(cfg.obs_encoder.model_path)
                )  # TODO
            if self.freeze:
                self.model.freeze()

        self.nets = nn.ModuleDict()
        if self.has_decoder:
            self.nets["encoder"] = self.model.nets["encoder"]
            self.nets["decoder"] = self.model.nets["decoder"]
        else:
            self.nets["encoder"] = self.model

        if self.layer_dims is not None:
            self.nets["mlp_encoder"] = MLP(
                input_dim=self.model.output_dim,
                layer_dims=self.layer_dims,
                output_dim=self.output_dim,
                activation=self.activation,
            )
        else:
            self.nets["mlp_encoder"] = nn.Identity()

        # check requires_grad of encoder and decoder
        # for name, net in self.nets.items():
        #     for param in net.parameters():
        #         print(f"{name} requires_grad: {param.requires_grad}")

    def forward(self, obs: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Images like (B, T, H, W, C) or (B, H, W, C) or (H, W, C) torch tensor or numpy ndarray of uint8.
        return latent like (B*T, D) or (B, D) or (D) torch tensor.
        """
        assert len(obs.shape) == 5 or len(obs.shape) == 4 or len(obs.shape) == 3
        if len(obs.shape) == 5:
            batch_size, seq_len, height, width, channel = obs.shape
        elif len(obs.shape) == 4:
            batch_size, height, width, channel = obs.shape
        else:  # len(obs.shape) == 3
            height, width, channel = obs.shape

        processed_obs = self.modality.process_obs(
            obs
        )  # to [0, 1] of [-1, C, H, W] torch float tensor

        # TODO
        if self.encoder_model in ["AutoEncoder", "VariationalAutoEncoder"]:
            latent, _, _ = self.nets["encoder"](
                processed_obs
            )  # (B, T, D) or (B, D) or (D)
        else:
            latent = self.nets["encoder"](processed_obs)
        if len(obs.shape) == 5:
            latent = latent.view(batch_size, seq_len, -1)  # (B, T, D)
        elif len(obs.shape) == 4:
            latent = latent.view(batch_size, -1)  # (B, D)
        else:  # len(obs.shape) == 3
            latent = latent.view(-1)  # (D)

        latent = self.nets["mlp_encoder"](latent)  # (B, T, D) or (B, D) or (D)
        return latent


class FloatVectorModalityEncoder(ModalityEncoderBase):
    def __init__(self, cfg: Dict, obs_name: str) -> None:
        self.cfg = cfg
        self.input_dim = cfg.obs_encoder.input_dim
        self.output_dim = cfg.obs_encoder.output_dim
        self.layer_dims = cfg.obs_encoder.layer_dims
        self.activation = eval("nn." + cfg.get("activation", "ReLU"))

        self.normalize = cfg.get("normalize", False)
        if self.normalize:
            mean, std = get_normalize_params(cfg.min, cfg.max)
        else:
            mean = 0.0
            std = 1.0

        super(FloatVectorModalityEncoder, self).__init__(
            obs_name=obs_name,
            modality=FloatVectorModality(
                name=obs_name, shape=self.input_dim, mean=mean, std=std
            ),
        )

        self.nets = (
            MLP(
                input_dim=self.input_dim,
                layer_dims=self.layer_dims,
                output_dim=self.output_dim,
                activation=self.activation,
            )
            if self.layer_dims
            else nn.Identity()
        )

    def forward(self, obs: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Vector like (B, T, D) or (B, D) or (D) torch tensor or numpy ndarray of float.
        return Vector like (B*T, D) or (B, D) or (D) torch tensor.
        """
        assert len(obs.shape) == 3 or len(obs.shape) == 2 or len(obs.shape) == 1
        if len(obs.shape) == 3:
            batch_size, seq_len, dim = obs.shape
        elif len(obs.shape) == 2:
            batch_size, dim = obs.shape
        else:  # len(obs.shape) == 1
            dim = obs.shape[0]
        processed_obs = self.modality.process_obs(obs)
        vector = self.nets(processed_obs)
        if len(obs.shape) == 3:
            vector = vector.view(batch_size, seq_len, -1)
        elif len(obs.shape) == 2:
            vector = vector.view(batch_size, -1)
        else:  # len(obs.shape) == 1
            vector = vector.view(-1)
        return vector


class ObservationEncoder(nn.Module):
    """
    Encodes observations into a latent space.
    """

    def __init__(self, cfg: Dict) -> None:
        super(ObservationEncoder, self).__init__()
        self.cfg = cfg
        self._build_encoder()

    def _build_encoder(self) -> None:
        self.nets = nn.ModuleDict()
        for key in self.cfg.keys():
            modality_encoder = eval(self.cfg[key]["modality"] + "Encoder")
            self.nets[key] = modality_encoder(self.cfg[key], key)

    def forward(self, obs_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        obs_dict is expected to be a dictionary with keys of self.obs_keys
        like {"image": image_obs, "robot_ee_pos": robot_ee_pos_obs}
        """
        obs_latents = []
        for key in self.cfg.keys():
            obs_latents.append(self.nets[key](obs_dict[key]))
        obs_latents = torch.cat(obs_latents, dim=-1)
        return obs_latents
