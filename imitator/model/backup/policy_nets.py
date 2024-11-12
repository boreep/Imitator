from abc import ABC, abstractmethod
from collections import OrderedDict

from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D

from imitator.models.base_nets import MLP, RNN, PositionalEncoding, GPT
import imitator.utils.tensor_utils as TensorUtils
from imitator.utils.obs_utils import (
    ObservationEncoder,
    FloatVectorModality,
    ImageModality,
    get_normalize_params,
)


"""
Policy Networks
flow : ObservationEncoder -> ActorCore[MLP, RNN, ...] -> MLPDecoder
"""


class Actor(ABC, nn.Module):
    """
    Base class for actor networks.
    """

    def __init__(self, cfg: Dict) -> None:
        super(Actor, self).__init__()
        self.cfg = cfg
        self.policy_type = cfg.network.policy.model
        self.action_dim = cfg.actions.dim

        if cfg.actions.normalize:
            action_mean, action_std = get_normalize_params(
                cfg.actions.min, cfg.actions.max
            )
        else:
            action_mean, action_std = 0.0, 1.0
        self.action_modality = eval(cfg.actions.modality)(
            name="actions", shape=cfg.actions.dim, mean=action_mean, std=action_std
        )

        self.nets = nn.ModuleDict()
        self.criterion = eval(
            "nn." + cfg.network.policy.get("criterion", "MSELoss") + "()"
        )
        self.training = True

        self.supervise_all_steps = cfg.network.policy.train.get(
            "supervise_all_steps", False
        )

        self.gmm = cfg.network.policy.gmm.get("enabled", False)
        if self.gmm:
            self.gmm_mode = cfg.network.policy.gmm.get("modes", 5)
            self.min_std = cfg.network.policy.gmm.get("min_std", 0.0001)
            self.low_noise_eval = cfg.network.policy.gmm.get("low_noise_eval", True)
            self.use_tanh = cfg.network.policy.gmm.get("use_tanh", False)
            self.gmm_activation = F.softplus

        self.mlp_decoder_layer_dim = cfg.network.policy.mlp_decoder.layer_dims
        self.mlp_activation = eval(
            "nn." + cfg.network.policy.mlp_decoder.get("activation", "ReLU")
        )
        self.mlp_decoder_output_dim = (
            self.action_dim * self.gmm_mode * 2 + self.gmm_mode
            if self.gmm
            else self.action_dim
        )
        self.mlp_decoder_output_activation = (
            nn.Tanh
            if (
                cfg.network.policy.mlp_decoder.get("squash_output", False)
                and not self.gmm
            )
            else None
        )

    @abstractmethod
    def _build_network(self):
        pass

    @abstractmethod
    def get_action(
        self,
        batch: Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]],
    ) -> torch.Tensor:
        pass

    @abstractmethod
    def forward_train(
        self,
        batch: Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]],
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        pass

    def compute_loss(self, predictions, actions, dists):
        if actions is None:
            return None
        if self.gmm:
            log_probs = dists.log_prob(actions)
            action_loss = -log_probs.mean()
        else:
            action_loss = self.criterion(predictions, actions)
        return action_loss

    def eval(self) -> "Actor":
        self.training = False
        return super(Actor, self).eval()

    def train(self, mode: bool = True) -> "Actor":
        self.training = True
        return super(Actor, self).train(mode)

    def mlp_decoder_postprocess(
        self,
        mlp_decoder_outputs: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
    ):
        outputs = OrderedDict()
        dists = None
        batch_size, _, _ = mlp_decoder_outputs.shape
        if actions is not None:
            actions = self.action_modality.process_obs(actions).reshape(
                batch_size, -1, self.action_dim
            )
        if self.gmm:
            means = mlp_decoder_outputs[
                :, :, : self.action_dim * self.gmm_mode
            ].reshape(batch_size, -1, self.gmm_mode, self.action_dim)
            scales = mlp_decoder_outputs[
                :,
                :,
                self.action_dim * self.gmm_mode : self.action_dim * self.gmm_mode * 2,
            ].reshape(batch_size, -1, self.gmm_mode, self.action_dim)
            logits = mlp_decoder_outputs[
                :, :, self.action_dim * self.gmm_mode * 2 :
            ].reshape(batch_size, -1, self.gmm_mode)

            if not self.use_tanh:  # TODO
                means = torch.tanh(means)
            if self.low_noise_eval and (not self.training):
                scales = torch.ones_like(means) * 1e-4
            else:
                scales = self.gmm_activation(scales) + self.min_std

            component_distribution = D.Normal(loc=means, scale=scales)
            component_distribution = D.Independent(component_distribution, 1)
            mixture_distribution = D.Categorical(logits=logits)

            dists = D.MixtureSameFamily(
                mixture_distribution=mixture_distribution,
                component_distribution=component_distribution,
            )

            if not self.supervise_all_steps:
                component_distribution = D.normal.Normal(
                    loc=dists.component_distribution.base_dist.loc[:, -1],
                    scale=dists.component_distribution.base_dist.scale[:, -1],
                )
                component_distribution = D.Independent(component_distribution, 1)
                mixture_distribution = D.Categorical(
                    logits=dists.mixture_distribution.logits[:, -1]
                )
                dists = D.MixtureSameFamily(
                    mixture_distribution=mixture_distribution,
                    component_distribution=component_distribution,
                )
                # TODO
                actions = actions[:, -1, :]
            outputs["predictions"] = dists.sample()
        else:  # not gmm
            outputs["predictions"] = mlp_decoder_outputs
        outputs["loss"] = self.compute_loss(outputs["predictions"], actions, dists)
        return outputs


class MLPActor(Actor):
    def __init__(self, cfg: Dict) -> None:
        super(MLPActor, self).__init__(cfg)
        self.mlp_input_dim = sum(
            [cfg.obs[key].obs_encoder.output_dim for key in cfg.obs.keys()]
        )  # sum of all obs_encoder output dims
        self.mlp_decoder_layer_dim = cfg.network.policy.mlp_decoder.layer_dims
        self.mlp_activation = eval(
            "nn." + cfg.network.policy.mlp_decoder.get("activation", "ReLU")
        )
        self.mlp_kwargs = cfg.network.policy.get("kwargs", {})
        self._build_network()

    def _build_network(self) -> None:
        self.nets["obs_encoder"] = ObservationEncoder(self.cfg.obs)
        self.nets["mlp_decoder"] = MLP(
            input_dim=self.mlp_input_dim,
            layer_dims=self.mlp_decoder_layer_dim,
            output_dim=self.mlp_decoder_output_dim,
            activation=self.mlp_activation,
            output_activation=self.mlp_decoder_output_activation,
        )

    def forward_train(
        self,
        batch: Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]],
    ) -> torch.Tensor:
        obs_latents = self.nets["obs_encoder"](batch["obs"])  # [B, T, D]
        mlp_decoder_outputs = self.nets["mlp_decoder"](obs_latents)  # [B, T, D]
        actions = batch.get("actions", None)
        outputs = self.mlp_decoder_postprocess(mlp_decoder_outputs, actions)
        return outputs

    def forward(
        self,
        batch: Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]],
    ) -> torch.Tensor:
        return self.forward_train(batch)

    def get_action(
        self,
        batch: Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch = TensorUtils.to_sequence(batch)  # [B, 1, D]
        actions = self.forward(batch)["predictions"]  # [B, D]
        actions = TensorUtils.squeeze(actions, 1)  # [B, D]
        actions = self.action_modality.unprocess_obs(actions)  # numpy ndarray
        return actions


class RNNActor(Actor):
    def __init__(self, cfg: Dict) -> None:
        super(RNNActor, self).__init__(cfg)
        self.rnn_type = cfg.network.policy.rnn.type  # ["LSTM" "GRU"]
        self.rnn_num_layers = cfg.network.policy.rnn.rnn_num_layers
        self.rnn_hidden_dim = cfg.network.policy.rnn.rnn_hidden_dim
        self.rnn_input_dim = sum(
            [cfg.obs[key].obs_encoder.output_dim for key in cfg.obs.keys()]
        )  # sum of all obs_encoder output dims
        self.rnn_horizon = cfg.network.policy.rnn.get("rnn_horizon", 10)
        self.open_loop = cfg.network.policy.rnn.get("open_loop", False)  # TODO
        self.rnn_kwargs = cfg.network.policy.rnn.get("kwargs", {})
        self.supervise_all_steps = cfg.network.policy.train.get(
            "supervise_all_steps", True
        )
        self._build_network()

    def _build_network(self) -> None:
        self.nets["obs_encoder"] = ObservationEncoder(self.cfg.obs)
        self.nets["mlp_decoder"] = MLP(
            input_dim=self.rnn_hidden_dim,
            layer_dims=self.mlp_decoder_layer_dim,
            output_dim=self.mlp_decoder_output_dim,
            activation=self.mlp_activation,
            output_activation=self.mlp_decoder_output_activation,
        )
        self.nets["rnn"] = RNN(
            rnn_input_dim=self.rnn_input_dim,
            rnn_hidden_dim=self.rnn_hidden_dim,
            rnn_num_layers=self.rnn_num_layers,
            rnn_type=self.rnn_type,
            rnn_kwargs=self.rnn_kwargs,
            per_step_net=self.nets["mlp_decoder"],
        )

    def get_rnn_init_state(
        self, batch_size: int, device: torch.device
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        return self.nets["rnn"].get_rnn_init_state(batch_size, device)

    def forward_train(
        self,
        batch: Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]],
        rnn_state: Optional[torch.Tensor] = None,
        return_rnn_state: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        outputs = OrderedDict()
        obs_latents = self.nets["obs_encoder"](batch["obs"])
        assert obs_latents.ndim == 3  # [B, T, D]
        mlp_decoder_outputs, rnn_state = self.nets["rnn"](
            inputs=obs_latents, rnn_state=rnn_state, return_rnn_state=True
        )
        actions = batch.get("actions", None)
        outputs = self.mlp_decoder_postprocess(mlp_decoder_outputs, actions)
        if return_rnn_state:
            return outputs, rnn_state
        else:
            return outputs

    def forward(
        self,
        batch: Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]],
        rnn_state: Optional[torch.Tensor] = None,
        return_rnn_state: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        return self.forward_train(batch, rnn_state, return_rnn_state)

    def get_action(
        self,
        batch: Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]],
        rnn_state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch = TensorUtils.to_sequence(batch)
        outputs, rnn_state = self.forward(batch, rnn_state, return_rnn_state=True)
        actions = outputs["predictions"]
        actions = self.action_modality.unprocess_obs(actions)  # numpy ndarray
        actions = actions[:, 0, :]
        return actions, rnn_state


class TransformerActor(Actor):
    """
    Actor with Transformer encoder and MLP decoder
    """

    def __init__(self, cfg: Dict) -> None:
        super(TransformerActor, self).__init__(cfg)
        self.transformer_type = cfg.network.policy.transformer.type  # TODO
        self.transformer_num_layers = cfg.network.policy.transformer.get(
            "num_layers", 6
        )
        self.transformer_num_heads = cfg.network.policy.transformer.get("num_heads", 8)
        self.transformer_embed_dim = cfg.network.policy.transformer.get(
            "embed_dim", 512
        )
        self.transformer_input_dim = sum(
            [cfg.obs[key].obs_encoder.output_dim for key in cfg.obs.keys()]
        )  # sum of all obs_encoder output dims
        self.transformer_kwargs = cfg.network.policy.transformer.get("kwargs", {})
        self.transformer_embed_dropout = cfg.network.policy.transformer.get(
            "embed_dropout", 0.1
        )
        self.transformer_attn_dropout = cfg.network.policy.transformer.get(
            "attn_dropout", 0.1
        )
        self.transformer_block_dropout = cfg.network.policy.transformer.get(
            "block_dropout", 0.1
        )
        self.transformer_activation = cfg.network.policy.transformer.get(
            "activation", "gelu"
        )
        self.context_length = cfg.network.policy.transformer.get("context_length", 10)
        self.supervise_all_steps = cfg.network.policy.train.get(
            "supervise_all_steps", True
        )
        self._build_network()

    def _build_network(self) -> None:
        self.params = nn.ParameterDict()
        self.params["embed_timestep"] = nn.Parameter(
            torch.zeros(1, self.context_length, self.transformer_embed_dim)
        )  # TODO : pos enc or something more
        self.nets["obs_encoder"] = ObservationEncoder(self.cfg.obs)  #
        self.nets["embedding"] = MLP(
            input_dim=self.transformer_input_dim,
            layer_dims=[],
            output_dim=self.transformer_embed_dim,
        )
        # self.nets["embedding"] = nn.Linear(self.transformer_input_dim, self.transformer_embed_dim)
        self.nets["embed_ln"] = nn.LayerNorm(self.transformer_embed_dim)
        self.nets["embed_drop"] = nn.Dropout(self.transformer_embed_dropout)
        self.nets["transformer"] = GPT(
            embed_dim=self.transformer_embed_dim,
            context_length=self.context_length,
            attn_dropout=self.transformer_attn_dropout,
            block_output_dropout=self.transformer_block_dropout,
            num_layers=self.transformer_num_layers,
            num_heads=self.transformer_num_heads,
            activation=self.transformer_activation,
        )
        self.nets["mlp_decoder"] = MLP(
            input_dim=self.transformer_embed_dim,
            layer_dims=self.mlp_decoder_layer_dim,
            output_dim=self.mlp_decoder_output_dim,
            activation=self.mlp_activation,
            output_activation=self.mlp_decoder_output_activation,
        )

    def embed_timesteps(self, embeddings):
        """
        Computes timestep-based embeddings (aka positional embeddings) to add to embeddings.
        Args:
            embeddings (torch.Tensor): embeddings prior to positional embeddings are computed
        Returns:
            time_embeddings (torch.Tensor): positional embeddings to add to embeddings
        """

        time_embeddings = self.params["embed_timestep"]
        return time_embeddings

    def input_embedding(
        self,
        inputs,
    ):
        """
        Process encoded observations into embeddings to pass to transformer,
        Adds timestep-based embeddings (aka positional embeddings) to inputs.
        Args:
            inputs (torch.Tensor): outputs from observation encoder
        Returns:
            embeddings (torch.Tensor): input embeddings to pass to transformer backbone.
        """
        # embeddings = self.nets["embed_encoder"](inputs)
        embeddings = self.nets["embedding"](inputs)
        time_embeddings = self.embed_timesteps(embeddings)
        embeddings = embeddings + time_embeddings
        embeddings = self.nets["embed_ln"](embeddings)
        embeddings = self.nets["embed_drop"](embeddings)

        return embeddings

    def forward_train(
        self,
        batch: Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]],
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        obs_latents = self.nets["obs_encoder"](batch["obs"])  # [B, T, D] falttened
        assert obs_latents.ndim == 3  # [B, T, D]
        transformer_embeddings = self.input_embedding(obs_latents)
        transformer_encoder_outputs = self.nets["transformer"].forward(
            transformer_embeddings
        )
        mlp_decoder_outputs = self.nets["mlp_decoder"](transformer_encoder_outputs)
        actions = batch.get("actions", None)
        outputs = self.mlp_decoder_postprocess(mlp_decoder_outputs, actions)
        return outputs

    def forward(
        self,
        batch: Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]],
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        return self.forward_train(batch)

    def get_action(
        self,
        batch: Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        actions = self.forward(batch)["predictions"]  # [B, T, D]
        actions = self.action_modality.unprocess_obs(actions)  # numpy ndarray
        if self.supervise_all_steps:  # get last timestep
            actions = actions[:, -1, :]
        else:
            actions = actions
        return actions  # [B, D], last timestep cause it uses framestack
