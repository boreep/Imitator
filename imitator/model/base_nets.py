"""
JAX/Flax implementation of base neural networks like MLPs, RNNs, Transformers, etc.
"""

import numpy as np
from typing import Callable, Optional, Sequence, Dict

import jax
import jax.numpy as jnp
import flax
import flax.linen as nn

default_init = nn.initializers.xavier_uniform


class MLP(nn.Module):
    hidden_dims: Sequence[int]
    activation: Callable = nn.relu
    activate_final: bool = False
    use_layer_norm: bool = False
    dropout_rate: Optional[float] = None

    @nn.compact
    def __call__(self, x: jax.Array, train: bool = False) -> jax.Array:
        for i, size in enumerate(self.hidden_dims):
            x = nn.Dense(size, kernel_init=default_init())(x)

            if i + 1 < len(self.hidden_dims) or self.activate_final:
                if self.dropout_rate is not None and self.dropout_rate > 0:
                    x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not train)
                if self.use_layer_norm:
                    x = nn.LayerNorm()(x)
                x = self.activation(x)
        return x


class RNN(nn.Module):
    rnn_input_dim: int
    rnn_hidden_dim: int
    rnn_num_layers: int
    rnn_type: str = "LSTM"
    rnn_kwargs: Optional[Dict] = None
    per_step_net: Optional[nn.Module] = None

    @nn.compact
    def __call__(self, x: jax.Array, train: bool = False) -> jax.Array:
        if self.rnn_type == "LSTM":
            rnn_cell = nn.LSTM
        elif self.rnn_type == "GRU":
            rnn_cell = nn.GRU
        else:
            raise ValueError(f"Unknown RNN type: {self.rnn_type}")

        for i in range(self.rnn_num_layers):
            x = rnn_cell(self.rnn_hidden_dim, name=f"rnn_{i}")(x)
        return x


if __name__ == "__main__":
    mlp = MLP(hidden_dims=[256, 128, 64])
    x = jnp.ones((32, 64))
    params = mlp.init(jax.random.PRNGKey(0), x)
    y = mlp.apply(params, x)
    print(y.shape)

    # rnn = RNN(hidden_dim=128, num_layers=2)
    # x = jnp.ones((32, 10, 64))
    # params = rnn.init(jax.random.PRNGKey(0), x)
    # y = rnn.apply(params, x)
    # print(y.shape)
