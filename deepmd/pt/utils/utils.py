# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Callable,
    Optional,
)

import numpy as np
import torch
import torch.nn.functional as F

from deepmd.model_format.common import PRECISION_DICT as NP_PRECISION_DICT

from .env import (
    DEVICE,
)
from .env import PRECISION_DICT as PT_PRECISION_DICT


def get_activation_fn(activation: str) -> Callable:
    """Returns the activation function corresponding to `activation`."""
    if activation.lower() == "relu":
        return F.relu
    elif activation.lower() == "gelu":
        return F.gelu
    elif activation.lower() == "tanh":
        return torch.tanh
    elif activation.lower() == "linear" or activation.lower() == "none":
        return lambda x: x
    else:
        raise RuntimeError(f"activation function {activation} not supported")


class ActivationFn(torch.nn.Module):
    def __init__(self, activation: Optional[str]):
        super().__init__()
        self.activation: str = activation if activation is not None else "linear"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns the tensor after applying activation function corresponding to `activation`."""
        # See jit supported types: https://pytorch.org/docs/stable/jit_language_reference.html#supported-type

        if self.activation.lower() == "relu":
            return F.relu(x)
        elif self.activation.lower() == "gelu":
            return F.gelu(x)
        elif self.activation.lower() == "tanh":
            return torch.tanh(x)
        elif self.activation.lower() == "linear" or self.activation.lower() == "none":
            return x
        else:
            raise RuntimeError(f"activation function {self.activation} not supported")


def to_numpy_array(
    xx: torch.Tensor,
) -> np.ndarray:
    if xx is None:
        return None
    assert xx is not None
    # Create a reverse mapping of PT_PRECISION_DICT
    reverse_precision_dict = {v: k for k, v in PT_PRECISION_DICT.items()}
    # Use the reverse mapping to find keys with the desired value
    prec = reverse_precision_dict.get(xx.dtype, None)
    prec = NP_PRECISION_DICT.get(prec, None)
    if prec is None:
        raise ValueError(f"unknown precision {xx.dtype}")
    return xx.detach().cpu().numpy().astype(prec)


def to_torch_tensor(
    xx: np.ndarray,
) -> torch.Tensor:
    if xx is None:
        return None
    assert xx is not None
    # Create a reverse mapping of NP_PRECISION_DICT
    reverse_precision_dict = {v: k for k, v in NP_PRECISION_DICT.items()}
    # Use the reverse mapping to find keys with the desired value
    prec = reverse_precision_dict.get(type(xx.flat[0]), None)
    prec = PT_PRECISION_DICT.get(prec, None)
    if prec is None:
        raise ValueError(f"unknown precision {xx.dtype}")
    return torch.tensor(xx, dtype=prec, device=DEVICE)