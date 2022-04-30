from abc import ABC
from os.path import join

import torch
from torch import nn, Tensor
from typing import Callable, OrderedDict, Optional, List, Iterable, Tuple

from torch.nn import Parameter

from ir.config import constants
from ir.config.hyperparams import Hyperparams
from ir.utils.logging import get_logging
logging = get_logging()


class _Net(ABC, nn.Module):
    def init_weights(self) -> None:
        pass

    def store(self, dirpath: str, epoch_num: int) -> None:
        filename: str = f'network_epoch={epoch_num}.pt'
        filepath: str = join(dirpath, filename)
        torch.save(self.state_dict(), filepath)


class DFFNet(_Net):
    def __init__(
            self,
            init_fn: Callable,
            activation_fn: Callable,
            init_state_dict: Optional[OrderedDict] = None,
            shape_wo_bias: Optional[Iterable[int]] = None,
            has_bias: Optional[bool] = None,
    ):
        """
        Build a deep feed-forward network, with a given activation function.
        There are two alternatives for determining the architecture of the network:
            1. pass a state_dict parameter; or
            2. explicitly set the layers shapes and whether there are biases.
        :param activation_fn:
        :param init_state_dict:
        :param shape_wo_bias: the width of the layers
        :param has_bias:
        """
        super().__init__()
        self._activation_fn: Callable = activation_fn
        self._init_state_dict: Optional[OrderedDict] = init_state_dict
        self._shape_wo_bias: Optional[Tuple[int, ...]] = shape_wo_bias
        self._init_fn: Callable = init_fn
        self._has_bias: Optional[bool] = has_bias
        self._layers: List[nn.Linear] = list()
        if self._init_state_dict is not None:
            raise NotImplementedError
        else:
            layer_idx: int
            d_in: int
            for layer_idx, d_in in enumerate(self._shape_wo_bias[:-1]):
                d_out: int = self._shape_wo_bias[layer_idx + 1]
                layer: nn.Linear = nn.Linear(d_in, d_out, bias=self._has_bias, dtype=torch.float64)
                self._init_fn(layer.weight, layer_idx=layer_idx)
                self._layers.append(layer)
                self.add_module(f'{constants.net.LAYER_NAME_PREFIX}{layer_idx + 1}', layer)
        logging.basicConfig(level=logging.INFO_IR)
        logging.info_ir(f'new net; init weights with init_fn: {self._init_fn}')

    def init_weights(self, init_fn: Optional[Callable] = None) -> None:
        if init_fn is not None:
            self._init_fn = init_fn
        logging.info_ir(f'init weights with init_fn: {self._init_fn}')
        layer: nn.Linear
        for idx, layer in enumerate(self._layers):
            self._init_fn(layer.weight, layer_idx=idx)
            if self._has_bias:
                self._init_fn(layer.bias, layer_idx=idx)

    def forward(self, X: Tensor) -> Tensor:
        layer: nn.Module
        for layer in self._layers[:-1]:
            X: Tensor = self._activation_fn(layer(X))
        return self._layers[-1](X)

    @property
    def frobenius_norm(self) -> float:
        """
        :return: The Frobenius norm of the network (i.e., of the $\theta$ vector, which is the vector of all parameters)
        """
        return torch.tensor([p.norm() ** 2 for p in self.parameters()]).sum().sqrt().item()

    def normalize_by_frobenius_norm(self, target_frobenius_norm: float = 1.) -> None:
        curr_frobenius_norm: float = self.frobenius_norm
        normalizing_scalar: float = target_frobenius_norm / curr_frobenius_norm
        p: Parameter
        for p in self.parameters():
            p *= normalizing_scalar

    def get_parameters_pt(self) -> Tensor:
        return torch.stack(list(self.parameters()))


def get_net(hyperparams: Hyperparams) -> DFFNet:
    ret: DFFNet = DFFNet(
        init_fn=hyperparams.net.INIT_FN,
        activation_fn=hyperparams.net.ACTIVATION_FN,
        shape_wo_bias=hyperparams.net.SHAPE_WO_BIAS,
        has_bias=hyperparams.net.HAS_BIAS,
    )
    return ret
