from typing import Tuple

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import TensorDataset

from ir.config.hyperparams import Hyperparams
from ir.modules import nets_init
from ir.modules.nets import DFFNet
from torch.nn import functional as f
from torch.nn.functional import normalize


class SimulationDataset(TensorDataset):
    def __init__(self, inputs_mtx: Tensor, labels_mtx: Tensor):
        super().__init__(inputs_mtx, labels_mtx)
        self._inputs_mtx: Tensor = inputs_mtx.double()
        self._labels_mtx: Tensor = labels_mtx.double()
        self.labels_mtx_rank: int = torch.linalg.matrix_rank(self._labels_mtx).item()

    @property
    def inputs_mtx(self) -> np.ndarray:
        return self._inputs_mtx.detach().numpy()

    @property
    def labels_mtx(self) -> np.ndarray:
        return self._labels_mtx.detach().numpy()

    def get_input_vector(self, idx: int) -> np.ndarray:
        return self.inputs_mtx[idx]

    def get_label_vector(self, idx: int) -> np.ndarray:
        return self.labels_mtx[idx]


@torch.no_grad()
def get_simulation_dataset(hyperparams: Hyperparams) -> SimulationDataset:
    # Inputs
    if hyperparams.data.IS_TEACHER_DATASET is True:
        inputs: Tensor = torch.rand(
            (hyperparams.data.NUM_OF_DATAPOINTS, hyperparams.net.INPUT_DIM), dtype=torch.float64
        )
    else:
        x1: Tensor = torch.tensor([1, .99], dtype=torch.float64)
        x2: Tensor = x1.clone()
        x2[0] *= -1
        inputs: Tensor = torch.stack([x1, x2])
    # NOTE: By default, SCALING_FACTOR == 1.
    inputs = normalize(inputs) / hyperparams.data.SCALING_FACTOR
    # Targets
    if hyperparams.data.IS_TEACHER_DATASET is True:
        teacher_shape_wo_bias: Tuple[int, ...] = \
            (hyperparams.net.INPUT_DIM,) + \
            ((hyperparams.data.TEACHER_WIDTH,) * (hyperparams.data.TEACHER_DEPTH - 1)) + \
            (hyperparams.net.LABEL_DIM,)
        teacher: DFFNet = DFFNet(
            init_fn=nets_init.xavier_normal,
            activation_fn=f.relu,
            shape_wo_bias=teacher_shape_wo_bias,
            has_bias=hyperparams.data.TEACHER_HAS_BIAS
        )
        teacher.normalize_by_frobenius_norm()
        outputs: Tensor = teacher(inputs)
        labels: Tensor = torch.sign(outputs)
        labels[labels == 0] = -1
    else:
        labels: Tensor = get_deterministic_labels(hyperparams=hyperparams)
    return SimulationDataset(inputs, labels)


def get_deterministic_labels(hyperparams: Hyperparams) -> Tensor:
    if hyperparams.net.LABEL_DIM == 1:
        raise NotImplementedError
    elif hyperparams.net.LABEL_DIM == 2:
        return torch.eye(2, dtype=torch.float64)
    else:
        raise NotImplementedError


def get_random_labels(hyperparams: Hyperparams) -> Tensor:
    if hyperparams.net.LABEL_DIM == 1:
        return _get_random_labels_in_R1()
    elif hyperparams.net.LABEL_DIM == 2:
        return _get_random_labels_in_R2()
    else:
        raise NotImplementedError


def _get_random_labels_in_R2() -> Tensor:
    rand: Tensor = torch.rand(2, dtype=torch.float64) * 2 * torch.pi
    return torch.stack((torch.cos(rand), torch.sin(rand)), dim=1)


def _get_random_labels_in_R1() -> Tensor:
    ret: Tensor = torch.bernoulli(torch.ones((2, 1), dtype=torch.float64) * .5)
    ret[ret == 0] = -1
    return ret
