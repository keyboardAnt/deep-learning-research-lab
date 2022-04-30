import typing
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import torch
from numpy.linalg import LinAlgError
from torch import nn, Tensor

from ir.modules.simulation_dataset import SimulationDataset


class SimulationTrack:
    def __init__(self, dataset: SimulationDataset):
        self.dataset: SimulationDataset = dataset
        self.model_state_dict_track: List[typing.OrderedDict[str, Tensor]] = []
        self._loss_track: List[float] = []

    @property
    def num_of_frames(self) -> int:
        return len(self.model_state_dict_track)

    @property
    def num_of_layers(self) -> int:
        return len(self.get_model_state_dict_record(frame_idx=0))

    def get_model_state_dict_record(self, frame_idx: int) -> typing.OrderedDict[str, Tensor]:
        return self.model_state_dict_track[frame_idx]

    def get_model_layer(self, frame_idx: int, layer_idx: int) -> Tensor:
        state_dict: OrderedDict = self.get_model_state_dict_record(frame_idx=frame_idx)
        return list(state_dict.values())[layer_idx]

    def get_model_neuron(self, frame_idx: int, layer_idx: int, neuron_idx: int) -> Tensor:
        layer: Tensor = self.get_model_layer(frame_idx=frame_idx, layer_idx=layer_idx)
        return layer[neuron_idx]

    @property
    def final_model_state_dict(self) -> typing.OrderedDict[str, Tensor]:
        return self.model_state_dict_track[-1]

    @property
    def loss_track(self) -> Tensor:
        return torch.tensor(self._loss_track, dtype=torch.float64)

    def get_loss_of_frame(self, frame_idx: int) -> float:
        return self.loss_track[frame_idx].item()

    def log_model_state_dict(self, model: nn.Module) -> None:
        self.model_state_dict_track.append(deepcopy(model.state_dict()))

    def log_loss(self, loss: float) -> None:
        self._loss_track.append(loss)

    def reset(self) -> None:
        self.model_state_dict_track = []
        self._loss_track = []

    def save(self, dirpath: Path):
        dirpath.mkdir(parents=True, exist_ok=True)
        np.save(str(dirpath / 'model_state_dict_track'), self.model_state_dict_track)
        np.save(str(dirpath / 'loss_track'), self.loss_track)


class SimulationAnalyzer:
    def __init__(self, simulation_track: SimulationTrack) -> None:
        self.track: SimulationTrack = simulation_track
        self._model_f2s_track: Optional[List[typing.OrderedDict[str, float]]] = None
        self._idx_of_last_rvalued_frame: Optional[int] = None

    @property
    def inputs_mtx_image(self) -> np.ndarray:
        mtx: np.ndarray = self.track.dataset.inputs_mtx
        return self.get_mtx_image(mtx)

    @property
    def labels_mtx_image(self) -> np.ndarray:
        mtx: np.ndarray = self.track.dataset.labels_mtx
        return self.get_mtx_image(mtx)

    @staticmethod
    def get_mtx_image(mtx: np.ndarray, source: Optional[np.ndarray] = None) -> np.ndarray:
        if source is None:
            t: np.ndarray = np.linspace(0, 2 * np.pi)
            unit_sphere: np.ndarray = np.array([np.cos(t), np.sin(t)])
            source = unit_sphere
        return np.matmul(mtx.T, source)

    @property
    def inputs_mtx_f2s(self) -> float:
        mtx: np.ndarray = self.track.dataset.inputs_mtx
        return self.get_mtx_f2s(mtx)

    @property
    def labels_mtx_f2s(self) -> float:
        mtx: np.ndarray = self.track.dataset.labels_mtx
        return self.get_mtx_f2s(mtx)

    @property
    def model_f2s_track(self) -> List[typing.OrderedDict[str, float]]:
        if self._model_f2s_track is None:
            ret: List[typing.OrderedDict[str, Union[Tensor, float]]] = deepcopy(self.track.model_state_dict_track)
            state_dict: typing.OrderedDict[str, Union[Tensor, float]]
            for state_dict in ret:
                layer_name: str
                layer: Tensor
                for layer_name, layer in state_dict.items():
                    state_dict[layer_name] = self.get_mtx_f2s(layer)
            self._model_f2s_track = ret
        return self._model_f2s_track

    @property
    def idx_of_last_rvalued_frame(self) -> int:
        """
        :return: The index of the last frame that contains a real valued loss. Namely, not a nan.
                 If all frames have nan, return `-1`. If all frames don't have nan, return `num_of_frames-1`.
        """
        if self._idx_of_last_rvalued_frame is None:
            idx_of_first_nan_frame: int = self.track.num_of_frames
            loss_is_nan: Tensor = self.track.loss_track.isnan().nonzero()
            try:
                idx_of_first_nan_frame = loss_is_nan[0].item()
            except IndexError:
                pass
            return idx_of_first_nan_frame - 1
        else:
            return self._idx_of_last_rvalued_frame

    def get_model_f2s_of_frame(self, frame_idx: int) -> typing.OrderedDict[str, float]:
        return self.model_f2s_track[frame_idx]

    def get_model_avg_f2s_of_frame(self, frame_idx: int, last_layer_is_nontrivial: bool) -> float:
        f2s_of_layers: List[float] = list(
            self.get_model_f2s_of_frame(frame_idx=frame_idx).values()
        )
        nontrivial_f2s_of_layers: List[float] = f2s_of_layers if last_layer_is_nontrivial else f2s_of_layers[:-1]
        return np.mean(nontrivial_f2s_of_layers).item()

    @staticmethod
    def get_mtx_f2s(mtx: Union[np.ndarray, Tensor]) -> float:
        if (mtx == 0).all():
            return np.nan
        try:
            # s: np.ndarray
            # _, s, _ = np.linalg.svd(mtx)
            # return s[0] / ((s ** 2).sum() ** 0.5)
            return (np.linalg.norm(mtx, ord='fro') / np.linalg.norm(mtx, ord=2)) ** 2
        except LinAlgError:
            return np.nan
