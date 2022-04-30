import typing
from copy import deepcopy
from pathlib import Path
from typing import List, Tuple, Union, Dict, Any
import imageio as imageio
import numpy as np
import pandas as pd
import pygifsicle as pygifsicle
from matplotlib.axes import Axes
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from torch import Tensor
from ir.config import constants
from ir.config.hyperparams import Hyperparams
from ir.modules.simulation_track_and_analyzer import SimulationAnalyzer
from ir.utils.slack import Slack
import seaborn as sns
import textwrap


class SimulationVisualizer:
    def __init__(
            self,
            hyperparams: Hyperparams,
            simulation_analyzer: SimulationAnalyzer,
            artifacts_dirpath: Path,
            random_seed: int,
            slack: Slack
    ) -> None:
        self._hyperparams: Hyperparams = hyperparams
        self._simulation_analyzer: SimulationAnalyzer = simulation_analyzer
        self._artifacts_dirpath: Path = artifacts_dirpath
        self._random_seed: int = random_seed
        self._slack: Slack = slack
        self._layers_name: List[str] = [
            f'{constants.net.LAYER_NAME_PREFIX}{layer_idx + 1}' for layer_idx in
            range(self._simulation_analyzer.track.num_of_layers)
        ]

    def create_frames_and_gif(self) -> None:
        frame_idx: int
        epoch_idx: int
        for frame_idx, epoch_idx in enumerate(self._hyperparams.training.EPOCHS_TO_FRAME_AND_LOG):
            # Init the figure/mosaic
            mosaic: List[List[str]] = [['f2s', 'simulation'],
                                       ['loss', 'simulation']]
            figsize: Tuple[int, int] = (25, 15)
            gridspec_kw: Dict[str, typing.Any] = {'width_ratios': (.45, .55)} #, 'wspace': .05, 'hspace': .005}
            fig: Figure
            axd: Dict[str, Axes]
            fig, axd = plt.subplot_mosaic(mosaic, figsize=figsize, gridspec_kw=gridspec_kw, constrained_layout=True)

            # Add a watermark
            watermark: str = f'random seed = {self._random_seed}; {self._artifacts_dirpath}; {self._hyperparams}'
            wrapper_suptitle = textwrap.TextWrapper(width=240)
            plt.suptitle(wrapper_suptitle.fill(watermark))

            # Globals
            cmap: ListedColormap = sns.color_palette('rocket', as_cmap=True)
            layers_color: np.ndarray = cmap(
                np.arange(self._simulation_analyzer.track.num_of_layers) / self._simulation_analyzer.track.num_of_layers
            )

            # F2S Ratio
            title_final_f2s: str = \
                f'Final f2s: {self._get_f2s_title(frame_idx=-1)}' \
                f' || First f2s: {self._get_f2s_title(frame_idx=0)}'
            wrapper_title = textwrap.TextWrapper(width=115)
            axd['f2s'].set_title(wrapper_title.fill(title_final_f2s))
            df_model_f2s_track: pd.DataFrame = pd.DataFrame.from_records(
                self._simulation_analyzer.model_f2s_track,
                index=self._hyperparams.training.EPOCHS_TO_FRAME_AND_LOG
            )
            df_model_f2s_track.columns = deepcopy(self._layers_name)
            sns.lineplot(data=df_model_f2s_track, palette=layers_color, ax=axd['f2s'])
            axd['f2s'].axvline(x=epoch_idx, color='grey')
            axd['f2s'].set_xlim(left=0-constants.vis.AXIS_LIM_MARGIN,
                                right=self._hyperparams.training.MAX_EPOCHS+constants.vis.AXIS_LIM_MARGIN)
            axd['f2s'].set_ylim(
                bottom=1-constants.vis.AXIS_LIM_MARGIN,
                top=self._hyperparams.net.MAX_F2S+constants.vis.AXIS_LIM_MARGIN
            )

            # Loss
            title_loss: str = f'Final loss = {self._simulation_analyzer.track.get_loss_of_frame(frame_idx=-1):.3e}' \
                              f' || First loss = {self._simulation_analyzer.track.get_loss_of_frame(frame_idx=0):.3e}'
            axd['loss'].set_title(title_loss)
            df_loss_track: pd.DataFrame = pd.DataFrame(
                self._simulation_analyzer.track.loss_track,
                index=self._hyperparams.training.EPOCHS_TO_FRAME_AND_LOG,
                columns=[f'{self._hyperparams.training.LOSS_FN}']
            )
            sns.lineplot(data=df_loss_track, ax=axd['loss'])
            axd['loss'].axvline(x=epoch_idx, color='grey')
            axd['loss'].set_xlim(left=0-constants.vis.AXIS_LIM_MARGIN,
                                 right=self._hyperparams.training.MAX_EPOCHS+constants.vis.AXIS_LIM_MARGIN)
            axd['loss'].set_ylim(bottom=0-constants.vis.AXIS_LIM_MARGIN, top=.5+constants.vis.AXIS_LIM_MARGIN)

            # Simulation
            axd['simulation'].set_title(f'Epoch {epoch_idx:}')
            axd['simulation'].grid()
            axd['simulation'].set_aspect('equal')
            xlim = (-1 * constants.frame.MODEL_AXES_LIM, constants.frame.MODEL_AXES_LIM)
            ylim = xlim
            axd['simulation'].set(xlim=xlim, ylim=ylim)
            start, end = axd['simulation'].get_xlim()
            axd['simulation'].xaxis.set_ticks(np.arange(start, end, 1))
            start, end = axd['simulation'].get_ylim()
            axd['simulation'].yaxis.set_ticks(np.arange(start, end, 1))

            # If inputs are in 2d
            if self._hyperparams.net.INPUT_DIM <= 2:
                # Plot unit sphere
                a_circle = plt.Circle((0, 0), 1, alpha=.1)
                axd['simulation'].add_artist(a_circle)
                # Plot SEES-ONLY boundaries
                x1: np.ndarray = self._simulation_analyzer.track.dataset.get_input_vector(0)
                x2: np.ndarray = self._simulation_analyzer.track.dataset.get_input_vector(1)
                t: np.ndarray = np.linspace(xlim[0], xlim[1], 100)
                axd['simulation'].plot(t, -t * x2[0] / x2[1], linestyle='dashed', label='ONLY1', c='b',
                                       alpha=constants.frame.MODEL_ALPHA)
                axd['simulation'].plot(t, -t * x1[0] / x1[1], linestyle='dashed', label='ONLY2', c='b',
                                       alpha=constants.frame.MODEL_ALPHA)
                # Plot the inputs
                X: np.ndarray = self._simulation_analyzer.track.dataset.inputs_mtx
                self._scatter_rows_of_a_mtx(mtx=X, row_name='x', color='blue', ax=axd['simulation'])
            if self._hyperparams.net.LABEL_DIM <= 2:
                # Plot the labels
                Y: np.ndarray = self._simulation_analyzer.track.dataset.labels_mtx
                to_scatter_labels: bool = len(Y) <= 2
                if to_scatter_labels:
                    self._scatter_rows_of_a_mtx(mtx=Y, row_name='y', color='green', ax=axd['simulation'])
                    # Plot the image of the labels matrix, if possible
                    if self._hyperparams.net.LABEL_DIM == 2:
                        Y_img: np.ndarray = self._simulation_analyzer.labels_mtx_image
                        self._plot_mtx_image(mtx_image=Y_img, mtx_name='Y', color='darkgreen', ax=axd['simulation'])

            frame_f2s: Dict[str, float] = {
                'X': self._simulation_analyzer.inputs_mtx_f2s,
                'Y': self._simulation_analyzer.labels_mtx_f2s
            }
            state_dict: typing.OrderedDict[str, Tensor] = self._simulation_analyzer.track.get_model_state_dict_record(
                frame_idx=frame_idx
            )
            layer_idx: int
            layer: Tensor
            for layer_idx, layer in enumerate(state_dict.values()):
                layer: np.ndarray = np.array(layer, dtype=np.float64)
                layer_name: str = self._layers_name[layer_idx]
                if self._hyperparams.net.WIDTH <= 2:
                    layer_color: Tuple[float, float, float, float] = layers_color[layer_idx]
                    neuron_name: str = f'{layer_name}_'
                    self._scatter_rows_of_a_mtx(
                        mtx=layer, row_name=neuron_name, color=layer_color, ax=axd['simulation']
                    )
                    is_layer_square_mtx: bool = (layer.shape[0] == 2) and (layer.shape[1] == 2)
                    if is_layer_square_mtx is True:
                        layer_image: np.ndarray = self._simulation_analyzer.get_mtx_image(mtx=layer)
                        self._plot_mtx_image(mtx_image=layer_image, mtx_name=layer_name, color=layer_color,
                                             ax=axd['simulation'])
                f2s: float = self._simulation_analyzer.get_mtx_f2s(layer)
                frame_f2s[layer_name] = f2s

            # Add legend & take a snapshot
            axd['simulation'].legend()
            fig.savefig(self.get_frame_dirpath(frame_idx))
            plt.close(fig)
            frame_f2s.pop('X')
        gif_filepath: Path = self._artifacts_dirpath / 'simulation.gif'
        single_frame_duration_in_sec: float = \
            constants.gif.total_duration / self._simulation_analyzer.track.num_of_frames
        with imageio.get_writer(gif_filepath, mode='I', duration=single_frame_duration_in_sec) as writer:
            filenames: List[str]
            for i in range(self._simulation_analyzer.track.num_of_frames):
                image = imageio.imread(self.get_frame_dirpath(i))
                writer.append_data(image)
        pygifsicle.optimize(gif_filepath)
        self._slack.send_file(f'{gif_filepath}')

    def _get_f2s_title(self, frame_idx: int) -> str:
        last_layer_is_nontrivial: bool = self._hyperparams.net.LABEL_DIM > 1
        avg_f2s_of_nontrivial_layers: float = self._simulation_analyzer.get_model_avg_f2s_of_frame(
            frame_idx=frame_idx, last_layer_is_nontrivial=last_layer_is_nontrivial
        )
        return f' Avg={avg_f2s_of_nontrivial_layers:.2e}; ' \
               + '; '.join([f'{self._layers_name[layer_idx]}={layer_f2s:.2e}' for layer_idx, layer_f2s
                            in enumerate(self._simulation_analyzer.get_model_f2s_of_frame(frame_idx=frame_idx).values())
                            ])

    @staticmethod
    def _scatter_rows_of_a_mtx(mtx: np.ndarray, row_name: str, color: Union[str, Tuple[float, ...]], ax: Axes) -> None:
        row_idx: int
        row: np.ndarray
        for row_idx, row in enumerate(mtx):
            # NOTE: The output dimension could be equal to 1
            if mtx.shape[-1] == 1:
                row = np.append(row, [0.])
            ax.scatter(*row, color=color, label=f'{row_name}{row_idx+1}')

    @staticmethod
    def _plot_mtx_image(
            mtx_image: np.ndarray,
            mtx_name: str,
            color: Union[str, Tuple[float, ...]],
            ax: Axes
    ) -> None:
        ax.plot(*mtx_image, color=color, label=f'{mtx_name}_img', alpha=constants.frame.MODEL_ALPHA)

    def get_frame_dirpath(self, frame_idx: int) -> str:
        return str(self._artifacts_dirpath / f'epoch_{frame_idx:03}.png')
