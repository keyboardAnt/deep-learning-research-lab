from pathlib import Path
from typing import List, Optional, Callable

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from ir.config import constants
from ir.config.hyperparams import Hyperparams

from ir.utils.logging import get_logging
from ir.utils.slack import Slack

logging = get_logging()


class AggregatingVisualizer:
    def __init__(self, hyperparams: Hyperparams, final_results: pd.DataFrame, artifacts_dirpath: Path, slack: Slack):
        self._hyperparams: Hyperparams = hyperparams
        self._df_final_results: pd.DataFrame = final_results
        f2s_columns_mask: np.ndarray = self._df_final_results.columns.str.startswith(
            constants.net.LAYER_NAME_PREFIX
        )
        self._f2s_columns: pd.Index = self._df_final_results.columns[f2s_columns_mask]
        get_loss_fn_name: Callable = lambda loss_fn: str(loss_fn).replace('(', '').replace(')', '')
        self._loss_fn_name: str = get_loss_fn_name(self._hyperparams.training.LOSS_FN)
        self._f2s_explanation: str = 'stable (numerical) rank'  #'squared ratio of Frobenius to spectral norms'
        self._artifacts_dirpath: Path = artifacts_dirpath
        self._slack: Slack = slack
        logging.basicConfig(level=logging.INFO_IR)

    def _get_df_zero_final_loss(self, threshold_zero_loss: float) -> pd.DataFrame:
        mask_zero_loss: pd.Series = \
            self._df_final_results[constants.agg.LAST_RVALUED_LOSS_COLUMN_NAME] < threshold_zero_loss
        return self._df_final_results[mask_zero_loss]

    def _get_simulations_with_final_loss_less_than(self, final_loss) -> pd.DataFrame:
        return self._df_final_results[self._df_final_results[constants.agg.LAST_RVALUED_LOSS_COLUMN_NAME] < final_loss]

    def _get_simulations_with_final_loss_equal_to(self, final_loss) -> pd.DataFrame:
        return self._df_final_results[self._df_final_results[constants.agg.LAST_RVALUED_LOSS_COLUMN_NAME] == final_loss]

    def _get_simulations_with_final_loss_greater_than(self, final_loss) -> pd.DataFrame:
        return self._df_final_results[self._df_final_results[constants.agg.LAST_RVALUED_LOSS_COLUMN_NAME] > final_loss]

    def _get_final_losses_report(
            self,
            less_than_losses: Optional[List[float]] = None
    ) -> str:
        ret: str = f'Final losses:'
        num_of_simulations_with_same_loss_treshold: int = 1
        equal_to_losses: List[float] = \
            (
                    self._df_final_results[constants.agg.LAST_RVALUED_LOSS_COLUMN_NAME].value_counts() >
                    num_of_simulations_with_same_loss_treshold
            ).index[self._df_final_results[constants.agg.LAST_RVALUED_LOSS_COLUMN_NAME].value_counts() >
                    num_of_simulations_with_same_loss_treshold].to_list()
        idx_less_than: int = 0
        idx_equal_to: int = 0
        while idx_less_than < len(less_than_losses) or idx_equal_to < len(equal_to_losses):
            less_than_loss: float = less_than_losses[idx_less_than] if idx_less_than < len(less_than_losses) else np.inf
            equal_to_loss: float = equal_to_losses[idx_equal_to] if idx_equal_to < len(equal_to_losses) else np.inf
            if less_than_loss <= equal_to_loss:
                simulations_with_loss_less_than: pd.DataFrame = \
                    self._get_simulations_with_final_loss_less_than(less_than_loss)
                num_of_simulations: int = len(simulations_with_loss_less_than)
                ret += '{:50s}'.format(f'\nThere are {num_of_simulations} simulations '
                                       f'with {self._hyperparams.training.LOSS_FN} < {less_than_loss:.3e}.')
                if num_of_simulations > 0:
                    worst_loss: float = \
                        simulations_with_loss_less_than[constants.agg.LAST_RVALUED_LOSS_COLUMN_NAME].max()
                    ret += f' Among them, the worst simulation has' \
                           f' {self._hyperparams.training.LOSS_FN} < {worst_loss:.3e}.'
                idx_less_than += 1
            else:
                num_of_simulations: int = len(self._get_simulations_with_final_loss_equal_to(equal_to_loss))
                ret += f'\nThere are {num_of_simulations} simulations' \
                       f' with {self._hyperparams.training.LOSS_FN} = {equal_to_loss:.3e}.'
                idx_equal_to += 1
        if len(equal_to_losses) > 0:
            largest_loss: float = equal_to_losses[-1]
            num_of_simulations = len(self._get_simulations_with_final_loss_greater_than(largest_loss))
            ret += f'\nThere are {num_of_simulations} simulations with ' \
                   f'{self._hyperparams.training.LOSS_FN} > {largest_loss:.3e}.'
        return ret

    def _displot_final_losses(self) -> None:
        g: sns.FacetGrid = sns.displot(self._df_final_results[constants.agg.LAST_RVALUED_LOSS_COLUMN_NAME])
        self._save_and_report_figure(fgrid=g, filename='final_losses_displot.png')

    def _get_df_final_f2s(self, threshold_zero_loss: float) -> pd.DataFrame:
        df_zero_final_loss: pd.DataFrame = self._get_df_zero_final_loss(threshold_zero_loss=threshold_zero_loss)
        nontrivial_f2s_columns: np.Index = \
            self._f2s_columns if self._hyperparams.net.LABEL_DIM > 1 else self._f2s_columns[:-1]
        return df_zero_final_loss[nontrivial_f2s_columns]

    def _displot_f2s(self, threshold_zero_loss: float) -> None:
        df_final_f2s: pd.DataFrame = self._get_df_final_f2s(threshold_zero_loss=threshold_zero_loss)
        g: sns.FacetGrid = sns.displot(df_final_f2s, multiple='dodge', kde=False)  # , alpha=.5
        num_of_simulations: int = len(df_final_f2s)

        g.set(
            xlim=(1-constants.vis.AXIS_LIM_MARGIN, self._hyperparams.net.MAX_F2S+constants.vis.AXIS_LIM_MARGIN),
            xlabel=self._f2s_explanation,
            ylabel='number of networks'
        )
        title: str = f'Final {self._f2s_explanation}' \
                     f'\n(of {num_of_simulations} simulations with {self._loss_fn_name} < {threshold_zero_loss})'
        self._report_text(title)
        self._save_and_report_figure(fgrid=g, filename='final_f2s_displot.png')

    def _get_mean_final_f2s(self, threshold_zero_loss: float) -> pd.DataFrame:
        df_final_f2s: pd.DataFrame = self._get_df_final_f2s(threshold_zero_loss=threshold_zero_loss)
        return df_final_f2s.mean()

    def _get_average_final_f2s_report(self, threshold_zero_loss: float) -> str:
        ret: str = f'The average final (Frob/spec)^2 of each layer, ' \
                   f'for simulations with {self._loss_fn_name} < {threshold_zero_loss}:\n'
        average_final_f2s: pd.DataFrame = \
            self._get_mean_final_f2s(threshold_zero_loss=threshold_zero_loss)
        precision: int = 4
        with pd.option_context('display.precision', precision):
            ret += f'{average_final_f2s}'
        ret += f'\nThe average of the averages: {average_final_f2s.mean():.4f}'
        return ret

    def _get_idx_of_last_rvalued_frame_report(self) -> str:
        counter: pd.Series = \
            self._df_final_results[constants.agg.IDX_OF_LAST_RVALUED_FRAME_COLUMN_NAME].value_counts()
        valid_idx: int = self._hyperparams.training.NUM_OF_EPOCHS_TO_FRAME_AND_LOG - 1
        invalid_mask: np.ndarray = counter.index < valid_idx
        num_of_invalid_simulations: int = counter[invalid_mask].sum()
        ret: str = \
            f'The number of invalid simulations (i.e., that suffer under/overflow): {num_of_invalid_simulations}.'
        if num_of_invalid_simulations > 0:
            ret += '\nHistogram of the index of the last real-valued frame (i.e., a frame without under/overflow):'
            ret += f'\n{counter}'
            ret += f'\nThe valid index of the last real-valued frame: {valid_idx}.'
        return ret

    def _save_and_report_figure(self, fgrid: sns.FacetGrid, filename: str):
        sns.despine()
        # plt.show()
        filepath: str = str(self._artifacts_dirpath / filename)
        fgrid.savefig(filepath)
        self._slack.send_file(filepath)

    def _report_text(self, text: str, mention: bool = False):
        logging.info_ir(text)
        self._slack.send_text(text, mention=mention)

    def report(
            self,
            threshold_zero_loss: float,
    ) -> None:
        self._report_text(f'### Report for {self._artifacts_dirpath.resolve()}', mention=True)
        self._report_text(str(self._hyperparams))
        self._displot_final_losses()
        less_than_losses: List[float] = np.sort(np.unique(np.array([threshold_zero_loss, 0.499, 1]))).tolist()
        final_losses_report: str = self._get_final_losses_report(
            less_than_losses=less_than_losses,
        )
        self._report_text(final_losses_report)
        self._displot_f2s(threshold_zero_loss=threshold_zero_loss)
        average_final_f2s_report: str = self._get_average_final_f2s_report(
            threshold_zero_loss=threshold_zero_loss
        )
        self._report_text(average_final_f2s_report)
        idx_of_last_rvalued_frame_report: str = self._get_idx_of_last_rvalued_frame_report()
        self._report_text(idx_of_last_rvalued_frame_report)
