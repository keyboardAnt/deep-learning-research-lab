import typing
from typing import List, Callable
from collections import OrderedDict
from pathlib import Path

import pandas as pd
import ray as ray
from ray.types import ObjectRef

from ir.config import constants
from ir.config.hyperparams import Hyperparams
from ir.config.random_seeds import set_random_seeds
from ir.modules.aggregating_visualizer import AggregatingVisualizer
from ir.modules.simulator import Simulator
from ir.modules.simulation_visualizer import SimulationVisualizer
from ir.modules.simulation_track_and_analyzer import SimulationTrack, SimulationAnalyzer
from ir.utils.slack import Slack, slack_sender


@ray.remote
def worker(
        random_seed: int,
        hyperparams: Hyperparams,
        simulator_artifacts_dirpath: Path,
        slack: Slack
) -> typing.OrderedDict[str, float]:
    set_random_seeds(random_seed)
    simulator = Simulator(hyperparams=hyperparams, artifacts_dirpath=simulator_artifacts_dirpath)
    simulation_track: SimulationTrack = simulator.run()
    simulation_analyzer: SimulationAnalyzer = SimulationAnalyzer(simulation_track=simulation_track)
    idx_of_last_rvalued_frame: int = simulation_analyzer.idx_of_last_rvalued_frame
    simulation_final_results: typing.OrderedDict[str, float] = OrderedDict(
        {
            constants.agg.IDX_OF_LAST_RVALUED_FRAME_COLUMN_NAME: idx_of_last_rvalued_frame,
            constants.agg.LAST_RVALUED_LOSS_COLUMN_NAME: simulation_track.get_loss_of_frame(
                frame_idx=idx_of_last_rvalued_frame)
        }
    )
    simulation_final_results.update(simulation_analyzer.get_model_f2s_of_frame(frame_idx=idx_of_last_rvalued_frame))
    simulation_visualizer: SimulationVisualizer = SimulationVisualizer(
        hyperparams=hyperparams,
        simulation_analyzer=simulation_analyzer,
        artifacts_dirpath=simulator.artifacts_dirpath,
        random_seed=random_seed,
        slack=slack
    )
    simulation_visualizer.create_frames_and_gif()
    return simulation_final_results


slack: Slack = Slack()


@slack_sender(slack=slack)
def main():
    hyperparams: Hyperparams = Hyperparams()
    print(hyperparams)

    slack.send_text(str(hyperparams))
    ray.init(local_mode=hyperparams.parser_args.ray_local_mode)
    all_simulation_final_results_ids: List[ObjectRef[OrderedDict[str, float]]] = []
    df_all_simulation_final_results: pd.DataFrame = pd.DataFrame()
    # Spawn multiple workers
    for i, random_seed in enumerate(range(
            hyperparams.random.SEED_OFFSET,
            hyperparams.random.SEED_OFFSET + hyperparams.training.NUM_OF_SIMULATORS
    )):
        simulator_artifacts_dirpath: Path = hyperparams.env.CURR_ARTIFACTS_DIRPATH / f'rseed_{random_seed:03d}'
        if hyperparams.parser_args.only_agg is False:
            simulator_artifacts_dirpath.mkdir(parents=True)
            simulation_final_results_id: ObjectRef[OrderedDict[str, float]] = worker.remote(
                random_seed=random_seed,
                hyperparams=hyperparams,
                simulator_artifacts_dirpath=simulator_artifacts_dirpath,
                slack=slack
            )
            all_simulation_final_results_ids.append(simulation_final_results_id)
    if hyperparams.parser_args.only_agg is False:
        # Get all final results
        while len(all_simulation_final_results_ids) > 0:
            simulation_final_results_id: ObjectRef[OrderedDict[str, float]] = all_simulation_final_results_ids.pop()
            simulation_final_results: OrderedDict[str, float] = ray.get(simulation_final_results_id)
            df_all_simulation_final_results = df_all_simulation_final_results.append(
                simulation_final_results,
                ignore_index=True
            )
    get_renamed_column: Callable = lambda col_name: col_name.replace('.weight', '')
    df_all_simulation_final_results.columns = list(map(get_renamed_column, df_all_simulation_final_results.columns))
    # Agg
    agg_dirpath: Path = hyperparams.env.CURR_ARTIFACTS_DIRPATH / constants.env.AGG_ARTIFACTS_DIRNAME
    agg_filepath: Path = agg_dirpath / 'all_simulation_final_results.csv'
    if hyperparams.parser_args.only_agg is True:
        df_all_simulation_final_results = pd.read_csv(agg_filepath)
    else:
        agg_dirpath.mkdir(parents=True, exist_ok=True)
        df_all_simulation_final_results.to_csv(str(agg_filepath))
    av: AggregatingVisualizer = AggregatingVisualizer(
        hyperparams=hyperparams,
        final_results=df_all_simulation_final_results,
        artifacts_dirpath=agg_dirpath,
        slack=slack
    )
    av.report(threshold_zero_loss=hyperparams.training.TARGET_LOSS)


if __name__ == '__main__':
    main()
