import functools
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, Namespace
from datetime import datetime
from pathlib import Path
from typing import Tuple, Callable, List, Dict, Optional

import numpy as np
from torch import nn
from torch.nn import functional as f

from ir.modules import nets_init
from ir.modules.loss_functions import SumOfExpLoss
from ir.modules.nets_init import uniform_from_ball, init_thm3


class HyperparamsParser(ArgumentParser):
    def __init__(self):
        super().__init__(formatter_class=ArgumentDefaultsHelpFormatter)

        # NETWORK
        self.add_argument('--depth', type=int, default=2, help='The depth of the learner network')
        self.add_argument('--width', type=int, default=2, help='Thw width of the learner network.')
        self.add_argument('--input_dim', type=int, default=-1,
                          help='The dimension of every input. Setting it to a negative value signals that the input'
                               ' dimension is equal to the width of the learner network.')
        self.add_argument('--label_dim', type=int, default=1,
                          help="The dimension of each label (and hence also of the output of network)")

        # DATASET
        self.add_argument('--scaling_factor', type=float, default=1,
                          help='Each input will be divided by this factor. This is a workaround for initializing the'
                               ' neurons to have a small norm.')
        self.add_argument('--teacher_depth', type=int, default=-1,
                          help='The depth of the teacher network. Setting width < 0 signals that there is no teacher'
                               ' network.')
        self.add_argument('--teacher_width', type=int, default=-1,
                          help='The width of the teacher network. Setting width < 0 signals that there is no teacher'
                               ' network.')
        self.add_argument('--num_of_datapoints', type=int, default=1000)

        # TRAINING
        self.add_argument('--loss_fn', type=str, choices=['mse', 'exp'], default='mse')
        self.add_argument('--lr', type=float, default=1e-4, help='Learning rate (a.k.a., step size.)')
        self.add_argument('--target_loss', type=float, default=-1)  # Default values are defined below
        self.add_argument('--num_of_simulators', type=int, default=6*48)
        self.add_argument('--max_epochs', type=int, default=int(3e6))
        self.add_argument('--init_max_norm', type=float, default=1e-4)
        self.add_argument('--init_fn', type=str, choices=['uniform_from_ball', 'thm3', 'xavier_normal'],
                          default='xavier_normal')
        self.add_argument('--log_interval_fn', type=str, choices=['geom', 'lin'], default='geom')

        # OPTIMIZER
        self.add_argument('--weight_decay', type=float, default=0)

        # RANDOM
        self.add_argument('--random_seed_offset', type=int, default=0)

        # ONLY AGGREGATING VISUALIZE
        self.add_argument('--only_agg', action='store_true', default=False, required=False,
                          help='Run only the AggregatingVisualizer for a single execution, given its timestamp string.')

        # ENV
        self.add_argument('--artifacts_dirname', nargs='?', type=str, default='', required=False,
                          help='Set a unique name (not necessarily a timestamp) for the current artifacts directory')

        # DEBUG
        self.add_argument('--ray_local_mode', dest='ray_local_mode', action='store_true')
        self.set_defaults(ray_local_mode=False)


class _Hyperparamers:
    def __repr__(self) -> str:
        return str(vars(self).items())


class Net(_Hyperparamers):
    def __init__(self, args: Namespace):
        self.DEPTH: int = args.depth
        self.WIDTH: int = args.width
        self.INPUT_DIM: int = args.input_dim if args.input_dim > 0 else self.WIDTH
        self.LABEL_DIM: int = args.label_dim
        self.SHAPE_WO_BIAS: Tuple = (self.INPUT_DIM,) + ((self.WIDTH,) * (self.DEPTH - 1)) + (self.LABEL_DIM,)
        self.ACTIVATION_FN: Callable = f.relu
        self.INIT_MAX_NORM: float = args.init_max_norm

        init_fns: Dict[str, Callable] = {
            'uniform_from_ball': functools.partial(uniform_from_ball, init_max_norm=self.INIT_MAX_NORM),
            'thm3': functools.partial(init_thm3, init_max_norm=self.INIT_MAX_NORM),
            'xavier_normal': nets_init.xavier_normal
        }

        self.INIT_FN: Callable = init_fns[args.init_fn]
        self.HAS_BIAS: bool = False
        self.MAX_F2S: float = self.WIDTH


class Data(_Hyperparamers):
    def __init__(self, args: Namespace):
        self.SCALING_FACTOR: float = args.scaling_factor
        self.TEACHER_DEPTH: int = args.teacher_depth
        self.TEACHER_WIDTH: int = args.teacher_width
        self.IS_TEACHER_DATASET: bool = (self.TEACHER_DEPTH > 0) or (self.TEACHER_WIDTH > 0)
        if self.IS_TEACHER_DATASET is True:
            assert all((self.TEACHER_DEPTH > 0, self.TEACHER_WIDTH > 0))
        self.TEACHER_HAS_BIAS: bool = False
        self.NUM_OF_DATAPOINTS: int = args.num_of_datapoints


class Training(_Hyperparamers):
    _LOSS_FNS: Dict[str, nn.Module] = {
        'mse': nn.MSELoss(),
        'exp': SumOfExpLoss()
    }
    _DEFAULT_TARGET_LOSSES: Dict[str, float] = {
        'mse': 1e-4,
        'exp': 1e-3
    }

    def __init__(self, args: Namespace):
        self.LOSS_FN: nn.Module = self._LOSS_FNS[args.loss_fn]
        self.TARGET_LOSS: float = \
            self._DEFAULT_TARGET_LOSSES[args.loss_fn] if args.target_loss < 0 else args.target_loss
        self.NUM_OF_SIMULATORS: int = args.num_of_simulators
        self.MAX_EPOCHS: int = args.max_epochs
        self.WANTED_NUM_OF_EPOCHS_TO_FRAME_AND_LOG: int = 20
        epochs_to_frame_and_log: Optional[List[int]] = None
        log_interval_fn: str = args.log_interval_fn
        if log_interval_fn == 'geom':
            epochs_to_frame_and_log: List[int] = np.unique(
                np.geomspace(1, self.MAX_EPOCHS, num=self.WANTED_NUM_OF_EPOCHS_TO_FRAME_AND_LOG, dtype=int)
            ).tolist()
        elif log_interval_fn == 'lin':
            epochs_to_frame_and_log: List[int] = np.unique(
                np.linspace(1, self.MAX_EPOCHS - 1, num=self.WANTED_NUM_OF_EPOCHS_TO_FRAME_AND_LOG, dtype=int)
            ).tolist()
        else:
            raise NotImplementedError
        self.EPOCHS_TO_FRAME_AND_LOG: List[int] = epochs_to_frame_and_log
        self.NUM_OF_EPOCHS_TO_FRAME_AND_LOG: int = len(self.EPOCHS_TO_FRAME_AND_LOG)


class Optimizer(_Hyperparamers):
    def __init__(self, args: Namespace):
        self.LR: float = args.lr
        self.WEIGHT_DECAY: float = args.weight_decay


class Random(_Hyperparamers):
    def __init__(self, args: Namespace):
        self.SEED_OFFSET: int = args.random_seed_offset


class Env(_Hyperparamers):
    def __init__(self, args: Namespace):
        artifacts_dirpath_remote: Path = Path.home() / 'data'
        artifacts_dirpath_local: Path = Path.cwd() / 'artifacts'
        current_time: str = str(datetime.now())
        artifacts_dirpath: Path = \
            artifacts_dirpath_remote if artifacts_dirpath_remote.exists() else artifacts_dirpath_local
        curr_artifacts_dirname: str = \
            args.artifacts_dirname if args.artifacts_dirname else current_time.replace(' ', '_')

        self.CURR_ARTIFACTS_DIRPATH: Path = artifacts_dirpath / curr_artifacts_dirname


class Hyperparams(Namespace):
    __instance = None

    def __new__(cls, *args, **kwargs):
        if not isinstance(cls.__instance, cls):
            cls.__instance = object.__new__(cls)
        return cls.__instance

    def __init__(self):
        self.parser_args: Namespace = HyperparamsParser().parse_args()
        super().__init__(
            net=Net(self.parser_args),
            data=Data(self.parser_args),
            training=Training(self.parser_args),
            optimizer=Optimizer(self.parser_args),
            random=Random(self.parser_args),
            env=Env(self.parser_args)
        )
        if self.parser_args.only_agg is True:
            assert self.parser_args.artifacts_dirname is not None
