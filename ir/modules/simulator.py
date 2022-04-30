from datetime import datetime

from pathlib import Path
from typing import Optional

import torch
from ignite.handlers import ModelCheckpoint
from torch import nn
from ignite.engine import Events, Engine, DeterministicEngine
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from ir.config.hyperparams import Hyperparams
from ir.modules import nets
from ir.modules.simulation_dataset import get_simulation_dataset, SimulationDataset
from ir.modules.nets import DFFNet
from ir.modules.simulation_track_and_analyzer import SimulationTrack
from ir.utils.logging import get_logging
logging = get_logging()


def early_stopping_score_fn(engine: Engine) -> float:
    loss_value: float = engine.state.output
    return -1 * loss_value


class Simulator:
    def __init__(self, hyperparams: Hyperparams, artifacts_dirpath: Path):
        self._hyperparams: Hyperparams = hyperparams
        self.artifacts_dirpath: Path = artifacts_dirpath
        self.artifacts_dirpath.mkdir(parents=True, exist_ok=True)
        self.dataset: SimulationDataset = get_simulation_dataset(hyperparams=self._hyperparams)
        self._model: DFFNet = nets.get_net(hyperparams=self._hyperparams)
        self._train_loader: DataLoader = DataLoader(self.dataset, batch_size=len(self.dataset))
        self._optimizer: Optimizer = torch.optim.SGD(
            self._model.parameters(),
            lr=hyperparams.optimizer.LR,
            weight_decay=hyperparams.optimizer.WEIGHT_DECAY
        )
        self._criterion: nn.Module = hyperparams.training.LOSS_FN
        self._simulation_track: SimulationTrack = SimulationTrack(dataset=self.dataset)
        self._trainer: DeterministicEngine = self._get_trainer()
        logging.basicConfig(level=logging.INFO_IR)

    def _get_trainer(self) -> DeterministicEngine:
        def update_model(engine: Engine, batch):
            inputs, targets = batch
            self._optimizer.zero_grad()
            outputs = self._model(inputs)
            loss = self._criterion(outputs, targets)
            loss.backward()
            self._optimizer.step()
            return loss.item()

        trainer = DeterministicEngine(update_model)

        def event_filter(engine, event) -> bool:
            return event in self._hyperparams.training.EPOCHS_TO_FRAME_AND_LOG

        @trainer.on(Events.EPOCH_COMPLETED(event_filter=event_filter))
        def log(engine: Engine) -> None:
            logging.info_ir(f"Epoch[{engine.state.epoch}] Loss: {engine.state.output:.15e}")
            self._simulation_track.log_model_state_dict(model=self._model)
            self._simulation_track.log_loss(loss=engine.state.output)

        filename_prefix: str = f"model_ts_{str(datetime.now()).replace(' ', '_')}"
        handler = ModelCheckpoint(str(self.artifacts_dirpath), filename_prefix, create_dir=True)
        trainer.add_event_handler(Events.COMPLETED, handler, {'epoch': self._model})

        @trainer.on(Events.COMPLETED)
        def save_track(engine: Engine) -> None:
            self._simulation_track.save(self.artifacts_dirpath)

        return trainer

    def run(self) -> SimulationTrack:
        self._trainer.run(self._train_loader, max_epochs=self._hyperparams.training.MAX_EPOCHS)
        return self._simulation_track
