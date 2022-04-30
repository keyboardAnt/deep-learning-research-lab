import torch
from torch import Tensor, nn


class SumOfExpLoss(nn.Module):
    def __init__(self):
        pass

    def __call__(self, outputs: Tensor, targets: Tensor) -> Tensor:
        return torch.exp(
            torch.multiply(outputs * targets, -1)
        ).sum()

    def __repr__(self) -> str:
        return 'SumOfExpLoss'
