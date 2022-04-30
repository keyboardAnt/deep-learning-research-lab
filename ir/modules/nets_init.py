import torch
from torch import Tensor


def uniform_from_ball(
        tensor: Tensor,
        init_max_norm: float,
        thetas_range_in_rad: float = 2 * torch.pi,
        thetas_shift_in_rad: float = 0.,
        **kwargs
) -> Tensor:
    assert tensor.shape == (2, 2)
    with torch.no_grad():
        rs: Tensor = torch.multiply(torch.sqrt(torch.rand(2)), init_max_norm)
        thetas: Tensor = torch.subtract(
            torch.multiply(torch.rand(2), thetas_range_in_rad),
            thetas_shift_in_rad
        )
        tensor[:, 0] = rs * torch.cos(thetas)
        tensor[:, 1] = rs * torch.sin(thetas)
        return tensor.double()


def init_thm3(layer: Tensor, layer_idx: int, init_max_norm: float = .5, **kwargs) -> Tensor:
    assert layer_idx in (0, 1)
    with torch.no_grad():
        if layer_idx == 0:
            layer = uniform_from_ball(layer, init_max_norm=init_max_norm)
        # The 2nd layer is all 0's
        else:
            torch.nn.init.constant_(layer, 0.)
        return layer.double()


def xavier_normal(tensor: Tensor, **kwargs) -> Tensor:
    return torch.nn.init.xavier_normal_(tensor)
