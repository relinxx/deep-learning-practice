import torch
from typing import Callable, Dict, List, Any
import torch.nn.functional as F


def section(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)



def evaluate_model(
    model: torch.nn.Module,
    criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    x_data: torch.Tensor,
    y_data: torch.Tensor,
    batch_size: int = 32
) -> Dict[str, float]: