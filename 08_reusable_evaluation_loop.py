import torch
from typing import Callable, Dict, List, Any
import torch.nn.functional as F


def section(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)

