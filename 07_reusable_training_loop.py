import torch
from typing import Callable, Dict, Any


def section(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    
def train_one_epoch(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    x_batch: torch.Tensor,
    y_batch: torch.Tensor
) -> Dict[str, float]:
    """
    Train model for one epoch on a single batch.

    Args:
        model: PyTorch model
        optimizer: Optimizer
        criterion: Loss function
        x_batch: Input batch
        y_batch: Target batch

    Returns:
        Dict with 'loss' and 'accuracy' (if classification)
    """
    model.train()
    optimizer.zero_grad()

    logits = model(x_batch)
    loss = criterion(logits, y_batch)

    loss.backward()
    optimizer.step()

    # For classification, compute accuracy
    if logits.shape == y_batch.shape:  # Regression
        accuracy = float('nan')
    else:  # Classification
        predictions = logits.argmax(dim=1)
        accuracy = (predictions == y_batch).float().mean().item()

    return {'loss': loss.item(), 'accuracy': accuracy}


def main() -> None:
    demo_training_loop()

    section("STEP 7 COMPLETE")
    print("Built reusable training loop function.")


if __name__ == "__main__":
    main()