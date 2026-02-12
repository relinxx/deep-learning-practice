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
    
    """
    Evaluate model on data with batching support.

    Args:
        model: PyTorch model
        criterion: Loss function
        x_data: Input data
        y_data: Target data
        batch_size: Batch size for evaluation

    Returns:
        Dict with evaluation metrics
    """
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for i in range(0, len(x_data), batch_size):
            x_batch = x_data[i:i+batch_size]
            y_batch = y_data[i:i+batch_size]

            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            total_loss += loss.item() * len(x_batch)

            if logits.shape != y_batch.shape:  # Classification
                predictions = logits.argmax(dim=1)
                total_correct += (predictions == y_batch).sum().item()

            total_samples += len(x_batch)

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples if total_samples > 0 else 0.0

    return {'loss': avg_loss, 'accuracy': accuracy}


def evaluate_with_metrics(
    model: torch.nn.Module,
    x_data: torch.Tensor,
    y_data: torch.Tensor,
    num_classes: int = None,
    batch_size: int = 32
) -> Dict[str, float]:
    """
    Evaluate model with comprehensive metrics.

    Args:
        model: PyTorch model
        x_data: Input data
        y_data: Target data
        num_classes: Number of classes for classification metrics
        batch_size: Batch size for evaluation

    Returns:
        Dict with detailed metrics
    """
    model.eval()
    all_predictions = []
    all_targets = []
    total_loss = 0.0

    criterion = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for i in range(0, len(x_data), batch_size):
            x_batch = x_data[i:i+batch_size]
            y_batch = y_data[i:i+batch_size]

            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            total_loss += loss.item() * len(x_batch)

            predictions = logits.argmax(dim=1)
            all_predictions.extend(predictions.tolist())
            all_targets.extend(y_batch.tolist())

    avg_loss = total_loss / len(x_data)
    accuracy = sum(p == t for p, t in zip(all_predictions, all_targets)) / len(all_targets)

    metrics = {
        'loss': avg_loss,
        'accuracy': accuracy,
    }

    if num_classes:
        # Confusion matrix
        confusion = torch.zeros(num_classes, num_classes)
        for t, p in zip(all_targets, all_predictions):
            confusion[t, p] += 1

        # Per-class accuracy
        per_class_acc = confusion.diag() / confusion.sum(dim=1)
        metrics['per_class_accuracy'] = per_class_acc.tolist()

        # Macro average
        metrics['macro_accuracy'] = per_class_acc.mean().item()

    return metrics


def demo_evaluation_loop() -> None:
    section("1) DEMO EVALUATION LOOP")

    # Simple classification model
    model = torch.nn.Sequential(
        torch.nn.Linear(10, 50),
        torch.nn.ReLU(),
        torch.nn.Linear(50, 3)
    )

    # Fake data
    x_test = torch.randn(100, 10)
    y_test = torch.randint(0, 3, (100,))

    # Basic evaluation
    basic_metrics = evaluate_model(model, torch.nn.CrossEntropyLoss(), x_test, y_test)
    print("Basic evaluation:")
    print(f"  Loss: {basic_metrics['loss']:.4f}")
    print(f"  Accuracy: {basic_metrics['accuracy']:.4f}")

    # Detailed evaluation
    detailed_metrics = evaluate_with_metrics(model, x_test, y_test, num_classes=3)
    print("\nDetailed evaluation:")
    print(f"  Loss: {detailed_metrics['loss']:.4f}")
    print(f"  Accuracy: {detailed_metrics['accuracy']:.4f}")
    print(f"  Macro Accuracy: {detailed_metrics['macro_accuracy']:.4f}")
    print(f"  Per-class Accuracy: {detailed_metrics['per_class_accuracy']}")

