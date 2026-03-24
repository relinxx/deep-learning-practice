import torch


def section(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)

def compare_sgd_implementations() -> None:
    section("1) MANUAL SGD VS TORCH.OPTIM.SGD")

    # Initialize parameters identically
    w_manual = torch.tensor(1.0, requires_grad=True)
    b_manual = torch.tensor(0.0, requires_grad=True)

    w_opt = torch.tensor(1.0, requires_grad=True)
    b_opt = torch.tensor(0.0, requires_grad=True)
        # Same data
    x_data = torch.tensor([1.0, 2.0, 3.0, 4.0])
    y_data = 2 * x_data + 1

    learning_rate = 0.01
    epochs = 100
    optimizer = torch.optim.SGD([w_opt, b_opt], lr=learning_rate)

    print("Initial: w =", w_manual.item(), "b =", b_manual.item())
    for epoch in range(epochs):
        # Manual SGD
        y_pred_manual = w_manual * x_data + b_manual
        loss_manual = torch.mean((y_pred_manual - y_data) ** 2)
        loss_manual.backward()

        with torch.no_grad():
            w_manual -= learning_rate * w_manual.grad
            b_manual -= learning_rate * b_manual.grad
            w_manual.grad.zero_()
        b_manual.grad.zero_()

        # PyTorch Optimizer SGD
        optimizer.zero_grad()
        y_pred_opt = w_opt * x_data + b_opt
        loss_opt = torch.mean((y_pred_opt - y_data) ** 2)
        loss_opt.backward()
        optimizer.step()
        if (epoch + 1) % 25 == 0:
            print(f"Epoch {epoch+1}:")
            print(f"  Manual: loss={loss_manual.item():.4f}, w={w_manual.item():.4f}, b={b_manual.item():.4f}")
            print(f"  Optimizer: loss={loss_opt.item():.4f}, w={w_opt.item():.4f}, b={b_opt.item():.4f}")

    print("\nFinal comparison:")
    print("Manual: w =", w_manual.item(), "b =", b_manual.item())
    print("Optimizer: w =", w_opt.item(), "b =", b_opt.item())
    print("Difference: w =", abs(w_manual.item() - w_opt.item()), "b =", abs(b_manual.item() - b_opt.item()))

def different_learning_rates() -> None:
    section("2) DIFFERENT LEARNING RATES")

    w = torch.tensor(1.0, requires_grad=True)
    b = torch.tensor(0.0, requires_grad=True)

    x_data = torch.tensor([1.0, 2.0, 3.0])
    y_data = 2 * x_data + 1
        rates = [0.001, 0.01, 0.1]

    for lr in rates:
        # Reset parameters
        w.data.fill_(1.0)
        b.data.fill_(0.0)
        w.grad = None
        b.grad = None
        
        optimizer = torch.optim.SGD([w, b], lr=lr)

        for _ in range(50):
            optimizer.zero_grad()
            y_pred = w * x_data + b
            loss = torch.mean((y_pred - y_data) ** 2)
            loss.backward()
            optimizer.step()
        print(f"LR {lr}: w={w.item():.4f}, b={b.item():.4f}, final_loss={loss.item():.4f}")

def train_model(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_val: torch.Tensor = None,
    y_val: torch.Tensor = None,
    epochs: int = 10,
    verbose: bool = True
) -> Dict[str, Any]:
    """
        Train model with optional validation.

    Args:
        model: PyTorch model
        optimizer: Optimizer
        criterion: Loss function
        x_train: Training inputs
        y_train: Training targets
        x_val: Validation inputs (optional)
        y_val: Validation targets (optional)
        epochs: Number of epochs
        verbose: Print progress

    Returns:
        Dict with training history
    """
    history = {
        'train_loss': [],
        'train_accuracy': [],
        'val_loss': [],
        'val_accuracy': []
    }

    for epoch in range(epochs):
        # Train
        train_metrics = train_one_epoch(model, optimizer, criterion, x_train, y_train)
        history['train_loss'].append(train_metrics['loss'])
        history['train_accuracy'].append(train_metrics['accuracy'])

        # Validate
        if x_val is not None and y_val is not None:
            val_metrics = evaluate_model(model, criterion, x_val, y_val)
            history['val_loss'].append(val_metrics['loss'])
            history['val_accuracy'].append(val_metrics['accuracy'])

        if verbose and (epoch + 1) % max(1, epochs // 10) == 0:
            print(f"Epoch {epoch+1}/{epochs}:")
            print(f"  Train Loss: {train_metrics['loss']:.4f}")
            if not torch.isnan(torch.tensor(train_metrics['accuracy'])):
                print(f"  Train Acc: {train_metrics['accuracy']:.4f}")
            
                   




def main() -> None:
    compare_sgd_implementations()
    different_learning_rates()

    section("STEP 6 COMPLETE")
    print("Compared manual gradient updates vs torch.optim.SGD.")


if __name__ == "__main__":
    main()