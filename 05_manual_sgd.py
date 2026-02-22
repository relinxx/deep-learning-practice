import torch


def section(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def manual_sgd_demo() -> None:
    section("1) MANUAL SGD ON LINEAR MODEL")

    # Tiny linear model: y = w * x + b
    w = torch.tensor(1.0, requires_grad=True)
    b = torch.tensor(0.0, requires_grad=True)

    # Fake data: y = 2*x + 1
    x_data = torch.tensor([1.0, 2.0, 3.0, 4.0])
    y_data = 2 * x_data + 1

    learning_rate = 0.01
    epochs = 100

    print("Initial: w =", w.item(), "b =", b.item())

    for epoch in range(epochs):
        # Forward pass
        y_pred = w * x_data + b
        loss = torch.mean((y_pred - y_data) ** 2)

        # Backward pass
        loss.backward()

        # Manual SGD update
        with torch.no_grad():
            w -= learning_rate * w.grad
            b -= learning_rate * b.grad

        # Zero gradients
        w.grad.zero_()
        b.grad.zero_()

        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}: loss = {loss.item():.4f}, w = {w.item():.4f}, b = {b.item():.4f}")

    print("Final: w =", w.item(), "b =", b.item())
    print("Target: w = 2.0, b = 1.0")


def compare_manual_vs_optimizer() -> None:
    section("2) MANUAL VS PYTORCH OPTIMIZER")

    # Same setup
    w_manual = torch.tensor(1.0, requires_grad=True)
    b_manual = torch.tensor(0.0, requires_grad=True)

    w_opt = torch.tensor(1.0, requires_grad=True)
    b_opt = torch.tensor(0.0, requires_grad=True)

    x_data = torch.tensor([1.0, 2.0, 3.0, 4.0])
    y_data = 2 * x_data + 1

    learning_rate = 0.01
    epochs = 50

    optimizer = torch.optim.SGD([w_opt, b_opt], lr=learning_rate)

    for epoch in range(epochs):
        # Manual
        y_pred_manual = w_manual * x_data + b_manual
        loss_manual = torch.mean((y_pred_manual - y_data) ** 2)
        loss_manual.backward()

        with torch.no_grad():
            w_manual -= learning_rate * w_manual.grad
            b_manual -= learning_rate * b_manual.grad

        w_manual.grad.zero_()
        b_manual.grad.zero_()

        # Optimizer
        optimizer.zero_grad()
        y_pred_opt = w_opt * x_data + b_opt
        loss_opt = torch.mean((y_pred_opt - y_data) ** 2)
        loss_opt.backward()
        optimizer.step()

    print("Manual: w =", w_manual.item(), "b =", b_manual.item())
    print("Optimizer: w =", w_opt.item(), "b =", b_opt.item())


def main() -> None:
    manual_sgd_demo()
    compare_manual_vs_optimizer()

    section("STEP 5 COMPLETE")
    print("Implemented manual SGD update for tiny linear model.")


if __name__ == "__main__":
    main()