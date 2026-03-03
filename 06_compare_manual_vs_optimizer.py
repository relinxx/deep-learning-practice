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

def main() -> None:
    compare_sgd_implementations()
    different_learning_rates()

    section("STEP 6 COMPLETE")
    print("Compared manual gradient updates vs torch.optim.SGD.")


if __name__ == "__main__":
    main()