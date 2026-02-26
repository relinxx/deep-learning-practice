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
def main() -> None:
    compare_sgd_implementations()
    different_learning_rates()

    section("STEP 6 COMPLETE")
    print("Compared manual gradient updates vs torch.optim.SGD.")


if __name__ == "__main__":
    main()