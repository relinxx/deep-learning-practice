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

def main() -> None:
    compare_sgd_implementations()
    different_learning_rates()

    section("STEP 6 COMPLETE")
    print("Compared manual gradient updates vs torch.optim.SGD.")


if __name__ == "__main__":
    main()