import torch


def section(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def gradient_accumulation_demo() -> None:
    section("1) GRADIENT ACCUMULATION WITHOUT ZERO_GRAD")

    x = torch.tensor(2.0, requires_grad=True)

    # First computation
    y1 = x ** 2
    y1.backward()
    print("After first backward (y = x^2):")
    print("x.grad =", x.grad.item())

    # Second computation without zero_grad
    y2 = x ** 3
    y2.backward()
    print("\nAfter second backward (y = x^3) without zero_grad:")
    print("x.grad =", x.grad.item())
    print("Expected: 2*x + 3*x^2 =", 2 * x.item() + 3 * x.item() ** 2)


def why_zero_grad_matters() -> None:
    section("2) WHY ZERO_GRAD MATTERS")

    x = torch.tensor(2.0, requires_grad=True)

    # Reset gradients
    x.grad = None

    # First loss
    loss1 = (x - 1) ** 2
    loss1.backward()
    grad1 = x.grad.item()
    print("Loss 1: (x-1)^2, grad =", grad1)

    # Second loss without zero_grad - gradients accumulate
    loss2 = (x - 3) ** 2
    loss2.backward()
    grad_accumulated = x.grad.item()
    print("Loss 2: (x-3)^2, grad =", grad_accumulated)
    print("Accumulated grad =", grad_accumulated)

    # Reset and compute separately
    x.grad = None
    loss2.backward()
    grad2 = x.grad.item()
    print("Loss 2 alone: grad =", grad2)
    print("Correct total grad =", grad1 + grad2)


def manual_vs_optimizer() -> None:
    section("3) MANUAL VS OPTIMIZER GRADIENT HANDLING")

    x = torch.tensor([1.0, 2.0], requires_grad=True)

    # Manual gradient accumulation
    loss1 = torch.sum(x ** 2)
    loss1.backward()
    manual_grad = x.grad.clone()

    # Without zero_grad, next backward accumulates
    loss2 = torch.sum((x - 1) ** 2)
    loss2.backward()
    accumulated_grad = x.grad.clone()

    print("Manual grad after loss1:", manual_grad)
    print("Accumulated grad after loss2:", accumulated_grad)

    # Optimizer handles zero_grad automatically
    x.grad = None
    optimizer = torch.optim.SGD([x], lr=0.1)
    optimizer.zero_grad()  # Explicit zero_grad
    loss1.backward()
    optimizer.step()
    print("After optimizer step:", x)


def main() -> None:
    gradient_accumulation_demo()
    why_zero_grad_matters()
    manual_vs_optimizer()

    section("STEP 4 COMPLETE")
    print("Implemented gradient accumulation examples.")


if __name__ == "__main__":
    main()