import torch


def section(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def scalar_function_demo() -> None:
    section("1) SCALAR FUNCTION AUTODIFF")

    x = torch.tensor(2.0, requires_grad=True)
    y = x ** 2 + 3 * x + 1

    print("x =", x.item())
    print("y = x^2 + 3x + 1 =", y.item())

    y.backward()
    print("dy/dx =", x.grad.item())


def vector_function_demo() -> None:
    section("2) VECTOR FUNCTION AUTODIFF")

    x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    y = torch.sum(x ** 2)

    print("x =", x)
    print("y = sum(x^2) =", y.item())

    y.backward()
    print("dy/dx =", x.grad)


def chain_rule_demo() -> None:
    section("3) CHAIN RULE EXAMPLE")

    x = torch.tensor(1.0, requires_grad=True)
    y = x ** 2
    z = y + 3 * x

    print("x =", x.item())
    print("y = x^2 =", y.item())
    print("z = y + 3x =", z.item())

    z.backward()
    print("dz/dx =", x.grad.item())


def vector_to_scalar_demo() -> None:
    section("4) VECTOR TO SCALAR")

    x = torch.tensor([1.0, 2.0], requires_grad=True)
    y = torch.tensor([3.0, 4.0], requires_grad=True)
    z = torch.sum(x * y)

    print("x =", x)
    print("y =", y)
    print("z = sum(x * y) =", z.item())

    z.backward()
    print("dz/dx =", x.grad)
    print("dz/dy =", y.grad)


def main() -> None:
    scalar_function_demo()
    vector_function_demo()
    chain_rule_demo()
    vector_to_scalar_demo()

    section("STEP 3 COMPLETE")
    print("Implemented automatic differentiation examples.")


if __name__ == "__main__":
    main()