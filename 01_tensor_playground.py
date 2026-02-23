import torch


def section(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def shapes_demo() -> None:
    section("1) SHAPES")

    scalar = torch.tensor(7)
    vector = torch.tensor([1, 2, 3, 4])
    matrix = torch.tensor([[1, 2, 3], [4, 5, 6]])
    tensor_3d = torch.randn(2, 3, 4)

    print("scalar shape:", scalar.shape)
    print("vector shape:", vector.shape)
    print("matrix shape:", matrix.shape)
    print("3D tensor shape:", tensor_3d.shape)


def dtypes_demo() -> None:
    section("2) DTYPES")

    x_int = torch.tensor([1, 2, 3], dtype=torch.int64)
    x_float = x_int.to(dtype=torch.float32)

    print("x_int dtype:", x_int.dtype)
    print("x_float dtype:", x_float.dtype)

    # Float operations are required for gradients.
    requires_grad_example = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    print("requires_grad tensor dtype:", requires_grad_example.dtype)


def broadcasting_demo() -> None:
    section("3) BROADCASTING")

    a = torch.ones(2, 3)
    b = torch.tensor([10.0, 20.0, 30.0])
    c = a + b

    print("a shape:", a.shape)
    print("b shape:", b.shape)
    print("a + b shape:", c.shape)


def indexing_demo() -> None:
    section("4) INDEXING AND SLICING")

    x = torch.arange(1, 13).view(3, 4)
    print("x:\n", x)

    print("first row:", x[0])
    print("last column:", x[:, -1])
    print("submatrix rows 0..1, cols 1..2:\n", x[0:2, 1:3])


def reshaping_demo() -> None:
    section("5) RESHAPING")

    x = torch.arange(1, 13)
    print("original x shape:", x.shape)

    x_view = x.view(3, 4)
    x_reshape = x.reshape(2, 6)

    print("x.view(3, 4) shape:", x_view.shape)
    print("x.reshape(2, 6) shape:", x_reshape.shape)

    x_unsqueeze = x.unsqueeze(0)
    x_squeeze = x_unsqueeze.squeeze(0)

    print("x.unsqueeze(0) shape:", x_unsqueeze.shape)
    print("squeezed back shape:", x_squeeze.shape)


def main() -> None:
    torch.manual_seed(42)

    shapes_demo()
    dtypes_demo()
    broadcasting_demo()
    indexing_demo()
    reshaping_demo()

    section("STEP 1 COMPLETE")
    print("You implemented the tensor playground basics.")


if __name__ == "__main__":
    
    main()
