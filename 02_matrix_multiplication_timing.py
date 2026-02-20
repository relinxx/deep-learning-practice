import time

import torch


def section(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def benchmark_cpu_matmul(size: int, repeats: int = 20) -> tuple[float, float, float]:
    a = torch.randn(size, size)
    b = torch.randn(size, size)

    # Warm-up reduces first-run overhead noise.
    _ = torch.matmul(a, b)
    _ = a @ b

    start_matmul = time.perf_counter()
    for _ in range(repeats):
        _ = torch.matmul(a, b)
    end_matmul = time.perf_counter()

    start_operator = time.perf_counter()
    for _ in range(repeats):
        _ = a @ b
    end_operator = time.perf_counter()

    matmul_ms = (end_matmul - start_matmul) * 1000 / repeats
    operator_ms = (end_operator - start_operator) * 1000 / repeats

    # Verify both forms produce effectively the same result.
    max_abs_diff = (torch.matmul(a, b) - (a @ b)).abs().max().item()

    return matmul_ms, operator_ms, max_abs_diff


def compare_sizes() -> None:
    section("1) CPU MATRIX MULTIPLICATION TIMING")

    torch.manual_seed(42)
    torch.set_num_threads(1)

    sizes = [128, 256, 512, 1024]
    repeats_map = {128: 200, 256: 100, 512: 40, 1024: 10}

    print("Threads:", torch.get_num_threads())
    print("Device: CPU")
    print("dtype: float32")
    print()
    print(f"{'size':>8} | {'matmul(ms)':>12} | {'@(ms)':>10} | {'max_abs_diff':>12}")
    print("-" * 52)

    for size in sizes:
        repeats = repeats_map[size]
        matmul_ms, operator_ms, max_abs_diff = benchmark_cpu_matmul(size, repeats=repeats)
        print(f"{size:>8} | {matmul_ms:>12.4f} | {operator_ms:>10.4f} | {max_abs_diff:>12.6f}")


def small_examples() -> None:
    section("2) SHAPE EXAMPLES")

    a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    b = torch.tensor([[5.0, 6.0], [7.0, 8.0]])

    print("A:\n", a)
    print("B:\n", b)
    print("A @ B:\n", a @ b)

    x = torch.randn(3, 5)
    y = torch.randn(5, 2)
    z = x @ y

    print("\n(3x5) @ (5x2) ->", z.shape)


def main() -> None:
    small_examples()
    compare_sizes()

    section("STEP 2 COMPLETE")
    print("Implemented matrix multiplication examples with CPU timing.")


if __name__ == "__main__":
    main()
