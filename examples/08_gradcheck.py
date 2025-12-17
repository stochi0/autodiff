import numpy as np

from beacongrad.tensor import Tensor
from beacongrad.utils import gradcheck
from beacongrad import ops


def test_scalar_elementwise():
    x = Tensor(np.random.randn(3, 2), requires_grad=True, dtype=np.float64)
    y = Tensor(np.random.randn(3, 2), requires_grad=True, dtype=np.float64)

    gradcheck(lambda x, y: (x + y).sum(), [x, y])
    gradcheck(lambda x, y: (x - y).sum(), [x, y])
    gradcheck(lambda x, y: (x * y).sum(), [x, y])
    gradcheck(lambda x, y: (x / (y + 1.5)).sum(), [x, y])  # avoid div by ~0

    # pow: positive base only
    base = Tensor(np.abs(np.random.randn(4, 3)) + 1.0, requires_grad=True, dtype=np.float64)
    exp = Tensor(np.random.randn(4, 3), requires_grad=True, dtype=np.float64)
    gradcheck(lambda a, b: (a**b).sum(), [base, exp])

    # activations (avoid relu kink at 0)
    z = Tensor(np.random.randn(5, 4) + 0.1, requires_grad=True, dtype=np.float64)
    gradcheck(lambda z: z.relu().sum(), [z])
    gradcheck(lambda z: z.tanh().sum(), [z])
    gradcheck(lambda z: z.sigmoid().sum(), [z])


def test_broadcasting():
    x = Tensor(np.random.randn(2, 3), requires_grad=True, dtype=np.float64)
    b = Tensor(np.random.randn(3), requires_grad=True, dtype=np.float64)
    gradcheck(lambda x, b: (x + b).sum(), [x, b])
    gradcheck(lambda x, b: (x * b).sum(), [x, b])


def test_sum_reshape_transpose():
    x = Tensor(np.random.randn(2, 3, 4), requires_grad=True, dtype=np.float64)
    gradcheck(lambda x: x.sum(axis=1).sum(), [x])
    gradcheck(lambda x: x.reshape(6, 4).sum(), [x])
    gradcheck(lambda x: x.transpose((2, 0, 1)).sum(), [x])


def test_matmul_grad():
    x = Tensor(np.random.randn(3, 4), requires_grad=True, dtype=np.float64)
    w = Tensor(np.random.randn(4, 5), requires_grad=True, dtype=np.float64)

    def f(x, w):
        return (x @ w).sum()

    gradcheck(f, [x, w])

    # batched
    a = Tensor(np.random.randn(2, 3, 4), requires_grad=True, dtype=np.float64)
    b = Tensor(np.random.randn(2, 4, 5), requires_grad=True, dtype=np.float64)
    gradcheck(lambda a, b: (a @ b).sum(), [a, b])


def test_cross_entropy():
    # small batch
    logits = Tensor(np.random.randn(4, 3), requires_grad=True, dtype=np.float64)
    target = np.array([0, 2, 1, 0], dtype=np.int64)
    gradcheck(lambda l: ops.cross_entropy(l, target), [logits])


if __name__ == "__main__":
    np.random.seed(0)
    test_scalar_elementwise()
    test_broadcasting()
    test_sum_reshape_transpose()
    test_matmul_grad()
    test_cross_entropy()
    print("All gradchecks passed.")


