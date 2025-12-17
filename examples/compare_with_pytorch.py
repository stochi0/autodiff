"""
BeaconGrad vs PyTorch parity checks.

This compares BeaconGrad forward + backward against PyTorch for a small set of models:
- Linear
- 2-layer MLP

Run:
  uv run python examples/compare_with_pytorch.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from beacongrad.tensor import Tensor
from beacongrad import ops


def _max_abs(x: np.ndarray) -> float:
    return float(np.max(np.abs(x))) if x.size else 0.0


def _max_err(a: np.ndarray, b: np.ndarray) -> float:
    return _max_abs(np.asarray(a) - np.asarray(b))


def _print_row(name: str, fwd: float, grad: float):
    print(f"{name:18s}  forward={fwd:.3e}  grad={grad:.3e}")


def _require_torch():
    try:
        import torch  # noqa: F401
    except Exception as e:
        raise SystemExit(
            "PyTorch is required for parity checks.\n"
            "Install it with uv:\n"
            "  uv pip install torch\n"
            f"Import error: {e}"
        )


def parity_linear():
    import torch

    np.random.seed(0)
    torch.manual_seed(0)
    dtype = torch.float64

    x_np = np.random.randn(7, 4).astype(np.float64)
    w_np = np.random.randn(3, 4).astype(np.float64)
    b_np = np.random.randn(3).astype(np.float64)

    # BeaconGrad
    x_bg = Tensor(x_np, requires_grad=True, dtype=np.float64)
    w_bg = Tensor(w_np, requires_grad=True, dtype=np.float64)
    b_bg = Tensor(b_np, requires_grad=True, dtype=np.float64)
    y_bg = (x_bg @ w_bg.T) + b_bg
    loss_bg = ops.mse_loss(y_bg, Tensor(np.zeros_like(y_bg.data), dtype=np.float64))
    loss_bg.backward()

    # PyTorch
    x_th = torch.from_numpy(x_np).to(dtype).requires_grad_(True)
    w_th = torch.from_numpy(w_np).to(dtype).requires_grad_(True)
    b_th = torch.from_numpy(b_np).to(dtype).requires_grad_(True)
    y_th = (x_th @ w_th.T) + b_th
    loss_th = torch.mean((y_th - torch.zeros_like(y_th)) ** 2)
    loss_th.backward()

    fwd_err = _max_err(y_bg.data, y_th.detach().cpu().numpy())
    dx_err = _max_err(x_bg.grad, x_th.grad.detach().cpu().numpy())
    dw_err = _max_err(w_bg.grad, w_th.grad.detach().cpu().numpy())
    db_err = _max_err(b_bg.grad, b_th.grad.detach().cpu().numpy())
    grad_err = max(dx_err, dw_err, db_err)

    print("\nLinear parity")
    print(f"  forward max error: {fwd_err:.3e}")
    print(f"  dx max error:      {dx_err:.3e}")
    print(f"  dW max error:      {dw_err:.3e}")
    print(f"  db max error:      {db_err:.3e}")

    return fwd_err, grad_err


def parity_mlp_2layer():
    import torch

    np.random.seed(0)
    torch.manual_seed(0)
    dtype = torch.float64

    bsz, din, dh, dout = 8, 5, 16, 3
    x_np = np.random.randn(bsz, din).astype(np.float64)
    w1_np = np.random.randn(dh, din).astype(np.float64)
    b1_np = np.random.randn(dh).astype(np.float64)
    w2_np = np.random.randn(dout, dh).astype(np.float64)
    b2_np = np.random.randn(dout).astype(np.float64)

    # BeaconGrad
    x_bg = Tensor(x_np, requires_grad=True, dtype=np.float64)
    w1_bg = Tensor(w1_np, requires_grad=True, dtype=np.float64)
    b1_bg = Tensor(b1_np, requires_grad=True, dtype=np.float64)
    w2_bg = Tensor(w2_np, requires_grad=True, dtype=np.float64)
    b2_bg = Tensor(b2_np, requires_grad=True, dtype=np.float64)

    h_bg = (x_bg @ w1_bg.T) + b1_bg
    h_bg = h_bg.relu()
    y_bg = (h_bg @ w2_bg.T) + b2_bg
    loss_bg = ops.mse_loss(y_bg, Tensor(np.zeros_like(y_bg.data), dtype=np.float64))
    loss_bg.backward()

    # PyTorch
    x_th = torch.from_numpy(x_np).to(dtype).requires_grad_(True)
    w1_th = torch.from_numpy(w1_np).to(dtype).requires_grad_(True)
    b1_th = torch.from_numpy(b1_np).to(dtype).requires_grad_(True)
    w2_th = torch.from_numpy(w2_np).to(dtype).requires_grad_(True)
    b2_th = torch.from_numpy(b2_np).to(dtype).requires_grad_(True)

    h_th = (x_th @ w1_th.T) + b1_th
    h_th = torch.relu(h_th)
    y_th = (h_th @ w2_th.T) + b2_th
    loss_th = torch.mean((y_th - torch.zeros_like(y_th)) ** 2)
    loss_th.backward()

    fwd_err = _max_err(y_bg.data, y_th.detach().cpu().numpy())
    errs = [
        _max_err(x_bg.grad, x_th.grad.detach().cpu().numpy()),
        _max_err(w1_bg.grad, w1_th.grad.detach().cpu().numpy()),
        _max_err(b1_bg.grad, b1_th.grad.detach().cpu().numpy()),
        _max_err(w2_bg.grad, w2_th.grad.detach().cpu().numpy()),
        _max_err(b2_bg.grad, b2_th.grad.detach().cpu().numpy()),
    ]
    grad_err = float(max(errs))

    print("\nMLP (2-layer) parity")
    print(f"  forward max error: {fwd_err:.3e}")
    print(f"  grad max error:    {grad_err:.3e}")

    return fwd_err, grad_err


if __name__ == "__main__":
    _require_torch()

    print("=" * 60)
    print("BeaconGrad vs PyTorch parity checks (float64)")
    print("=" * 60)

    rows = []

    fwd, grad = parity_linear()
    rows.append(("Linear", fwd, grad))

    fwd, grad = parity_mlp_2layer()
    rows.append(("MLP", fwd, grad))

    print("\nSummary")
    print("Model\tForward max error\tGrad max error")
    for name, fwd, grad in rows:
        print(f"{name}\t{fwd:.3e}\t{grad:.3e}")

    # Tight checks (as requested). Print-first, then assert.
    for name, fwd, grad in rows:
        if fwd >= 1e-6 or grad >= 1e-6:
            raise SystemExit(f"Parity check failed for {name}: fwd={fwd:.3e}, grad={grad:.3e}")

