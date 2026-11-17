"""Single-device FP32 SAM example; synthetic data only."""
import math
import torch
from torch import nn


def negative_ic(prediction, target):
    prediction = prediction.reshape(-1)
    target = target.reshape(-1)
    prediction = prediction - prediction.mean()
    target = target - target.mean()
    denominator = (prediction.square().sum() * target.square().sum()).sqrt()
    return -(prediction * target).sum() / (denominator + 1e-8)


def sam_step(model, optimizer, closure, rho=0.05):
    """closure() computes a scalar loss on the SAME preloaded batch."""
    if not math.isfinite(rho) or rho < 0:
        raise ValueError("rho must be finite and nonnegative")
    params = [p for group in optimizer.param_groups
              for p in group["params"] if p.requires_grad]
    if not params or len({p.device for p in params}) != 1:
        raise ValueError("Use one CPU or CUDA device")
    device = params[0].device
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("This example supports CPU/CUDA only")
    if any(p.dtype != torch.float32 for p in params):
        raise ValueError("This example uses FP32, without AMP")
    batch_norm = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
                  nn.SyncBatchNorm)
    if any(isinstance(module, batch_norm) for module in model.modules()):
        raise ValueError("BatchNorm needs separate running-stat handling")

    model.train()
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state(device) if device.type == "cuda" else None

    # 1. First backward: find the uphill direction.
    optimizer.zero_grad(set_to_none=True)
    loss = closure()
    if loss.ndim != 0 or not torch.isfinite(loss).item():
        raise ValueError("The first loss must be a finite scalar")
    loss.backward()
    active = [p for p in params if p.grad is not None]
    if not active:
        raise RuntimeError("No parameter gradients")
    grad_norm = torch.stack([
        p.grad.detach().square().sum() for p in active
    ]).sum().sqrt()
    if not torch.isfinite(grad_norm).item():
        raise FloatingPointError("Nonfinite first gradient")

    # 2. One global L2 radius; no gradient graph through the perturbation.
    original = [p.detach().clone() for p in active]
    perturbations = [p.grad.detach() * (rho / (grad_norm + 1e-12))
                     for p in active]
    try:
        with torch.no_grad():
            for p, perturbation in zip(active, perturbations):
                p.add_(perturbation)

        # 3. Replay dropout randomness and recompute on the SAME batch.
        optimizer.zero_grad(set_to_none=True)
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state(cuda_rng, device)
        perturbed_loss = closure()
        if perturbed_loss.ndim != 0 or not torch.isfinite(perturbed_loss).item():
            raise FloatingPointError("Nonfinite perturbed loss")
        perturbed_loss.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all().item()
               for p in params):
            raise FloatingPointError("Nonfinite second gradient")
    except Exception:
        optimizer.zero_grad(set_to_none=True)
        raise
    finally:
        # copy_ restores the original weights exactly, including on failure.
        with torch.no_grad():
            for p, saved in zip(active, original):
                p.copy_(saved)

    # 4. Keep second-pass gradients; do ONE AdamW/SGD update at original weights.
    optimizer.step()
    perturbation_norm = torch.stack([
        e.square().sum() for e in perturbations
    ]).sum().sqrt()
    return {
        "loss": loss.detach().item(),
        "perturbed_loss": perturbed_loss.detach().item(),
        "gradient_norm": grad_norm.item(),
        "perturbation_norm": perturbation_norm.item(),
    }


class GRURegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(6, 64, num_layers=2, dropout=0.1, batch_first=True)
        self.head = nn.Linear(64, 1)

    def forward(self, x):
        sequence, _ = self.gru(x)
        return self.head(sequence[:, -1]).squeeze(-1)


def main():
    torch.manual_seed(7)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = GRURegressor().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    # x: (batch, time, fields). Synthetic labels only check code execution.
    x = torch.randn(64, 30, 6, device=device)
    y = 0.6 * x[:, -1, 0] - 0.3 * x[:, -1, 1]
    closure = lambda: negative_ic(model(x), y)
    # Native GRU path: avoid cuDNN-owned dropout state in this teaching demo.
    with torch.backends.cudnn.flags(enabled=False):
        stats = sam_step(model, optimizer, closure, rho=0.05)
    print({key: round(value, 6) for key, value in stats.items()})


if __name__ == "__main__":
    main()
