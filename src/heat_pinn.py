"""PINN para a equação 1D do calor com solução analítica de referência.

Executar na raiz do projeto:
    python src/heat_pinn.py
"""
from pathlib import Path
import csv
import os
import time
import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt

SEED = int(os.environ.get("PINN_SEED", "2026"))
torch.manual_seed(SEED)
np.random.seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures"
RES_DIR = ROOT / "results"
FIG_DIR.mkdir(exist_ok=True)
RES_DIR.mkdir(exist_ok=True)

ALPHA = 1.0
T_MAX = 0.5
N_F = 4000
N_B = 256
N_0 = 256
STEPS = 8000
LR = 1e-3
LBFGS_STEPS = 100

def exact_solution(x, t):
    return torch.sin(torch.pi * x) * torch.exp(-ALPHA * torch.pi**2 * t)

class MLP(nn.Module):
    def __init__(self, width=64, depth=4):
        super().__init__()
        layers = [nn.Linear(2, width), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.Tanh()]
        layers += [nn.Linear(width, 1)]
        self.net = nn.Sequential(*layers)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, xt):
        return self.net(xt)

def residual(model, xt):
    xt = xt.detach().clone().requires_grad_(True)
    u = model(xt)
    grad_u = torch.autograd.grad(
        u, xt, grad_outputs=torch.ones_like(u), create_graph=True
    )[0]
    u_t = grad_u[:, 1:2]
    u_x = grad_u[:, 0:1]
    grad_ux = torch.autograd.grad(
        u_x, xt, grad_outputs=torch.ones_like(u_x), create_graph=True
    )[0]
    u_xx = grad_ux[:, 0:1]
    return u_t - ALPHA * u_xx

def sample_training_points():
    # Interior collocation points
    xf = torch.rand(N_F, 1, device=DEVICE)
    tf = torch.rand(N_F, 1, device=DEVICE) * T_MAX
    xt_f = torch.cat([xf, tf], dim=1)

    # Initial condition
    x0 = torch.rand(N_0, 1, device=DEVICE)
    t0 = torch.zeros_like(x0)
    xt_0 = torch.cat([x0, t0], dim=1)
    u0 = exact_solution(x0, t0)

    # Dirichlet boundaries u(0,t)=u(1,t)=0
    tb = torch.rand(N_B // 2, 1, device=DEVICE) * T_MAX
    xt_l = torch.cat([torch.zeros_like(tb), tb], dim=1)
    xt_r = torch.cat([torch.ones_like(tb), tb], dim=1)
    xt_b = torch.cat([xt_l, xt_r], dim=0)
    ub = torch.zeros_like(tb).repeat(2, 1)
    return xt_f, xt_0, u0, xt_b, ub

def main():
    model = MLP().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    xt_f, xt_0, u0, xt_b, ub = sample_training_points()
    history = []
    start = time.perf_counter()

    for step in range(1, STEPS + 1):
        optimizer.zero_grad()
        r = residual(model, xt_f)
        loss_f = torch.mean(r**2)
        loss_0 = torch.mean((model(xt_0) - u0)**2)
        loss_b = torch.mean((model(xt_b) - ub)**2)
        loss = loss_f + loss_0 + loss_b
        loss.backward()
        optimizer.step()
        history.append((step, float(loss.detach().cpu()),
                        float(loss_f.detach().cpu()),
                        float(loss_0.detach().cpu()),
                        float(loss_b.detach().cpu())))
        if step == 1 or step % 500 == 0:
            print(f"step={step:5d} loss={loss.item():.4e} "
                  f"(PDE={loss_f.item():.2e}, IC={loss_0.item():.2e}, BC={loss_b.item():.2e})")

    # Segunda fase: refinamento com L-BFGS
    print("\nIniciando refinamento com L-BFGS...")
    optimizer_lbfgs = torch.optim.LBFGS(
        model.parameters(),
        lr=1.0,
        max_iter=LBFGS_STEPS,
        max_eval=LBFGS_STEPS * 2,
        tolerance_grad=1e-9,
        tolerance_change=1e-12,
        history_size=50,
        line_search_fn="strong_wolfe",
    )
    lbfgs_calls = 0
    lbfgs_history = []

    def closure():
        nonlocal lbfgs_calls
        optimizer_lbfgs.zero_grad()
        r = residual(model, xt_f)
        loss_f = torch.mean(r**2)
        loss_0 = torch.mean((model(xt_0) - u0)**2)
        loss_b = torch.mean((model(xt_b) - ub)**2)
        loss = loss_f + loss_0 + loss_b
        lbfgs_history.append((
            lbfgs_calls + 1,
            float(loss.detach().cpu()),
            float(loss_f.detach().cpu()),
            float(loss_0.detach().cpu()),
            float(loss_b.detach().cpu()),
        ))
        loss.backward()
        lbfgs_calls += 1
        if lbfgs_calls % 10 == 0:
            print(
                f"L-BFGS avaliação={lbfgs_calls:4d} "
                f"loss={loss.item():.4e}"
            )
        return loss

    optimizer_lbfgs.step(closure)
    print(f"Total de avaliações da closure L-BFGS: {lbfgs_calls}")

    train_seconds = time.perf_counter() - start
    # Independent evaluation grid
    nx, nt = 201, 101
    x = torch.linspace(0, 1, nx, device=DEVICE)
    t = torch.linspace(0, T_MAX, nt, device=DEVICE)
    X, T = torch.meshgrid(x, t, indexing="ij")
    xt_eval = torch.stack([X.reshape(-1), T.reshape(-1)], dim=1)
    with torch.no_grad():
        pred = model(xt_eval).reshape(nx, nt)
        truth = exact_solution(X, T)
        error = pred - truth
        rel_l2 = torch.linalg.norm(error) / torch.linalg.norm(truth)
        rmse = torch.sqrt(torch.mean(error**2))
        max_abs = torch.max(torch.abs(error))

    # Save history and metrics
    with (RES_DIR / f"heat_loss_history_seed_{SEED}.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "loss_total", "loss_pde", "loss_initial", "loss_boundary"])
        writer.writerows(history)
    with (RES_DIR / f"heat_lbfgs_loss_seed_{SEED}.csv").open(
        "w", newline=""
    ) as f:
        writer = csv.writer(f)
        writer.writerow([
            "evaluation", "loss_total", "loss_pde",
            "loss_initial", "loss_boundary"
        ])
        writer.writerows(lbfgs_history)
    with (RES_DIR / f"heat_metrics_seed_{SEED}.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value", "device", "seed", "steps", "train_seconds"])
        writer.writerow(["relative_l2_error", rel_l2.item(), str(DEVICE), SEED, STEPS, train_seconds])
        writer.writerow(["rmse", rmse.item(), str(DEVICE), SEED, STEPS, train_seconds])
        writer.writerow(["max_absolute_error", max_abs.item(), str(DEVICE), SEED, STEPS, train_seconds])

    # Plot reference, PINN and absolute error
    x_np = X.detach().cpu().numpy()
    t_np = T.detach().cpu().numpy()
    pred_np = pred.detach().cpu().numpy()
    truth_np = truth.detach().cpu().numpy()
    err_np = np.abs(pred_np - truth_np)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    for ax, data, title in zip(
        axes, [truth_np, pred_np, err_np],
        ["Solução analítica", "PINN", "Erro absoluto"]
    ):
        im = ax.imshow(data, origin="lower", aspect="auto",
                       extent=[0, T_MAX, 0, 1])
        ax.set_xlabel("Tempo t")
        ax.set_ylabel("Posição x")
        ax.set_title(title)
        fig.colorbar(im, ax=ax)
    fig.savefig(FIG_DIR / "heat_solution_comparison.png", dpi=180)
    plt.close(fig)

    hist = np.array(history)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogy(hist[:, 0], hist[:, 1], label="Total")
    ax.semilogy(hist[:, 0], hist[:, 2], label="EDP")
    ax.semilogy(hist[:, 0], hist[:, 3], label="Inicial")
    ax.semilogy(hist[:, 0], hist[:, 4], label="Contorno")
    ax.set_xlabel("Passo de otimização")
    ax.set_ylabel("Perda (escala log)")
    ax.set_title("Convergência do PINN — equação do calor")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "heat_training_loss.png", dpi=180)
    plt.close(fig)

    print("\nMétricas calculadas (não interpretar sem avaliar convergência):")
    print(f"Dispositivo: {DEVICE}")
    print(f"Tempo de treinamento: {train_seconds:.2f} s")
    print(f"Erro L2 relativo: {rel_l2.item():.6e}")
    print(f"RMSE: {rmse.item():.6e}")
    print(f"Erro absoluto máximo: {max_abs.item():.6e}")
    print(f"Gráficos: {FIG_DIR}")
    print(f"Métricas: {RES_DIR}")

if __name__ == "__main__":
    main()
