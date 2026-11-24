"""PINN para a equação de Schrödinger 1D de partícula livre.

Atenção metodológica:
A referência gaussiana livre abaixo é analítica em domínio infinito, mas não é
exatamente periódica no intervalo finito usado pelo PINN. O código impõe condições
periódicas aproximadas; portanto, este script é um protótipo exploratório e NÃO deve
ser usado para afirmar precisão científica sem estudo de sensibilidade do domínio.

Executar na raiz do projeto:
    python src/schrodinger_pinn.py
"""
from pathlib import Path
import csv
import time
import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt

SEED = 2026
torch.manual_seed(SEED)
np.random.seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures"
RES_DIR = ROOT / "results"
FIG_DIR.mkdir(exist_ok=True)
RES_DIR.mkdir(exist_ok=True)

XMIN, XMAX = -np.pi, np.pi
TMAX = 0.25
SIGMA = 0.7
X0 = -0.7
K0 = 1.5
HBAR_OVER_2M = 0.5
N_F, N_B, N_0 = 5000, 256, 512
STEPS, LR = 12000, 1e-3

def exact_free_packet(x, t):
    """Solução analítica complexa de pacote gaussiano livre em domínio infinito."""
    # Psi(x,0) = exp(-(x-x0)^2/(2*sigma^2)) * exp(i*k0*x)
    tau = 2.0 * HBAR_OVER_2M * t / (SIGMA**2)
    denom = 1.0 + 1j * tau
    prefactor = (1.0 / torch.sqrt(denom))
    phase = torch.exp(1j * K0 * x)
    envelope = torch.exp(-((x - X0)**2) / (2.0 * SIGMA**2 * denom))
    dispersion_phase = torch.exp(-1j * (K0**2) * HBAR_OVER_2M * t)
    return prefactor * envelope * phase * dispersion_phase

class ComplexMLP(nn.Module):
    def __init__(self, width=64, depth=4):
        super().__init__()
        layers = [nn.Linear(2, width), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.Tanh()]
        layers += [nn.Linear(width, 2)]
        self.net = nn.Sequential(*layers)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, xt):
        out = self.net(xt)
        return out[:, 0:1], out[:, 1:2]

def derivatives(model, xt):
    xt = xt.detach().clone().requires_grad_(True)
    re, im = model(xt)
    grad_re = torch.autograd.grad(re, xt, torch.ones_like(re), create_graph=True)[0]
    grad_im = torch.autograd.grad(im, xt, torch.ones_like(im), create_graph=True)[0]
    re_t, re_x = grad_re[:, 1:2], grad_re[:, 0:1]
    im_t, im_x = grad_im[:, 1:2], grad_im[:, 0:1]
    re_xx = torch.autograd.grad(re_x, xt, torch.ones_like(re_x), create_graph=True)[0][:, 0:1]
    im_xx = torch.autograd.grad(im_x, xt, torch.ones_like(im_x), create_graph=True)[0][:, 0:1]
    return re, im, re_t, im_t, re_xx, im_xx

def main():
    model = ComplexMLP().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    xf = XMIN + (XMAX - XMIN) * torch.rand(N_F, 1, device=DEVICE)
    tf = TMAX * torch.rand(N_F, 1, device=DEVICE)
    xt_f = torch.cat([xf, tf], dim=1)

    x0 = XMIN + (XMAX - XMIN) * torch.rand(N_0, 1, device=DEVICE)
    t0 = torch.zeros_like(x0)
    xt_0 = torch.cat([x0, t0], dim=1)
    psi0 = exact_free_packet(x0, t0)

    tb = TMAX * torch.rand(N_B, 1, device=DEVICE)
    xt_left = torch.cat([torch.full_like(tb, XMIN), tb], dim=1)
    xt_right = torch.cat([torch.full_like(tb, XMAX), tb], dim=1)

    history = []
    start = time.perf_counter()
    for step in range(1, STEPS + 1):
        optimizer.zero_grad()
        re, im, re_t, im_t, re_xx, im_xx = derivatives(model, xt_f)
        # i psi_t = -c psi_xx, c = hbar/(2m)
        # Real: -im_t + c*re_xx = 0; Imag: re_t + c*im_xx = 0
        r_re = -im_t + HBAR_OVER_2M * re_xx
        r_im = re_t + HBAR_OVER_2M * im_xx
        loss_f = torch.mean(r_re**2 + r_im**2)

        re0, im0 = model(xt_0)
        loss_0 = torch.mean((re0 - psi0.real)**2 + (im0 - psi0.imag)**2)

        rel, iml = model(xt_left)
        rer, imr = model(xt_right)
        loss_b = torch.mean((rel-rer)**2 + (iml-imr)**2)
        # Match first spatial derivatives at boundaries to encourage periodic smoothness
        xbl = xt_left.detach().clone().requires_grad_(True)
        xbr = xt_right.detach().clone().requires_grad_(True)
        rel2, iml2 = model(xbl)
        rer2, imr2 = model(xbr)
        drel = torch.autograd.grad(rel2, xbl, torch.ones_like(rel2), create_graph=True)[0][:, 0:1]
        drer = torch.autograd.grad(rer2, xbr, torch.ones_like(rer2), create_graph=True)[0][:, 0:1]
        diml = torch.autograd.grad(iml2, xbl, torch.ones_like(iml2), create_graph=True)[0][:, 0:1]
        dimr = torch.autograd.grad(imr2, xbr, torch.ones_like(imr2), create_graph=True)[0][:, 0:1]
        loss_b = loss_b + torch.mean((drel-drer)**2 + (diml-dimr)**2)

        loss = loss_f + loss_0 + loss_b
        loss.backward()
        optimizer.step()
        history.append((step, float(loss.detach().cpu()), float(loss_f.detach().cpu()),
                        float(loss_0.detach().cpu()), float(loss_b.detach().cpu())))
        if step == 1 or step % 1000 == 0:
            print(f"step={step:5d} loss={loss.item():.4e} "
                  f"(PDE={loss_f.item():.2e}, IC={loss_0.item():.2e}, periodic={loss_b.item():.2e})")

    train_seconds = time.perf_counter() - start

    nx, nt = 301, 81
    x = torch.linspace(XMIN, XMAX, nx, device=DEVICE).reshape(-1, 1)
    t = torch.linspace(0, TMAX, nt, device=DEVICE).reshape(-1, 1)
    X, T = torch.meshgrid(x.squeeze(1), t.squeeze(1), indexing="ij")
    xt_eval = torch.stack([X.reshape(-1), T.reshape(-1)], dim=1)
    with torch.no_grad():
        re_pred, im_pred = model(xt_eval)
        psi_pred = torch.complex(re_pred.reshape(nx, nt), im_pred.reshape(nx, nt))
        psi_ref = exact_free_packet(X.reshape(-1, 1), T.reshape(-1, 1)).reshape(nx, nt)
        abs_err = torch.abs(psi_pred - psi_ref)
        rel_l2 = torch.linalg.norm(psi_pred - psi_ref) / torch.linalg.norm(psi_ref)
        rmse = torch.sqrt(torch.mean(abs_err**2))
        prob_pred = torch.abs(psi_pred)**2
        prob_ref = torch.abs(psi_ref)**2

    with (RES_DIR / "schrodinger_loss_history.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "loss_total", "loss_pde", "loss_initial", "loss_periodic"])
        writer.writerows(history)
    with (RES_DIR / "schrodinger_metrics.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value", "device", "seed", "steps", "train_seconds",
                         "reference_note"])
        note = "Free Gaussian analytic solution on infinite domain; finite periodic PINN mismatch"
        writer.writerow(["relative_l2_error", rel_l2.item(), str(DEVICE), SEED, STEPS, train_seconds, note])
        writer.writerow(["complex_abs_rmse", rmse.item(), str(DEVICE), SEED, STEPS, train_seconds, note])
        writer.writerow(["probability_integral_prediction_t0", torch.trapezoid(prob_pred[:,0], x.squeeze()).item(),
                         str(DEVICE), SEED, STEPS, train_seconds, note])
        writer.writerow(["probability_integral_reference_t0", torch.trapezoid(prob_ref[:,0], x.squeeze()).item(),
                         str(DEVICE), SEED, STEPS, train_seconds, note])

    # Plots: probability density and absolute complex error
    x_np = X.detach().cpu().numpy()
    t_np = T.detach().cpu().numpy()
    pp = prob_pred.detach().cpu().numpy()
    pr = prob_ref.detach().cpu().numpy()
    ee = abs_err.detach().cpu().numpy()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    for ax, data, title in zip(
        axes, [pr, pp, ee],
        ["Densidade de probabilidade (referência)", "Densidade de probabilidade (PINN)",
         "Erro absoluto complexo"]
    ):
        im = ax.imshow(data, origin="lower", aspect="auto",
                       extent=[0, TMAX, XMIN, XMAX])
        ax.set_xlabel("Tempo t")
        ax.set_ylabel("Posição x")
        ax.set_title(title)
        fig.colorbar(im, ax=ax)
    fig.savefig(FIG_DIR / "schrodinger_comparison.png", dpi=180)
    plt.close(fig)

    hist = np.array(history)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogy(hist[:,0], hist[:,1], label="Total")
    ax.semilogy(hist[:,0], hist[:,2], label="EDP")
    ax.semilogy(hist[:,0], hist[:,3], label="Inicial")
    ax.semilogy(hist[:,0], hist[:,4], label="Periodicidade")
    ax.set_xlabel("Passo de otimização")
    ax.set_ylabel("Perda (escala log)")
    ax.set_title("Convergência do PINN — Schrödinger")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "schrodinger_training_loss.png", dpi=180)
    plt.close(fig)

    print("\nMétricas exploratórias — verificar a incompatibilidade de domínio descrita no cabeçalho.")
    print(f"Dispositivo: {DEVICE}")
    print(f"Tempo de treinamento: {train_seconds:.2f} s")
    print(f"Erro L2 relativo: {rel_l2.item():.6e}")
    print(f"RMSE complexo: {rmse.item():.6e}")
    print(f"Gráficos: {FIG_DIR}")
    print(f"Métricas: {RES_DIR}")

if __name__ == "__main__":
    main()
