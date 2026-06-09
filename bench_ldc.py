import time, threading, subprocess, os, sys
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
import numpy as np
import matplotlib
matplotlib.use("Agg")          # works without a display too
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap

# ── NVIDIA green palette ──────────────────────────────────
NVIDIA_GREEN  = "#76B900"
DARK_BG       = "#0F0F0F"
PANEL_BG      = "#1A1A1A"
GRID_COLOR    = "#2A2A2A"
TEXT_COLOR    = "#E8E8E8"
ACCENT2       = "#00D4FF"   # cyan accent
ACCENT3       = "#FF6B35"   # orange for second GPU line

plt.rcParams.update({
    "figure.facecolor":  DARK_BG,
    "axes.facecolor":    PANEL_BG,
    "axes.edgecolor":    "#333333",
    "axes.labelcolor":   TEXT_COLOR,
    "axes.titlecolor":   TEXT_COLOR,
    "xtick.color":       TEXT_COLOR,
    "ytick.color":       TEXT_COLOR,
    "grid.color":        GRID_COLOR,
    "grid.linewidth":    0.6,
    "text.color":        TEXT_COLOR,
    "font.family":       "DejaVu Sans",
    "legend.facecolor":  PANEL_BG,
    "legend.edgecolor":  "#444444",
})

# ══════════════════════════════════════════════════════════
#  GPU INFO
# ══════════════════════════════════════════════════════════
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("\n" + "═"*60)
print("   PHYSICSNEMO SYM — LDC PINN GPU BENCHMARK")
print("═"*60)
gpu_name = "CPU"
if torch.cuda.is_available():
    props    = torch.cuda.get_device_properties(0)
    gpu_name = props.name
    vram_gb  = props.total_memory / 1e9
    print(f"  GPU          : {gpu_name}")
    print(f"  VRAM Total   : {vram_gb:.2f} GB")
    print(f"  CUDA Version : {torch.version.cuda}")
    print(f"  Torch        : {torch.__version__}")
else:
    print("  ⚠  No CUDA GPU — running on CPU")
print("═"*60 + "\n")

# ══════════════════════════════════════════════════════════
#  BACKGROUND GPU MONITOR
# ══════════════════════════════════════════════════════════
gpu_log = {"util": [], "mem": [], "temp": [], "power": [],
           "cur_util": 0, "cur_mem": 0, "cur_temp": 0, "cur_power": 0}
_mon_active = True

def _monitor():
    while _mon_active:
        try:
            raw = subprocess.check_output([
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,temperature.gpu,power.draw",
                "--format=csv,noheader,nounits"
            ]).decode().strip().split(",")
            u, m, t, p = [float(x.strip()) for x in raw]
            gpu_log["cur_util"]  = u
            gpu_log["cur_mem"]   = m
            gpu_log["cur_temp"]  = t
            gpu_log["cur_power"] = p
        except:
            pass
        time.sleep(1)

threading.Thread(target=_monitor, daemon=True).start()

# ══════════════════════════════════════════════════════════
#  SAMPLERS  (Lid-Driven Cavity 0.1 × 0.1 m)
# ══════════════════════════════════════════════════════════
def sample_interior(n=4096):
    xy = torch.rand(n, 2, device=device) * 0.1 - 0.05
    return xy.requires_grad_(True)

def sample_boundary(n=512):
    s  = lambda: torch.rand(n, 1, device=device) * 0.1 - 0.05
    lo = torch.full((n, 1), -0.05, device=device)
    hi = torch.full((n, 1),  0.05, device=device)
    bottom = torch.cat([s(), lo], 1)
    top    = torch.cat([s(), hi], 1)
    left   = torch.cat([lo, s()], 1)
    right  = torch.cat([hi, s()], 1)
    return bottom, top, left, right

# ══════════════════════════════════════════════════════════
#  PINN — 8 × 512 Tanh  →  (u, v, p)
# ══════════════════════════════════════════════════════════
class PINN(nn.Module):
    def __init__(self, depth=8, width=512):
        super().__init__()
        layers = [nn.Linear(2, width), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.Tanh()]
        layers += [nn.Linear(width, 3)]
        self.net = nn.Sequential(*layers)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, xy):
        return self.net(xy)

model     = PINN().to(device)
optimizer = Adam(model.parameters(), lr=1e-3)
scheduler = ExponentialLR(optimizer, gamma=0.9995)
n_params  = sum(p.numel() for p in model.parameters())
print(f"  Network : 8 × 512  |  params : {n_params:,}\n")

# ══════════════════════════════════════════════════════════
#  PHYSICS LOSS  (incompressible N-S, ν = 0.001)
# ══════════════════════════════════════════════════════════
NU = 0.001

def _grad(f, xy):
    return torch.autograd.grad(f, xy,
        grad_outputs=torch.ones_like(f),
        create_graph=True, retain_graph=True)[0]

def physics_loss(xy):
    out = model(xy)
    u, v, p = out[:,0:1], out[:,1:2], out[:,2:3]
    ug = _grad(u, xy);  vg = _grad(v, xy);  pg = _grad(p, xy)
    ux, uy = ug[:,0:1], ug[:,1:2]
    vx, vy = vg[:,0:1], vg[:,1:2]
    px, py = pg[:,0:1], pg[:,1:2]
    uxx = _grad(ux, xy)[:,0:1];  uyy = _grad(uy, xy)[:,1:2]
    vxx = _grad(vx, xy)[:,0:1];  vyy = _grad(vy, xy)[:,1:2]
    cont  = (ux + vy).pow(2).mean()
    mom_x = (u*ux + v*uy + px - NU*(uxx+uyy)).pow(2).mean()
    mom_y = (u*vx + v*vy + py - NU*(vxx+vyy)).pow(2).mean()
    return cont + mom_x + mom_y

def boundary_loss():
    bottom, top, left, right = sample_boundary()
    ns  = torch.cat([bottom, left, right], 0)
    o   = model(ns)
    lns = o[:,0].pow(2).mean() + o[:,1].pow(2).mean()
    ol  = model(top)
    ll  = (ol[:,0]-1).pow(2).mean() + ol[:,1].pow(2).mean()
    return lns + ll

# ══════════════════════════════════════════════════════════
#  TRAINING LOOP
# ══════════════════════════════════════════════════════════
STEPS   = 2000
WARMUP  = 50
LOG     = 100

# history buffers
hist = {
    "step": [], "pde": [], "bc": [], "total": [],
    "its":  [], "util": [], "mem": [], "temp": [], "power": []
}

print(f"{'─'*72}")
print(f"{'Step':>6} │ {'PDE':>10} │ {'BC':>10} │ {'Total':>10} │"
      f" {'it/s':>6} │ {'GPU%':>5} │ {'MB':>6} │ {'°C':>4}")
print(f"{'─'*72}")

step_times = []
best_loss  = float("inf")
os.makedirs("outputs", exist_ok=True)
t_start = time.time()

for step in range(1, STEPS + 1):
    t0 = time.perf_counter()

    xy       = sample_interior(4096)
    optimizer.zero_grad()
    l_pde    = physics_loss(xy)
    l_bc     = boundary_loss()
    loss     = l_pde + 10.0 * l_bc
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    scheduler.step()

    dt = time.perf_counter() - t0
    step_times.append(dt)

    # record GPU stats every step
    gpu_log["util"].append(gpu_log["cur_util"])
    gpu_log["mem"].append(gpu_log["cur_mem"])
    gpu_log["temp"].append(gpu_log["cur_temp"])
    gpu_log["power"].append(gpu_log["cur_power"])

    if step <= WARMUP:
        continue

    if step % LOG == 0 or step == STEPS:
        recent = step_times[-LOG:]
        its    = 1.0 / (sum(recent) / len(recent))
        lv     = loss.item()
        if lv < best_loss:
            best_loss = lv
            torch.save(model.state_dict(), "outputs/best_model.pt")

        hist["step"].append(step)
        hist["pde"].append(l_pde.item())
        hist["bc"].append(l_bc.item())
        hist["total"].append(lv)
        hist["its"].append(its)
        hist["util"].append(gpu_log["cur_util"])
        hist["mem"].append(gpu_log["cur_mem"])
        hist["temp"].append(gpu_log["cur_temp"])
        hist["power"].append(gpu_log["cur_power"])

        print(f"{step:>6} │ {l_pde.item():>10.3e} │ {l_bc.item():>10.3e} │"
              f" {lv:>10.3e} │ {its:>6.1f} │"
              f" {gpu_log['cur_util']:>4.0f}% │"
              f" {gpu_log['cur_mem']:>5.0f}M │"
              f" {gpu_log['cur_temp']:>3.0f}°")

_mon_active = False
t_total = time.time() - t_start
bench   = step_times[WARMUP:]
avg_its = 1.0 / (sum(bench) / len(bench))

print(f"\n{'═'*60}")
print("  BENCHMARK RESULTS")
print(f"{'═'*60}")
print(f"  GPU          : {gpu_name}")
print(f"  Total time   : {t_total:.1f} s")
print(f"  Avg it/s     : {avg_its:.1f}   ← share this number")
print(f"  Best loss    : {best_loss:.4e}")
print(f"  Peak GPU mem : {max(gpu_log['mem'], default=0):.0f} MB")
print(f"  Avg GPU util : {sum(gpu_log['util'])/max(len(gpu_log['util']),1):.0f}%")
print(f"  Avg temp     : {sum(gpu_log['temp'])/max(len(gpu_log['temp']),1):.0f} °C")
print(f"  Avg power    : {sum(gpu_log['power'])/max(len(gpu_log['power']),1):.0f} W")
print(f"{'═'*60}\n")

# ══════════════════════════════════════════════════════════
#  VELOCITY FIELD PREDICTION (for plotting)
# ══════════════════════════════════════════════════════════
model.eval()
with torch.no_grad():
    N   = 80
    lin = torch.linspace(-0.05, 0.05, N, device=device)
    gx, gy = torch.meshgrid(lin, lin, indexing="ij")
    xy_grid = torch.stack([gx.flatten(), gy.flatten()], dim=1)
    pred    = model(xy_grid).cpu().numpy()
    U  = pred[:,0].reshape(N, N)
    V  = pred[:,1].reshape(N, N)
    P  = pred[:,2].reshape(N, N)
    Sp = np.sqrt(U**2 + V**2)

x_np = lin.cpu().numpy()
X, Y = np.meshgrid(x_np, x_np, indexing="ij")

# ══════════════════════════════════════════════════════════
#  PLOT 1 — TRAINING DASHBOARD  (6 panels)
# ══════════════════════════════════════════════════════════
fig1 = plt.figure(figsize=(18, 11), facecolor=DARK_BG)
fig1.suptitle(f"PhysicsNeMo Sym — LDC PINN Benchmark\n{gpu_name}",
              fontsize=15, fontweight="bold", color=NVIDIA_GREEN, y=0.97)

gs = gridspec.GridSpec(2, 3, figure=fig1,
                       hspace=0.42, wspace=0.38,
                       left=0.07, right=0.96, top=0.90, bottom=0.08)

steps = hist["step"]

def styled_ax(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=11, fontweight="bold",
                 color=NVIDIA_GREEN, pad=8)
    ax.set_xlabel(xlabel, fontsize=9, color=TEXT_COLOR)
    ax.set_ylabel(ylabel, fontsize=9, color=TEXT_COLOR)
    ax.grid(True, alpha=0.4)
    ax.tick_params(labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333333")

# — Loss curves —
ax0 = fig1.add_subplot(gs[0, 0])
ax0.semilogy(steps, hist["pde"],   color=NVIDIA_GREEN,  lw=2,   label="PDE (physics)")
ax0.semilogy(steps, hist["bc"],    color=ACCENT2,        lw=2,   label="Boundary cond.")
ax0.semilogy(steps, hist["total"], color="#FFFFFF",       lw=2.5, label="Total", ls="--")
ax0.legend(fontsize=8, framealpha=0.3)
styled_ax(ax0, "Training Loss", "Step", "Loss (log)")

# — it/s throughput —
ax1 = fig1.add_subplot(gs[0, 1])
ax1.plot(steps, hist["its"], color=NVIDIA_GREEN, lw=2.5, marker="o",
         markersize=5, markerfacecolor=DARK_BG, markeredgecolor=NVIDIA_GREEN)
ax1.axhline(avg_its, color=ACCENT2, ls="--", lw=1.5, alpha=0.8,
            label=f"Avg: {avg_its:.1f} it/s")
ax1.fill_between(steps, hist["its"], alpha=0.15, color=NVIDIA_GREEN)
ax1.legend(fontsize=9, framealpha=0.3)
styled_ax(ax1, "Training Speed", "Step", "Iterations / second")

# — GPU Utilisation —
ax2 = fig1.add_subplot(gs[0, 2])
all_util = gpu_log["util"][WARMUP:]
xs_util  = list(range(WARMUP, STEPS))
if all_util:
    ax2.fill_between(xs_util, all_util, alpha=0.25, color=NVIDIA_GREEN)
    ax2.plot(xs_util, all_util, color=NVIDIA_GREEN, lw=1.2)
ax2.set_ylim(0, 105)
styled_ax(ax2, "GPU Utilisation (%)", "Step", "Utilisation %")

# — VRAM usage —
ax3 = fig1.add_subplot(gs[1, 0])
all_mem = gpu_log["mem"][WARMUP:]
if all_mem:
    ax3.fill_between(xs_util, all_mem, alpha=0.25, color=ACCENT2)
    ax3.plot(xs_util, all_mem, color=ACCENT2, lw=1.5)
styled_ax(ax3, "VRAM Usage (MB)", "Step", "Memory (MB)")

# — Temperature —
ax4 = fig1.add_subplot(gs[1, 1])
all_temp = gpu_log["temp"][WARMUP:]
if all_temp:
    ax4.plot(xs_util, all_temp, color=ACCENT3, lw=1.8)
    ax4.fill_between(xs_util, all_temp, alpha=0.2, color=ACCENT3)
styled_ax(ax4, "GPU Temperature (°C)", "Step", "Temp °C")

# — Power draw —
ax5 = fig1.add_subplot(gs[1, 2])
all_pwr = gpu_log["power"][WARMUP:]
if all_pwr:
    ax5.plot(xs_util, all_pwr, color="#FFD700", lw=1.8)
    ax5.fill_between(xs_util, all_pwr, alpha=0.2, color="#FFD700")
styled_ax(ax5, "Power Draw (W)", "Step", "Power (W)")

fig1.savefig("outputs/training_dashboard.png", dpi=150, bbox_inches="tight",
             facecolor=DARK_BG)
print("  Saved → outputs/training_dashboard.png")

# ══════════════════════════════════════════════════════════
#  PLOT 2 — PHYSICS FIELD  (4 panels)
# ══════════════════════════════════════════════════════════
fig2 = plt.figure(figsize=(18, 8), facecolor=DARK_BG)
fig2.suptitle(
    f"LDC Flow Field — PINN Prediction\n{gpu_name}  |  Re = 100  |  0.1 × 0.1 m cavity",
    fontsize=14, fontweight="bold", color=NVIDIA_GREEN, y=0.97)

gs2 = gridspec.GridSpec(1, 4, figure=fig2,
                        wspace=0.35, left=0.05, right=0.97,
                        top=0.88, bottom=0.10)

cmap_vel   = LinearSegmentedColormap.from_list(
    "nvidia_vel", [DARK_BG, "#003300", NVIDIA_GREEN, "#CCFF88", "white"])
cmap_pres  = LinearSegmentedColormap.from_list(
    "nvidia_pres", [DARK_BG, "#001133", ACCENT2, "white"])
cmap_vort  = LinearSegmentedColormap.from_list(
    "nvidia_vort", ["#FF2200", DARK_BG, NVIDIA_GREEN])

def field_ax(fig, gs_pos, data, cmap, title, cbar_label):
    ax = fig.add_subplot(gs_pos)
    im = ax.pcolormesh(X*100, Y*100, data, cmap=cmap,
                       shading="auto", rasterized=True)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=8, colors=TEXT_COLOR)
    cb.set_label(cbar_label, fontsize=8, color=TEXT_COLOR)
    cb.outline.set_edgecolor("#444444")
    ax.set_title(title, fontsize=11, fontweight="bold",
                 color=NVIDIA_GREEN, pad=8)
    ax.set_xlabel("x (cm)", fontsize=9, color=TEXT_COLOR)
    ax.set_ylabel("y (cm)", fontsize=9, color=TEXT_COLOR)
    ax.tick_params(labelsize=8)
    for sp in ax.spines.values():
        sp.set_edgecolor("#333333")
    return ax

# u velocity
ax_u = field_ax(fig2, gs2[0], U, cmap_vel, "x-Velocity  u (m/s)", "m/s")

# v velocity
ax_v = field_ax(fig2, gs2[1], V, cmap_vel, "y-Velocity  v (m/s)", "m/s")

# pressure
ax_p = field_ax(fig2, gs2[2], P, cmap_pres, "Pressure  p (Pa)", "Pa")

# speed + streamlines
ax_s = fig2.add_subplot(gs2[3])
im_s = ax_s.pcolormesh(X*100, Y*100, Sp, cmap=cmap_vel,
                        shading="auto", rasterized=True)
# streamlines (coarser grid for clarity)
stride = 4
ax_s.streamplot(x_np[::stride]*100, x_np[::stride]*100,
                U[::stride, ::stride].T, V[::stride, ::stride].T,
                color="white", linewidth=0.7, density=1.2, arrowsize=0.9,
                arrowstyle="->")
cb_s = fig2.colorbar(im_s, ax=ax_s, fraction=0.046, pad=0.04)
cb_s.ax.tick_params(labelsize=8, colors=TEXT_COLOR)
cb_s.set_label("Speed (m/s)", fontsize=8, color=TEXT_COLOR)
cb_s.outline.set_edgecolor("#444444")
ax_s.set_title("Speed + Streamlines", fontsize=11, fontweight="bold",
               color=NVIDIA_GREEN, pad=8)
ax_s.set_xlabel("x (cm)", fontsize=9, color=TEXT_COLOR)
ax_s.set_ylabel("y (cm)", fontsize=9, color=TEXT_COLOR)
ax_s.tick_params(labelsize=8)
for sp in ax_s.spines.values():
    sp.set_edgecolor("#333333")

fig2.savefig("outputs/flow_field.png", dpi=150, bbox_inches="tight",
             facecolor=DARK_BG)
print("  Saved → outputs/flow_field.png")

# ══════════════════════════════════════════════════════════
#  PLOT 3 — BENCHMARK SUMMARY CARD
# ══════════════════════════════════════════════════════════
fig3, ax = plt.subplots(figsize=(10, 5), facecolor=DARK_BG)
ax.set_facecolor(DARK_BG)
ax.axis("off")

# GPU name banner
ax.text(0.5, 0.93, "PINN BENCHMARK RESULT",
        ha="center", va="top", fontsize=13, fontweight="bold",
        color=NVIDIA_GREEN, transform=ax.transAxes)
ax.text(0.5, 0.82, gpu_name,
        ha="center", va="top", fontsize=17, fontweight="bold",
        color="white", transform=ax.transAxes)
ax.axhline(0.75, color=NVIDIA_GREEN, lw=1.5, xmin=0.05, xmax=0.95,
           transform=ax.transAxes)

metrics = [
    ("Avg Speed",    f"{avg_its:.1f} it/s",     NVIDIA_GREEN),
    ("Total Time",   f"{t_total:.1f} s",          ACCENT2),
    ("Best Loss",    f"{best_loss:.2e}",           "white"),
    ("Peak VRAM",    f"{max(gpu_log['mem'], default=0):.0f} MB",  "#FFD700"),
    ("Avg GPU Util", f"{sum(gpu_log['util'])/max(len(gpu_log['util']),1):.0f}%", ACCENT3),
    ("Avg Power",    f"{sum(gpu_log['power'])/max(len(gpu_log['power']),1):.0f} W", "#FF88FF"),
]

for i, (label, value, color) in enumerate(metrics):
    col = i % 3
    row = i // 3
    x   = 0.12 + col * 0.32
    y   = 0.55 - row * 0.30
    ax.text(x, y, label, ha="center", va="center",
            fontsize=9, color="#888888", transform=ax.transAxes)
    ax.text(x, y - 0.10, value, ha="center", va="center",
            fontsize=18, fontweight="bold", color=color,
            transform=ax.transAxes)

ax.text(0.5, 0.03,
        "Run bench_ldc.py on both machines and compare  Avg Speed (it/s)",
        ha="center", va="bottom", fontsize=9, color="#666666",
        transform=ax.transAxes, style="italic")

fig3.savefig("outputs/benchmark_card.png", dpi=150, bbox_inches="tight",
             facecolor=DARK_BG)
print("  Saved → outputs/benchmark_card.png")
print("\n  ✓ All plots saved to ./outputs/\n")
plt.close("all")
