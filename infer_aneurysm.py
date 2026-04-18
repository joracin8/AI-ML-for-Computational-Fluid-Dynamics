"""
Standalone Inference Script — Aneurysm PINN
=============================================
Loads the trained flow_network.0.pth checkpoint and runs inference
on points sampled from your geometry (STL or a regular grid).

Usage:
    python infer_aneurysm.py

Outputs:
    pinn_output.csv  — ready to feed into validate_pinn_vs_ccm.py
"""

import os
import numpy as np
import torch
import pandas as pd

# PhysicsNeMo / Modulus imports
from physicsnemo.sym.hydra import instantiate_arch, PhysicsNeMoConfig
from physicsnemo.sym.key import Key
from physicsnemo.sym.geometry.tessellation import Tessellation
from omegaconf import OmegaConf

# ─────────────────────────────────────────────
# CONFIG — edit these paths to match your setup
# ─────────────────────────────────────────────

CHECKPOINT_PATH = "./outputs/flow_network.0.pth"   # path to your .pth file
STL_INTERIOR    = "./stl_files/aneurysm_closed.stl" # closed interior STL
OUTPUT_CSV      = "pinn_output.csv"                 # output file for validator

# Normalization params (must match training)
CENTER = (0.01151354, 0.03322, 0.0)
SCALE  = 12.753

# Number of points to sample from geometry for inference
N_POINTS = 50_000

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ─────────────────────────────────────────────
# STEP 1: Load geometry and sample points
# ─────────────────────────────────────────────

print("=" * 55)
print("STEP 1: Sampling interior points from STL")
print("=" * 55)

interior_mesh = Tessellation.from_stl(STL_INTERIOR, airtight=True)

# Translate and scale (same as training)
interior_mesh = interior_mesh.translate([-c for c in CENTER])
interior_mesh = interior_mesh.scale(SCALE)

# Sample interior points
samples = interior_mesh.sample_interior(N_POINTS)
x = samples["x"].astype(np.float32)
y = samples["y"].astype(np.float32)
z = samples["z"].astype(np.float32)

print(f"  Sampled {len(x)} interior points")
print(f"  x: [{x.min():.4f}, {x.max():.4f}]")
print(f"  y: [{y.min():.4f}, {y.max():.4f}]")
print(f"  z: [{z.min():.4f}, {z.max():.4f}]")

# ─────────────────────────────────────────────
# STEP 2: Build the same network arch as training
# ─────────────────────────────────────────────

print("\n" + "=" * 55)
print("STEP 2: Instantiating network architecture")
print("=" * 55)

# Minimal config that mirrors your conf/config.yaml arch block
# Adjust layer_size and nr_layers if your config differs
arch_cfg = OmegaConf.create({
    "arch": {
        "fully_connected": {
            "_target_": "physicsnemo.sym.models.fully_connected.FullyConnectedArch",
            "layer_size": 512,
            "nr_layers": 6,
            "skip_connections": False,
            "activation_fn": "silu",
            "adaptive_activations": False,
            "weight_norm": True,
        }
    }
})

flow_net = instantiate_arch(
    input_keys=[Key("x"), Key("y"), Key("z")],
    output_keys=[Key("u"), Key("v"), Key("w"), Key("p")],
    cfg=arch_cfg.arch.fully_connected,
)

# ─────────────────────────────────────────────
# STEP 3: Load the checkpoint
# ─────────────────────────────────────────────

print("\n" + "=" * 55)
print("STEP 3: Loading checkpoint")
print("=" * 55)

state_dict = torch.load(CHECKPOINT_PATH, map_location=DEVICE)

# PhysicsNeMo wraps weights under a "model" key in some versions
if "model" in state_dict:
    state_dict = state_dict["model"]

flow_net.load_state_dict(state_dict)
flow_net.to(DEVICE)
flow_net.eval()

print(f"  Loaded: {CHECKPOINT_PATH}")
print(f"  Device: {DEVICE}")

# ─────────────────────────────────────────────
# STEP 4: Run inference in batches
# ─────────────────────────────────────────────

print("\n" + "=" * 55)
print("STEP 4: Running inference")
print("=" * 55)

BATCH_SIZE = 8192

x_t = torch.tensor(x, dtype=torch.float32).unsqueeze(-1)
y_t = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)
z_t = torch.tensor(z, dtype=torch.float32).unsqueeze(-1)

u_list, v_list, w_list, p_list = [], [], [], []

n_batches = int(np.ceil(len(x) / BATCH_SIZE))

with torch.no_grad():
    for i in range(n_batches):
        s = i * BATCH_SIZE
        e = min(s + BATCH_SIZE, len(x))

        inputs = {
            "x": x_t[s:e].to(DEVICE),
            "y": y_t[s:e].to(DEVICE),
            "z": z_t[s:e].to(DEVICE),
        }

        preds = flow_net(inputs)

        u_list.append(preds["u"].cpu().numpy().squeeze())
        v_list.append(preds["v"].cpu().numpy().squeeze())
        w_list.append(preds["w"].cpu().numpy().squeeze())
        p_list.append(preds["p"].cpu().numpy().squeeze())

        if (i + 1) % 10 == 0 or (i + 1) == n_batches:
            print(f"  Batch {i+1}/{n_batches} done")

u_pred = np.concatenate(u_list)
v_pred = np.concatenate(v_list)
w_pred = np.concatenate(w_list)
p_pred = np.concatenate(p_list)

vel_mag = np.sqrt(u_pred**2 + v_pred**2 + w_pred**2)

print(f"\n  Velocity magnitude : mean={vel_mag.mean():.4f}, max={vel_mag.max():.4f}")
print(f"  Pressure           : mean={p_pred.mean():.4f}, range=[{p_pred.min():.4f}, {p_pred.max():.4f}]")

# ─────────────────────────────────────────────
# STEP 5: Save to CSV
# ─────────────────────────────────────────────

print("\n" + "=" * 55)
print("STEP 5: Saving output CSV")
print("=" * 55)

df = pd.DataFrame({
    "x": x,
    "y": y,
    "z": z,
    "u": u_pred,
    "v": v_pred,
    "w": w_pred,
    "p": p_pred,
    "vel_mag": vel_mag,
})

df.to_csv(OUTPUT_CSV, index=False)
print(f"  Saved {len(df)} rows → {OUTPUT_CSV}")
print(f"\n  ✅ Done. Feed '{OUTPUT_CSV}' into validate_pinn_vs_ccm.py")
print("=" * 55)
