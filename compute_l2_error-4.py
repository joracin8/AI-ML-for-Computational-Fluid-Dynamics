import torch
import numpy as np
import pandas as pd


# ── 1. Load checkpoint ────────────────────────────────────────────────────────
checkpoint_path = r"outputs\aneurysm\flow_network.0.pth"
sd = torch.load(checkpoint_path, map_location="cpu")

# Strip '_impl.' prefix
state_dict = {k.replace("_impl.", ""): v for k, v in sd.items()}


# ── 2. Manual forward pass matching PhysicsNeMo architecture ─────────────────
# Architecture: 6 hidden layers (512) + final layer (4 outputs)
# Activation: sin
# Weight norm: W_effective = weight_g * (weight / ||weight||)

def apply_weight_norm(weight, weight_g):
    """PhysicsNeMo custom weight normalization."""
    norm = weight.norm(dim=1, keepdim=True)  # (out, 1)
    return weight_g * weight / norm          # (out, in)


def forward(x):
    # Hidden layers 0-5
    for i in range(6):
        W = apply_weight_norm(
            state_dict[f"layers.{i}.linear.weight"],
            state_dict[f"layers.{i}.linear.weight_g"]
        )
        b = state_dict[f"layers.{i}.linear.bias"]
        x = torch.sin(x @ W.T + b)

    # Final layer (no weight_g)
    W = state_dict["final_layer.linear.weight"]
    b = state_dict["final_layer.linear.bias"]
    x = x @ W.T + b
    return x


# ── 3. Load StarCCM+ CSV ──────────────────────────────────────────────────────
csv_path = r"starccm\starccm_pressure.csv"
df = pd.read_csv(csv_path)
print(f"✅ CSV loaded: {len(df)} points")

# ── 4. Normalize coordinates ──────────────────────────────────────────────────
center = (0.01151354, 0.03322, 0.0)
scale  = 12.753

x = (df["X (m)"].values - center[0]) * scale
y = (df["Y (m)"].values - center[1]) * scale
z = (df["Z (m)"].values - center[2]) * scale

coords = torch.tensor(np.stack([x, y, z], axis=1), dtype=torch.float32)

# ── 5. Run inference ──────────────────────────────────────────────────────────
with torch.no_grad():
    preds = forward(coords).numpy()

# Print all output stats to verify
print("\n── Output column statistics ──")
col_names = ["u", "v", "w", "p"]
for i, name in enumerate(col_names):
    print(f"  [{i}] {name}: min={preds[:,i].min():.4f}, "
          f"max={preds[:,i].max():.4f}, "
          f"mean={preds[:,i].mean():.4f}")

true_p = df["Pressure (Pa)"].values.astype(np.float32)
print(f"\n── StarCCM+ pressure statistics ──")
print(f"  min={true_p.min():.4f}, max={true_p.max():.4f}, "
      f"mean={true_p.mean():.4f}")

# ── 6. Compute L2 error ───────────────────────────────────────────────────────
pred_p = preds[:, 3]

l2_abs = np.sqrt(np.mean((pred_p - true_p) ** 2))
l2_rel = l2_abs / np.sqrt(np.mean(true_p ** 2))

print("\n========================================")
print(f"  L2 Absolute Error : {l2_abs:.4f} Pa")
print(f"  L2 Relative Error : {l2_rel * 100:.2f} %")
print("========================================")

# ── 7. Save comparison CSV ────────────────────────────────────────────────────
out = pd.DataFrame({
    "X (m)": df["X (m)"],
    "Y (m)": df["Y (m)"],
    "Z (m)": df["Z (m)"],
    "true_p (Pa)": true_p,
    "pred_p (Pa)": pred_p,
    "error_p (Pa)": np.abs(pred_p - true_p),
})
out.to_csv("pressure_comparison.csv", index=False)
print("✅ Saved: pressure_comparison.csv")
