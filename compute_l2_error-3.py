import torch
import torch.nn as nn
import numpy as np
import pandas as pd


# ── 1. Define network WITH weight normalization (matching PhysicsNeMo) ────────
class LinearLayer(nn.Module):
    def __init__(self, in_f, out_f):
        super().__init__()
        self.linear = nn.utils.weight_norm(nn.Linear(in_f, out_f))

    def forward(self, x):
        return torch.sin(self.linear(x))


class FinalLayer(nn.Module):
    def __init__(self, in_f, out_f):
        super().__init__()
        self.linear = nn.utils.weight_norm(nn.Linear(in_f, out_f))

    def forward(self, x):
        return self.linear(x)


class FullyConnected(nn.Module):
    def __init__(self, in_features=3, out_features=4, num_layers=6, hidden=512):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(LinearLayer(in_features, hidden))
        for _ in range(num_layers - 1):
            self.layers.append(LinearLayer(hidden, hidden))
        self.final_layer = FinalLayer(hidden, out_features)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.final_layer(x)


# ── 2. Load trained weights ───────────────────────────────────────────────────
checkpoint_path = r"outputs\aneurysm\flow_network.0.pth"
state_dict = torch.load(checkpoint_path, map_location="cpu")

# Strip '_impl.' prefix
new_state_dict = {}
for k, v in state_dict.items():
    new_key = k.replace("_impl.", "")
    new_state_dict[new_key] = v

model = FullyConnected(in_features=3, out_features=4, num_layers=6, hidden=512)
model.load_state_dict(new_state_dict, strict=True)
model.eval()
print("✅ Model loaded successfully with weight normalization")

# ── 3. Load StarCCM+ CSV ──────────────────────────────────────────────────────
csv_path = r"starccm\starccm_pressure.csv"
df = pd.read_csv(csv_path)
print(f"✅ CSV loaded: {len(df)} points")
print(f"   Columns: {list(df.columns)}")

# ── 4. Normalize coordinates (same as training) ───────────────────────────────
center = (0.01151354, 0.03322, 0.0)
scale  = 12.753

x = (df["X (m)"].values - center[0]) * scale
y = (df["Y (m)"].values - center[1]) * scale
z = (df["Z (m)"].values - center[2]) * scale

coords = torch.tensor(np.stack([x, y, z], axis=1), dtype=torch.float32)

# ── 5. Run inference ──────────────────────────────────────────────────────────
with torch.no_grad():
    preds = model(coords).numpy()

# Print all 4 output stats to verify which column is pressure
print("\n── Output column statistics ──")
col_names = ["u", "v", "w", "p"]
for i, name in enumerate(col_names):
    print(f"  [{i}] {name}: min={preds[:,i].min():.4f}, "
          f"max={preds[:,i].max():.4f}, "
          f"mean={preds[:,i].mean():.4f}")

true_p = df["Pressure (Pa)"].values.astype(np.float32)
print(f"\n── StarCCM+ pressure statistics ──")
print(f"  min={true_p.min():.4f}, max={true_p.max():.4f}, mean={true_p.mean():.4f}")

# ── 6. Compute L2 error for pressure (index 3) ───────────────────────────────
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
