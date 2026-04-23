import torch
import torch.nn as nn
import numpy as np
import pandas as pd

# ── 1. Define the same network architecture as training ──────────────────────
class FullyConnected(nn.Module):
    def __init__(self, in_features=3, out_features=4, layers=6, hidden=512):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(in_features, hidden))
        for _ in range(layers - 1):
            self.layers.append(nn.Linear(hidden, hidden))
        self.final_layer = nn.Linear(hidden, out_features)

    def forward(self, x):
        for layer in self.layers:
            x = torch.sin(layer(x))
        return self.final_layer(x)

# ── 2. Load trained weights ───────────────────────────────────────────────────
checkpoint_path = r"outputs\aneurysm\flow_network.0.pth"
state_dict = torch.load(checkpoint_path, map_location="cpu")

# Strip '_impl.' prefix if present
new_state_dict = {}
for k, v in state_dict.items():
    new_key = k.replace("_impl.", "")
    new_state_dict[new_key] = v

model = FullyConnected(in_features=3, out_features=4, layers=6, hidden=512)
model.load_state_dict(new_state_dict)
model.eval()
print("✅ Model loaded successfully")

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

# Output order: u, v, w, p  (index 3 = pressure)
pred_p = preds[:, 3]
true_p = df["Pressure (Pa)"].values.astype(np.float32)

# ── 6. Compute L2 error ───────────────────────────────────────────────────────
l2_abs = np.sqrt(np.mean((pred_p - true_p) ** 2))
l2_rel = l2_abs / np.sqrt(np.mean(true_p ** 2))

print("\n========================================")
print(f"  L2 Absolute Error : {l2_abs:.4f} Pa")
print(f"  L2 Relative Error : {l2_rel * 100:.2f} %")
print("========================================")

# ── 7. Save comparison CSV for ParaView ──────────────────────────────────────
out = pd.DataFrame({
    "X (m)": df["X (m)"],
    "Y (m)": df["Y (m)"],
    "Z (m)": df["Z (m)"],
    "true_p (Pa)": true_p,
    "pred_p (Pa)": pred_p,
    "error_p (Pa)": np.abs(pred_p - true_p),
})
out.to_csv("pressure_comparison.csv", index=False)
print("\n✅ Saved: pressure_comparison.csv")
