# L2 Error computation using PhysicsNeMo instantiate_arch
# Based on working inference script

import os
import numpy as np
import torch
import pandas as pd

import physicsnemo.sym
from physicsnemo.sym.hydra import to_absolute_path, instantiate_arch, PhysicsNeMoConfig
from physicsnemo.sym.key import Key


def normalize_invar(invar, center, scale, dims=2):
    invar["x"] -= center[0]
    invar["y"] -= center[1]
    invar["z"] -= center[2]
    invar["x"] *= scale
    invar["y"] *= scale
    invar["z"] *= scale
    if "area" in invar.keys():
        invar["area"] *= scale**dims
    return invar


def load_model(checkpoint_dir, cfg):
    flow_net = instantiate_arch(
        input_keys=[Key("x"), Key("y"), Key("z")],
        output_keys=[Key("u"), Key("v"), Key("w"), Key("p")],
        cfg=cfg.arch.fully_connected,
    )
    ckpt_files = sorted(
        [f for f in os.listdir(checkpoint_dir) if f.startswith("flow_network")],
        key=lambda f: int(f.split(".")[1]) if f.split(".")[1].isdigit() else 0,
    )
    if not ckpt_files:
        raise FileNotFoundError(f"No checkpoint found in {checkpoint_dir}")

    ckpt_path = os.path.join(checkpoint_dir, ckpt_files[-1])
    print(f"✅ Loading checkpoint: {ckpt_path}")
    state_dict = torch.load(ckpt_path, map_location="cpu")
    flow_net.load_state_dict(state_dict)
    flow_net.eval()
    return flow_net


def run_inference(flow_net, invar, batch_size=4096,
                  device="cuda" if torch.cuda.is_available() else "cpu"):
    x = torch.tensor(invar["x"], dtype=torch.float32)
    y = torch.tensor(invar["y"], dtype=torch.float32)
    z = torch.tensor(invar["z"], dtype=torch.float32)
    flow_net = flow_net.to(device)

    u_list, v_list, w_list, p_list = [], [], [], []
    with torch.no_grad():
        for i in range(0, len(x), batch_size):
            xb = x[i:i+batch_size].to(device)
            yb = y[i:i+batch_size].to(device)
            zb = z[i:i+batch_size].to(device)
            out = flow_net({"x": xb, "y": yb, "z": zb})
            u_list.append(out["u"].cpu().numpy())
            v_list.append(out["v"].cpu().numpy())
            w_list.append(out["w"].cpu().numpy())
            p_list.append(out["p"].cpu().numpy())

    return {
        "u": np.concatenate(u_list).squeeze(),
        "v": np.concatenate(v_list).squeeze(),
        "w": np.concatenate(w_list).squeeze(),
        "p": np.concatenate(p_list).squeeze(),
    }


@physicsnemo.sym.main(config_path="conf", config_name="config")
def run(cfg: PhysicsNeMoConfig) -> None:

    # ── Parameters (same as training) ────────────────────────────────────────
    center = (0.01151354, 0.03322, 0.0)
    scale  = 12.753
    checkpoint_dir = to_absolute_path("./outputs/aneurysm")

    # ── Load model ────────────────────────────────────────────────────────────
    flow_net = load_model(checkpoint_dir, cfg)

    # ── Load StarCCM+ CSV ─────────────────────────────────────────────────────
    csv_path = to_absolute_path("./starccm/starccm_pressure.csv")
    df = pd.read_csv(csv_path)
    print(f"✅ CSV loaded: {len(df)} points")

    # ── Normalize coordinates ─────────────────────────────────────────────────
    invar = {
        "x": df["X (m)"].values.reshape(-1, 1).astype(np.float32),
        "y": df["Y (m)"].values.reshape(-1, 1).astype(np.float32),
        "z": df["Z (m)"].values.reshape(-1, 1).astype(np.float32),
    }
    invar = normalize_invar(invar, center, scale, dims=3)

    # ── Run inference ─────────────────────────────────────────────────────────
    print(f"[inference] Running on {len(df)} StarCCM+ points...")
    results = run_inference(flow_net, invar)

    pred_p = results["p"]
    true_p = df["Pressure (Pa)"].values.astype(np.float32)

    # ── Print statistics ──────────────────────────────────────────────────────
    print("\n── Pressure statistics ──")
    print(f"  PINN  pred_p : min={pred_p.min():.4f}, max={pred_p.max():.4f}, mean={pred_p.mean():.4f}")
    print(f"  CCM   true_p : min={true_p.min():.4f}, max={true_p.max():.4f}, mean={true_p.mean():.4f}")

    # ── L2 error ──────────────────────────────────────────────────────────────
    l2_abs = np.sqrt(np.mean((pred_p - true_p) ** 2))
    l2_rel = l2_abs / np.sqrt(np.mean(true_p ** 2))

    print("\n========================================")
    print(f"  L2 Absolute Error (p) : {l2_abs:.4f} Pa")
    print(f"  L2 Relative Error (p) : {l2_rel * 100:.2f} %")
    print("========================================")

    # ── Save comparison CSV ───────────────────────────────────────────────────
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


if __name__ == "__main__":
    run()
