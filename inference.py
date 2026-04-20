# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import numpy as np
import torch
import csv

import physicsnemo.sym
from physicsnemo.sym.hydra import to_absolute_path, instantiate_arch, PhysicsNeMoConfig
from physicsnemo.sym.key import Key
from physicsnemo.sym.geometry.tessellation import Tessellation


# ─────────────────────────────────────────────
# Helper: normalize coordinates (same as training)
# ─────────────────────────────────────────────
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


# ─────────────────────────────────────────────
# Load trained model from checkpoint
# ─────────────────────────────────────────────
def load_model(checkpoint_dir: str, cfg: PhysicsNeMoConfig):
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
        raise FileNotFoundError(
            f"No flow_network checkpoint found in {checkpoint_dir}"
        )

    ckpt_path = os.path.join(checkpoint_dir, ckpt_files[-1])
    print(f"[inference] Loading checkpoint: {ckpt_path}")
    state_dict = torch.load(ckpt_path, map_location="cpu")
    flow_net.load_state_dict(state_dict)
    flow_net.eval()
    return flow_net


# ─────────────────────────────────────────────
# Run inference on normalised point cloud
# ─────────────────────────────────────────────
def run_inference(
    flow_net,
    invar: dict,
    batch_size: int = 4096,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    """
    invar : dict with keys 'x', 'y', 'z' as (N,1) numpy arrays,
            already normalized to training space.
    """
    x = torch.tensor(invar["x"], dtype=torch.float32)
    y = torch.tensor(invar["y"], dtype=torch.float32)
    z = torch.tensor(invar["z"], dtype=torch.float32)

    flow_net = flow_net.to(device)

    u_list, v_list, w_list, p_list = [], [], [], []

    with torch.no_grad():
        for i in range(0, len(x), batch_size):
            xb = x[i : i + batch_size].to(device)
            yb = y[i : i + batch_size].to(device)
            zb = z[i : i + batch_size].to(device)

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


# ─────────────────────────────────────────────
# Save results to CSV
# ─────────────────────────────────────────────
def save_to_csv(invar: dict, results: dict, out_path: str):
    x = invar["x"].squeeze()
    y = invar["y"].squeeze()
    z = invar["z"].squeeze()
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y", "z", "u", "v", "w", "p"])
        for i in range(len(x)):
            writer.writerow([
                x[i], y[i], z[i],
                results["u"][i], results["v"][i],
                results["w"][i], results["p"][i],
            ])
    print(f"[inference] Results saved to: {out_path}")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
@physicsnemo.sym.main(config_path="conf", config_name="config")
def run(cfg: PhysicsNeMoConfig) -> None:

    # Parameters — identical to training script
    center = (-18.40381048596882, -50.285383353981196, 12.848136936899031)
    scale = 0.4

    # Checkpoint directory
    checkpoint_dir = to_absolute_path("./outputs/aneurysm_flow/checkpoints")

    # Load trained model
    flow_net = load_model(checkpoint_dir, cfg)

    # Load interior mesh and sample points
    point_path = to_absolute_path("./stl_files")
    interior_mesh = Tessellation.from_stl(
        point_path + "/aneurysm_closed.stl", airtight=True
    )

    N_points = 10000
    invar = interior_mesh.sample_interior(N_points)
    # invar has keys 'x', 'y', 'z' as (N,1) numpy arrays

    # Normalize — same transform applied during training
    invar = normalize_invar(invar, center, scale, dims=3)

    # Run inference
    print(f"[inference] Evaluating {N_points} interior points ...")
    results = run_inference(flow_net, invar)

    # Summary statistics
    vel_mag = np.sqrt(results["u"]**2 + results["v"]**2 + results["w"]**2)
    print("\n──────── Inference Summary ────────")
    print(f"  Points evaluated : {N_points}")
    print(f"  |U| max          : {vel_mag.max():.6f}")
    print(f"  |U| mean         : {vel_mag.mean():.6f}")
    print(f"  p   max          : {results['p'].max():.6f}")
    print(f"  p   min          : {results['p'].min():.6f}")
    print(f"  Pressure drop    : {results['p'].max() - results['p'].min():.6f}")
    print("───────────────────────────────────\n")

    # Save to CSV
    out_path = to_absolute_path("./inference_results.csv")
    save_to_csv(invar, results, out_path)


if __name__ == "__main__":
    run()
