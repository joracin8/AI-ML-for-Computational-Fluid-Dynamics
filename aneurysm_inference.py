# =============================================================================
# Custom Inference Script for Aneurysm PINN
# Uses trained checkpoint to predict u, v, w, p at custom points
# No OpenFOAM CSV required
# =============================================================================

import os
import numpy as np
import torch
import csv

import physicsnemo.sym
from physicsnemo.sym.hydra import to_absolute_path, instantiate_arch, PhysicsNeMoConfig
from physicsnemo.sym.solver import Solver
from physicsnemo.sym.domain import Domain
from physicsnemo.sym.domain.inferencer import PointwiseInferencer
from physicsnemo.sym.key import Key
from physicsnemo.sym.eq.pdes.navier_stokes import NavierStokes
from physicsnemo.sym.eq.pdes.basic import NormalDotVec
from physicsnemo.sym.geometry.tessellation import Tessellation


# =============================================================================
# YOUR CUSTOM PARAMETERS (from your modified config)
# =============================================================================
nu              = 7.989e-05
inlet_vel       = 1.5
center          = (0.01151354, 0.03322, 0.0)
scale           = 12.753
inlet_center    = (-0.1469, -0.4996, 0.0)
inlet_area_raw  = 5.0179867253e-05
inlet_area      = inlet_area_raw * (scale ** 2)
inlet_radius    = np.sqrt(inlet_area / np.pi)

# ⚠️ Update inlet_normal once verified from your STL
inlet_normal    = (0.8526, -0.428, 0.299)

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def normalize_invar(invar, center, scale, dims=2):
    """Normalize coordinates using your geometry's center and scale."""
    invar["x"] -= center[0]
    invar["y"] -= center[1]
    invar["z"] -= center[2]
    invar["x"] *= scale
    invar["y"] *= scale
    invar["z"] *= scale
    if "area" in invar.keys():
        invar["area"] *= scale ** dims
    return invar


def save_to_csv(filename, invar, outvar):
    """Save inference results to a CSV file (readable by ParaView)."""
    keys = list(invar.keys()) + list(outvar.keys())
    rows = zip(*[invar[k].flatten() for k in invar.keys()],
               *[outvar[k].flatten() for k in outvar.keys()])
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(keys)
        writer.writerows(rows)
    print(f"  Saved: {filename}")


def save_to_vtk(filename, invar, outvar):
    """Save inference results as VTK legacy format for ParaView."""
    x = invar["x"].flatten()
    y = invar["y"].flatten()
    z = invar["z"].flatten()
    n = len(x)

    with open(filename, "w") as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write("PINN Inference Output\n")
        f.write("ASCII\n")
        f.write("DATASET POLYDATA\n")
        f.write(f"POINTS {n} float\n")
        for i in range(n):
            f.write(f"{x[i]:.6e} {y[i]:.6e} {z[i]:.6e}\n")
        f.write(f"\nVERTICES {n} {2*n}\n")
        for i in range(n):
            f.write(f"1 {i}\n")
        f.write(f"\nPOINT_DATA {n}\n")

        # Write velocity as a vector field
        if all(k in outvar for k in ["u", "v", "w"]):
            u = outvar["u"].flatten()
            v = outvar["v"].flatten()
            w = outvar["w"].flatten()
            f.write("VECTORS velocity float\n")
            for i in range(n):
                f.write(f"{u[i]:.6e} {v[i]:.6e} {w[i]:.6e}\n")

        # Write pressure as scalar field
        if "p" in outvar:
            p = outvar["p"].flatten()
            f.write("\nSCALARS pressure float 1\n")
            f.write("LOOKUP_TABLE default\n")
            for i in range(n):
                f.write(f"{p[i]:.6e}\n")

    print(f"  Saved: {filename}")


# =============================================================================
# MAIN INFERENCE FUNCTION
# =============================================================================

@physicsnemo.sym.main(config_path="conf", config_name="config")
def run(cfg: PhysicsNeMoConfig) -> None:

    # ------------------------------------------------------------------
    # 1. Load STL meshes (same as training)
    # ------------------------------------------------------------------
    point_path = to_absolute_path("./stl_files")
    interior_mesh = Tessellation.from_stl(
        point_path + "/aneurysm_closed.stl", airtight=True
    )

    def normalize_mesh(mesh, center, scale):
        mesh = mesh.translate([-c for c in center])
        mesh = mesh.scale(scale)
        return mesh

    interior_mesh = normalize_mesh(interior_mesh, center, scale)

    # ------------------------------------------------------------------
    # 2. Rebuild the same network architecture as training
    # ------------------------------------------------------------------
    ns = NavierStokes(nu=nu * scale, rho=1.0, dim=3, time=False)
    normal_dot_vel = NormalDotVec(["u", "v", "w"])

    flow_net = instantiate_arch(
        input_keys=[Key("x"), Key("y"), Key("z")],
        output_keys=[Key("u"), Key("v"), Key("w"), Key("p")],
        cfg=cfg.arch.fully_connected,
    )
    nodes = (
        ns.make_nodes()
        + normal_dot_vel.make_nodes()
        + [flow_net.make_node(name="flow_network")]
    )

    # ------------------------------------------------------------------
    # 3. Create inference points
    #    Option A: Sample directly from your STL interior (recommended)
    #    Option B: Uniform grid (may include points outside geometry)
    # ------------------------------------------------------------------

    print("\n--- Generating inference points ---")

    # OPTION A: Sample from interior mesh (cleanest approach)
    N_POINTS = 50000   # ← increase for higher resolution
    interior_samples = interior_mesh.sample_interior(nr_points=N_POINTS)

    invar_sampled = {
        "x": interior_samples["x"],
        "y": interior_samples["y"],
        "z": interior_samples["z"],
    }
    print(f"  Sampled {N_POINTS} interior points from STL")

    # OPTION B: Uniform grid (uncomment to use instead)
    # --------------------------------------------------------
    # Nx, Ny, Nz = 50, 50, 50
    # # Set these bounds based on your normalized geometry
    # x_vals = np.linspace(-1.0, 1.0, Nx)
    # y_vals = np.linspace(-1.0, 1.0, Ny)
    # z_vals = np.linspace(-0.5, 0.5, Nz)
    # xx, yy, zz = np.meshgrid(x_vals, y_vals, z_vals)
    # invar_sampled = {
    #     "x": xx.flatten()[:, None].astype(np.float32),
    #     "y": yy.flatten()[:, None].astype(np.float32),
    #     "z": zz.flatten()[:, None].astype(np.float32),
    # }
    # print(f"  Created uniform grid: {Nx}x{Ny}x{Nz} = {Nx*Ny*Nz} points")
    # --------------------------------------------------------

    # ------------------------------------------------------------------
    # 4. Add inferencer to domain
    # ------------------------------------------------------------------
    domain = Domain()

    inferencer = PointwiseInferencer(
        nodes=nodes,
        invar=invar_sampled,
        output_names=["u", "v", "w", "p"],
        batch_size=4096,   # GPU batch size during inference
    )
    domain.add_inferencer(inferencer, "flow_field")

    # ------------------------------------------------------------------
    # 5. Run in eval mode
    # ------------------------------------------------------------------
    slv = Solver(cfg, domain)
    slv.solve()

    # ------------------------------------------------------------------
    # 6. Load results and export
    # ------------------------------------------------------------------
    print("\n--- Exporting results ---")

    results_path = to_absolute_path("./outputs/inferencers/flow_field.npz")

    if os.path.exists(results_path):
        data = np.load(results_path)

        invar_out = {
            "x": data["x"],
            "y": data["y"],
            "z": data["z"],
        }
        outvar_out = {
            "u": data["u"],
            "v": data["v"],
            "w": data["w"],
            "p": data["p"],
        }

        # Print quick stats
        print("\n--- Flow Field Statistics ---")
        print(f"  u  : min={data['u'].min():.4f}, max={data['u'].max():.4f}")
        print(f"  v  : min={data['v'].min():.4f}, max={data['v'].max():.4f}")
        print(f"  w  : min={data['w'].min():.4f}, max={data['w'].max():.4f}")
        print(f"  p  : min={data['p'].min():.4f}, max={data['p'].max():.4f}")

        # Save outputs
        os.makedirs("./inference_output", exist_ok=True)
        save_to_csv("./inference_output/flow_results.csv", invar_out, outvar_out)
        save_to_vtk("./inference_output/flow_results.vtk", invar_out, outvar_out)

        print("\n✓ Done! Open flow_results.vtk in ParaView to visualize.")

    else:
        print(f"Results not found at {results_path}")
        print("Check ./outputs/inferencers/ for output files.")


if __name__ == "__main__":
    run()
