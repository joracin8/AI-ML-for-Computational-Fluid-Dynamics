"""
Aneurysm CFD Training Dashboard
--------------------------------
A Streamlit control panel for the PhysicsNeMo aneurysm Navier-Stokes example.

Lets you:
  - Edit geometry parameters (center, inlet_center, scale, inlet_normal) and boundary
    condition parameters (nu, inlet_vel, areas, Q_scaled, rho, lambda weighting)
  - Point at your STL files and OpenFOAM validation CSV
  - Launch training as a background subprocess (matches normal PhysicsNeMo/Hydra usage)
  - Watch live loss curves while it trains
  - Browse and download results once finished

Run with:  streamlit run app.py
"""

import os
import sys
import json
import glob
import signal
import subprocess
import time
from datetime import datetime

import streamlit as st
import pandas as pd

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    HAS_TB = True
except ImportError:
    HAS_TB = False


# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
APP_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_SCRIPT = os.path.join(APP_DIR, "aneurysm_train.py")
PARAMS_FILE = os.path.join(APP_DIR, "run_params.json")
HYDRA_OVERRIDES_FILE = os.path.join(APP_DIR, "hydra_overrides.json")
RUN_LOG = os.path.join(APP_DIR, "run.log")
PID_FILE = os.path.join(APP_DIR, "run.pid")
OUTPUTS_DIR = os.path.join(APP_DIR, "outputs")  # Hydra default output root

DEFAULTS = {
    "nu": 7.989e-05,
    "inlet_vel": 1.5,
    "center": [0.01151354, 0.03322, 0.0],
    "scale": 12.753,
    "inlet_center": [-0.1469, -0.4996, 0.0],
    "inlet_area_raw": 5.0179867253e-05,
    "inlet_normal": [0.8526, -0.428, 0.299],
    "outlet_area_raw": 12.0773,
    "integral_continuity_1_value": 0.006122,
    "integral_continuity_2_value": -0.006122,
    "lambda_weighting_1": 0.1,
    "lambda_weighting_2": 0.1,
    "rho": 1.0,
    "time_dependent": False,
    "stl_dir": "./stl_files",
    "openfoam_csv": "./openfoam/aneurysm_parabolicInlet_sol0.csv",
}

# Hydra cfg overrides — NOT part of run_params.json. These live in conf/config.yaml
# and are passed as command-line arguments when launching the subprocess, since
# Hydra config is resolved once at process startup.
HYDRA_DEFAULTS = {
    "arch.fully_connected.layer_size": 256,
    "arch.fully_connected.nr_layers": 6,
    "training.max_steps": 1500000,
    "training.rec_results_freq": 1000,
    "training.rec_constraint_freq": 5000,
    "scheduler.decay_rate": 0.95,
    "scheduler.decay_steps": 15000,
    "batch_size.inlet": 1100,
    "batch_size.outlet": 650,
    "batch_size.no_slip": 5200,
    "batch_size.interior": 6000,
    "batch_size.integral_continuity": 1100,
}

FLAGGED_PARAMS = {"inlet_normal", "outlet_area_raw"}  # still need verification from STL/CAD

st.set_page_config(page_title="Aneurysm CFD Dashboard", layout="wide")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_current_params():
    """Load saved run_params.json, merged on top of DEFAULTS.

    Merging (not replacing) means any keys missing from an older or hand-edited
    run_params.json fall back to DEFAULTS instead of raising a KeyError later
    when the UI tries to read them.
    """
    params = dict(DEFAULTS)
    if os.path.exists(PARAMS_FILE):
        try:
            with open(PARAMS_FILE, "r") as f:
                saved = json.load(f)
            params.update(saved)
        except (json.JSONDecodeError, OSError):
            pass  # corrupt/unreadable file — fall back to defaults
    return params


def load_current_hydra_overrides():
    overrides = dict(HYDRA_DEFAULTS)
    if os.path.exists(HYDRA_OVERRIDES_FILE):
        try:
            with open(HYDRA_OVERRIDES_FILE, "r") as f:
                saved = json.load(f)
            overrides.update(saved)
        except (json.JSONDecodeError, OSError):
            pass
    return overrides


def save_params(params):
    with open(PARAMS_FILE, "w") as f:
        json.dump(params, f, indent=2)


def save_hydra_overrides(overrides):
    with open(HYDRA_OVERRIDES_FILE, "w") as f:
        json.dump(overrides, f, indent=2)


def is_running():
    if not os.path.exists(PID_FILE):
        return False
    with open(PID_FILE, "r") as f:
        pid = int(f.read().strip())
    if os.name == "nt":
        # os.kill(pid, 0) is unreliable on Windows; query via tasklist instead.
        try:
            out = subprocess.check_output(
                ["tasklist", "/FI", f"PID eq {pid}"], text=True, stderr=subprocess.DEVNULL
            )
            return str(pid) in out
        except Exception:
            return False
    else:
        try:
            os.kill(pid, 0)  # signal 0 = check existence only
            return True
        except (OSError, ProcessLookupError):
            return False


def get_pid():
    if os.path.exists(PID_FILE):
        with open(PID_FILE, "r") as f:
            return int(f.read().strip())
    return None


def launch_training(hydra_overrides=None):
    """Launch aneurysm_train.py as a background subprocess.

    Returns (success: bool, message: str). Never raises — any failure to spawn
    the subprocess is caught and reported back as a message instead of crashing
    the Streamlit app.
    """
    try:
        log_f = open(RUN_LOG, "w")
        # Use sys.executable (the exact Python running this Streamlit app) rather
        # than a bare "python" string — on Windows/conda, bare command resolution
        # inside subprocess.Popen can fail or pick the wrong interpreter even when
        # "python" works fine when typed directly into the terminal.
        cmd = [sys.executable, TRAIN_SCRIPT]
        if hydra_overrides:
            for key, val in hydra_overrides.items():
                cmd.append(f"{key}={val}")

        creationflags = 0
        if os.name == "nt":
            # Detach fully from the parent console so closing/losing the terminal
            # window doesn't send a Ctrl-Break/close signal to the training process.
            creationflags = subprocess.CREATE_NEW_PROCESS_GROUP

        proc = subprocess.Popen(
            cmd,
            cwd=APP_DIR,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            creationflags=creationflags,
        )
        with open(PID_FILE, "w") as f:
            f.write(str(proc.pid))
        return True, f"Launched (PID {proc.pid})."
    except Exception as e:
        return False, f"Failed to launch training: {e}"


def stop_training():
    pid = get_pid()
    if pid:
        try:
            if os.name == "nt":
                subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True)
            else:
                os.kill(pid, signal.SIGTERM)
        except (OSError, ProcessLookupError):
            pass
    if os.path.exists(PID_FILE):
        os.remove(PID_FILE)


def find_latest_run_dir():
    """Hydra writes to outputs/<date>/<time>/ relative to the script's cwd."""
    if not os.path.isdir(OUTPUTS_DIR):
        return None
    date_dirs = sorted(glob.glob(os.path.join(OUTPUTS_DIR, "*", "*")))
    return date_dirs[-1] if date_dirs else None


def find_monitor_csvs(run_dir):
    if not run_dir:
        return []
    return sorted(glob.glob(os.path.join(run_dir, "monitors", "*.csv")))


def find_tb_events(run_dir):
    if not run_dir:
        return []
    return glob.glob(os.path.join(run_dir, "**", "events.out.tfevents.*"), recursive=True)


def read_tb_scalars(event_files):
    """Pull scalar series (loss curves) out of TensorBoard event files."""
    series = {}
    if not HAS_TB:
        return series
    for ef in event_files:
        try:
            ea = EventAccumulator(ef)
            ea.Reload()
            for tag in ea.Tags().get("scalars", []):
                events = ea.Scalars(tag)
                steps = [e.step for e in events]
                vals = [e.value for e in events]
                if tag not in series:
                    series[tag] = (steps, vals)
                else:
                    # merge if multiple event files cover the same tag
                    s, v = series[tag]
                    series[tag] = (s + steps, v + vals)
        except Exception:
            continue
    return series


def tail_log(n_lines=60):
    if not os.path.exists(RUN_LOG):
        return ""
    with open(RUN_LOG, "r", errors="ignore") as f:
        lines = f.readlines()
    return "".join(lines[-n_lines:])


def vec3_input(label, key_prefix, default_vec, flagged=False):
    flag = " ⚠️" if flagged else ""
    st.caption(f"{label}{flag}")
    cols = st.columns(3)
    out = []
    for i, axis in enumerate(["x", "y", "z"]):
        val = cols[i].number_input(
            axis, value=float(default_vec[i]), key=f"{key_prefix}_{axis}",
            format="%.6g",
        )
        out.append(val)
    return out


# ---------------------------------------------------------------------------
# Sidebar — run controls
# ---------------------------------------------------------------------------
st.sidebar.title("Run Control")

running = is_running()
status_color = "🟢" if running else "⚪"
st.sidebar.markdown(f"**Status:** {status_color} {'Training running' if running else 'Idle'}")

if running:
    st.sidebar.write(f"PID: {get_pid()}")
    if st.sidebar.button("Stop training", type="primary"):
        stop_training()
        st.sidebar.success("Sent stop signal.")
        st.rerun()
else:
    if st.sidebar.button("Launch training", type="primary"):
        save_params(st.session_state["params"])
        save_hydra_overrides(st.session_state["hydra"])
        success, msg = launch_training(st.session_state["hydra"])
        if success:
            st.sidebar.success(msg)
            time.sleep(1)
            st.rerun()
        else:
            st.sidebar.error(msg)

st.sidebar.divider()
auto_refresh = st.sidebar.checkbox("Auto-refresh (10s)", value=running)
if st.sidebar.button("Refresh now"):
    st.rerun()

st.sidebar.divider()
st.sidebar.caption(
    "Training launches `aneurysm_train.py` as a background process from this "
    "app's folder, matching normal PhysicsNeMo/Hydra usage. Outputs land in "
    "`outputs/<date>/<time>/` next to the script."
)


# ---------------------------------------------------------------------------
# Main layout
# ---------------------------------------------------------------------------
st.title("Aneurysm CFD Training Dashboard")
st.caption("PhysicsNeMo Navier-Stokes aneurysm example — custom geometry control panel")

tab_params, tab_training, tab_monitor, tab_results = st.tabs(
    ["Parameters", "Architecture & Training", "Live Monitor", "Results"]
)

if "params" not in st.session_state:
    st.session_state["params"] = load_current_params()
if "hydra" not in st.session_state:
    st.session_state["hydra"] = load_current_hydra_overrides()
P = st.session_state["params"]
H = st.session_state["hydra"]

# --- Parameters tab -------------------------------------------------------
with tab_params:
    if running:
        st.warning("Training is currently running. Stop it before editing parameters for a new run.")

    # ===================== GEOMETRY =====================
    st.subheader("Geometry")
    st.caption("Mesh normalization, scaling, and surface orientation. Derived from your STL/CAD.")

    c1, c2 = st.columns(2)
    with c1:
        P["center"] = vec3_input("center (mesh normalization origin)", "center", P["center"])
    with c2:
        P["inlet_center"] = vec3_input("inlet_center", "inlet_center", P["inlet_center"])

    P["scale"] = st.number_input("scale", value=float(P["scale"]), format="%.6g", key="scale")

    st.caption("inlet_normal ⚠️ — verify against your STL's actual inlet face normal before relying on results.")
    P["inlet_normal"] = vec3_input("inlet_normal (unit vector)", "inlet_normal", P["inlet_normal"], flagged=True)

    st.divider()

    # ===================== BOUNDARY CONDITIONS =====================
    st.subheader("Boundary conditions")
    st.caption("Inlet/outlet flow conditions and fluid properties applied at the domain boundaries.")

    c1, c2, c3 = st.columns(3)
    with c1:
        P["nu"] = st.number_input("nu (kinematic viscosity)", value=float(P["nu"]), format="%.6g", key="nu")
        P["inlet_vel"] = st.number_input("inlet_vel", value=float(P["inlet_vel"]), format="%.6g", key="inlet_vel")
    with c2:
        P["inlet_area_raw"] = st.number_input(
            "inlet_area_raw (pre-scale)", value=float(P["inlet_area_raw"]), format="%.6g", key="inlet_area_raw"
        )
        P["outlet_area_raw"] = st.number_input(
            "outlet_area_raw ⚠️ (pre-scale)",
            value=float(P["outlet_area_raw"]), format="%.6g", key="outlet_area_raw",
        )
        st.caption("outlet_area = outlet_area_raw × scale². Currently unused downstream — see README.")
    with c3:
        P["integral_continuity_1_value"] = st.number_input(
            "Q_scaled — outlet (normal_dot_vel)", value=float(P["integral_continuity_1_value"]),
            format="%.6g", key="q1",
        )
        P["integral_continuity_2_value"] = st.number_input(
            "Q_scaled — integral plane (normal_dot_vel)", value=float(P["integral_continuity_2_value"]),
            format="%.6g", key="q2",
        )

    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        P["rho"] = st.number_input("rho (fluid density)", value=float(P["rho"]), format="%.6g", key="rho")
    with c2:
        P["time_dependent"] = st.checkbox("Time-dependent (unsteady) flow", value=bool(P["time_dependent"]), key="time_dependent")
        st.caption("Unchecked = steady-state Navier-Stokes (matches original example)")

    c1, c2 = st.columns(2)
    with c1:
        P["lambda_weighting_1"] = st.number_input(
            "lambda_weighting — integral_continuity_1", value=float(P["lambda_weighting_1"]),
            format="%.6g", key="lw1",
        )
    with c2:
        P["lambda_weighting_2"] = st.number_input(
            "lambda_weighting — integral_continuity_2", value=float(P["lambda_weighting_2"]),
            format="%.6g", key="lw2",
        )

    st.divider()
    st.subheader("File locations")
    P["stl_dir"] = st.text_input("STL folder", value=P["stl_dir"])
    st.caption("Expects: aneurysm_inlet.stl, aneurysm_outlet.stl, aneurysm_noslip.stl, aneurysm_integral.stl, aneurysm_closed.stl")
    P["openfoam_csv"] = st.text_input("OpenFOAM validation CSV (optional)", value=P["openfoam_csv"])

    st.divider()
    col_save, col_reset = st.columns(2)
    with col_save:
        if st.button("Save parameters"):
            save_params(P)
            st.success(f"Saved to run_params.json")
    with col_reset:
        if st.button("Reset to current table defaults"):
            st.session_state["params"] = dict(DEFAULTS)
            st.rerun()

    with st.expander("Preview run_params.json"):
        st.json(P)

# --- Architecture & Training tab -------------------------------------------------------
with tab_training:
    if running:
        st.warning("Training is currently running. Stop it before changing these for a new run — they only take effect at launch.")

    st.caption(
        "These settings live in Hydra's config (not run_params.json) and are passed as "
        "command-line overrides when training launches, e.g. `arch.fully_connected.layer_size=512`."
    )

    st.subheader("Network architecture")
    c1, c2 = st.columns(2)
    with c1:
        H["arch.fully_connected.layer_size"] = st.number_input(
            "layer_size", value=int(H["arch.fully_connected.layer_size"]), step=32, key="layer_size",
        )
    with c2:
        H["arch.fully_connected.nr_layers"] = st.number_input(
            "nr_layers", value=int(H["arch.fully_connected.nr_layers"]), step=1, key="nr_layers",
        )

    st.divider()
    st.subheader("Training schedule")
    c1, c2, c3 = st.columns(3)
    with c1:
        H["training.max_steps"] = st.number_input(
            "max_steps", value=int(H["training.max_steps"]), step=10000, key="max_steps",
        )
    with c2:
        H["training.rec_results_freq"] = st.number_input(
            "rec_results_freq", value=int(H["training.rec_results_freq"]), step=100, key="rec_results_freq",
        )
    with c3:
        H["training.rec_constraint_freq"] = st.number_input(
            "rec_constraint_freq", value=int(H["training.rec_constraint_freq"]), step=100, key="rec_constraint_freq",
        )

    st.divider()
    st.subheader("Learning rate decay (exponential scheduler)")
    c1, c2 = st.columns(2)
    with c1:
        H["scheduler.decay_rate"] = st.number_input(
            "decay_rate", value=float(H["scheduler.decay_rate"]), format="%.4g", key="decay_rate",
        )
    with c2:
        H["scheduler.decay_steps"] = st.number_input(
            "decay_steps", value=int(H["scheduler.decay_steps"]), step=1000, key="decay_steps",
        )

    st.divider()
    st.subheader("Batch sizes")
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        H["batch_size.inlet"] = st.number_input("inlet", value=int(H["batch_size.inlet"]), step=100, key="bs_inlet")
    with c2:
        H["batch_size.outlet"] = st.number_input("outlet", value=int(H["batch_size.outlet"]), step=100, key="bs_outlet")
    with c3:
        H["batch_size.no_slip"] = st.number_input("no_slip", value=int(H["batch_size.no_slip"]), step=100, key="bs_no_slip")
    with c4:
        H["batch_size.interior"] = st.number_input("interior", value=int(H["batch_size.interior"]), step=100, key="bs_interior")
    with c5:
        H["batch_size.integral_continuity"] = st.number_input(
            "integral_continuity", value=int(H["batch_size.integral_continuity"]), step=100, key="bs_integral",
        )

    st.divider()
    col_save, col_reset = st.columns(2)
    with col_save:
        if st.button("Save training config"):
            save_hydra_overrides(H)
            st.success("Saved to hydra_overrides.json")
    with col_reset:
        if st.button("Reset to config.yaml defaults"):
            st.session_state["hydra"] = dict(HYDRA_DEFAULTS)
            st.rerun()

    with st.expander("Preview command-line overrides"):
        st.code(" ".join(f"{k}={v}" for k, v in H.items()), language="bash")

# --- Live Monitor tab -------------------------------------------------------
with tab_monitor:
    st.subheader("Training log (tail)")
    log_text = tail_log(80)
    st.code(log_text if log_text else "No log yet — launch a run to see output here.", language="text")

    st.subheader("Loss curves")
    run_dir = find_latest_run_dir()
    if not run_dir:
        st.info("No run directory found yet under `outputs/`. Loss curves appear once training starts writing TensorBoard logs.")
    elif not HAS_TB:
        st.warning("`tensorboard` package not installed in this environment, so event files can't be parsed here. "
                    "Install it with `pip install tensorboard`, or run `tensorboard --logdir outputs` separately and open it alongside this dashboard.")
    else:
        events = find_tb_events(run_dir)
        if not events:
            st.info(f"No TensorBoard event files found yet in {run_dir}")
        else:
            scalars = read_tb_scalars(events)
            loss_tags = [t for t in scalars if "loss" in t.lower()]
            other_tags = [t for t in scalars if t not in loss_tags]

            if loss_tags:
                st.caption(f"Run directory: `{run_dir}`")
                for tag in loss_tags:
                    steps, vals = scalars[tag]
                    df = pd.DataFrame({"step": steps, tag: vals}).drop_duplicates("step").sort_values("step")
                    st.line_chart(df, x="step", y=tag, height=220)
            else:
                st.info("No loss scalars found yet — training may still be initializing.")

            if other_tags:
                with st.expander("Other monitored metrics (e.g. pressure_drop)"):
                    for tag in other_tags:
                        steps, vals = scalars[tag]
                        df = pd.DataFrame({"step": steps, tag: vals}).drop_duplicates("step").sort_values("step")
                        st.line_chart(df, x="step", y=tag, height=200)

# --- Results tab -------------------------------------------------------
with tab_results:
    run_dir = find_latest_run_dir()
    if not run_dir:
        st.info("No completed or in-progress run found yet.")
    else:
        st.caption(f"Latest run directory: `{run_dir}`")

        st.subheader("Monitor CSVs")
        csvs = find_monitor_csvs(run_dir)
        if csvs:
            for csv_path in csvs:
                name = os.path.basename(csv_path)
                df = pd.read_csv(csv_path)
                st.write(f"**{name}**")
                st.line_chart(df.set_index(df.columns[0]))
                with open(csv_path, "rb") as f:
                    st.download_button(f"Download {name}", f, file_name=name, key=csv_path)
        else:
            st.info("No monitor CSVs found yet (written under `<run_dir>/monitors/`).")

        st.subheader("Checkpoints")
        ckpts = sorted(glob.glob(os.path.join(run_dir, "*.pth")))
        if ckpts:
            for ck in ckpts:
                st.write(f"- `{os.path.basename(ck)}`  ({os.path.getsize(ck) / 1e6:.1f} MB)")
        else:
            st.info("No checkpoint files found yet.")

        st.subheader("Validator outputs (VTK / images)")
        vtu_files = glob.glob(os.path.join(run_dir, "validators", "*"))
        if vtu_files:
            for vf in sorted(vtu_files)[:50]:
                st.write(f"- `{os.path.relpath(vf, run_dir)}`")
        else:
            st.info("No validator output files found yet.")

if auto_refresh and running:
    time.sleep(10)
    st.rerun()
