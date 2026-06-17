# Aneurysm CFD Training Dashboard

A Streamlit control panel for the PhysicsNeMo aneurysm Navier-Stokes example,
built around your custom-geometry parameter table.

## What's in this folder

- `app.py` — the Streamlit dashboard (parameters, launch/stop, live monitor, results)
- `aneurysm_train.py` — your training script, modified so the parameters in the
  dashboard's "Needs verification" and "Core physics & geometry" sections are read
  from `run_params.json` at runtime instead of being hardcoded. If `run_params.json`
  doesn't exist, it falls back to the values currently in your summary table.
- `conf/config.yaml` — the Hydra config (batch sizes, network architecture, training
  steps, scheduler). Same structure PhysicsNeMo examples normally use.
- `requirements.txt` — Python packages the dashboard itself needs (NOT physicsnemo,
  which you already have installed).

## One-time setup on your office GPU machine

1. Copy this whole folder onto the machine, next to (or replacing) wherever you
   currently keep the aneurysm example script.
2. Put your 5 STL files in a `stl_files/` subfolder (or point the dashboard at
   wherever they already live — there's a text field for the path).
3. If you have OpenFOAM validation data, put the CSV wherever you like and point
   the dashboard at it (optional — training runs fine without it).
4. Install the dashboard's own dependencies (separate from physicsnemo):

   ```bash
   pip install -r requirements.txt
   ```

## Running it

```bash
streamlit run app.py
```

This opens in your browser (usually `http://localhost:8501`). If you're running it
on a remote/headless machine and viewing it from your own laptop, use:

```bash
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

and then browse to `http://<office-machine-ip>:8501` from your laptop.

## Using the dashboard

**Parameters tab** — edit `inlet_normal` and `outlet_area_raw` (still flagged ⚠️ in
your table) plus the core physics/geometry values: `nu`, `inlet_vel`, `scale`,
`inlet_area_raw`, the two `Q_scaled` integral-continuity values, `center`,
`inlet_center`, fluid density `rho`, the (placeholder) time-dependent flow toggle,
and the lambda weighting for each integral continuity constraint. Click **Save
parameters** to write `run_params.json`, or just click **Launch training** in the
sidebar, which saves automatically before starting.

**Architecture & Training tab** — edit network architecture (`layer_size`,
`nr_layers`), training schedule (`max_steps`, result/checkpoint recording frequency),
learning-rate decay (`decay_rate`, `decay_steps`), and per-constraint batch sizes
(inlet, outlet, no-slip, interior, integral continuity). These come from Hydra's
config rather than `run_params.json`, so they're passed as command-line overrides
when training launches (you can preview the exact command line in this tab) and
saved separately to `hydra_overrides.json`.

**Launch training** (sidebar) — starts `aneurysm_train.py` as a background process,
exactly like running `python aneurysm_train.py` from the terminal. Streamlit stays
responsive while it trains. Output and errors are captured to `run.log` in this folder.

**Stop training** (sidebar) — sends a termination signal to the running process.
Useful if you spot a bad parameter early and want to fix it without waiting hours.

**Live Monitor tab** — tails `run.log` and reads loss curves directly from the
TensorBoard event files PhysicsNeMo writes under `outputs/<date>/<time>/`. Check
"Auto-refresh" in the sidebar to have it update every 10 seconds while training runs.

**Results tab** — once training has produced output, lists checkpoint files (`.pth`),
monitor CSVs (like your pressure-drop monitor) with inline charts and download
buttons, and any validator output files.

## Notes / things to double check

- `outlet_area_raw` is still flagged because the original code never gave it a name —
  it computed `outlet_radius` once and never used it again downstream (`np.sqrt(...)`
  on its own line with no assignment, in both the original and your version). Verify
  this value against your actual outlet face area in your CAD/STL before trusting
  results that depend on it. As written, the dashboard exposes it for editing, but
  since the original script doesn't appear to use this variable in any constraint or
  boundary condition, changing it currently has **no effect on the simulation** — flag
  this to confirm whether your version of the script uses it elsewhere, or whether
  it's vestigial.
- `inlet_normal` **does** matter — it directly sets the inlet velocity direction in the
  parabolic profile. Get this right from your STL's actual inlet face normal.
- **`time_dependent` checkbox is exposed but not fully wired to true transient behavior.**
  Flipping it changes `NavierStokes(..., time=True)`, but the rest of the script (network
  input keys, constraint outvars, Solver config) is still set up for steady-state, matching
  the original example. Making this a real transient simulation needs additional changes:
  adding `Key("t")` to the network's input_keys, time-varying boundary/initial conditions,
  and a `training.max_steps`/time-stepping setup suited to unsteady flow. Treat this toggle
  as a placeholder for that future work, not a working unsteady-flow switch yet.
- Training is long-running (the config defaults to 1.5M steps, matching the original
  example). The dashboard doesn't change that; it's just a window into it.
- If `tensorboard` isn't installed, the Live Monitor tab will still show the log tail,
  just not the parsed loss charts. `pip install tensorboard` fixes that.
- Architecture and training settings (layer size, batch sizes, max steps, decay rate)
  are passed as Hydra command-line overrides at launch time, not stored in
  `run_params.json`. They're saved separately to `hydra_overrides.json` so your last
  settings persist between dashboard restarts.
