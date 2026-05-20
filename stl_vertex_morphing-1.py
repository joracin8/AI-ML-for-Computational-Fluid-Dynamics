# =============================================================================
# STL Vertex Morphing for Aneurysm Geometry Parameterization
# Compatible with your modified PhysicsNeMo Sym aneurysm PINN code
#
# Your modified parameters (from project screenshot):
#   nu           = 7.989e-05
#   inlet_vel    = 1.5
#   center       = (0.01151354, 0.03322, 0.0)
#   scale        = 12.753
#   inlet_center = (-0.1469, -0.4996, 0.0)
#   inlet_area   = 5.0179867253e-05 * (12.753**2)
#   Q_scaled     = 0.006122
#   Re           = 150
# =============================================================================

import os
import copy
import numpy as np
from stl import mesh as stl_mesh
from physicsnemo.sym.geometry.tessellation import Tessellation


# =============================================================================
# SECTION 1: CORE MORPH UTILITIES
# These are the building blocks. All morphing functions operate directly
# on raw numpy-stl mesh vertex arrays (mesh.vectors shape: [N_triangles, 3, 3])
# =============================================================================

def get_vertices(raw_mesh):
    """
    Extract unique vertex positions from a numpy-stl mesh.
    Returns:
        unique_verts : (M, 3) array of unique xyz positions
        indices      : (N*3,) array mapping each triangle vertex to unique_verts
    """
    flat = raw_mesh.vectors.reshape(-1, 3)           # (N*3, 3)
    unique_verts, indices = np.unique(flat, axis=0, return_inverse=True)
    return unique_verts, indices


def set_vertices(raw_mesh, unique_verts, indices):
    """
    Write morphed unique vertices back into the mesh.vectors array.
    Call this after modifying unique_verts in-place or returning new ones.
    """
    raw_mesh.vectors = unique_verts[indices].reshape(-1, 3, 3)


def compute_centroid(raw_mesh):
    """Compute the centroid (mean vertex position) of the mesh."""
    flat = raw_mesh.vectors.reshape(-1, 3)
    return flat.mean(axis=0)


# =============================================================================
# SECTION 2: MORPHING STRATEGIES
# Each function takes a raw numpy-stl Mesh and morph parameters,
# returns a NEW morphed Mesh (original is never modified).
# =============================================================================

# ── 2A. UNIFORM SCALE MORPH ─────────────────────────────────────────────────
def morph_uniform_scale(raw_mesh, scale_factor, pivot=None):
    """
    Scale all vertices uniformly around a pivot point.

    Parameters
    ----------
    raw_mesh     : numpy-stl Mesh object
    scale_factor : float  (e.g. 1.1 = 10% larger, 0.9 = 10% smaller)
    pivot        : (3,) array or None. If None, uses mesh centroid.

    Physical meaning for your aneurysm:
        Simulates a uniformly larger/smaller vessel — useful for studying
        the effect of vessel caliber on hemodynamics at Re=150.

    Example
    -------
    # Generate 5 sizes from 90% to 110% of baseline
    for sf in np.linspace(0.9, 1.1, 5):
        morphed = morph_uniform_scale(raw_mesh, sf)
    """
    morphed = copy.deepcopy(raw_mesh)
    verts, idx = get_vertices(morphed)

    if pivot is None:
        pivot = verts.mean(axis=0)

    verts_centered = verts - pivot
    verts_scaled   = verts_centered * scale_factor
    set_vertices(morphed, verts_scaled + pivot, idx)
    return morphed


# ── 2B. DIRECTIONAL STRETCH MORPH ───────────────────────────────────────────
def morph_directional_stretch(raw_mesh, axis, stretch_factor, pivot=None):
    """
    Stretch the geometry along ONE axis only (x=0, y=1, z=2).

    Parameters
    ----------
    axis           : int   0=x, 1=y, 2=z
    stretch_factor : float >1 elongates, <1 compresses along that axis

    Physical meaning for your aneurysm:
        Stretching along the vessel axis changes the aspect ratio of the
        aneurysm sac — clinically relevant (elongated vs. globular sacs
        have different rupture risk profiles).

    Example
    -------
    # Elongate aneurysm sac by 20% along z-axis
    morphed = morph_directional_stretch(raw_mesh, axis=2, stretch_factor=1.2)
    """
    morphed = copy.deepcopy(raw_mesh)
    verts, idx = get_vertices(morphed)

    if pivot is None:
        pivot = verts.mean(axis=0)

    verts_c = verts - pivot
    scale_vec = np.ones(3)
    scale_vec[axis] = stretch_factor
    verts_morphed = verts_c * scale_vec
    set_vertices(morphed, verts_morphed + pivot, idx)
    return morphed


# ── 2C. RADIAL BULGE MORPH (Aneurysm Sac Inflation) ─────────────────────────
def morph_radial_bulge(raw_mesh, sac_center, sac_radius, bulge_amount,
                       falloff_sharpness=2.0):
    """
    Inflate or deflate vertices within a spherical region of influence.
    This is the most physically meaningful morph for aneurysm geometry.

    Parameters
    ----------
    sac_center       : (3,) array  — center of the aneurysm sac in NORMALIZED coords
    sac_radius       : float       — radius of influence sphere (normalized units)
    bulge_amount     : float       — outward displacement magnitude
                                     positive = inflate, negative = deflate
    falloff_sharpness: float       — controls how sharply displacement falls off
                                     at the edge of the influence sphere

    Physical meaning:
        Directly models aneurysm growth — a positive bulge_amount simulates
        sac enlargement. Combined with your Re=150 setup, you can study how
        sac size affects wall shear stress and pressure distribution.

    Displacement field:
        d(v) = bulge_amount * max(0, 1 - (dist/sac_radius)^falloff)^falloff
        Direction: radially outward from sac_center

    Example
    -------
    # Inflate sac by 10% of its current radius
    sac_c  = np.array([0.0, 0.0, 0.05])   # adjust to your geometry
    morphed = morph_radial_bulge(raw_mesh, sac_c, sac_radius=0.08,
                                  bulge_amount=0.005)
    """
    morphed = copy.deepcopy(raw_mesh)
    verts, idx = get_vertices(morphed)

    sac_center = np.asarray(sac_center)

    # Vector from sac center to each vertex
    diff = verts - sac_center                            # (M, 3)
    dist = np.linalg.norm(diff, axis=1, keepdims=True)  # (M, 1)

    # Avoid division by zero at the center
    safe_dist = np.where(dist < 1e-12, 1e-12, dist)
    direction = diff / safe_dist                         # unit outward vectors

    # Weight: smooth falloff to zero at sac_radius
    t = dist / sac_radius                                # normalized distance
    weight = np.maximum(0.0, 1.0 - t ** falloff_sharpness) ** falloff_sharpness

    # Apply displacement
    displacement = bulge_amount * weight * direction     # (M, 3)
    verts_morphed = verts + displacement
    set_vertices(morphed, verts_morphed, idx)
    return morphed


# ── 2D. NECK CONSTRICTION MORPH ─────────────────────────────────────────────
def morph_neck_constriction(raw_mesh, neck_center, neck_axis,
                             neck_half_length, constriction_factor):
    """
    Constrict or widen the aneurysm neck — the junction between the parent
    vessel and the aneurysm sac.

    Parameters
    ----------
    neck_center       : (3,) array  — midpoint of the neck in normalized coords
    neck_axis         : (3,) array  — unit vector along the vessel axis at neck
    neck_half_length  : float       — axial half-width of the neck region
    constriction_factor: float      — <1 narrows, >1 widens the neck

    Physical meaning:
        Neck width is one of the most important geometric parameters in
        cerebral aneurysm hemodynamics. A narrow neck (aspect ratio > 2)
        is associated with higher rupture risk and poor flow washout.

    Example
    -------
    neck_c    = np.array([-0.05, 0.0, 0.0])
    neck_axis = np.array([1.0, 0.0, 0.0])   # vessel runs along x
    morphed = morph_neck_constriction(raw_mesh, neck_c, neck_axis,
                                       neck_half_length=0.03,
                                       constriction_factor=0.85)
    """
    morphed = copy.deepcopy(raw_mesh)
    verts, idx = get_vertices(morphed)

    neck_center = np.asarray(neck_center)
    neck_axis   = np.asarray(neck_axis) / np.linalg.norm(neck_axis)

    # Project each vertex onto the neck axis to find axial position
    diff        = verts - neck_center
    axial_proj  = (diff @ neck_axis)[:, np.newaxis] * neck_axis  # axial component
    radial_vec  = diff - axial_proj                               # radial component

    # Axial distance from neck center
    axial_dist = np.abs(diff @ neck_axis)

    # Smooth weight: 1 at neck center, 0 at neck_half_length away
    weight = np.maximum(0.0, 1.0 - (axial_dist / neck_half_length) ** 2)

    # Scale radial component: interpolate between original and constricted
    scale_field = 1.0 + (constriction_factor - 1.0) * weight[:, np.newaxis]
    verts_morphed = neck_center + axial_proj + radial_vec * scale_field
    set_vertices(morphed, verts_morphed, idx)
    return morphed


# ── 2E. SURFACE NOISE MORPH (Rugosity / Wall Irregularity) ──────────────────
def morph_surface_noise(raw_mesh, amplitude, seed=42):
    """
    Add small random normal perturbations along surface normals.
    Models vessel wall rugosity or small irregularities.

    Parameters
    ----------
    amplitude : float  — max displacement as fraction of bounding box size
                         e.g. 0.002 = 0.2% of bbox diagonal
    seed      : int    — random seed for reproducibility

    Physical meaning:
        Real vessel walls are not perfectly smooth. Surface rugosity affects
        near-wall flow and wall shear stress calculations.

    Example
    -------
    morphed = morph_surface_noise(raw_mesh, amplitude=0.001, seed=7)
    """
    morphed = copy.deepcopy(raw_mesh)
    rng = np.random.default_rng(seed)

    # Compute bounding box diagonal as reference length
    flat    = morphed.vectors.reshape(-1, 3)
    bbox_diag = np.linalg.norm(flat.max(axis=0) - flat.min(axis=0))
    max_disp  = amplitude * bbox_diag

    # Perturb each triangle vertex along its face normal
    normals = morphed.normals / (
        np.linalg.norm(morphed.normals, axis=1, keepdims=True) + 1e-12
    )  # (N, 3) unit normals

    noise = rng.uniform(-max_disp, max_disp, size=(len(normals), 1))
    for i in range(3):
        morphed.vectors[:, i, :] += noise * normals
    return morphed


# =============================================================================
# SECTION 3: MORPH ALL STL FILES CONSISTENTLY
# Your geometry has 5 STL files that must stay watertight and consistent.
# This function applies the SAME morph to all of them.
# =============================================================================

def apply_morph_to_all_meshes(stl_paths, morph_fn, morph_kwargs,
                               output_dir=None, tag="morphed"):
    """
    Load all 5 aneurysm STL files, apply the same morph function,
    and return both the raw numpy-stl meshes and Tessellation objects.

    Parameters
    ----------
    stl_paths   : dict  with keys: inlet, outlet, noslip, integral, interior
    morph_fn    : callable — one of the morph_* functions above
    morph_kwargs: dict   — keyword arguments for morph_fn (excluding raw_mesh)
    output_dir  : str or None — if given, saves morphed STLs here
    tag         : str   — suffix added to saved filenames

    Returns
    -------
    raw_meshes   : dict of numpy-stl Mesh objects (for inspection)
    tess_meshes  : dict of PhysicsNeMo Tessellation objects (for constraints)

    Example
    -------
    stl_paths = {
        "inlet"    : "./stl_files/aneurysm_inlet.stl",
        "outlet"   : "./stl_files/aneurysm_outlet.stl",
        "noslip"   : "./stl_files/aneurysm_noslip.stl",
        "integral" : "./stl_files/aneurysm_integral.stl",
        "interior" : "./stl_files/aneurysm_closed.stl",
    }
    raw, tess = apply_morph_to_all_meshes(
        stl_paths,
        morph_fn=morph_radial_bulge,
        morph_kwargs=dict(sac_center=np.array([0,0,0.05]),
                          sac_radius=0.08,
                          bulge_amount=0.005),
        output_dir="./morphed_stl_files",
        tag="bulge_01"
    )
    """
    airtight_map = {
        "inlet":    False,
        "outlet":   False,
        "noslip":   False,
        "integral": False,
        "interior": True,
    }

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    raw_meshes  = {}
    tess_meshes = {}

    for name, path in stl_paths.items():
        # Load raw STL
        raw = stl_mesh.Mesh.from_file(path)

        # Apply morph
        morphed_raw = morph_fn(raw, **morph_kwargs)

        # Optionally save morphed STL
        if output_dir:
            out_name = f"{name}_{tag}.stl"
            out_path = os.path.join(output_dir, out_name)
            morphed_raw.save(out_path)
            print(f"  Saved: {out_path}")

        # Wrap in PhysicsNeMo Tessellation
        tess = Tessellation(morphed_raw, airtight=airtight_map[name])

        raw_meshes[name]  = morphed_raw
        tess_meshes[name] = tess

    return raw_meshes, tess_meshes


# =============================================================================
# SECTION 4: PARAMETRIC SWEEP GENERATOR
# Generates a list of (tag, tess_meshes) tuples over a parameter range.
# Plug each tess_meshes dict directly into your training loop.
# =============================================================================

def parametric_sweep(stl_paths, morph_fn, param_name, param_values,
                     fixed_kwargs=None, output_dir=None):
    """
    Generate morphed geometry sets by sweeping ONE parameter while keeping
    all others fixed.

    Parameters
    ----------
    stl_paths    : dict  — as in apply_morph_to_all_meshes
    morph_fn     : callable — morph function to sweep
    param_name   : str   — name of the parameter to sweep (kwarg name)
    param_values : list  — values to sweep over
    fixed_kwargs : dict  — other fixed kwargs for morph_fn
    output_dir   : str   — base dir; subdirs created per sweep value

    Returns
    -------
    sweep_results : list of (param_value, tess_meshes_dict)

    Example — sweep bulge_amount from 0 to 0.015 in 6 steps:
    -------
    results = parametric_sweep(
        stl_paths,
        morph_fn     = morph_radial_bulge,
        param_name   = "bulge_amount",
        param_values = np.linspace(0.0, 0.015, 6),
        fixed_kwargs = dict(
            sac_center=np.array([0.0, 0.0, 0.05]),
            sac_radius=0.08
        ),
        output_dir="./sweep_bulge"
    )
    for val, tess in results:
        print(f"bulge_amount = {val:.4f}, interior vertices ready")
    """
    fixed_kwargs = fixed_kwargs or {}
    sweep_results = []

    for val in param_values:
        kwargs = {**fixed_kwargs, param_name: val}
        tag    = f"{param_name}_{val:.5f}".replace(".", "p")
        sub_dir = os.path.join(output_dir, tag) if output_dir else None

        print(f"\n[sweep] {param_name} = {val:.5f}")
        _, tess_meshes = apply_morph_to_all_meshes(
            stl_paths, morph_fn, kwargs,
            output_dir=sub_dir, tag=tag
        )
        sweep_results.append((val, tess_meshes))

    return sweep_results


# =============================================================================
# SECTION 5: DROP-IN REPLACEMENT FOR YOUR run() FUNCTION
# Shows exactly how to wire morph into your existing aneurysm PINN code.
# Replace the STL loading + normalize_mesh block with this.
# =============================================================================

def load_morphed_aneurysm_geometry(
    stl_dir,
    morph_fn=None,
    morph_kwargs=None,
    center=(0.01151354, 0.03322, 0.0),   # YOUR modified center
    scale=12.753,                          # YOUR modified scale
):
    """
    Load, morph (optionally), normalize, and return all 5 Tessellation objects
    ready for use in PointwiseBoundaryConstraint / PointwiseInteriorConstraint.

    Parameters
    ----------
    stl_dir     : str  — path to folder containing the 5 STL files
    morph_fn    : callable or None — pass None to use unmodified geometry
    morph_kwargs: dict or None
    center      : tuple — YOUR geometry's normalization center
    scale       : float — YOUR geometry's normalization scale

    Returns
    -------
    dict with keys: inlet, outlet, noslip, integral, interior
    Each value is a normalized PhysicsNeMo Tessellation object.

    Usage in your run() function:
    ──────────────────────────────
    meshes = load_morphed_aneurysm_geometry(
        stl_dir    = to_absolute_path("./stl_files"),
        morph_fn   = morph_radial_bulge,
        morph_kwargs = dict(
            sac_center   = np.array([0.0, 0.0, 0.0]),   # adjust!
            sac_radius   = 0.08,
            bulge_amount = 0.005
        ),
    )
    inlet_mesh    = meshes["inlet"]
    outlet_mesh   = meshes["outlet"]
    noslip_mesh   = meshes["noslip"]
    integral_mesh = meshes["integral"]
    interior_mesh = meshes["interior"]
    """
    stl_paths = {
        "inlet"   : os.path.join(stl_dir, "aneurysm_inlet.stl"),
        "outlet"  : os.path.join(stl_dir, "aneurysm_outlet.stl"),
        "noslip"  : os.path.join(stl_dir, "aneurysm_noslip.stl"),
        "integral": os.path.join(stl_dir, "aneurysm_integral.stl"),
        "interior": os.path.join(stl_dir, "aneurysm_closed.stl"),
    }
    airtight_map = {
        "inlet": False, "outlet": False, "noslip": False,
        "integral": False, "interior": True,
    }

    result = {}
    for name, path in stl_paths.items():
        raw = stl_mesh.Mesh.from_file(path)

        # Apply morphing if requested
        if morph_fn is not None and morph_kwargs is not None:
            raw = morph_fn(raw, **morph_kwargs)

        # Build Tessellation
        tess = Tessellation(raw, airtight=airtight_map[name])

        # Normalize: translate then scale (matches your normalize_mesh logic)
        tess = tess.translate([-c for c in center])
        tess = tess.scale(scale)

        result[name] = tess

    return result


# =============================================================================
# SECTION 6: MESH QUALITY CHECK
# Always validate morphed meshes before training — a bad morph can cause
# inverted triangles or self-intersections that break the SDF sampling.
# =============================================================================

def check_mesh_quality(raw_mesh, name="mesh"):
    """
    Basic quality checks on a morphed numpy-stl mesh.
    Prints warnings if issues are found.

    Checks:
      - Any degenerate (zero-area) triangles
      - Any NaN or Inf vertices
      - Bounding box and centroid (sanity check)
    """
    verts = raw_mesh.vectors.reshape(-1, 3)
    normals = raw_mesh.normals
    areas = np.linalg.norm(normals, axis=1) / 2.0

    print(f"\n[Quality Check: {name}]")
    print(f"  Triangles   : {len(raw_mesh.vectors)}")
    print(f"  BBox min    : {verts.min(axis=0)}")
    print(f"  BBox max    : {verts.max(axis=0)}")
    print(f"  Centroid    : {verts.mean(axis=0)}")

    n_degenerate = np.sum(areas < 1e-15)
    if n_degenerate > 0:
        print(f"  ⚠ WARNING: {n_degenerate} degenerate (zero-area) triangles found!")
    else:
        print(f"  ✓ No degenerate triangles")

    if np.any(~np.isfinite(verts)):
        print(f"  ⚠ WARNING: NaN or Inf vertices detected!")
    else:
        print(f"  ✓ All vertices are finite")

    print(f"  Min area    : {areas.min():.4e}")
    print(f"  Max area    : {areas.max():.4e}")


# =============================================================================
# SECTION 7: QUICK-START EXAMPLE
# Run this file directly to test morphing on your geometry.
# =============================================================================

if __name__ == "__main__":
    import sys

    STL_DIR = "./stl_files"

    if not os.path.isdir(STL_DIR):
        print(f"STL directory '{STL_DIR}' not found.")
        print("Set STL_DIR to your stl_files folder and re-run.")
        sys.exit(0)

    # ── Example 1: Radial Bulge Sweep ─────────────────────────────────────
    print("=" * 60)
    print("Example: Radial Bulge Parametric Sweep")
    print("=" * 60)

    stl_paths = {
        "inlet"   : f"{STL_DIR}/aneurysm_inlet.stl",
        "outlet"  : f"{STL_DIR}/aneurysm_outlet.stl",
        "noslip"  : f"{STL_DIR}/aneurysm_noslip.stl",
        "integral": f"{STL_DIR}/aneurysm_integral.stl",
        "interior": f"{STL_DIR}/aneurysm_closed.stl",
    }

    # Sweep bulge_amount from 0 (baseline) to 0.01 in 4 steps
    results = parametric_sweep(
        stl_paths    = stl_paths,
        morph_fn     = morph_radial_bulge,
        param_name   = "bulge_amount",
        param_values = np.linspace(0.0, 0.01, 4),
        fixed_kwargs = dict(
            sac_center        = np.array([0.0, 0.0, 0.05]),  # ← adjust to your sac
            sac_radius        = 0.08,                          # ← adjust to your sac
            falloff_sharpness = 2.0,
        ),
        output_dir = "./morphed_stl_files"
    )

    print(f"\n✓ Generated {len(results)} geometry variants")
    for val, tess_dict in results:
        print(f"  bulge_amount={val:.4f} → meshes: {list(tess_dict.keys())}")

    # ── Quality check on the most-morphed variant ──────────────────────────
    most_morphed_val, most_morphed_tess = results[-1]
    raw_check = stl_mesh.Mesh.from_file(
        f"./morphed_stl_files/interior_bulge_amount_{most_morphed_val:.5f}"
        .replace(".", "p") + ".stl"
    )
    check_mesh_quality(raw_check, name=f"interior (bulge={most_morphed_val:.4f})")
