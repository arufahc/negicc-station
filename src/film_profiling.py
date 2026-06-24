#!/usr/bin/env python3
"""
Film Profiling Library — ICC profile generation from captured IT8 data.

Implements the full negative-film profiling pipeline:
1. Load a film profile JSON (patches, film_base, crosstalk matrix)
2. Combine with IT8 reference XYZ data
3. Compute crosstalk correction, TRC curves, and generate TI3 for ArgyllCMS
4. Build ICC profiles via colprof
5. Apply negative-to-positive correction to captured images

This module ports the logic from ../negicc/build_prof.py to work with the
JSON profile format produced by ui_film_profiling.py.
"""

import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
import re

import numpy as np
import pandas as pd
from scipy import interpolate
from sklearn.linear_model import LinearRegression

# Optional: colour-science for chromatic adaptation
try:
    import colour
    HAS_COLOUR = True
except ImportError:
    HAS_COLOUR = False


def parse_shutter_speed(shutter_str):
    """Parses a shutter speed string (e.g. '1/8s', '0.5s') into numerator and denominator."""
    s = shutter_str.rstrip('s')
    if '/' in s:
        parts = s.split('/')
        return int(parts[0]), int(parts[1])
    else:
        val = float(s)
        if val.is_integer():
            return int(val), 1
        else:
            return int(round(val * 10.0)), 10


# ---------------------------------------------------------------------------
# FilmProfile: Load and hold a film profile JSON
# ---------------------------------------------------------------------------

DEFAULT_NORMALIZATION_TARGET = 55000.0


class FilmProfile:
    """Represents a loaded film profile JSON with patch measurements and metadata."""

    def __init__(self, profile_json_path_or_dict):
        """Load profile JSON and extract all data.

        Parameters:
            profile_json_path_or_dict: Path to the film profile JSON file or a dict.
        """
        if isinstance(profile_json_path_or_dict, dict):
            data = profile_json_path_or_dict
            self.path = None
            basename = "Custom Profile"
        else:
            with open(profile_json_path_or_dict, 'r') as f:
                data = json.load(f)
            self.path = profile_json_path_or_dict
            basename = os.path.splitext(os.path.basename(profile_json_path_or_dict))[0]

        self.raw_data = data
        # Strip the "profile_" prefix and timestamp suffix
        parts = basename.split('_')
        if parts[0] == 'profile' and len(parts) >= 2:
            # e.g. "profile_Portra 400_20260622_205408" -> "Portra 400"
            # Find the timestamp portion (digits only) and keep everything before it
            name_parts = []
            for p in parts[1:]:
                if p.isdigit() and len(p) >= 8:
                    break
                name_parts.append(p)
            self.film_name = ' '.join(name_parts) if name_parts else basename
        else:
            self.film_name = basename

        # Camera name
        self.camera_name = data.get('camera_name', 'Unknown')

        # Crosstalk correction matrix (3x3)
        ct_profile = data.get('crosstalk_profile', {})
        cc_matrix = ct_profile.get('crosstalk_correction_matrix', None)
        if cc_matrix is not None:
            self.crosstalk_matrix = np.array(cc_matrix, dtype=np.float64)
        else:
            self.crosstalk_matrix = np.eye(3, dtype=np.float64)

        # Targets — use first target by default
        targets = data.get('targets', [])
        if not targets:
            raise ValueError("Profile JSON contains no targets.")
        target = targets[0]
        self.target_name = target.get('name', 'Target 1')
        self.target_iso = target.get('iso', 100)
        self.target_shutter = target.get('shutter', '1/8s')
        self.patches = target.get('patches', {})

        # Film base values
        fb = data.get('film_base', {})
        self.normalization_target = data.get('normalization_target', DEFAULT_NORMALIZATION_TARGET)
        self.film_base = {
            'r_avg': fb.get('r', {}).get('avg', 0.0),
            'g_avg': fb.get('g', {}).get('avg', 0.0),
            'b_avg': fb.get('b', {}).get('avg', 0.0),
        }
        self.film_base_iso = fb.get('iso', 100)
        self.film_base_shutter = fb.get('shutter', '1/8s')

        # Self-contained profile elements if present
        # icc_profile_bytes holds the raw ICC binary (decoded from base64 in JSON)
        icc_b64 = data.get('icc_profile_base64', None)
        # Also check first target for backward compatibility
        if icc_b64 is None and targets:
            icc_b64 = targets[0].get('icc_profile_base64', None)
        if icc_b64 is not None:
            import base64
            self.icc_profile_bytes = base64.b64decode(icc_b64)
        else:
            self.icc_profile_bytes = None

    def get_patch_rgb(self, patch_name):
        """Return (r, g, b) for a patch."""
        p = self.patches.get(patch_name, {})
        return p.get('r', 0.0), p.get('g', 0.0), p.get('b', 0.0)

    def get_film_base_rgb(self):
        """Return (r_avg, g_avg, b_avg) for the film base."""
        return (self.film_base['r_avg'],
                self.film_base['g_avg'],
                self.film_base['b_avg'])

    def build_training_dataframe(self, ref_xyz_path):
        """Create a pandas DataFrame matching negicc's build_prof.py format.

        Columns: patch, r, g, b, refR, refG, refB, refX, refY, refZ

        Parameters:
            ref_xyz_path: Path to the reference XYZ JSON file (data/R190808_ref.json).

        Returns:
            pd.DataFrame indexed by patch name.
        """
        # Load reference XYZ
        with open(ref_xyz_path, 'r') as f:
            ref_data = json.load(f)
        ref_patches = ref_data.get('patches', {})

        rows = []
        for patch_name in sorted(self.patches.keys()):
            p = self.patches[patch_name]
            ref = ref_patches.get(patch_name, {})

            rows.append({
                'patch': patch_name,
                'r': p.get('r', 0.0),
                'g': p.get('g', 0.0),
                'b': p.get('b', 0.0),
                'refR': 0,  # unused when XYZ is available
                'refG': 0,
                'refB': 0,
                'refX': ref.get('X', 0.0),
                'refY': ref.get('Y', 0.0),
                'refZ': ref.get('Z', 0.0),
            })

        df = pd.DataFrame(rows)
        df.set_index('patch', inplace=True)
        return df


# ---------------------------------------------------------------------------
# ICC Profile Building Pipeline
# ---------------------------------------------------------------------------

def _run_cmd(cmd, cwd=None, check=True, log_cb=None):
    """Run a subprocess command, capturing output and optional line-by-line log streaming."""
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=cwd,
        bufsize=1
    )
    
    stdout_lines = []
    while True:
        line = process.stdout.readline()
        if not line and process.poll() is not None:
            break
        if line:
            stripped = line.rstrip('\r\n')
            if log_cb:
                log_cb(stripped)
            stdout_lines.append(line)
            
    process.wait()
    full_output = "".join(stdout_lines)
    
    class ProcessResult:
        def __init__(self, returncode, stdout, stderr):
            self.returncode = returncode
            self.stdout = stdout
            self.stderr = stderr
            
    if check and process.returncode != 0:
        raise RuntimeError(
            f"Command failed: {' '.join(cmd)}\n"
            f"stdout: {full_output}\n"
            f"stderr: "
        )
    return ProcessResult(process.returncode, full_output, "")



def _chromatic_adaptation_to_d50(df):
    """Perform chromatic adaptation of reference XYZ values to D50.

    Normalizes to max value of 100 as expected by ArgyllCMS colprof.
    Results are written to norm_refX, norm_refY, norm_refZ columns.

    Returns:
        D50 white point XYZ.
    """
    D50_XYZ = np.array([0.9642, 1.0, 0.8249])

    unadapted_XYZ = np.array(df[['refX', 'refY', 'refZ']])

    # If colour-science is available and we have a test white,
    # perform chromatic adaptation. Otherwise just use as-is (assume D50).
    adapted_XYZ = unadapted_XYZ  # default: no adaptation needed for D50 ref

    # Normalize to max value of 100 (what colprof expects)
    max_val = adapted_XYZ.max()
    if max_val > 0:
        norm_XYZ = adapted_XYZ.T / max_val * 100
    else:
        norm_XYZ = adapted_XYZ.T

    df['norm_refX'] = norm_XYZ[0]
    df['norm_refY'] = norm_XYZ[1]
    df['norm_refZ'] = norm_XYZ[2]

    return D50_XYZ


def _estimate_crosstalk_correction(df, fit_intercept=True):
    """Estimate crosstalk correction coefficients from training data.

    Uses linear regression to fit r,g,b -> refR/refG/refB channels,
    then normalizes by primary signal.

    Returns:
        (r_coef, g_coef, b_coef) — each a 1x3 numpy array.
    """
    x = df[['r', 'g', 'b']]

    def estimate_coef(channel):
        reg = LinearRegression(fit_intercept=fit_intercept, copy_X=True)
        y = df[channel]
        reg.fit(x, y)
        return reg.coef_, reg.intercept_

    r_coef, _ = estimate_coef('norm_refX')
    g_coef, _ = estimate_coef('norm_refY')
    b_coef, _ = estimate_coef('norm_refZ')

    # Normalize by primary signal
    r_coef = r_coef / r_coef[0]
    g_coef = g_coef / g_coef[1]
    b_coef = b_coef / b_coef[2]

    return r_coef, g_coef, b_coef


def _compute_total_mse_in_gb(df, r_coef, g_coef, b_coef, gs_cell):
    """Compute total mean-square-error in G and B channels when scaling
    using a specific grayscale cell for color balance."""
    gs_cell_rgb = np.array([df['r'][gs_cell], df['g'][gs_cell], df['b'][gs_cell]])
    crosstalk_mat = np.array([r_coef, g_coef, b_coef])
    corrected_gs_cell_rgb = crosstalk_mat.dot(gs_cell_rgb)

    # Scale G and B to match R
    scaled_mat = np.array([
        r_coef,
        g_coef * corrected_gs_cell_rgb[0] / corrected_gs_cell_rgb[1],
        b_coef * corrected_gs_cell_rgb[0] / corrected_gs_cell_rgb[2]
    ])

    gs_indices = ['gs' + str(x) for x in range(24)]
    gs_indices = [idx for idx in gs_indices if idx in df.index]
    gs = df.loc[gs_indices]

    corrected_gs_rgb = np.matmul(
        gs[['r', 'g', 'b']], scaled_mat.T
    ).T.to_numpy()

    gr_ratio = corrected_gs_rgb[1] / corrected_gs_rgb[0]
    br_ratio = corrected_gs_rgb[2] / corrected_gs_rgb[0]
    gr_mse = np.mean(np.square(gr_ratio - np.ones(len(gr_ratio))))
    br_mse = np.mean(np.square(br_ratio - np.ones(len(br_ratio))))
    return gr_mse + br_mse


def _find_best_gs_cell(df, r_coef, g_coef, b_coef):
    """Find the grayscale patch that minimizes total GB MSE."""
    best_err = float('inf')
    best_cell = 'gs14'  # default fallback
    for i in range(24):
        cell = f'gs{i}'
        if cell not in df.index:
            continue
        try:
            err = _compute_total_mse_in_gb(df, r_coef, g_coef, b_coef, cell)
            if err < best_err:
                best_err = err
                best_cell = cell
        except (ZeroDivisionError, KeyError):
            continue
    return best_cell


def _estimate_trc_curves(corrected_gs_rgb, luminance, debug=False):
    """Estimate TRC curves using PchipInterpolator.

    Parameters:
        corrected_gs_rgb: 3xN array of corrected grayscale RGB values.
        luminance: 1D array of normalized luminance values.
        debug: If True, plots characteristic and TRC curves using matplotlib.

    Returns:
        (r_curve, g_curve, b_curve) — each a 4096-point array.
    """
    train_gs_luminance = np.append(np.insert(luminance.tolist(), 0, 1), 0)
    corrected_gs_r, corrected_gs_g, corrected_gs_b = corrected_gs_rgb

    # Extend with boundary points
    corrected_gs_r = np.append(np.insert(corrected_gs_r, 0, 0), 65535)
    corrected_gs_g = np.append(np.insert(corrected_gs_g, 0, 0), 65535)
    corrected_gs_b = np.append(np.insert(corrected_gs_b, 0, 0), 65535)

    # Fix non-strictly-increasing sequences
    def fix_strictly_increasing(d):
        for i in range(len(d) - 1):
            if d[i + 1] <= d[i]:
                d[i + 1] = d[i] + 1

    fix_strictly_increasing(corrected_gs_r)
    fix_strictly_increasing(corrected_gs_g)
    fix_strictly_increasing(corrected_gs_b)

    interp_r = interpolate.PchipInterpolator(corrected_gs_r, train_gs_luminance)
    interp_g = interpolate.PchipInterpolator(corrected_gs_g, train_gs_luminance)
    interp_b = interpolate.PchipInterpolator(corrected_gs_b, train_gs_luminance)

    xs = np.linspace(0, 65536, 4096)
    r_curve = np.clip(interp_r(xs), 0, 1)
    g_curve = np.clip(interp_g(xs), 0, 1)
    b_curve = np.clip(interp_b(xs), 0, 1)

    if debug:
        try:
            import matplotlib.pyplot as plt
            # Pick a o_min such that optical density of the lightest patch has OD=0.
            o_min = corrected_gs_rgb.max()
            r_density = np.vectorize(lambda x: math.log10(o_min / x) if x > 0 else 0.0)(corrected_gs_rgb[0])
            g_density = np.vectorize(lambda x: math.log10(o_min / x) if x > 0 else 0.0)(corrected_gs_rgb[1])
            b_density = np.vectorize(lambda x: math.log10(o_min / x) if x > 0 else 0.0)(corrected_gs_rgb[2])
            log_lx = np.vectorize(lambda x: math.log10(x / luminance.max()) if x > 0 else -10.0)(luminance)

            plt.figure("Measured Characteristic Curves")
            plt.plot(log_lx, r_density, 'rx-', label='R density over Log(lx)')
            plt.plot(log_lx, g_density, 'gx-', label='G density over Log(lx)')
            plt.plot(log_lx, b_density, 'bx-', label='B density over Log(lx)')
            plt.title('Measured Characteristic Curves')
            plt.xlabel('Log-Exposure (lx)')
            plt.ylabel('Density')
            plt.legend()
            plt.grid(True)
            plt.show()

            # Plot a graph to show the interpolated tone curves.
            plt.figure("Computed TRC Curves")
            plt.title('Computed TRC Curves')
            plt.plot(corrected_gs_r, train_gs_luminance, 'r*', xs, r_curve, 'r-')
            plt.plot(corrected_gs_g, train_gs_luminance, 'g*', xs, g_curve, 'g-')
            plt.plot(corrected_gs_b, train_gs_luminance, 'b*', xs, b_curve, 'b-')
            plt.xlabel('Input value')
            plt.ylabel('Output value')
            plt.grid(True)
            plt.show()
        except ImportError:
            print("Warning: matplotlib is required for plotting but could not be imported.")
        except Exception as e:
            print(f"Warning: Failed to plot graphs: {e}")

    return r_curve, g_curve, b_curve


def _write_ti3(filename, df):
    """Write a TI3 file for colprof from the DataFrame."""
    with open(filename, 'w') as f:
        f.write("""CTI3

DESCRIPTOR "Argyll Calibration Target chart information 3"
ORIGINATOR "negicc-station film_profiling"
DEVICE_CLASS "INPUT"
COLOR_REP "XYZ_RGB"

NUMBER_OF_FIELDS 7
BEGIN_DATA_FORMAT
SAMPLE_ID XYZ_X XYZ_Y XYZ_Z RGB_R RGB_G RGB_B
END_DATA_FORMAT

""")
        f.write(f"NUMBER_OF_SETS {len(df)}\n")
        f.write("BEGIN_DATA\n")
        for idx, row in df.iterrows():
            f.write(f"{idx} {row['norm_refX']:.6f} {row['norm_refY']:.6f} "
                    f"{row['norm_refZ']:.6f} {row['pos_r']:.6f} "
                    f"{row['pos_g']:.6f} {row['pos_b']:.6f}\n")
        f.write("END_DATA\n")


def build_icc_profile(profile, ref_xyz_path, output_dir,
                      mid_grey_patch='gs14',
                      whitest_patch_scaling=0.75,
                      mid_grey_scaling=10000,
                      progress_callback=None,
                      debug=False):
    """Full ICC profile build pipeline.

    Follows the same steps as negicc/build_prof.py.

    Parameters:
        profile: FilmProfile instance.
        ref_xyz_path: Path to IT8 reference XYZ JSON file.
        output_dir: Directory to write output ICC profiles.
        mid_grey_patch: Grayscale patch for pre-scaling (default 'gs14').
        whitest_patch_scaling: Scaling factor for whitest patch luminance.
        mid_grey_scaling: Target value for mid-grey patch after correction.
        progress_callback: Optional callable(step_name, detail_msg) for progress.

    Returns:
        dict with keys:
            'clut_icc_path', 'matrix_icc_path', 'profcheck_output',
            'correction_matrix', 'trc_curves', 'log_messages'
    """
    log_messages = []

    def log(msg):
        log_messages.append(msg)
        if progress_callback:
            progress_callback("Building", msg)

    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Build training DataFrame
    log("Step 1: Loading profile and reference data...")
    
    # 1. Compute exposure ratio between film base capture and target capture
    base_num, base_den = parse_shutter_speed(profile.film_base_shutter)
    t_base = base_num / base_den
    iso_base = profile.film_base_iso

    target_num, target_den = parse_shutter_speed(profile.target_shutter)
    t_target = target_num / target_den
    iso_target = profile.target_iso

    exposure_profile_fb = t_base * (iso_base / 100.0)
    exposure_target = t_target * (iso_target / 100.0)
    exposure_ratio = exposure_profile_fb / exposure_target if exposure_target > 0 else 1.0

    # 2. Scale factors to map film base (at target exposure) to normalization_target
    fb_r = profile.film_base['r_avg']
    fb_g = profile.film_base['g_avg']
    fb_b = profile.film_base['b_avg']

    target_val = profile.normalization_target
    scale_r = (target_val / fb_r) * exposure_ratio if fb_r > 0 else 1.0
    scale_g = (target_val / fb_g) * exposure_ratio if fb_g > 0 else 1.0
    scale_b = (target_val / fb_b) * exposure_ratio if fb_b > 0 else 1.0

    # 3. Create a temporary copy of profile where patches are scaled
    import copy
    profile_scaled = copy.deepcopy(profile)
    for p_name, p_val in profile_scaled.patches.items():
        profile_scaled.patches[p_name] = {
            'r': p_val.get('r', 0.0) * scale_r,
            'g': p_val.get('g', 0.0) * scale_g,
            'b': p_val.get('b', 0.0) * scale_b
        }

    df = profile_scaled.build_training_dataframe(ref_xyz_path)
    log(f"  Loaded {len(df)} patches from profile (scaled to {target_val} film base reference).")

    # Step 2: Chromatic adaptation
    log("Step 2: Performing chromatic adaptation to D50...")
    _chromatic_adaptation_to_d50(df)

    # Step 3: Bypass crosstalk correction estimation
    log("Step 3: Bypassing crosstalk correction (using identity matrix)...")
    r_coef = np.array([1.0, 0.0, 0.0])
    g_coef = np.array([0.0, 1.0, 0.0])
    b_coef = np.array([0.0, 0.0, 1.0])
    log(f"  R coef: {r_coef}")
    log(f"  G coef: {g_coef}")
    log(f"  B coef: {b_coef}")

    crosstalk_correction_mat = np.array([r_coef, g_coef, b_coef])

    # Step 4: Find optimal grayscale cell for color balance
    log("Step 4: Finding optimal grayscale cell for color balance...")
    color_balance_cell = _find_best_gs_cell(df, r_coef, g_coef, b_coef)
    log(f"  Selected color balance cell: {color_balance_cell}")

    
    '''
    # Scale G and B coefficients for color balance
    corrected_cb_rgb = crosstalk_correction_mat.dot(
        np.array([df['r'][color_balance_cell],
                  df['g'][color_balance_cell],
                  df['b'][color_balance_cell]])
    )
    crosstalk_correction_mat = np.array([
        r_coef,
        g_coef * corrected_cb_rgb[0] / corrected_cb_rgb[1],
        b_coef * corrected_cb_rgb[0] / corrected_cb_rgb[2]
    ])
    '''

    # Step 5: Estimate TRC curves
    log("Step 5: Estimating TRC curves from grayscale patches...")
    gs_indices = ['gs' + str(x) for x in range(24)]
    gs_indices = [idx for idx in gs_indices if idx in df.index]
    gs = df.loc[gs_indices]
    gs_rgb = np.array([gs['r'].tolist(), gs['g'].tolist(), gs['b'].tolist()])

    '''
    # Scale the correction matrix by mid-grey scaling
    if mid_grey_patch in df.index:
        mid_grey_rgb = df.loc[mid_grey_patch][['r', 'g', 'b']].to_numpy()
        avg_corrected_mid = np.average(crosstalk_correction_mat.dot(mid_grey_rgb))
        if avg_corrected_mid > 0:
            global_scale_factor = mid_grey_scaling / avg_corrected_mid
        else:
            global_scale_factor = 1.0
    else:
        global_scale_factor = 1.0

    log(f"  Scale correction matrix by: {global_scale_factor:.6f}")
    crosstalk_correction_mat *= global_scale_factor
    '''

    # Compute corrected grayscale values
    corrected_gs_rgb = np.matmul(crosstalk_correction_mat, gs_rgb)

    luminance = gs['refY'] / (gs['refY'].max() / whitest_patch_scaling)

    r_curve, g_curve, b_curve = _estimate_trc_curves(corrected_gs_rgb, luminance, debug=debug)
    log(f"  TRC curves computed ({len(r_curve)} points each).")

    # Step 6: Compute positive RGB values for all patches
    log("Step 6: Computing positive RGB values...")
    corrected_all_rgb = np.clip(
        np.matmul(crosstalk_correction_mat,
                  np.array([df['r'].tolist(), df['g'].tolist(), df['b'].tolist()])),
        0, 65535
    )

    # Apply TRC curves using interpolation
    r_interp = interpolate.interp1d(
        np.linspace(0, 65536, len(r_curve)), r_curve,
        kind='linear', bounds_error=False, fill_value=(r_curve[0], r_curve[-1])
    )
    g_interp = interpolate.interp1d(
        np.linspace(0, 65536, len(g_curve)), g_curve,
        kind='linear', bounds_error=False, fill_value=(g_curve[0], g_curve[-1])
    )
    b_interp = interpolate.interp1d(
        np.linspace(0, 65536, len(b_curve)), b_curve,
        kind='linear', bounds_error=False, fill_value=(b_curve[0], b_curve[-1])
    )

    pos_r = r_interp(corrected_all_rgb[0]) * 100
    pos_g = g_interp(corrected_all_rgb[1]) * 100
    pos_b = b_interp(corrected_all_rgb[2]) * 100

    df['pos_r'] = pos_r
    df['pos_g'] = pos_g
    df['pos_b'] = pos_b

    # Also store corrected (non-curved) values
    corrected_df = corrected_all_rgb / 65535 * 100
    df['corrected_r'] = corrected_df[0]
    df['corrected_g'] = corrected_df[1]
    df['corrected_b'] = corrected_df[2]

    # Step 7: Write TI3 and run colprof
    log("Step 7: Running ArgyllCMS colprof (cLUT profile)...")
    tmpdir = tempfile.mkdtemp(prefix="negicc_prof_")
    
    def run_and_log(cmd, cwd=None):
        log(f"$ {' '.join(cmd)}")
        return _run_cmd(cmd, cwd=cwd, log_cb=log)

    try:
        ti3_path = os.path.join(tmpdir, "build_prof")
        _write_ti3(ti3_path + ".ti3", df)

        # cLUT profile
        run_and_log([
            'colprof', '-v',
            '-ax',   # XYZ cLUT
            '-qu',   # high quality (45 grid points)
            '-kz',
            '-u',
            '-bn',   # no B2A profiles
            '-ni', '-np', '-no',  # linear input/output curves
            'build_prof'
        ], cwd=tmpdir)

        shutil.copy(os.path.join(tmpdir, "build_prof.icc"), os.path.join(tmpdir, "clut_raw.icc"))

        # Step 8: Matrix profile
        log("Step 8: Running ArgyllCMS colprof (matrix profile)...")
        run_and_log([
            'colprof', '-v',
            '-am',   # XYZ Matrix
            '-qh',
            '-kz',
            '-u',
            '-bn',
            '-ni', '-np', '-no',
            'build_prof'
        ], cwd=tmpdir)

        shutil.copy(os.path.join(tmpdir, "build_prof.icc"), os.path.join(tmpdir, "mat_raw.icc"))

        # Step 8.5: Write build_prof.h and run make_icc
        log("Step 8.5: Compiling and running make_icc to merge curves and crosstalk matrix...")
        header_path = os.path.join(tmpdir, "build_prof.h")
        with open(header_path, 'w') as f:
            f.write("double crosstalk_correction_mat[] = {\n")
            mat = crosstalk_correction_mat.flatten()
            f.write(", ".join(f" {v:.15f}" for v in mat))
            f.write("\n};\n\n")

            f.write(f"#define CURVE_POINTS {len(r_curve)}\n")

            def write_curve_to_file(name, curve):
                f.write(f"float {name}[CURVE_POINTS] = {{\n")
                f.write(", ".join(f" {v:.7f}" for v in curve))
                f.write("\n};\n\n")

            write_curve_to_file('b_curve', b_curve)
            write_curve_to_file('g_curve', g_curve)
            write_curve_to_file('r_curve', r_curve)

        make_icc_src_dir = os.path.dirname(os.path.abspath(__file__))
        make_icc_exe = os.path.join(tmpdir, "make_icc")
        run_and_log([
            'gcc', '-O3', f'-I{tmpdir}',
            'make_icc.c', '-o', make_icc_exe,
            '-llcms2'
        ], cwd=make_icc_src_dir)

        os.makedirs(os.path.join(tmpdir, "icc_out"), exist_ok=True)
        run_and_log([
            make_icc_exe,
            profile.film_name,
            "clut_raw.icc",
            "mat_raw.icc"
        ], cwd=tmpdir)

        clut_icc_path = os.path.join(output_dir, f"{profile.film_name} cLUT.icc")
        shutil.copy(os.path.join(tmpdir, "icc_out", f"{profile.film_name} cLUT.icc"), clut_icc_path)
        log(f"  cLUT profile written to: {os.path.basename(clut_icc_path)}")

        # Step 9: Profile check
        log("Step 9: Running profcheck...")
        # Write check TI3 with corrected (non-curved) RGB values
        check_ti3_path = os.path.join(tmpdir, "check_prof")
        # For profcheck, use corrected RGB values instead of positive
        df_check = df.copy()
        df_check['pos_r'] = df_check['corrected_r']
        df_check['pos_g'] = df_check['corrected_g']
        df_check['pos_b'] = df_check['corrected_b']
        _write_ti3(check_ti3_path + ".ti3", df_check)

        shutil.copy(clut_icc_path, os.path.join(tmpdir, "check.icc"))
        profcheck_result = run_and_log(
            ['profcheck', '-v2', '-k', 'check_prof.ti3', 'check.icc'],
            cwd=tmpdir
        )
        profcheck_output = profcheck_result.stdout
        # profcheck_output is already logged line-by-line via run_and_log

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    return {
        'clut_icc_path': clut_icc_path,
        'profcheck_output': profcheck_output,
        'correction_matrix': crosstalk_correction_mat,
        'trc_curves': (r_curve, g_curve, b_curve),
        'log_messages': log_messages,
    }


# ---------------------------------------------------------------------------
# Negative-to-Positive Image Processing
# ---------------------------------------------------------------------------

def apply_negative_correction(image_arr, profile, icc_profile_path=None):
    """Apply the full negative-to-positive pipeline to a captured image.

    Follows the same steps as negicc/neg_process.cc:
    1. Apply crosstalk correction matrix
    2. Divide by film base -> transmittance
    3. Negative inversion: -log10(T)
    4. Normalize density to 0..1 (density / 3.0)
    5. Optionally apply ICC profile via Pillow ImageCms

    Parameters:
        image_arr: np.ndarray (H, W, 3) float32 or uint16, linear RGB.
        profile: FilmProfile instance.
        icc_profile_path: Path to generated ICC profile (optional).
            If None, returns the density-mapped image without ICC correction.

    Returns:
        np.ndarray (H, W, 3) uint8, corrected positive image in sRGB.
    """
    h, w = image_arr.shape[:2]
    arr = image_arr.astype(np.float32)

    # Step 1: Crosstalk correction
    corr_mat = profile.crosstalk_matrix.astype(np.float64)
    flat = arr.reshape(-1, 3).astype(np.float64)
    corrected = flat @ corr_mat.T
    corrected = corrected.reshape(h, w, 3).astype(np.float32)

    # Step 2 & 3: Transmittance and negative inversion
    base_r, base_g, base_b = profile.get_film_base_rgb()
    base = np.array([base_r, base_g, base_b], dtype=np.float32)

    # Transmittance = pixel / film_base
    transmittance = corrected / base[np.newaxis, np.newaxis, :]
    transmittance = np.clip(transmittance, 1e-6, None)

    # Density = -log10(T)
    density = -np.log10(transmittance)

    # Step 4: Normalize density to 0..1 (typical range 0..3)
    normalized = np.clip(density / 3.0, 0.0, 1.0)

    # Step 5: ICC profile correction
    if icc_profile_path and os.path.exists(icc_profile_path):
        try:
            from PIL import Image, ImageCms
            # Convert float 0..1 to uint16 for ICC transform
            img_u16 = (normalized * 65535).astype(np.uint16)
            pil_img = Image.fromarray(img_u16, mode='RGB')

            input_profile = ImageCms.getOpenProfile(icc_profile_path)
            srgb_profile = ImageCms.createProfile('sRGB')
            transform = ImageCms.buildTransform(
                input_profile, srgb_profile,
                'RGB', 'RGB',
                renderingIntent=ImageCms.Intent.PERCEPTUAL
            )
            corrected_img = ImageCms.applyTransform(pil_img, transform)
            result = np.array(corrected_img, dtype=np.uint8)
        except ImportError:
            # Fallback: convert to 8-bit without ICC
            result = (normalized * 255).astype(np.uint8)
        except Exception as e:
            print(f"ICC transform failed: {e}, falling back to simple conversion")
            result = (normalized * 255).astype(np.uint8)
    else:
        # No ICC profile: simple 0..1 -> 0..255 mapping
        result = (normalized * 255).astype(np.uint8)

    return result


def download_and_parse_reference_file(url_or_path, cache_dir, prompt_zip_callback=None):
    """
    Downloads the file if it's a URL, or reads it from a local path,
    caches it. If it is a zip file, lists the files inside and uses prompt_zip_callback
    to ask the user which file to use (or selects a default if none provided/only one file).
    If it is a .txt/.it8 file, parses it directly.
    Returns: (patches_dict, loaded_filename)
    """
    if url_or_path.startswith(('http://', 'https://')):
        import hashlib
        url_hash = hashlib.sha256(url_or_path.encode('utf-8')).hexdigest()
        tmp_cache_dir = os.path.join("/tmp", f"negicc_ref_cache_{url_hash}")
        os.makedirs(tmp_cache_dir, exist_ok=True)
        filename = os.path.basename(url_or_path)
        cache_path = os.path.join(tmp_cache_dir, filename)
        
        # Download if not already cached in /tmp
        if not os.path.exists(cache_path):
            print(f"Downloading reference from {url_or_path} to cache: {cache_path}...")
            urllib.request.urlretrieve(url_or_path, cache_path)
        else:
            print(f"Using cached reference file from: {cache_path}")
        local_path = cache_path
        reference_dir = tmp_cache_dir
    else:
        local_path = url_or_path
        reference_dir = os.path.dirname(os.path.abspath(url_or_path))

    if not os.path.exists(local_path):
        raise FileNotFoundError(f"Reference file not found: {local_path}")

    # Check if zip file
    is_zip = zipfile.is_zipfile(local_path)
    
    if is_zip:
        with zipfile.ZipFile(local_path, 'r') as z:
            ref_filenames = [name for name in z.namelist() if name.lower().endswith(('.txt', '.it8'))]
            if not ref_filenames:
                raise ValueError("No .txt or .it8 files found in the ZIP archive.")
            
            selected_file = None
            if len(ref_filenames) == 1:
                selected_file = ref_filenames[0]
            else:
                # If there are multiple files, use callback if provided, otherwise fallback to heuristics
                if prompt_zip_callback:
                    selected_file = prompt_zip_callback(ref_filenames)
                
                if not selected_file:
                    # Fallback heuristics
                    zip_basename = os.path.splitext(os.path.basename(url_or_path))[0]
                    # 1. Prefer file matching base name and not in Extras
                    for name in ref_filenames:
                        if "extras" not in name.lower() and "macosx" not in name.lower():
                            base_name_in_zip = os.path.splitext(os.path.basename(name))[0]
                            if base_name_in_zip.lower() == zip_basename.lower():
                                selected_file = name
                                break
                    # 2. Prefer any file not in Extras
                    if not selected_file:
                        for name in ref_filenames:
                            if "extras" not in name.lower() and "liesmich" not in name.lower() and "readme" not in name.lower():
                                selected_file = name
                                break
                    # 3. Fallback to first found
                    if not selected_file:
                        selected_file = ref_filenames[0]
            
            print(f"Selected reference file from ZIP: {selected_file}")
            ref_content = z.read(selected_file).decode('utf-8', errors='ignore')
            loaded_filename = os.path.basename(selected_file)
    else:
        with open(local_path, 'r', encoding='utf-8', errors='ignore') as f:
            ref_content = f.read()
        loaded_filename = os.path.basename(local_path)

    lines = ref_content.splitlines()
    
    begin_data_idx = -1
    fields = []
    for idx, line in enumerate(lines):
        line = line.strip()
        if line == "BEGIN_DATA":
            begin_data_idx = idx
            break
        if line.startswith("SAMPLE_ID"):
            fields = re.split(r'\s+', line)
    
    if begin_data_idx == -1:
        raise ValueError("Invalid format: BEGIN_DATA section not found in target file.")
        
    if not fields:
        for idx in range(begin_data_idx - 1, -1, -1):
            line = lines[idx].strip()
            if "SAMPLE_ID" in line or "XYZ_X" in line:
                fields = re.split(r'\s+', line)
                break
    
    if not fields:
        raise ValueError("Could not find column headers (e.g. SAMPLE_ID, XYZ_X).")
        
    col_map = {}
    for i, col in enumerate(fields):
        c_upper = col.upper()
        if c_upper in ('SAMPLE_ID', 'PATCH'):
            col_map['patch'] = i
        elif c_upper in ('XYZ_X', 'REFX', 'X'):
            col_map['X'] = i
        elif c_upper in ('XYZ_Y', 'REFY', 'Y'):
            col_map['Y'] = i
        elif c_upper in ('XYZ_Z', 'REFZ', 'Z'):
            col_map['Z'] = i
            
    if 'patch' not in col_map or 'X' not in col_map or 'Y' not in col_map or 'Z' not in col_map:
        raise ValueError(f"Could not map all required columns. Found fields: {fields}")
        
    patches = {}
    for idx in range(begin_data_idx + 1, len(lines)):
        line = lines[idx].strip()
        if not line or line.startswith("END_DATA"):
            continue
        parts = re.split(r'\s+', line)
        if len(parts) <= max(col_map.values()):
            continue
        
        patch_name = parts[col_map['patch']].lower()
        try:
            x_val = float(parts[col_map['X']])
            y_val = float(parts[col_map['Y']])
            z_val = float(parts[col_map['Z']])
            patches[patch_name] = {'X': x_val, 'Y': y_val, 'Z': z_val}
        except ValueError:
            continue
            
    if not patches:
        raise ValueError("No valid patch data parsed.")
        
    return patches, loaded_filename, reference_dir


def convert_raw_image(img, profile, clut_path=None, shutter_str=None, exposure_comp=1.0, half=True, film_base_rgb=None, film_base_img=None, pipeline="cpp"):
    """Converts RAW image to positive sRGB using the C++ backend and row-wise film base scaling."""
    if pipeline == "python":
        import color_conversion
        return color_conversion.convert_raw_to_tiff( # wait, to_numpy vs convert_raw_to_tiff?
            # actually convert_raw_image was returning numpy array, but convert_raw_to_tiff saves file.
            # let's route to color_conversion or standard flow
            img=img, profile=profile, output_path="", colorspace="srgb", clut_path=clut_path,
            shutter_str=shutter_str, exposure_comp=exposure_comp, half=half,
            film_base_rgb=film_base_rgb, film_base_img=film_base_img
        )
    # Resolve ICC data: prefer in-memory bytes, then fall back to clut_path file
    icc_bytes = None
    if clut_path is None:
        if getattr(profile, 'icc_profile_bytes', None):
            icc_bytes = profile.icc_profile_bytes
        else:
            raise ValueError("clut_path is None and profile has no self-contained ICC profile bytes.")

    # 1. Determine scanned film base (crosstalk-corrected)
    if film_base_rgb is not None:
        fb_r, fb_g, fb_b = film_base_rgb
    else:
        fb_r = profile.film_base['r_avg']
        fb_g = profile.film_base['g_avg']
        fb_b = profile.film_base['b_avg']

    # 2. Compute exposure ratio
    if shutter_str is not None:
        scan_num, scan_den = parse_shutter_speed(shutter_str)
        t_scan = scan_num / scan_den
    else:
        t_scan = img.shutter_speed

    # Shutter speed and ISO of film base (use film_base_img if provided, otherwise fallback to profile metadata)
    if film_base_img is not None:
        t_base = film_base_img.shutter_speed
        iso_base = film_base_img.iso
    else:
        base_num, base_den = parse_shutter_speed(profile.film_base_shutter)
        t_base = base_num / base_den
        iso_base = profile.film_base_iso

    iso_scan = img.iso

    # Exposure: t * ISO
    exposure_profile = t_base * (iso_base / 100.0)
    exposure_scan = t_scan * (iso_scan / 100.0)
    exposure_ratio = exposure_profile / exposure_scan if exposure_scan > 0 else 1.0

    # Scale factors to map film base at current exposure to normalization_target
    target_val = profile.normalization_target
    scale_r = (target_val / fb_r) * exposure_ratio if fb_r > 0 else 1.0
    scale_g = (target_val / fb_g) * exposure_ratio if fb_g > 0 else 1.0
    scale_b = (target_val / fb_b) * exposure_ratio if fb_b > 0 else 1.0

    # Merge normalization scale factors into crosstalk matrix on the fly (row-wise!)
    # This applies scaling *after* crosstalk correction: diag(scales) * M
    raw_crosstalk = np.array(profile.crosstalk_matrix)
    scales = np.array([scale_r, scale_g, scale_b])
    merged_matrix = raw_crosstalk * scales[:, np.newaxis]
    flat_merged_matrix = merged_matrix.flatten().tolist()

    kwargs = dict(
        half=half,
        crosstalk_matrix=flat_merged_matrix,
        output_profile_path="srgb",
        profile_film_base=None,
        film_base=None,
        exposure_comp=exposure_comp,
        pipeline=pipeline
    )
    if icc_bytes is not None:
        kwargs['it8_profile_bytes'] = icc_bytes
    else:
        kwargs['it8_profile_path'] = clut_path

    return img.to_numpy(**kwargs)


def convert_raw_to_tiff(img, profile, output_path, colorspace="srgb", clut_path=None, shutter_str=None, exposure_comp=1.0, half=True, film_base_rgb=None, film_base_img=None, pipeline="cpp"):
    """Converts RAW image and saves directly to TIFF in C++ or CUDA without NumPy image copy."""
    if pipeline == "python":
        import color_conversion
        return color_conversion.convert_raw_to_tiff(
            img=img, profile=profile, output_path=output_path, colorspace=colorspace,
            clut_path=clut_path, shutter_str=shutter_str, exposure_comp=exposure_comp,
            half=half, film_base_rgb=film_base_rgb, film_base_img=film_base_img
        )

    # Resolve ICC data: prefer in-memory bytes, then fall back to clut_path file
    icc_bytes = None
    if clut_path is None:
        if getattr(profile, 'icc_profile_bytes', None):
            icc_bytes = profile.icc_profile_bytes
        else:
            raise ValueError("clut_path is None and profile has no self-contained ICC profile bytes.")

    # 1. Determine scanned film base (crosstalk-corrected)
    if film_base_rgb is not None:
        fb_r, fb_g, fb_b = film_base_rgb
    else:
        fb_r = profile.film_base['r_avg']
        fb_g = profile.film_base['g_avg']
        fb_b = profile.film_base['b_avg']

    # 2. Compute exposure ratio
    if shutter_str is not None:
        scan_num, scan_den = parse_shutter_speed(shutter_str)
        t_scan = scan_num / scan_den
    else:
        t_scan = img.shutter_speed

    # Shutter speed and ISO of film base
    if film_base_img is not None:
        t_base = film_base_img.shutter_speed
        iso_base = film_base_img.iso
    else:
        base_num, base_den = parse_shutter_speed(profile.film_base_shutter)
        t_base = base_num / base_den
        iso_base = profile.film_base_iso

    iso_scan = img.iso

    # Exposure: t * ISO
    exposure_profile = t_base * (iso_base / 100.0)
    exposure_scan = t_scan * (iso_scan / 100.0)
    exposure_ratio = exposure_profile / exposure_scan if exposure_scan > 0 else 1.0

    # Scale factors to map film base at current exposure to normalization_target
    target_val = profile.normalization_target
    scale_r = (target_val / fb_r) * exposure_ratio if fb_r > 0 else 1.0
    scale_g = (target_val / fb_g) * exposure_ratio if fb_g > 0 else 1.0
    scale_b = (target_val / fb_b) * exposure_ratio if fb_b > 0 else 1.0

    # Merge normalization scale factors into crosstalk matrix on the fly
    raw_crosstalk = np.array(profile.crosstalk_matrix)
    scales = np.array([scale_r, scale_g, scale_b])
    merged_matrix = raw_crosstalk * scales[:, np.newaxis]
    flat_merged_matrix = merged_matrix.flatten().tolist()

    kwargs = dict(
        output_path=output_path,
        half=half,
        crosstalk_matrix=flat_merged_matrix,
        output_profile_path=colorspace,
        profile_film_base=None,
        film_base=None,
        exposure_comp=exposure_comp,
        pipeline=pipeline
    )
    if icc_bytes is not None:
        kwargs['it8_profile_bytes'] = icc_bytes
    else:
        kwargs['it8_profile_path'] = clut_path

    return img.write_tiff(**kwargs)
