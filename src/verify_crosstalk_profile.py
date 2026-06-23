import os
import sys
import json
import numpy as np

# Ensure the project src directory is in path
project_dir = os.path.dirname(os.path.abspath(__file__))
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)

import negicc_station
import crosstalk_calibration

def main():
    profile_path = os.path.join(project_dir, "../profiles/ILCE-7RM4_crosstalk_profile.json")
    calib = crosstalk_calibration.CrosstalkCalibration.load(profile_path)
    
    matrix = calib.M_corr.tolist()
    matrix_np = calib.M_corr
    flat_matrix = calib.M_corr.flatten().tolist()
    
    print("=== Loaded Profile Correction Matrix ===")
    for row in matrix:
        print(f"  [ {row[0]:.6f}, {row[1]:.6f}, {row[2]:.6f} ]")
    print("========================================")

    for name in ["red", "green", "blue"]:
        filename = f"{name}.ARW"
        filepath = os.path.join(project_dir, "../", filename)
        if not os.path.exists(filepath):
            print(f"Error: {filename} not found at {filepath}")
            continue

        print(f"\nProcessing {filename}...")
        img = negicc_station.CapturedImage(
            type=0,  # CAPTURE_SINGLE
            shutter_speed=1.0,
            iso=100,
            filepaths=[filepath]
        )

        # 1. Load RAW to NumPy (uncorrected, full resolution)
        arr_raw = img.to_numpy(half=False, crosstalk_matrix=None)
        H, W, C = arr_raw.shape
        cy, cx = H / 2.0, W / 2.0
        S = min(H, W)
        r_circle = S / 6.0

        # Mask circular center area (1/3 of shorter side)
        y, x = np.ogrid[:H, :W]
        mask = (x - cx)**2 + (y - cy)**2 <= r_circle**2

        # Extract raw circle pixels and calculate mean
        pixels_raw = arr_raw[mask]
        mean_raw = np.mean(pixels_raw, axis=0)

        # 2. Apply correction in Python via NumPy dot product and clamp (matching C++ + 0.5f rounding)
        pixels_cc_py = calib.apply(pixels_raw)
        mean_cc_py = np.mean(pixels_cc_py, axis=0)

        # 3. Load corrected RAW from C++ backend directly
        arr_cc_cpp = img.to_numpy(half=False, crosstalk_matrix=flat_matrix)
        pixels_cc_cpp = arr_cc_cpp[mask]
        mean_cc_cpp = np.mean(pixels_cc_cpp, axis=0)

        print(f"Resolution: {W}x{H}, Circle radius: {r_circle:.1f} pixels")
        print(f"  Raw RGB Mean:     R={mean_raw[0]:.4f}, G={mean_raw[1]:.4f}, B={mean_raw[2]:.4f}")
        print(f"  CC Mean (Python): R={mean_cc_py[0]:.4f}, G={mean_cc_py[1]:.4f}, B={mean_cc_py[2]:.4f}")
        print(f"  CC Mean (C++):    R={mean_cc_cpp[0]:.4f}, G={mean_cc_cpp[1]:.4f}, B={mean_cc_cpp[2]:.4f}")
        
        # Check max difference between Python NumPy and C++ backend results
        diff = np.abs(mean_cc_py.astype(np.float32) - mean_cc_cpp.astype(np.float32))
        max_diff = np.max(diff)
        print(f"  Parity Max Diff:  {max_diff:.6f} LSB")
        
        img.discard()

if __name__ == "__main__":
    main()
