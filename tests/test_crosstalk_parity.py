import os
import sys
import subprocess
import numpy as np

def main():
    print("========================================")
    print("STARTING CROSSTALK PARITY TEST")
    print("========================================")

    project_dir = os.path.abspath(os.path.dirname(__file__))
    parent_dir = os.path.abspath(os.path.join(project_dir, ".."))
    sys.path.insert(0, os.path.join(parent_dir, "src"))

    # 1. Import Python modules
    print("Importing negicc_station CPython module and crosstalk_calibration...")
    try:
        import negicc_station
        import crosstalk_calibration
    except ImportError as e:
        print(f"ERROR: Failed to import required modules: {e}")
        sys.exit(1)

    # 2. Load the checked-in crosstalk profile
    profile_path = os.path.join(parent_dir, "profiles/ILCE-7RM4_crosstalk_profile.json")
    if not os.path.exists(profile_path):
        print(f"ERROR: Crosstalk profile not found at {profile_path}")
        sys.exit(1)

    print(f"Loading crosstalk profile: {profile_path}")
    calib = crosstalk_calibration.CrosstalkCalibration.load(profile_path)
    flat_matrix = calib.M_corr.flatten().tolist()

    # 3. Decompress the checked-in reference ARW file for offline testing
    raw_file = os.path.join(parent_dir, "test_imgs/test_capture_ref.ARW")
    compressed_file = raw_file + ".xz"
    
    decompressed_locally = False
    if not os.path.exists(raw_file):
        if os.path.exists(compressed_file):
            print(f"[*] Decompressing {compressed_file}...")
            subprocess.run(["xz", "-d", "-k", compressed_file], check=True)
            decompressed_locally = True
        else:
            print(f"ERROR: Reference ARW file not found at {raw_file} or {compressed_file}")
            sys.exit(1)

    # 4. Initialize CapturedImage
    print(f"Initializing CapturedImage offline with {raw_file}...")
    try:
        img = negicc_station.CapturedImage(
            type=0,  # CAPTURE_SINGLE
            shutter_speed=0.125,
            iso=100,
            filepaths=[raw_file]
        )
    except Exception as e:
        print(f"ERROR: Instantiation failed: {e}")
        if decompressed_locally and os.path.exists(raw_file):
            os.remove(raw_file)
        sys.exit(1)

    try:
        # 5. Extract uncorrected raw pixels via Python
        print("Extracting uncorrected RAW image into NumPy...")
        arr_raw = img.to_numpy(half=False, crosstalk_matrix=None)
        
        # 6. Apply correction in Python via NumPy/crosstalk_calibration library
        print("Applying crosstalk correction in Python...")
        arr_cc_py = calib.apply(arr_raw)

        # 7. Load corrected RAW from C++ backend directly
        print("Applying crosstalk correction in C++ backend...")
        arr_cc_cpp = img.to_numpy(half=False, crosstalk_matrix=flat_matrix)

        # 8. Verify Parity
        print("Comparing Python NumPy results with C++ backend results...")
        
        # Check shapes
        assert arr_cc_py.shape == arr_cc_cpp.shape, f"Shape mismatch: {arr_cc_py.shape} vs {arr_cc_cpp.shape}"
        assert arr_cc_py.dtype == arr_cc_cpp.dtype, f"Dtype mismatch: {arr_cc_py.dtype} vs {arr_cc_cpp.dtype}"
        
        # Compute pixel-wise absolute difference
        diff = np.abs(arr_cc_py.astype(np.int32) - arr_cc_cpp.astype(np.int32))
        max_diff = np.max(diff)
        
        # Because of floating point precision limits (C++ uses float, NumPy uses float64 by default)
        # and clipping/rounding (+0.5f), a 1 LSB rounding discrepancy on rare edge cases is acceptable,
        # but the vast majority of pixels must match. Let's assert max_diff <= 1 and mean_diff is extremely close to 0.
        mean_diff = np.mean(diff)
        
        print(f"  Maximum pixel-wise difference: {max_diff} LSB")
        print(f"  Mean pixel-wise difference:    {mean_diff:.6f} LSB")
        
        assert max_diff <= 1, f"ERROR: Maximum difference ({max_diff} LSB) exceeds the 1 LSB rounding limit!"
        assert mean_diff < 0.01, f"ERROR: Mean difference ({mean_diff:.6f} LSB) is too high!"
        
        print("  [PASS] Python and C++ crosstalk correction parity verified successfully!")

    finally:
        # Clean up
        img.discard()
        if decompressed_locally and os.path.exists(raw_file):
            print(f"[*] Removing decompressed reference RAW file: {raw_file}")
            os.remove(raw_file)

    print("\n========================================")
    print("CROSSTALK PARITY TEST PASSED: SUCCESS")
    print("========================================")

if __name__ == "__main__":
    main()
