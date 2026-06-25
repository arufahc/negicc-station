#!/usr/bin/env python3
import os
import sys
import gc
import subprocess
import numpy as np

def compute_diff(arr1, arr2):
    diff = np.abs(arr1.astype(np.int32) - arr2.astype(np.int32))
    return np.max(diff), np.mean(diff)

def main():
    print("==================================================")
    print("STARTING COMPREHENSIVE PIPELINE PARITY TEST SUITE")
    print("==================================================")

    project_dir = os.path.abspath(os.path.dirname(__file__))
    parent_dir = os.path.abspath(os.path.join(project_dir, ".."))
    sys.path.insert(0, os.path.join(parent_dir, "src"))

    # 1. Import required modules
    try:
        import negicc_station
        import film_profiling
        import crosstalk_calibration
        from film_profiling import FilmProfile
    except ImportError as e:
        print(f"ERROR: Failed to import required modules: {e}")
        sys.exit(1)

    # 2. Paths to profile files
    profile_path = os.path.join(parent_dir, "profiles/profile_Portra400_20260623_170610.json")
    crosstalk_path = os.path.join(parent_dir, "profiles/ILCE-7RM4_crosstalk_profile.json")
    
    if not os.path.exists(profile_path):
        print(f"ERROR: Film profile not found at {profile_path}")
        sys.exit(1)
    if not os.path.exists(crosstalk_path):
        print(f"ERROR: Crosstalk profile not found at {crosstalk_path}")
        sys.exit(1)

    print("Loading film profile and crosstalk profile...")
    profile = FilmProfile(profile_path)
    # Load ICC profile bytes if needed
    icc_path = os.path.join(parent_dir, "profiles/profile_Portra400_20260623_170610.icc")
    if not getattr(profile, 'icc_profile_bytes', None):
        if os.path.exists(icc_path):
            with open(icc_path, 'rb') as f:
                profile.icc_profile_bytes = f.read()
        else:
            for root, dirs, files in os.walk(os.path.join(parent_dir, "profiles")):
                for f in files:
                    if f.endswith(".icc"):
                        with open(os.path.join(root, f), 'rb') as f_icc:
                            profile.icc_profile_bytes = f_icc.read()
                        break
                if profile.icc_profile_bytes:
                    break

    calib = crosstalk_calibration.CrosstalkCalibration.load(crosstalk_path)
    flat_crosstalk = calib.M_corr.flatten().tolist()

    # 3. Decompress reference RAW file
    raw_file = os.path.join(parent_dir, "test_imgs/test_capture_ref.ARW")
    compressed_file = raw_file + ".xz"
    decompressed_locally = False
    
    if not os.path.exists(raw_file):
        if os.path.exists(compressed_file):
            print(f"[*] Decompressing {compressed_file}...")
            subprocess.run(["xz", "-d", "-k", compressed_file], check=True)
            decompressed_locally = True
        else:
            print(f"ERROR: Reference ARW file not found at {raw_file}")
            sys.exit(1)

    print("Initializing CapturedImage...")
    img = negicc_station.CapturedImage(
        type=0,  # CAPTURE_SINGLE
        shutter_speed=0.125,
        iso=100,
        filepaths=[raw_file]
    )

    tests_failed = 0

    try:
        # ==================================================
        # COMBINATION 1: LINEAR RAW ONLY (NO CROSSTALK, NO ICC)
        # ==================================================
        print("\n--- COMBINATION 1: Linear RAW Only ---")
        
        # 1) Python
        arr_c1_py = img.to_numpy(half=True, crosstalk_matrix=None)
        
        # 2) C++ CPU
        arr_c1_cpp = img.to_numpy(half=True, crosstalk_matrix=None, pipeline="cpp")
        
        # 3) CUDA
        arr_c1_cuda = img.to_numpy(half=True, crosstalk_matrix=None, pipeline="cuda")
        
        # 4) CUDA with Fallback
        os.environ["FORCE_CUDA_FALLBACK"] = "1"
        arr_c1_fallback = img.to_numpy(half=True, crosstalk_matrix=None, pipeline="cuda")
        if "FORCE_CUDA_FALLBACK" in os.environ:
            del os.environ["FORCE_CUDA_FALLBACK"]

        # Assertions: All must match exactly
        for name, arr in [("C++ CPU", arr_c1_cpp), ("CUDA", arr_c1_cuda), ("CUDA Fallback", arr_c1_fallback)]:
            max_d, mean_d = compute_diff(arr_c1_py, arr)
            print(f"  Python vs {name}: Max Diff = {max_d} LSB, Mean Diff = {mean_d:.6f} LSB")
            if max_d != 0:
                print(f"  [FAIL] Linear RAW mismatch between Python and {name}!")
                tests_failed += 1
            else:
                print(f"  [PASS] Python vs {name} match exactly.")

        # Keep Python raw array for next combination, clean up rest
        del arr_c1_cpp, arr_c1_cuda, arr_c1_fallback
        gc.collect()

        # ==================================================
        # COMBINATION 2: LINEAR RAW WITH CROSSTALK CORRECTION ONLY
        # ==================================================
        print("\n--- COMBINATION 2: Linear RAW + Crosstalk Only ---")
        
        # 1) Python
        arr_c2_py = calib.apply(arr_c1_py)
        
        # 2) C++ CPU
        arr_c2_cpp = img.to_numpy(half=True, crosstalk_matrix=flat_crosstalk, pipeline="cpp")
        
        # 3) CUDA
        arr_c2_cuda = img.to_numpy(half=True, crosstalk_matrix=flat_crosstalk, pipeline="cuda")
        
        # 4) CUDA with Fallback
        os.environ["FORCE_CUDA_FALLBACK"] = "1"
        arr_c2_fallback = img.to_numpy(half=True, crosstalk_matrix=flat_crosstalk, pipeline="cuda")
        if "FORCE_CUDA_FALLBACK" in os.environ:
            del os.environ["FORCE_CUDA_FALLBACK"]

        # Assertions:
        max_d_py_cpp, mean_d_py_cpp = compute_diff(arr_c2_py, arr_c2_cpp)
        print(f"  Python vs C++ CPU: Max Diff = {max_d_py_cpp} LSB, Mean Diff = {mean_d_py_cpp:.6f} LSB")
        if max_d_py_cpp > 1:
            print(f"  [FAIL] Python vs C++ CPU crosstalk discrepancy exceeds 1 LSB limit!")
            tests_failed += 1
        else:
            print("  [PASS] Python vs C++ CPU crosstalk correction parity verified.")

        for name, arr in [("CUDA", arr_c2_cuda), ("CUDA Fallback", arr_c2_fallback)]:
            max_d, mean_d = compute_diff(arr_c2_cpp, arr)
            print(f"  C++ CPU vs {name}: Max Diff = {max_d} LSB, Mean Diff = {mean_d:.6f} LSB")
            if max_d != 0:
                print(f"  [FAIL] Crosstalk mismatch between C++ CPU and {name}!")
                tests_failed += 1
            else:
                print(f"  [PASS] C++ CPU vs {name} match exactly.")

        del arr_c1_py, arr_c2_py, arr_c2_cpp, arr_c2_cuda, arr_c2_fallback
        gc.collect()

        # ==================================================
        # COMBINATION 3: LINEAR RAW WITH ICC TARGET CONVERSION
        # ==================================================
        print("\n--- COMBINATION 3: Linear RAW + Crosstalk + ICC Target Conversion ---")
        
        # 1) Python
        arr_c3_py = film_profiling.convert_raw_image(img, profile, half=True, pipeline="python")
        
        # 2) C++ CPU
        arr_c3_cpp = film_profiling.convert_raw_image(img, profile, half=True, pipeline="cpp")
        
        # 3) CUDA
        arr_c3_cuda = film_profiling.convert_raw_image(img, profile, half=True, pipeline="cuda")
        
        # 4) CUDA with Fallback
        os.environ["FORCE_CUDA_FALLBACK"] = "1"
        arr_c3_fallback = film_profiling.convert_raw_image(img, profile, half=True, pipeline="cuda")
        if "FORCE_CUDA_FALLBACK" in os.environ:
            del os.environ["FORCE_CUDA_FALLBACK"]

        # Assertions
        max_d_py_cuda, mean_d_py_cuda = compute_diff(arr_c3_py, arr_c3_cuda)
        print(f"  Python vs CUDA: Max Diff = {max_d_py_cuda} LSB, Mean Diff = {mean_d_py_cuda:.6f} LSB")
        if max_d_py_cuda > 1:
            print(f"  [FAIL] Python vs CUDA conversion discrepancy exceeds 1 LSB limit!")
            tests_failed += 1
        else:
            print("  [PASS] Python vs CUDA conversion parity verified.")

        max_d_cpp_fallback, mean_d_cpp_fallback = compute_diff(arr_c3_cpp, arr_c3_fallback)
        print(f"  C++ CPU vs CUDA Fallback: Max Diff = {max_d_cpp_fallback} LSB, Mean Diff = {mean_d_cpp_fallback:.6f} LSB")
        if max_d_cpp_fallback != 0:
            print(f"  [FAIL] C++ CPU and CUDA Fallback outputs are not identical!")
            tests_failed += 1
        else:
            print("  [PASS] C++ CPU vs CUDA Fallback parity verified (identical code path).")

        max_d_cuda_cpp, mean_d_cuda_cpp = compute_diff(arr_c3_cuda, arr_c3_cpp)
        print(f"  CUDA vs C++ CPU (CMM quantization check): Max Diff = {max_d_cuda_cpp} LSB, Mean Diff = {mean_d_cuda_cpp:.6f} LSB")
        if max_d_cuda_cpp > 5000 or mean_d_cuda_cpp > 150:
            print(f"  [FAIL] Conversion difference is unusually large: Max={max_d_cuda_cpp}, Mean={mean_d_cuda_cpp:.2f}")
            tests_failed += 1
        else:
            print("  [PASS] CUDA vs C++ CPU comparison is within expected fixed-point LCMS vs float32 limits.")

        del arr_c3_py, arr_c3_cpp, arr_c3_cuda, arr_c3_fallback
        gc.collect()

    finally:
        # Clean up CapturedImage
        img.discard()
        if decompressed_locally and os.path.exists(raw_file):
            print(f"[*] Removing decompressed reference RAW file: {raw_file}")
            os.remove(raw_file)

    print("\n==============================================")
    if tests_failed > 0:
        print(f"PARITY SUITE FAILED with {tests_failed} test failure(s).")
        print("==============================================")
        sys.exit(1)
    else:
        print("ALL COMPREHENSIVE PARITY TESTS PASSED: SUCCESS")
        print("==============================================")
        sys.exit(0)

if __name__ == "__main__":
    main()
