#!/usr/bin/env python3
import os
import sys
import time
import subprocess
import numpy as np

def main():
    print("==================================================")
    print("BENCHMARK & PARITY TEST FOR UINT8 PREVIEW KERNEL")
    print("==================================================")

    project_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(project_dir)
    sys.path.insert(0, project_dir)

    try:
        import negicc_station
        import film_profiling
        from film_profiling import FilmProfile
    except ImportError as e:
        print(f"ERROR: Failed to import modules: {e}")
        sys.exit(1)

    # 1. Load profiles
    profile_path = os.path.join(parent_dir, "profiles/profile_Portra400_20260623_170610.json")
    if not os.path.exists(profile_path):
        print(f"ERROR: Profile not found at {profile_path}")
        sys.exit(1)

    profile = FilmProfile(profile_path)
    icc_bytes = None
    if getattr(profile, 'icc_profile_bytes', None):
        icc_bytes = profile.icc_profile_bytes
    else:
        # Load from disk
        icc_path = os.path.join(parent_dir, "profiles/profile_Portra400_20260623_170610.icc")
        if os.path.exists(icc_path):
            with open(icc_path, 'rb') as f:
                icc_bytes = f.read()
                profile.icc_profile_bytes = icc_bytes

    # 2. Decompress RAW image
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

    # Calculate flat crosstalk matrix scaled with film base
    p_fb_r = profile.film_base['r_avg']
    p_fb_g = profile.film_base['g_avg']
    p_fb_b = profile.film_base['b_avg']
    base_num, base_den = film_profiling.parse_shutter_speed(profile.film_base_shutter)
    t_base = base_num / base_den
    iso_base = profile.film_base_iso

    # Scanned image parameters
    t_scan = 0.125
    iso_scan = 100
    exposure_profile = t_base * (iso_base / 100.0)
    exposure_scan = t_scan * (iso_scan / 100.0)
    exposure_ratio = exposure_profile / exposure_scan

    target_val = profile.normalization_target
    scale_r = (target_val / p_fb_r) * exposure_ratio
    scale_g = (target_val / p_fb_g) * exposure_ratio
    scale_b = (target_val / p_fb_b) * exposure_ratio

    raw_crosstalk = np.array(profile.crosstalk_matrix)
    scales = np.array([scale_r, scale_g, scale_b])
    merged_matrix = raw_crosstalk * scales[:, np.newaxis]
    flat_matrix = merged_matrix.flatten().tolist()

    img = negicc_station.CapturedImage(
        type=0,
        shutter_speed=t_scan,
        iso=iso_scan,
        filepaths=[raw_file]
    )

    try:
        # ==================================================
        # TIMING: 16-BIT PIPELINE + CPU DOWNSCALING
        # ==================================================
        print("\n--- Running 16-bit CUDA Pipeline + CPU 16->8bit Downscale ---")
        
        # Cold Run
        t0 = time.perf_counter()
        arr_16_cold = img.to_numpy(
            half=True,  # always True for preview
            crosstalk_matrix=flat_matrix,
            it8_profile_bytes=icc_bytes,
            pipeline="cuda",
            to_uint8=False
        )
        arr_8_cpu_cold = (arr_16_cold >> 8).astype(np.uint8)
        t_cold_16 = time.perf_counter() - t0
        print(f"  Cold Run Time: {t_cold_16:.4f} seconds")

        # Warm Runs (Cache)
        warm_times_16 = []
        for i in range(5):
            t0 = time.perf_counter()
            arr_16 = img.to_numpy(
                half=True,
                crosstalk_matrix=flat_matrix,
                it8_profile_bytes=icc_bytes,
                pipeline="cuda",
                to_uint8=False
            )
            arr_8_cpu = (arr_16 >> 8).astype(np.uint8)
            warm_times_16.append(time.perf_counter() - t0)
        t_warm_16 = np.mean(warm_times_16)
        print(f"  Average Warm Run Time (Cached): {t_warm_16:.4f} seconds")

        # ==================================================
        # TIMING: NEW 8-BIT CUDA PREVIEW KERNEL
        # ==================================================
        print("\n--- Running New 8-bit CUDA Preview Pipeline (Direct uint8 output) ---")
        
        # Cold Run
        t0 = time.perf_counter()
        arr_8_gpu_cold = img.to_numpy(
            half=True,  # always True for preview
            crosstalk_matrix=flat_matrix,
            it8_profile_bytes=icc_bytes,
            pipeline="cuda",
            to_uint8=True
        )
        t_cold_8 = time.perf_counter() - t0
        print(f"  Cold Run Time: {t_cold_8:.4f} seconds")

        # Warm Runs (Cache)
        warm_times_8 = []
        for i in range(5):
            t0 = time.perf_counter()
            arr_8_gpu = img.to_numpy(
                half=True,
                crosstalk_matrix=flat_matrix,
                it8_profile_bytes=icc_bytes,
                pipeline="cuda",
                to_uint8=True
            )
            warm_times_8.append(time.perf_counter() - t0)
        t_warm_8 = np.mean(warm_times_8)
        print(f"  Average Warm Run Time (Cached): {t_warm_8:.4f} seconds")

        # ==================================================
        # COMPARISON & SPEEDUP SUMMARY
        # ==================================================
        print("\n================ BENCHMARK SUMMARY ================")
        print(f"16-bit CUDA + CPU Downscale (Cold):  {t_cold_16:.4f} s")
        print(f"New 8-bit CUDA Preview (Cold):       {t_cold_8:.4f} s (Speedup: {t_cold_16/t_cold_8:.2f}x)")
        print(f"16-bit CUDA + CPU Downscale (Warm):  {t_warm_16:.4f} s")
        print(f"New 8-bit CUDA Preview (Warm):       {t_warm_8:.4f} s (Speedup: {t_warm_16/t_warm_8:.2f}x)")
        
        latency_saved_ms = (t_warm_16 - t_warm_8) * 1000.0
        print(f"  --> Net Saved Latency per UI Refresh Frame: {latency_saved_ms:.2f} ms")
        print("==================================================\n")

        # ==================================================
        # PARITY TEST (16-bit >> 8 on CPU vs 8-bit GPU Output)
        # ==================================================
        print("--- PARITY COMPARISON: 16-bit CPU-Shifted vs 8-bit GPU ---")
        diff = np.abs(arr_8_cpu_cold.astype(np.int32) - arr_8_gpu_cold.astype(np.int32))
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)

        print(f"  Maximum pixel-wise difference: {max_diff} LSB (out of 255)")
        print(f"  Mean pixel-wise difference:    {mean_diff:.6f} LSB")

        # Because one uses (float_val * 65535) rounded to uint16, then shifted >> 8
        # and the other uses (float_val * 255) rounded to uint8,
        # slight rounding differences can occur, but the maximum difference MUST be <= 1 LSB!
        if max_diff <= 1:
            print("  [PASS] Parity verified successfully! Discrepancy is within 1 LSB rounding limit.")
        else:
            print(f"  [FAIL] Large discrepancy detected! Max Diff = {max_diff} LSB")
            sys.exit(1)

    finally:
        img.discard()
        if decompressed_locally and os.path.exists(raw_file):
            print(f"[*] Removing decompressed reference RAW file: {raw_file}")
            os.remove(raw_file)

if __name__ == "__main__":
    main()
