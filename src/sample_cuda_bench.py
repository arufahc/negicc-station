#!/usr/bin/env python3
import os
import sys
import time
import numpy as np

src_dir = os.path.dirname(os.path.abspath(__file__))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

import negicc_station
import film_profiling
from film_profiling import FilmProfile

def main():
    profile_path = "profiles/profile_Portra400_20260623_170610.json"
    raw_path = "sample.ARW"
    
    if not os.path.exists(profile_path):
        profiles_dir = "profiles"
        if os.path.exists(profiles_dir):
            for f in os.listdir(profiles_dir):
                if f.endswith(".json") and "crosstalk" not in f:
                    profile_path = os.path.join(profiles_dir, f)
                    break
                    
    if not os.path.exists(raw_path):
        raw_path = "test_imgs/sample_portra400.ARW"

    print(f"Using Film Profile: {profile_path}")
    print(f"Using RAW Image: {raw_path}")

    profile = FilmProfile(profile_path)
    if not getattr(profile, 'icc_profile_bytes', None):
        icc_path = None
        for root, dirs, files in os.walk("profiles"):
            for f in files:
                if f.endswith(".icc"):
                    icc_path = os.path.join(root, f)
                    break
            if icc_path:
                break
        
        if icc_path:
            print(f"Loading built ICC profile from: {icc_path}")
            with open(icc_path, 'rb') as f:
                profile.icc_profile_bytes = f.read()
        else:
            print("No ICC profile found. Please run sample_build_and_convert.py first to build the profile.")
            sys.exit(1)

    print("Profile ICC bytes loaded successfully.")

    for half in [True, False]:
        print("\n" + "="*50)
        print(f"BENCHMARKING {'HALF' if half else 'FULL'}-SIZE IMAGES")
        print("="*50)
        
        img_cpu = negicc_station.CapturedImage(
            type=0,
            shutter_speed=0.125,
            iso=100,
            filepaths=[raw_path]
        )
        
        print("Running CPU (cpp) Cold Run...")
        t0 = time.perf_counter()
        arr_cpu_cold = film_profiling.convert_raw_image(img_cpu, profile, half=half, pipeline="cpp")
        t_cpu_cold = time.perf_counter() - t0
        print(f"  CPU Cold Run: {t_cpu_cold:.4f} seconds")
        
        print("Running CPU (cpp) Warm Run...")
        t0 = time.perf_counter()
        arr_cpu_warm = film_profiling.convert_raw_image(img_cpu, profile, half=half, pipeline="cpp")
        t_cpu_warm = time.perf_counter() - t0
        print(f"  CPU Warm Run: {t_cpu_warm:.4f} seconds")
        
        t_raw_load = t_cpu_cold - t_cpu_warm
        print(f"  Estimated CPU RAW loading + demosaic time: {t_raw_load:.4f} seconds")

        img_cuda = negicc_station.CapturedImage(
            type=0,
            shutter_speed=0.125,
            iso=100,
            filepaths=[raw_path]
        )
        
        print("Running CUDA Cold Run...")
        t0 = time.perf_counter()
        arr_cuda_cold = film_profiling.convert_raw_image(img_cuda, profile, half=half, pipeline="cuda")
        t_cuda_cold = time.perf_counter() - t0
        print(f"  CUDA Cold Run: {t_cuda_cold:.4f} seconds")
        
        print("Running CUDA Warm Run (repeated to show cache effect)...")
        cuda_warm_times = []
        for i in range(5):
            t0 = time.perf_counter()
            arr_cuda_warm = film_profiling.convert_raw_image(img_cuda, profile, half=half, pipeline="cuda")
            elapsed = time.perf_counter() - t0
            cuda_warm_times.append(elapsed)
            print(f"    Warm Run #{i+1}: {elapsed:.4f} seconds")
            
        t_cuda_warm = np.mean(cuda_warm_times)
        print(f"  Average CUDA Warm Run: {t_cuda_warm:.4f} seconds")
        
        latency_savings = t_cuda_cold - t_cuda_warm
        percent_savings = (latency_savings / t_cuda_cold) * 100.0
        print(f"\n  --> Latency Savings: {latency_savings:.4f} seconds ({percent_savings:.1f}% reduction)")
        
        t_upload_convert = t_cuda_cold - t_raw_load - t_cuda_warm
        print(f"  --> Estimated Host-to-Device upload + float32 conversion time: {t_upload_convert:.4f} seconds")
        
        np.testing.assert_array_equal(arr_cuda_cold, arr_cuda_warm, err_msg="Parity check failed between CUDA cold and warm runs!")
        print("  Parity verification passed: Warm and cold runs produce identical output arrays.")

if __name__ == "__main__":
    main()
