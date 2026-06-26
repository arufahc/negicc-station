#!/usr/bin/env python3
import os
import sys
import argparse
import json
import tempfile
import numpy as np
import imageio

# Add src/ to python path
src_dir = os.path.dirname(os.path.abspath(__file__))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

import negicc_station
import film_profiling
from film_profiling import FilmProfile, download_and_parse_reference_file

def parse_shutter_speed(shutter_str):
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

def main():
    parser = argparse.ArgumentParser(description="Build profile and convert raw ARW image to TIFF.")
    parser.add_argument("--profile", default="profiles/profile_Portra400_20260623_170610.json",
                        help="Path to the input film profile JSON.")
    parser.add_argument("--reference", default="http://www.colorreference.de/targets/R190808.zip",
                        help="URL or local path of the IT8 reference file.")
    parser.add_argument("--raw", default="sample.ARW",
                        help="Path to the input raw image (e.g. sample.ARW).")
    parser.add_argument("--output", default="build/sample_converted.tiff",
                        help="Path to the output TIFF file.")
    parser.add_argument("--full", action="store_true",
                        help="Use full size rendering instead of half size.")
    parser.add_argument("--exposure-comp", type=float, default=1.0,
                        help="Exposure compensation factor.")
    parser.add_argument("--colorspace", choices=["srgb", "srgb-g10"], default="srgb",
                        help="Working space profile (default: srgb)")
    parser.add_argument("--pipeline", choices=["cpp", "cuda", "python"], default="cpp",
                        help="Select conversion pipeline (default: cpp)")
    parser.add_argument("--compare", action="store_true",
                        help="Run Python, C++ CPU and CUDA pipelines and compare results.")
    parser.add_argument("--rebuild-profile-json", default=None,
                        help="Path to save the rebuilt self-contained film profile JSON.")
    
    args = parser.parse_args()

    # Determine final values from named arguments
    profile_path = args.profile
    raw_path = args.raw
    output_path = args.output

    # 1. Load film profile
    print(f"Loading Film Profile: {profile_path}")
    if not os.path.exists(profile_path):
        print(f"Error: Profile file {profile_path} not found.")
        sys.exit(1)

    if args.rebuild_profile_json:
        import base64
        with open(profile_path, 'r') as f:
            profile_data = json.load(f)

        print(f"Loading reference targets from: {args.reference}")
        cache_dir = tempfile.gettempdir()
        patches, loaded_filename, reference_dir, illuminant = download_and_parse_reference_file(
            args.reference, cache_dir, prompt_zip_callback=None
        )
        ref_base_name = os.path.splitext(os.path.basename(loaded_filename))[0]
        out_json_path = os.path.join(reference_dir, f"{ref_base_name}_ref.json")

        ref_data = {
            "description": "IT8.7/2 Reference XYZ values",
            "source": args.reference,
            "illuminant": illuminant,
            "patches": patches
        }
        with open(out_json_path, 'w') as f:
            json.dump(ref_data, f, indent=2)

        print(f"Loaded {len(patches)} reference patches to {out_json_path} (Illuminant: {illuminant})")

        targets = profile_data.get('targets', [])
        print(f"Rebuilding {len(targets)} targets in the profile...")

        output_profiles_dir = "profiles"
        os.makedirs(output_profiles_dir, exist_ok=True)

        temp_profile = FilmProfile(profile_path)

        for idx, target_dict in enumerate(targets):
            name = target_dict.get('name', f"Target {idx + 1}")
            print(f"\n--- Rebuilding Target {idx + 1}/{len(targets)}: '{name}' ---")

            temp_profile.target_name = name
            temp_profile.target_iso = target_dict.get('iso', 100)
            temp_profile.target_shutter = target_dict.get('shutter', '1/8s')
            temp_profile.patches = target_dict.get('patches', {})
            temp_profile.icc_profile_bytes = None

            res = film_profiling.build_icc_profile(
                temp_profile,
                out_json_path,
                output_profiles_dir,
                progress_callback=lambda step, detail: print(f"  [{step}] {detail}")
            )

            clut_path = res['clut_icc_path']
            with open(clut_path, 'rb') as f_icc:
                icc_bytes = f_icc.read()
            icc_b64 = base64.b64encode(icc_bytes).decode('utf-8')

            target_dict['icc_profile_base64'] = icc_b64
            target_dict['profcheck_output'] = res['profcheck_output']
            target_dict['log_messages'] = res['log_messages']

            if os.path.exists(clut_path):
                os.remove(clut_path)

        print(f"\nSaving rebuilt profile JSON to: {args.rebuild_profile_json}")
        with open(args.rebuild_profile_json, 'w') as f:
            json.dump(profile_data, f, indent=4)
        print("Rebuild complete. Exiting.")
        sys.exit(0)

    profile = FilmProfile(profile_path)
    print(f"Film Profile Name: {profile.film_name}")

    sc_profile = None

    # Check if the profile is already self-contained
    if profile.icc_profile_bytes:
        print("Profile is already self-contained (ICC bytes in memory). Skipping profile building...")
        sc_profile = profile
    else:
        # We need to build the ICC profile
        # Download and parse reference file
        print(f"Loading reference targets from: {args.reference}")
        cache_dir = tempfile.gettempdir()
        patches, loaded_filename, reference_dir, illuminant = download_and_parse_reference_file(
            args.reference, cache_dir, prompt_zip_callback=None
        )
        ref_base_name = os.path.splitext(os.path.basename(loaded_filename))[0]
        out_json_path = os.path.join(reference_dir, f"{ref_base_name}_ref.json")
        
        ref_data = {
            "description": "IT8.7/2 Reference XYZ values",
            "source": args.reference,
            "illuminant": illuminant,
            "patches": patches
        }
        with open(out_json_path, 'w') as f:
            json.dump(ref_data, f, indent=2)
        
        print(f"Loaded {len(patches)} reference patches to {out_json_path}")

        # Build ICC Profile
        output_profiles_dir = "profiles"
        os.makedirs(output_profiles_dir, exist_ok=True)
        
        def log_cb(step, detail):
            print(f"[{step}] {detail}")
            
        print("Building ICC profile...")
        res = film_profiling.build_icc_profile(
            profile,
            out_json_path,
            output_profiles_dir,
            progress_callback=log_cb
        )
        clut_path = res['clut_icc_path']
        print(f"Built ICC Profile saved to: {clut_path}")
        
        # Create self-contained profile by attaching ICC bytes directly in memory
        import copy
        
        with open(clut_path, 'rb') as f_icc:
            icc_bytes = f_icc.read()
        
        # Attach ICC bytes directly — no need to write trc_curves or a temp JSON
        sc_profile = copy.copy(profile)
        sc_profile.icc_profile_bytes = icc_bytes
        print(f"ICC profile loaded into memory ({len(icc_bytes)} bytes). No temp file needed.")

    # 4. Decompress RAW if needed
    actual_raw_path = raw_path
    if actual_raw_path.endswith('.xz'):
        xz_path = actual_raw_path
        actual_raw_path = actual_raw_path[:-3]  # Strip '.xz'
        if not os.path.exists(actual_raw_path) and os.path.exists(xz_path):
            import subprocess
            print(f"Decompressing RAW target: {xz_path} -> {actual_raw_path}...")
            subprocess.run(['xz', '-d', '-k', xz_path], check=True)
    elif not os.path.exists(actual_raw_path) and os.path.exists(actual_raw_path + '.xz'):
        xz_path = actual_raw_path + '.xz'
        import subprocess
        print(f"Decompressing RAW target: {xz_path} -> {actual_raw_path}...")
        subprocess.run(['xz', '-d', '-k', xz_path], check=True)

    print(f"Loading RAW image: {actual_raw_path}")
    if not os.path.exists(actual_raw_path):
        print(f"Error: Raw file {actual_raw_path} not found.")
        sys.exit(1)
        
    half_size = not args.full
    img = negicc_station.CapturedImage(
        type=0,  # CAPTURE_SINGLE
        shutter_speed=0.125,  # 1/8s
        iso=100,
        filepaths=[actual_raw_path]
    )
    
    if args.compare:
        import time
        import color_conversion
        
        # 1. Run Python conversion
        py_output_path = output_path.replace(".tiff", "_py.tiff").replace(".tif", "_py.tif")
        print(f"Converting and saving TIFF (Python pipeline) to: {py_output_path}...")
        t0 = time.perf_counter()
        py_success = color_conversion.convert_raw_to_tiff(
            img=img,
            profile=sc_profile,
            output_path=py_output_path,
            colorspace=args.colorspace,
            clut_path=None,
            shutter_str="1/8s",
            exposure_comp=args.exposure_comp,
            half=half_size
        )
        t_py = time.perf_counter() - t0
        
        # 2. Run C++ CPU conversion
        cpp_output_path = output_path.replace(".tiff", "_cpp.tiff").replace(".tif", "_cpp.tif")
        print(f"Converting and saving TIFF (C++ CPU pipeline) to: {cpp_output_path}...")
        t0 = time.perf_counter()
        cpp_success = film_profiling.convert_raw_to_tiff(
            img=img,
            profile=sc_profile,
            output_path=cpp_output_path,
            colorspace=args.colorspace,
            clut_path=None,
            shutter_str="1/8s",
            exposure_comp=args.exposure_comp,
            half=half_size,
            pipeline="cpp"
        )
        t_cpp = time.perf_counter() - t0

        # 3. Run CUDA conversion
        cuda_output_path = output_path.replace(".tiff", "_cuda.tiff").replace(".tif", "_cuda.tif")
        print(f"Converting and saving TIFF (CUDA pipeline) to: {cuda_output_path}...")
        t0 = time.perf_counter()
        cuda_success = film_profiling.convert_raw_to_tiff(
            img=img,
            profile=sc_profile,
            output_path=cuda_output_path,
            colorspace=args.colorspace,
            clut_path=None,
            shutter_str="1/8s",
            exposure_comp=args.exposure_comp,
            half=half_size,
            pipeline="cuda"
        )
        t_cuda = time.perf_counter() - t0
        
        if not (cpp_success and py_success and cuda_success):
            print("Error: Conversion failed on one or both pipelines.")
            sys.exit(1)
            
        # Read and compare pixel data
        print("Loading output images for comparison...")
        arr_cpp = imageio.imread(cpp_output_path)
        arr_py = imageio.imread(py_output_path)
        arr_cuda = imageio.imread(cuda_output_path)
        
        # Compute absolute differences
        diff_cpp_py = np.abs(arr_cpp.astype(np.int32) - arr_py.astype(np.int32))
        max_diff_cpp_py = np.max(diff_cpp_py)
        mean_diff_cpp_py = np.mean(diff_cpp_py)

        diff_cuda_py = np.abs(arr_cuda.astype(np.int32) - arr_py.astype(np.int32))
        max_diff_cuda_py = np.max(diff_cuda_py)
        mean_diff_cuda_py = np.mean(diff_cuda_py)

        diff_cuda_cpp = np.abs(arr_cuda.astype(np.int32) - arr_cpp.astype(np.int32))
        max_diff_cuda_cpp = np.max(diff_cuda_cpp)
        mean_diff_cuda_cpp = np.mean(diff_cuda_cpp)
        
        print("\n=== PROCESSING TIMES ===")
        print(f"Python pipeline:  {t_py:.4f} seconds")
        print(f"C++ CPU pipeline: {t_cpp:.4f} seconds (Speedup vs Python: {t_py/t_cpp:.1f}x)")
        print(f"CUDA pipeline:    {t_cuda:.4f} seconds (Speedup vs Python: {t_py/t_cuda:.1f}x, vs CPU: {t_cpp/t_cuda:.1f}x)")

        print("\n=== PARITY COMPARISON RESULTS ===")
        print(f"C++ CPU vs Python:")
        print(f"  Max Pixel-wise Difference: {max_diff_cpp_py} LSB (out of 65535)")
        print(f"  Mean Pixel-wise Difference: {mean_diff_cpp_py:.6f} LSB")
        print(f"CUDA vs Python:")
        print(f"  Max Pixel-wise Difference: {max_diff_cuda_py} LSB")
        print(f"  Mean Pixel-wise Difference: {mean_diff_cuda_py:.6f} LSB")
        print(f"CUDA vs C++ CPU:")
        print(f"  Max Pixel-wise Difference: {max_diff_cuda_cpp} LSB")
        print(f"  Mean Pixel-wise Difference: {mean_diff_cuda_cpp:.6f} LSB")
        
        # Verify ICC tag embedding
        from imageio.plugins.tifffile import _tifffile
        tif_cpp = _tifffile.TiffFile(cpp_output_path)
        tif_py = _tifffile.TiffFile(py_output_path)
        tif_cuda = _tifffile.TiffFile(cuda_output_path)
        
        has_icc_cpp = 34675 in [t.code for t in tif_cpp.pages[0].tags.values()]
        has_icc_py = 34675 in [t.code for t in tif_py.pages[0].tags.values()]
        has_icc_cuda = 34675 in [t.code for t in tif_cuda.pages[0].tags.values()]
        
        print(f"C++ CPU TIFF has ICC profile tag (34675): {has_icc_cpp}")
        print(f"Python TIFF has ICC profile tag (34675):  {has_icc_py}")
        print(f"CUDA TIFF has ICC profile tag (34675):    {has_icc_cuda}")
        print("==================================\n")
        
        # CPU/CUDA tetrahedral matching comparison
        if mean_diff_cuda_cpp <= 5.0 and max_diff_cuda_cpp <= 50:
            print("PARITY CHECK PASSED: CUDA and C++ CPU match very closely.")
        else:
            print("WARNING: Large difference between CUDA and C++ CPU conversions.")
            
    else:
        import time
        t0 = time.perf_counter()
        if args.pipeline == "python":
            import color_conversion
            print(f"Converting and saving TIFF (Python pipeline) to: {output_path}...")
            success = color_conversion.convert_raw_to_tiff(
                img=img,
                profile=sc_profile,
                output_path=output_path,
                colorspace=args.colorspace,
                clut_path=None,
                shutter_str="1/8s",
                exposure_comp=args.exposure_comp,
                half=half_size
            )
        else:
            print(f"Converting and saving TIFF ({args.pipeline.upper()} pipeline) to: {output_path}...")
            success = film_profiling.convert_raw_to_tiff(
                img=img,
                profile=sc_profile,
                output_path=output_path,
                colorspace=args.colorspace,
                clut_path=None,
                shutter_str="1/8s",
                exposure_comp=args.exposure_comp,
                half=half_size,
                pipeline=args.pipeline
            )
        elapsed = time.perf_counter() - t0
        if success:
            print(f"TIFF written successfully in {elapsed:.4f} seconds using {args.pipeline} pipeline!")
        else:
            print(f"Error: TIFF conversion failed using {args.pipeline} pipeline.")
            sys.exit(1)

if __name__ == "__main__":
    main()
