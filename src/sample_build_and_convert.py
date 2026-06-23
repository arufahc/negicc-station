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
    parser.add_argument("--profile", default="profiles/profile_Portra 400_20260623_000121.json",
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
    parser.add_argument("--gamma", type=float, default=1.0,
                        help="Post correction gamma.")
    
    args = parser.parse_args()
    
    # 1. Download and parse reference file
    print(f"Loading reference targets from: {args.reference}")
    cache_dir = tempfile.gettempdir()
    patches, loaded_filename, reference_dir = download_and_parse_reference_file(
        args.reference, cache_dir, prompt_zip_callback=None
    )
    ref_base_name = os.path.splitext(os.path.basename(loaded_filename))[0]
    out_json_path = os.path.join(reference_dir, f"{ref_base_name}_ref.json")
    
    ref_data = {
        "description": "IT8.7/2 Reference XYZ values",
        "source": args.reference,
        "patches": patches
    }
    with open(out_json_path, 'w') as f:
        json.dump(ref_data, f, indent=2)
    
    print(f"Loaded {len(patches)} reference patches to {out_json_path}")
    
    # 2. Load film profile
    print(f"Loading Film Profile: {args.profile}")
    profile = FilmProfile(args.profile)
    print(f"Film Profile Name: {profile.film_name}")
    
    # 3. Build ICC Profile
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
    
    # 4. Convert RAW to TIFF using C++ to_numpy and save with imageio
    print(f"Loading RAW image: {args.raw}")
    if not os.path.exists(args.raw):
        print(f"Error: Raw file {args.raw} not found.")
        sys.exit(1)
        
    half_size = not args.full
    img = negicc_station.CapturedImage(
        type=0,  # CAPTURE_SINGLE
        shutter_speed=0.125,  # 1/8s
        iso=100,
        filepaths=[args.raw]
    )
    
    if hasattr(film_profiling, 'convert_raw_image'):
        print(f"Applying dynamic film base scaling and IT8 profile (half size: {half_size})...")
        arr = film_profiling.convert_raw_image(
            img=img,
            profile=profile,
            clut_path=clut_path,
            shutter_str="1/8s",
            exposure_comp=args.exposure_comp,
            post_correction_gamma=args.gamma,
            half=half_size
        )
    else:
        print(f"Applying HEAD clean conversion (no dynamic scaling, half size: {half_size})...")
        raw_crosstalk = np.array(profile.crosstalk_matrix)
        flat_merged_matrix = raw_crosstalk.flatten().tolist()
        arr = img.to_numpy(
            half=half_size,
            crosstalk_matrix=flat_merged_matrix,
            it8_profile_path=clut_path,
            output_profile_path="srgb",
            profile_film_base=None,
            film_base=None,
            exposure_comp=args.exposure_comp,
            post_correction_gamma=args.gamma
        )
    
    print(f"Output array shape: {arr.shape}, dtype: {arr.dtype}")
    print(f"Writing TIFF to: {args.output}")
    imageio.imwrite(args.output, arr)
    print("TIFF written successfully!")

if __name__ == "__main__":
    main()
