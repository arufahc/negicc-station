#!/usr/bin/env python3
import os
import sys
import argparse
import numpy as np
import time
import json

project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_dir)
sys.path.insert(0, os.path.join(project_dir, 'src'))

# Preload Sony CrSDK
lib_path = os.path.join(project_dir, 'venv/bin/libCr_Core.so')
if os.path.exists(lib_path):
    import ctypes
    ctypes.CDLL(lib_path)

import negicc_station
from film_profiling import FilmProfile
from target_selection import find_best_target_index

def main():
    parser = argparse.ArgumentParser(description="CLI tool for profile selection and auto-gain search using CUDA histograms.")
    parser.add_argument("raw_image", help="Path to the captured ARW raw image.")
    parser.add_argument("profile", help="Path to the film profile JSON.")
    parser.add_argument("-t", "--target", type=int, default=None, help="Force a specific target index (default: auto-select).")
    parser.add_argument("-s", "--shutter", type=str, default="1/8", help="Shutter speed of the scan (e.g. 1/8).")
    parser.add_argument("-i", "--iso", type=int, default=100, help="ISO of the scan.")
    
    args = parser.parse_args()
    
    print(f"Loading Profile: {args.profile}")
    profile = FilmProfile(args.profile)
    
    # Parse scan shutter
    scan_shutter = 0.125
    if '/' in args.shutter:
        n, d = args.shutter.split('/')
        scan_shutter = float(n) / float(d)
    else:
        scan_shutter = float(args.shutter)
        
    print(f"Loading Raw Image: {args.raw_image}")
    img = negicc_station.CapturedImage(type=0, shutter_speed=scan_shutter, iso=args.iso, filepaths=[args.raw_image])
    
    # Decode raw image to numpy (half size) for target selection
    print("Decoding raw image...")
    # we need the uint16 raw pixels, not converted to srgb yet, just crosstalk corrected
    # easiest way is to call get_bayer_image() and downsample, or just run the pipeline with linear output
    # but target_selection.py expects just a numpy array.
    # We can use to_numpy with pipeline="opencv" and no profile, but it's slow.
    # The new cuda pipeline can output linear RGB quickly!
    lin_rgb = img.to_numpy(
        half=True, pipeline="cuda", to_uint8=False,
        output_profile_path="linear"
    )
    
    # Get film base from raw_image black border (approximated here by top border or profile)
    # For a real CLI, we should extract film base from the actual image edges.
    print("Extracting film base RGB...")
    film_base_rgb = np.mean(lin_rgb[:10, :, :], axis=(0,1))
    print(f"Detected Film Base RGB: {film_base_rgb[0]:.1f}, {film_base_rgb[1]:.1f}, {film_base_rgb[2]:.1f}")
    
    base_shutter = profile.raw_data.get('exposure', {}).get('shutter_speed', 0.125)
    base_iso = profile.raw_data.get('exposure', {}).get('iso', 100)
    
    target_idx = args.target
    if target_idx is None:
        print("Auto-selecting best profile target based on dynamic range...")
        target_idx, mean_t = find_best_target_index(
            profile, lin_rgb, film_base_rgb, 
            scan_shutter=scan_shutter, scan_iso=args.iso, 
            base_shutter=base_shutter, base_iso=base_iso
        )
        print(f"Selected Target Index: {target_idx} (mean transmittance: {mean_t:.4f})")
    else:
        print(f"Using forced Target Index: {target_idx}")
        
    # Isolate target in profile
    prof_data = json.loads(json.dumps(profile.raw_data))
    if 'targets' in prof_data and target_idx < len(prof_data['targets']):
        tgt = prof_data['targets'][target_idx]
        prof_data['targets'] = [tgt]
        if 'icc_profile_base64' in tgt:
            prof_data['icc_profile_base64'] = tgt['icc_profile_base64']
            
    temp_profile = FilmProfile(prof_data)
    if temp_profile.icc_profile_bytes is None and getattr(profile, 'icc_profile_bytes', None):
        temp_profile.icc_profile_bytes = profile.icc_profile_bytes
        
    print("Running high-speed CUDA gain search...")
    
    # Global gains
    gains = np.arange(0.1, 3.1, 0.1).astype(np.float32)
    
    base_exp = base_shutter * base_iso
    scan_exp = scan_shutter * args.iso
    exposure_ratio = base_exp / scan_exp if scan_exp > 0 else 1.0
    
    target_val = getattr(temp_profile, 'normalization_target', 55000.0)
    fb_r, fb_g, fb_b = film_base_rgb
    scale_r = (target_val / fb_r) * exposure_ratio if fb_r > 0 else 1.0
    scale_g = (target_val / fb_g) * exposure_ratio if fb_g > 0 else 1.0
    scale_b = (target_val / fb_b) * exposure_ratio if fb_b > 0 else 1.0
    
    scales = np.array([scale_r, scale_g, scale_b])
    merged_matrix = np.array(temp_profile.crosstalk_matrix) * scales[:, np.newaxis]
    flat_cc = merged_matrix.flatten().astype(float).tolist()
    
    # Ensure raw is decoded so we can get its dimensions
    w = 0
    h = 0
    if hasattr(img, 'width') and hasattr(img, 'height'):
        w = img.width // 2
        h = img.height // 2
    else:
        # fallback, hardcoded for a7r4 half size if missing
        w = 4752
        h = 3168
        
    cW = int(0.8 * w)
    cH = int(0.8 * h)
    
    t0 = time.time()
    hists_global = img.search_gains_histogram(
        half=True, crop_w=cW, crop_h=cH,
        crosstalk_matrix=flat_cc,
        global_gains=gains.tolist(),
        g_gains=[1.0] * len(gains), b_gains=[1.0] * len(gains),
        it8_profile_bytes=temp_profile.icc_profile_bytes,
        film_base=film_base_rgb.astype(int).tolist(),
        profile_film_base=[int(x) for x in temp_profile.get_film_base_rgb()]
    )
    
    bin_centers = np.arange(65536) / 65535.0 * 100.0
    best_gain = 1.0
    best_L_diff = float('inf')
    
    for i, gain in enumerate(gains):
        hist_L = hists_global[i, 0, :]
        total = np.sum(hist_L)
        if total > 0:
            mean_L = np.sum(hist_L * bin_centers) / total
            if abs(mean_L - 50.0) < best_L_diff:
                best_L_diff = abs(mean_L - 50.0)
                best_gain = float(gain)
                
    print(f"Optimal Global Gain found: {best_gain:.2f}")
    
    # GB Gain search
    g_gains = []
    b_gains = []
    for g in np.arange(0.8, 1.21, 0.05):
        for b in np.arange(0.8, 1.21, 0.05):
            g_gains.append(float(g))
            b_gains.append(float(b))
            
    hists_gb = img.search_gains_histogram(
        half=True, crop_w=cW, crop_h=cH,
        crosstalk_matrix=flat_cc,
        global_gains=[best_gain] * len(g_gains),
        g_gains=g_gains, b_gains=b_gains,
        it8_profile_bytes=temp_profile.icc_profile_bytes,
        film_base=film_base_rgb.astype(int).tolist(),
        profile_film_base=[int(x) for x in temp_profile.get_film_base_rgb()]
    )
    
    bin_centers_ab = (np.arange(65536) / 65535.0 * 255.0) - 128.0
    best_g = 1.0
    best_b = 1.0
    min_cast = float('inf')
    
    for i, (g, b) in enumerate(zip(g_gains, b_gains)):
        hist_a = hists_gb[i, 1, :]
        hist_b = hists_gb[i, 2, :]
        tot_a = np.sum(hist_a)
        tot_b = np.sum(hist_b)
        if tot_a > 0 and tot_b > 0:
            mean_a_sq = np.sum(hist_a * (bin_centers_ab**2)) / tot_a
            mean_b_sq = np.sum(hist_b * (bin_centers_ab**2)) / tot_b
            cast = mean_a_sq + mean_b_sq
            if cast < min_cast:
                min_cast = cast
                best_g = float(g)
                best_b = float(b)
                
    t1 = time.time()
    print(f"Optimal Green Gain found: {best_g:.2f}")
    print(f"Optimal Blue Gain found:  {best_b:.2f}")
    print(f"Time taken for gain search: {t1 - t0:.3f} seconds")

if __name__ == "__main__":
    main()
