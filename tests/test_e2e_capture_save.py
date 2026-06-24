import os
import sys
import glob
import numpy as np

# Add src to python path
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))
import negicc_station
from film_profiling import FilmProfile
import color_conversion
from target_selection import find_best_target_index
import json

def get_lib_path():
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(project_dir, 'venv/bin/libCr_Core.so')

import ctypes
lib_path = get_lib_path()
if os.path.exists(lib_path):
    ctypes.CDLL(lib_path)

def test_e2e():
    test_imgs_dir = os.path.join(os.path.dirname(__file__), '../test_imgs')
    arw_file = os.path.join(test_imgs_dir, 'sample_portra400.ARW')
    
    if not os.path.exists(arw_file):
        print(f"Skipping e2e test, {arw_file} not found.")
        return
        
    print(f"Loading {arw_file}...")
    img = negicc_station.CapturedImage(
        type=0,
        shutter_speed=1/60.0,
        iso=100,
        filepaths=[arw_file]
    )
    
    print("Extracting linear numpy array...")
    raw_linear = img.to_numpy(half=True)
    print(f"Shape: {raw_linear.shape}")
    
    # Find profile
    profiles_dir = os.path.join(os.path.dirname(__file__), '../profiles')
    json_files = glob.glob(os.path.join(profiles_dir, "profile_Portra400*.json"))
    if not json_files:
        print("Skipping target selection, no Portra 400 profile found.")
        return
        
    profile_path = json_files[0]
    print(f"Loading profile {profile_path}...")
    profile = FilmProfile(profile_path)
    
    # Mock film base (center of image)
    h, w = raw_linear.shape[:2]
    film_base_rgb = tuple(np.mean(raw_linear[h//2-10:h//2+10, w//2-10:w//2+10], axis=(0,1)))
    print(f"Mock Film Base RGB: {film_base_rgb}")
    
    # Test Target Selection
    best_idx, dist = find_best_target_index(profile, raw_linear, film_base_rgb)
    print(f"Selected target {best_idx} with dist {dist:.2f}")
    
    # Mock target profile injection
    prof_data = json.loads(json.dumps(profile.raw_data))
    prof_data['targets'] = [prof_data['targets'][best_idx]]
    temp_profile = FilmProfile(prof_data)
    if getattr(profile, 'icc_profile_bytes', None):
        temp_profile.icc_profile_bytes = profile.icc_profile_bytes
        
    # Convert and Save
    out_tiff = os.path.join(os.path.dirname(__file__), 'test_out.tiff')
    print(f"Converting and saving to {out_tiff}...")
    color_conversion.convert_raw_to_tiff(
        img=img, profile=temp_profile, output_path=out_tiff,
        exposure_comp=1.0, half=True, film_base_rgb=film_base_rgb
    )
    
    assert os.path.exists(out_tiff), "TIFF not created!"
    
    # Test TIFF in-place editing
    from ui_capture import set_tiff_orientation_inplace, get_exif_orientation
    
    tag_val = get_exif_orientation(hflip=True, vflip=False, rot_cw=90)
    print(f"Setting EXIF orientation tag to {tag_val}...")
    set_tiff_orientation_inplace(out_tiff, tag_val)
    
    from PIL import Image
    im = Image.open(out_tiff)
    orientation = im.tag_v2.get(274)
    assert orientation == tag_val, f"Expected {tag_val}, got {orientation}"
    print("Success!")
    
    os.remove(out_tiff)

def test_e2e_crosstalk_only():
    test_imgs_dir = os.path.join(os.path.dirname(__file__), '../test_imgs')
    arw_file = os.path.join(test_imgs_dir, 'sample_portra400.ARW')
    
    if not os.path.exists(arw_file):
        print(f"Skipping crosstalk-only E2E test, {arw_file} not found.")
        return
        
    print(f"Loading {arw_file} for crosstalk-only test...")
    img = negicc_station.CapturedImage(
        type=0,
        shutter_speed=1/60.0,
        iso=100,
        filepaths=[arw_file]
    )
    
    profile_path = os.path.join(os.path.dirname(__file__), '../profiles/ILCE-7RM4_crosstalk_profile.json')
    print(f"Loading profile {profile_path}...")
    profile = FilmProfile(profile_path)
    
    # Save TIFF using crosstalk matrix
    out_tiff = os.path.join(os.path.dirname(__file__), 'test_out_crosstalk.tiff')
    print(f"Converting and saving to {out_tiff} using crosstalk matrix...")
    matrix = [val for row in profile.crosstalk_matrix for val in row]
    img.write_tiff(out_tiff, half=True, crosstalk_matrix=matrix)
    
    assert os.path.exists(out_tiff), "Crosstalk TIFF not created!"
    
    # Clean up
    os.remove(out_tiff)
    print("Crosstalk-only E2E test passed!")

if __name__ == "__main__":
    test_e2e()
    test_e2e_crosstalk_only()
