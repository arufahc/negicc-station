import os
import sys
import zipfile
import shutil

# Add src to python path
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from ui_capture import read_arw_metadata, unpack_archive_and_find_arws

def test_exif_parsing():
    arw_file = os.path.join(os.path.dirname(__file__), '../test_imgs/sample_portra400.ARW')
    if not os.path.exists(arw_file):
        print(f"Skipping test_exif_parsing, {arw_file} not found.")
        return
        
    print(f"Reading EXIF metadata from {arw_file}...")
    shutter, iso = read_arw_metadata(arw_file)
    print(f"Parsed Shutter Speed: {shutter}s, ISO: {iso}")
    
    # We know the reference sample_portra400.ARW has shutter speed of 0.125s (1/8s) and ISO 100
    assert abs(shutter - 0.125) < 1e-5, f"Expected 0.125, got {shutter}"
    assert iso == 100, f"Expected 100, got {iso}"
    print("EXIF parsing test passed!")

def test_archive_unpacking():
    # 1. Create a dummy zip archive containing a mock raw file
    test_dir = os.path.dirname(__file__)
    tmp_dir = os.path.join(test_dir, 'tmp_test_archive')
    os.makedirs(tmp_dir, exist_ok=True)
    
    mock_arw_path = os.path.join(tmp_dir, 'mock_image.ARW')
    with open(mock_arw_path, 'w') as f:
        f.write("mock raw content")
        
    zip_path = os.path.join(test_dir, 'mock_archive.zip')
    with zipfile.ZipFile(zip_path, 'w') as z:
        z.write(mock_arw_path, arcname='mock_image.ARW')
        
    # Clean up mock file and directory
    shutil.rmtree(tmp_dir)
    
    # 2. Extract archive using our helper
    extract_dest = os.path.join(test_dir, 'extracted_test_archive')
    print(f"Extracting archive {zip_path} to {extract_dest}...")
    arw_files = unpack_archive_and_find_arws(zip_path, extract_dest)
    print(f"Unpacked ARW files: {arw_files}")
    
    assert len(arw_files) == 1, f"Expected 1 ARW file, got {len(arw_files)}"
    assert os.path.basename(arw_files[0]) == 'mock_image.ARW', f"Expected mock_image.ARW, got {arw_files[0]}"
    
    # Clean up zip and extracted files
    if os.path.exists(zip_path):
        os.remove(zip_path)
    if os.path.exists(extract_dest):
        shutil.rmtree(extract_dest)
        
    print("Archive unpacking test passed!")

if __name__ == "__main__":
    test_exif_parsing()
    test_archive_unpacking()
    print("ALL OFFLINE LOAD TESTS PASSED!")
