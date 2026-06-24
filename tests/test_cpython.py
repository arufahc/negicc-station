import os
import sys
import subprocess
import numpy as np

def main():
    print("========================================")
    project_dir = os.path.abspath(os.path.dirname(__file__))
    parent_dir = os.path.abspath(os.path.join(project_dir, ".."))

    # 1. Import Python extension
    print("Importing negicc_station CPython module...")
    try:
        import negicc_station
    except ImportError as e:
        print(f"ERROR: Failed to import negicc_station: {e}")
        sys.exit(1)

    print("Successfully imported negicc_station!")

    # 2. Test camera connection query
    print("Testing is_camera_connected() function...")
    conn = negicc_station.is_camera_connected()
    print(f"  is_camera_connected() returned: {conn}")
    assert isinstance(conn, bool), "is_camera_connected() must return a boolean"

    # 3. Decompress the checked-in reference ARW file for offline parity testing
    raw_file = os.path.abspath(os.path.join(project_dir, "../test_imgs/test_capture_ref.ARW"))
    compressed_file = raw_file + ".xz"
    if not os.path.exists(raw_file):
        if os.path.exists(compressed_file):
            print(f"[*] Decompressing {compressed_file}...")
            subprocess.run(["xz", "-d", "-k", compressed_file], check=True)
        else:
            print(f"ERROR: Reference ARW file not found at {raw_file} or {compressed_file}")
            sys.exit(1)

    print(f"Initializing CapturedImage offline with {raw_file}...")
    try:
        py_img = negicc_station.CapturedImage(
            type=0,  # CAPTURE_SINGLE
            shutter_speed=0.125,
            iso=100,
            filepaths=[raw_file]
        )
    except Exception as e:
        print(f"ERROR: Instantiation failed: {e}")
        sys.exit(1)

    print("Offline CapturedImage wrapper successfully created!")
    print(f"  Shutter speed: {py_img.shutter_speed}s")
    print(f"  ISO:           {py_img.iso}")
    print(f"  Capture type:  {py_img.capture_type}")
    print(f"  Filepaths:     {py_img.filepaths}")

    # Assert properties
    assert py_img.shutter_speed == 0.125
    assert py_img.iso == 100
    assert py_img.capture_type == 0
    assert py_img.filepaths == [raw_file]

    # 4. Generate NumPy arrays from Python
    print("Converting to Python NumPy array (full-size)...")
    arr_full = py_img.to_numpy(half=False)
    print(f"  Full-size NumPy array shape: {arr_full.shape}, dtype: {arr_full.dtype}")

    print("Converting to Python NumPy array (half-size)...")
    arr_half = py_img.to_numpy(half=True)
    print(f"  Half-size NumPy array shape: {arr_half.shape}, dtype: {arr_half.dtype}")

    # 5. Run C++ test program to write TIFFs and read header size
    cpp_executable = os.path.join(parent_dir, "build/cpp_test_tiff")
    if not os.path.exists(cpp_executable):
        print(f"ERROR: C++ test executable not found at {cpp_executable}. Build it first.")
        sys.exit(1)

    tiff_full_path = "linear_full_cpp.tif"
    tiff_half_path = "linear_half_cpp.tif"

    # Run C++ for full-size
    print(f"Running C++ test program to save full-size TIFF to {tiff_full_path}...")
    cmd_full = [cpp_executable, tiff_full_path, "0", raw_file]
    proc_full = subprocess.run(cmd_full, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc_full.returncode != 0:
        print(f"ERROR: C++ tiff program failed:\n{proc_full.stderr}")
        sys.exit(1)

    # Parse header size from C++ stdout
    header_size = 0
    for line in proc_full.stdout.splitlines():
        if line.startswith("TIFF_HEADER_SIZE:"):
            header_size = int(line.split(":")[1])
            break

    if header_size == 0:
        print("ERROR: Failed to parse TIFF_HEADER_SIZE from C++ program.")
        sys.exit(1)
    print(f"Parsed TIFF header size: {header_size} bytes")

    # Run C++ for half-size
    print(f"Running C++ test program to save half-size TIFF to {tiff_half_path}...")
    cmd_half = [cpp_executable, tiff_half_path, "1", raw_file]
    proc_half = subprocess.run(cmd_half, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc_half.returncode != 0:
        print(f"ERROR: C++ tiff program failed:\n{proc_half.stderr}")
        sys.exit(1)

    # 6. Read raw pixel payloads from C++ TIFF files and compare with Python NumPy arrays
    # Full-size comparison
    print("Comparing Python full-size NumPy array with C++ output TIFF payload...")
    with open(tiff_full_path, "rb") as f:
        tiff_data = f.read()
    pixel_bytes = tiff_data[-arr_full.size * 2:]
    cpp_arr_full = np.frombuffer(pixel_bytes, dtype=np.uint16).reshape(arr_full.shape)

    if not np.array_equal(arr_full, cpp_arr_full):
        print("ERROR: Full-size arrays do not match!")
        sys.exit(1)
    print("  [PASS] Full-size arrays match exactly!")

    # Half-size comparison
    print("Comparing Python half-size NumPy array with C++ output TIFF payload...")
    with open(tiff_half_path, "rb") as f:
        tiff_data_half = f.read()
    pixel_bytes_half = tiff_data_half[-arr_half.size * 2:]
    cpp_arr_half = np.frombuffer(pixel_bytes_half, dtype=np.uint16).reshape(arr_half.shape)

    if not np.array_equal(arr_half, cpp_arr_half):
        print("ERROR: Half-size arrays do not match!")
        sys.exit(1)
    print("  [PASS] Half-size arrays match exactly!")

    # 7. Clean up temporary files (including decompressed reference ARW)
    print("Cleaning up temporary TIFF files...")
    if os.path.exists(tiff_full_path):
        os.remove(tiff_full_path)
    if os.path.exists(tiff_half_path):
        os.remove(tiff_half_path)
    if os.path.exists(raw_file):
        print(f"[*] Removing decompressed reference RAW file: {raw_file}")
        os.remove(raw_file)

    print("\n========================================")
    print("ALL CPYTHON INTEGRATION TESTS PASSED: SUCCESS")
    print("========================================")

if __name__ == "__main__":
    main()
