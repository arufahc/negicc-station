import os
import sys
import subprocess
import numpy as np

def main():
    print("========================================")
    print("STARTING LIVE CAPTURE AND PARITY TEST")
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

    cpp_executable = os.path.join(parent_dir, "build/cpp_test_tiff")
    if not os.path.exists(cpp_executable):
        print(f"ERROR: C++ test executable not found at {cpp_executable}. Build it first.")
        sys.exit(1)

    # Resolve header size by running C++ test once on a dummy run or parse from size
    # Struct tiff_hdr size is 1376 bytes
    header_size = 1376

    # =========================================================================
    # TEST CASE 1: Live SINGLE Capture & Parity
    # =========================================================================
    print("\n----------------------------------------")
    print("TEST CASE 1: Live SINGLE Capture")
    print("----------------------------------------")
    
    # Capture single shot (1/8s exposure)
    print("CPython triggering live SINGLE capture with shutter 1/8s...")
    try:
        py_img_single = negicc_station.capture(type=0, shutter_num=1, shutter_den=8)
    except Exception as e:
        print(f"ERROR: Live SINGLE capture failed: {e}")
        sys.exit(1)

    print("Live SINGLE capture succeeded!")
    print(f"  ISO:           {py_img_single.iso}")
    print(f"  Shutter speed: {py_img_single.shutter_speed}s")
    print(f"  Filepaths:     {py_img_single.filepaths}")

    assert len(py_img_single.filepaths) == 1
    raw_file_single = py_img_single.filepaths[0]

    # Full-size CPython conversion
    print("Converting SINGLE capture to CPython full-size NumPy array...")
    arr_full_py = py_img_single.to_numpy(half=False)
    
    # Half-size CPython conversion
    print("Converting SINGLE capture to CPython half-size NumPy array...")
    arr_half_py = py_img_single.to_numpy(half=True)

    # C++ full-size TIFF output
    tiff_full_cpp = "live_single_full_cpp.tif"
    print(f"Running C++ test program to convert SINGLE raw to TIFF: {tiff_full_cpp}...")
    cmd_full = [cpp_executable, tiff_full_cpp, "0", raw_file_single]
    proc_full = subprocess.run(cmd_full, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc_full.returncode != 0:
        print(f"ERROR: C++ tiff program failed:\n{proc_full.stderr}")
        sys.exit(1)

    # C++ half-size TIFF output
    tiff_half_cpp = "live_single_half_cpp.tif"
    print(f"Running C++ test program to convert SINGLE raw to TIFF: {tiff_half_cpp}...")
    cmd_half = [cpp_executable, tiff_half_cpp, "1", raw_file_single]
    proc_half = subprocess.run(cmd_half, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc_half.returncode != 0:
        print(f"ERROR: C++ tiff program failed:\n{proc_half.stderr}")
        sys.exit(1)

    # Read and compare pixel payloads
    print("Comparing Python full-size NumPy array with C++ output TIFF payload...")
    with open(tiff_full_cpp, "rb") as f:
        tiff_data = f.read()
    cpp_arr_full = np.frombuffer(tiff_data[header_size:], dtype=np.uint16).reshape(arr_full_py.shape)
    assert np.array_equal(arr_full_py, cpp_arr_full), "ERROR: Full-size SINGLE pixel mismatch!"
    print("  [PASS] Full-size SINGLE arrays match exactly!")

    print("Comparing Python half-size NumPy array with C++ output TIFF payload...")
    with open(tiff_half_cpp, "rb") as f:
        tiff_data_half = f.read()
    cpp_arr_half = np.frombuffer(tiff_data_half[header_size:], dtype=np.uint16).reshape(arr_half_py.shape)
    assert np.array_equal(arr_half_py, cpp_arr_half), "ERROR: Half-size SINGLE pixel mismatch!"
    print("  [PASS] Half-size SINGLE arrays match exactly!")

    # Clean up single files
    py_img_single.discard()
    if os.path.exists(tiff_full_cpp):
        os.remove(tiff_full_cpp)
    if os.path.exists(tiff_half_cpp):
        os.remove(tiff_half_cpp)
    print("Temporary files for SINGLE test cleaned up successfully.")

    # =========================================================================
    # TEST CASE 2: Live SONY_PIXEL_SHIFT_4 Capture & Parity
    # =========================================================================
    print("\n----------------------------------------")
    print("TEST CASE 2: Live SONY_PIXEL_SHIFT_4 Capture")
    print("----------------------------------------")

    # Capture 4-shot pixel shift (1/125s exposure)
    print("CPython triggering live SONY_PIXEL_SHIFT_4 capture with shutter 1/125s...")
    try:
        py_img_ps = negicc_station.capture(type=1, shutter_num=1, shutter_den=125)
    except Exception as e:
        print(f"ERROR: Live SONY_PIXEL_SHIFT_4 capture failed: {e}")
        sys.exit(1)

    print("Live SONY_PIXEL_SHIFT_4 capture succeeded!")
    print(f"  ISO:           {py_img_ps.iso}")
    print(f"  Shutter speed: {py_img_ps.shutter_speed}s")
    print(f"  Filepaths:     {py_img_ps.filepaths}")

    assert len(py_img_ps.filepaths) == 4
    raw_files_ps = py_img_ps.filepaths

    # Full-size CPython conversion
    print("Converting SONY_PIXEL_SHIFT_4 to CPython full-size NumPy array...")
    arr_full_py_ps = py_img_ps.to_numpy(half=False)

    # Half-size CPython conversion
    print("Converting SONY_PIXEL_SHIFT_4 to CPython half-size NumPy array...")
    arr_half_py_ps = py_img_ps.to_numpy(half=True)

    # C++ full-size TIFF output
    tiff_full_cpp_ps = "live_ps_full_cpp.tif"
    print(f"Running C++ test program to merge and convert SONY_PIXEL_SHIFT_4 raws to TIFF: {tiff_full_cpp_ps}...")
    cmd_full_ps = [cpp_executable, tiff_full_cpp_ps, "0"] + raw_files_ps
    proc_full_ps = subprocess.run(cmd_full_ps, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc_full_ps.returncode != 0:
        print(f"ERROR: C++ tiff program failed:\n{proc_full_ps.stderr}")
        sys.exit(1)

    # C++ half-size TIFF output
    tiff_half_cpp_ps = "live_ps_half_cpp.tif"
    print(f"Running C++ test program to merge and convert SONY_PIXEL_SHIFT_4 raws to TIFF: {tiff_half_cpp_ps}...")
    cmd_half_ps = [cpp_executable, tiff_half_cpp_ps, "1"] + raw_files_ps
    proc_half_ps = subprocess.run(cmd_half_ps, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc_half_ps.returncode != 0:
        print(f"ERROR: C++ tiff program failed:\n{proc_half_ps.stderr}")
        sys.exit(1)

    # Read and compare pixel payloads
    print("Comparing Python full-size NumPy array with C++ output TIFF payload...")
    with open(tiff_full_cpp_ps, "rb") as f:
        tiff_data_ps = f.read()
    cpp_arr_full_ps = np.frombuffer(tiff_data_ps[header_size:], dtype=np.uint16).reshape(arr_full_py_ps.shape)
    assert np.array_equal(arr_full_py_ps, cpp_arr_full_ps), "ERROR: Full-size SONY_PIXEL_SHIFT_4 pixel mismatch!"
    print("  [PASS] Full-size SONY_PIXEL_SHIFT_4 arrays match exactly!")

    print("Comparing Python half-size NumPy array with C++ output TIFF payload...")
    with open(tiff_half_cpp_ps, "rb") as f:
        tiff_data_half_ps = f.read()
    cpp_arr_half_ps = np.frombuffer(tiff_data_half_ps[header_size:], dtype=np.uint16).reshape(arr_half_py_ps.shape)
    assert np.array_equal(arr_half_py_ps, cpp_arr_half_ps), "ERROR: Half-size SONY_PIXEL_SHIFT_4 pixel mismatch!"
    print("  [PASS] Half-size SONY_PIXEL_SHIFT_4 arrays match exactly!")

    # Clean up pixel shift files
    py_img_ps.discard()
    if os.path.exists(tiff_full_cpp_ps):
        os.remove(tiff_full_cpp_ps)
    if os.path.exists(tiff_half_cpp_ps):
        os.remove(tiff_half_cpp_ps)
    print("Temporary files for SONY_PIXEL_SHIFT_4 test cleaned up successfully.")

    print("\n========================================")
    print("LIVE CAPTURE AND PARITY TEST SUCCESS: ALL PASSED")
    print("========================================")

if __name__ == "__main__":
    main()
