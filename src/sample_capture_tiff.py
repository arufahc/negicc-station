#!/usr/bin/env python3
"""
Simple capture and convert demo using the negicc_station extension.
Run with: ./venv/bin/python3 src/sample_capture_tiff.py
"""
import negicc_station

def main():
    # 1. Capture live single-shot (type=0) at 1/8s shutter speed
    print("Triggering single capture...")
    img = negicc_station.capture(type=0, shutter_num=1, shutter_den=8)
    
    print(f"Captured: {img.filepaths} (ISO={img.iso}, Shutter={img.shutter_speed}s)")

    # 2. Convert to a 16-bit RGB NumPy array (half-size downsampled)
    print("Converting to NumPy array...")
    arr = img.to_numpy(half=True)
    print(f"NumPy array shape: {arr.shape}, dtype: {arr.dtype}")

    # 3. Save directly to a 16-bit linear TIFF file
    print("Saving to linear TIFF...")
    img.write_tiff("output_linear.tif", half=True)

    # 4. Clean up the temporary RAW file from disk
    img.discard()
    print("Temporary files cleaned up.")

if __name__ == "__main__":
    main()
