#!/usr/bin/env python3
"""
Test script for the auto-exposure library.
Uses the modular auto_exposure.py implementation.
"""

import os
import sys
import time

# Ensure project root is in python path
project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_dir)
sys.path.insert(0, os.path.join(project_dir, 'src'))

# Preload the Sony CrSDK shared library from the virtual environment
lib_path = os.path.join(project_dir, 'venv/bin/libCr_Core.so')
if os.path.exists(lib_path):
    import ctypes
    ctypes.CDLL(lib_path)

import auto_exposure

def test_capture_func(idx):
    shutter_str = auto_exposure.SHUTTER_SPEEDS[idx]
    print(f"[{idx}] Capturing at shutter speed {shutter_str} (single-shot, half-size)...")
    return auto_exposure.capture_exposure_frame(shutter_str, half=True)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test script for the auto-exposure library.")
    parser.add_argument("start_speed", nargs="?", default="1/8s", help="Shutter speed string to start the search from (default: 1/8s)")
    parser.add_argument("-c", "--channel", default=None, help="Channel to maximize: ALL, R, G, B (default: ALL)")
    parser.add_argument("pos_channel", nargs="?", default=None, help="Channel (positional) if not using -c/--channel")

    args = parser.parse_args()

    # Resolve channel
    channel = "ALL"
    if args.channel is not None:
        channel = args.channel
    elif args.pos_channel is not None:
        channel = args.pos_channel

    channel = channel.upper()
    if channel not in ["ALL", "R", "G", "B"]:
        parser.error(f"argument channel: invalid choice: '{channel}' (choose from 'ALL', 'R', 'G', 'B')")
        
    print(f"Testing auto-exposure library starting from {args.start_speed} using channel {channel}...")
    
    opt_speed, steps = auto_exposure.run_auto_exposure(
        start_shutter_str=args.start_speed,
        capture_func=test_capture_func,
        progress_callback=None,
        channel=channel
    )
    
    print(f"\nOptimal speed found: {opt_speed}")
    print("\n--- Steps summary ---")
    for i, (idx, avg_dr, ch_dr) in enumerate(steps):
        print(f"Step {i+1}: Index {idx} ({auto_exposure.SHUTTER_SPEEDS[idx]}) -> R: {ch_dr[0]:.1f}, G: {ch_dr[1]:.1f}, B: {ch_dr[2]:.1f} | Avg: {avg_dr:.1f}")
