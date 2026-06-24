#!/usr/bin/env python3
"""
Auto-Exposure library for negicc-station.
Provides functions to search for optimal shutter speed by maximizing average dynamic range.
"""

import numpy as np

class AnnotatedArray(np.ndarray):
    def __new__(cls, input_array, **kwargs):
        obj = np.asarray(input_array).view(cls)
        for k, v in kwargs.items():
            setattr(obj, k, v)
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.__dict__.update(getattr(obj, '__dict__', {}))

SHUTTER_SPEEDS = [
    "30s", "25s", "20s", "15s", "13s", "10s", "8s", "6s", "5s", "4s", "3.2s", "2.5s", "2s", "1.6s", "1.3s", "1s",
    "0.8s", "0.6s", "0.5s", "0.4s", "1/3s", "1/4s", "1/5s", "1/6s", "1/8s", "1/10s", "1/13s", "1/15s", "1/20s",
    "1/25s", "1/30s", "1/40s", "1/50s", "1/60s", "1/80s", "1/100s", "1/125s", "1/160s", "1/200s", "1/250s",
    "1/320s", "1/400s", "1/500s", "1/640s", "1/800s", "1/1000s", "1/1250s", "1/1600s", "1/2000s", "1/2500s",
    "1/3200s", "1/4000s", "1/5000s", "1/6400s", "1/8000s"
]

def parse_shutter_speed(shutter_str):
    """Parses user-friendly string (e.g. '1/125s' or '2.5s') into (numerator, denominator) integers."""
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

def capture_exposure_frame(shutter_str, half=True, session=None):
    """
    Helper function to capture a single frame at a given shutter speed string.
    Loads negicc_station, triggers capture, converts to numpy array,
    discards image to free C++ memory, and returns the numpy array.
    """
    import negicc_station
    num, den = parse_shutter_speed(shutter_str)
    if session is not None:
        img = session.capture(type=0, shutter_num=num, shutter_den=den)
    else:
        img = negicc_station.capture(type=0, shutter_num=num, shutter_den=den)
    if not img:
        raise RuntimeError("Camera capture returned Null/None. The download might have failed or the camera is unresponsive.")
    try:
        arr = img.to_numpy(half=half)
        arr_annotated = AnnotatedArray(arr, iso=img.iso)
    finally:
        img.discard()
    return arr_annotated

def calculate_dynamic_range(arr):
    """Calculates the dynamic range for each channel and the average, on the center square
    (size equal to 2/3 of the shorter side of the image) to guide the auto-exposure search.
    Enforces that the 95th percentile of each channel must be below 13107.2 (80% of 16384).
    If p95 exceeds this limit, the channel is penalized to guide the auto-exposure search.
    """
    H, W, C = arr.shape
    square_size = int(min(H, W) * 2 // 3)
    y_start = (H - square_size) // 2
    x_start = (W - square_size) // 2
    cropped = arr[y_start:y_start+square_size, x_start:x_start+square_size, :]
    
    # Decimate if cropped array is large to make percentile calculations lightning fast
    cH, cW = cropped.shape[:2]
    total_pixels = cH * cW
    if total_pixels > 200000:
        step = int(np.sqrt(total_pixels / 200000))
        step = max(1, step)
        cropped = cropped[::step, ::step, :]
        
    OVEREXPOSURE_THRESHOLD = 13107.2  # 80% of 16384
    
    dr_channels = []
    # Loop over R, G, B channels
    for c in range(3):
        channel_data = cropped[:, :, c]
        p95 = np.percentile(channel_data, 95)
        p5 = np.percentile(channel_data, 5)
        dr = p95 - p5
        
        # Penalize if the 95th percentile exceeds the safety threshold
        if p95 > OVEREXPOSURE_THRESHOLD:
            excess = p95 - OVEREXPOSURE_THRESHOLD
            penalty = 100000.0 + 10000.0 * excess
            dr -= penalty
            
        dr_channels.append(dr)
        
    avg_dr = np.mean(dr_channels)
    return avg_dr, tuple(dr_channels)

def run_auto_exposure(start_shutter_str, capture_func, progress_callback=None, channel='ALL'):
    """
    Runs the auto-exposure hill-climbing search.
    
    Parameters:
      - start_shutter_str: The shutter speed string to start the search from.
      - capture_func: A callback function `capture_func(idx)` that returns a uint16 numpy array.
      - progress_callback: A callback function `progress_callback(step_idx, shutter_str, (dr_r, dr_g, dr_b), avg_dr)`
                           called after each capture step.
      - channel: The channel to maximize. 'ALL' (default) maximizes average dynamic range of all channels,
                 or 'R', 'G', 'B' to maximize only that specific channel's dynamic range.
                           
    Returns:
      - optimal_shutter_str: The shutter speed string with the maximum average dynamic range.
      - steps: List of steps taken, each step is a tuple (idx, avg_dr, (dr_r, dr_g, dr_b))
    """
    if start_shutter_str not in SHUTTER_SPEEDS:
        raise ValueError(f"Shutter speed '{start_shutter_str}' is not supported.")
        
    channel = channel.upper()
    if channel not in ['ALL', 'R', 'G', 'B']:
        raise ValueError(f"Invalid channel '{channel}'. Must be one of 'ALL', 'R', 'G', 'B'.")
        
    start_idx = SHUTTER_SPEEDS.index(start_shutter_str)
    steps = []
    
    # Inner helper to perform capture, measurement, print to stdout, and notify callback
    def evaluate_step(idx):
        shutter_str = SHUTTER_SPEEDS[idx]
        arr = capture_func(idx)
        iso = getattr(arr, 'iso', 100)
        
        # Calculate raw uncompensated dynamic range (p95 - p5) on center 2/3 square region
        H, W = arr.shape[:2]
        square_size = int(min(H, W) * 2 // 3)
        y_start = (H - square_size) // 2
        x_start = (W - square_size) // 2
        cropped = arr[y_start:y_start+square_size, x_start:x_start+square_size, :]
        raw_drs = []
        for c in range(3):
            ch_data = cropped[:, :, c]
            raw_drs.append(np.percentile(ch_data, 95) - np.percentile(ch_data, 5))
        raw_avg = sum(raw_drs) / 3.0
        
        # Calculate penalized DR for search logic
        avg_dr, (dr_r, dr_g, dr_b) = calculate_dynamic_range(arr)
        
        # Format metrics and print to stdout (showing uncompensated values)
        print(f"Auto-Exposure Step [Index {idx}] Shutter: {shutter_str} (ISO {iso}) -> R: {raw_drs[0]:.1f}, G: {raw_drs[1]:.1f}, B: {raw_drs[2]:.1f} | Avg DR: {raw_avg:.1f}")
        sys.stdout.flush()
        
        if progress_callback:
            progress_callback(idx, shutter_str, iso, raw_drs, raw_avg)
            
        return avg_dr, (dr_r, dr_g, dr_b)
        
    def get_metric(avg_val, ch_vals):
        if channel == 'ALL':
            return avg_val
        elif channel == 'R':
            return ch_vals[0]
        elif channel == 'G':
            return ch_vals[1]
        elif channel == 'B':
            return ch_vals[2]
        return avg_val

    curr_idx = start_idx
    curr_avg, curr_ch = evaluate_step(curr_idx)
    steps.append((curr_idx, curr_avg, curr_ch))
    curr_metric = get_metric(curr_avg, curr_ch)
    
    left_idx = curr_idx - 1  # Longer exposure (scale up time)
    right_idx = curr_idx + 1 # Shorter exposure (scale down time)
    
    left_metric = -1.0
    right_metric = -1.0
    
    if left_idx >= 0:
        left_avg, left_ch = evaluate_step(left_idx)
        steps.append((left_idx, left_avg, left_ch))
        left_metric = get_metric(left_avg, left_ch)
        
    if right_idx < len(SHUTTER_SPEEDS):
        right_avg, right_ch = evaluate_step(right_idx)
        steps.append((right_idx, right_avg, right_ch))
        right_metric = get_metric(right_avg, right_ch)
        
    # Determine direction of expansion
    best_metric = curr_metric
    direction = 0
    
    if left_metric > best_metric and left_metric >= right_metric:
        best_metric = left_metric
        direction = -1
        curr_idx = left_idx
    elif right_metric > best_metric and right_metric >= left_metric:
        best_metric = right_metric
        direction = 1
        curr_idx = right_idx
        
    if direction != 0:
        # Continue in the chosen direction until it is no longer expanding
        while True:
            next_idx = curr_idx + direction
            if next_idx < 0 or next_idx >= len(SHUTTER_SPEEDS):
                break
                
            next_avg, next_ch = evaluate_step(next_idx)
            steps.append((next_idx, next_avg, next_ch))
            next_metric = get_metric(next_avg, next_ch)
            
            if next_metric > best_metric:
                best_metric = next_metric
                curr_idx = next_idx
            else:
                break
                
    optimal_idx = curr_idx
    optimal_shutter = SHUTTER_SPEEDS[optimal_idx]
    
    return optimal_shutter, steps
