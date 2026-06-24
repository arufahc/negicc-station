import numpy as np

def find_best_target_index(profile, raw_image, film_base_rgb, scan_shutter=None, scan_iso=100, base_shutter=None, base_iso=100):
    """
    Selects the best target from a FilmProfile based on the dynamic range of the raw image.
    Adjusts raw measurements by the exposure ratio between the captured scan and the film base
    to ensure transmittance calculations are physically correct.
    """
    if not hasattr(profile, 'raw_data') or 'targets' not in profile.raw_data:
        return 0, 0.0 # Default to first target if no targets list
        
    targets = profile.raw_data['targets']
    if len(targets) <= 1:
        return 0, 0.0

    # 1. Extract the 2/3 center square of the shorter side
    h, w = raw_image.shape[:2]
    shorter_side = min(h, w)
    square_size = int(shorter_side * 2 / 3)
    y_start = (h - square_size) // 2
    x_start = (w - square_size) // 2
    
    center_square = raw_image[y_start:y_start+square_size, x_start:x_start+square_size]
    
    # 2. Crosstalk correction
    # Assuming center_square is float or we convert to float
    cc_img = center_square.astype(np.float32)
    
    # Apply matrix if available
    if hasattr(profile, 'crosstalk_matrix') and profile.crosstalk_matrix is not None:
        M = profile.crosstalk_matrix
        cc_img = np.dot(cc_img, M.T)
        
    cc_img = np.clip(cc_img, 0, 65535)
    
    # 3. Compute 2% and 98% percentiles
    # For simplicity and stability, we can compute percentiles on the Green channel
    # Or average of RGB. Let's use the Green channel as it represents luminance well.
    p2 = np.percentile(cc_img[..., 1], 2)
    p98 = np.percentile(cc_img[..., 1], 98)
    
    # 4. Transmittance computed from captured film base
    fb_g = film_base_rgb[1]
    if fb_g <= 0:
        fb_g = 1.0 # Prevent division by zero

    # Compute exposure ratio: exposure_base / exposure_scan
    t_scan = scan_shutter if scan_shutter is not None else 1.0
    iso_scan = scan_iso if scan_iso is not None else 100
    
    if base_shutter is not None:
        t_base = base_shutter
        iso_base = base_iso if base_iso is not None else 100
    else:
        import sys
        print("[Warning] target_selection: base_shutter is None. Falling back to profile film_base_shutter and film_base_iso.", file=sys.stdout)
        sys.stdout.flush()
        film_base_shutter = getattr(profile, 'film_base_shutter', None)
        if film_base_shutter:
            from auto_exposure import parse_shutter_speed
            base_num, base_den = parse_shutter_speed(film_base_shutter)
            t_base = base_num / base_den
        else:
            t_base = 1.0
        iso_base = getattr(profile, 'film_base_iso', 100)

    exposure_base = t_base * (iso_base / 100.0)
    exposure_scan = t_scan * (iso_scan / 100.0)
    exposure_ratio = exposure_base / exposure_scan if exposure_scan > 0 else 1.0
        
    t_2 = (p2 / fb_g) * exposure_ratio
    t_98 = (p98 / fb_g) * exposure_ratio

    fb_r = film_base_rgb[0] if film_base_rgb[0] > 0 else 1.0
    fb_b = film_base_rgb[2] if film_base_rgb[2] > 0 else 1.0
    p2_r = np.percentile(cc_img[..., 0], 2)
    p98_r = np.percentile(cc_img[..., 0], 98)
    p2_b = np.percentile(cc_img[..., 2], 2)
    p98_b = np.percentile(cc_img[..., 2], 98)
    t_2_r = (p2_r / fb_r) * exposure_ratio
    t_98_r = (p98_r / fb_r) * exposure_ratio
    t_2_b = (p2_b / fb_b) * exposure_ratio
    t_98_b = (p98_b / fb_b) * exposure_ratio

    import sys
    print(f"[Profile Selection] Transmittance percentiles (2% / 98%):\n"
          f"  Red:   2%={t_2_r:.6f}, 98%={t_98_r:.6f}\n"
          f"  Green: 2%={t_2:.6f}, 98%={t_98:.6f} (Used for matching)\n"
          f"  Blue:  2%={t_2_b:.6f}, 98%={t_98_b:.6f}", file=sys.stdout)
    sys.stdout.flush()
    
    best_target_idx = 0
    min_dist_to_mid_grey = float('inf')
    
    for i, target in enumerate(targets):
        patches = target.get('patches', {})
        
        # Extract gs0 to gs23 green transmittance
        gs_transmittances = []
        for j in range(24):
            key = f"gs{j}"
            if key in patches and 'g' in patches[key]:
                # We need to normalize the target patch by the target's film base to get transmittance
                # Wait, the profile JSON patches are already absolute raw values.
                # The profile has its own film base:
                prof_fb_g = profile.raw_data.get('film_base', {}).get('g', {}).get('avg', 1.0)
                patch_t = patches[key]['g'] / prof_fb_g
                gs_transmittances.append(patch_t)
            else:
                # Fallback if missing
                gs_transmittances.append(1.0 - (j / 23.0)) 
                
        # Find enclosing patches for t_98 and t_2
        # gs0 is the highest transmittance (whitest/clearest), gs23 is lowest (blackest/densest)
        # We find where t_98 fits
        idx_98 = 0
        for j in range(24):
            if gs_transmittances[j] <= t_98:
                idx_98 = j
                break
        
        idx_2 = 23
        for j in range(23, -1, -1):
            if gs_transmittances[j] >= t_2:
                idx_2 = j
                break
                
        # Mid-grey is between gs11 and gs12 (index 11.5)
        center_idx = (idx_98 + idx_2) / 2.0
        dist = abs(center_idx - 11.5)
        
        if dist < min_dist_to_mid_grey:
            min_dist_to_mid_grey = dist
            best_target_idx = i
            
    return best_target_idx, min_dist_to_mid_grey

