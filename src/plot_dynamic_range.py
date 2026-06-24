#!/usr/bin/env python3
import os
import json
import matplotlib.pyplot as plt
import numpy as np

def parse_shutter(shutter_str):
    shutter_str = shutter_str.rstrip('s')
    if '/' in shutter_str:
        num, den = shutter_str.split('/')
        return float(num) / float(den)
    return float(shutter_str)

def main():
    # Relative path from repository root
    json_path = "profiles/profile_Portra400_20260624_012038.json"
    if not os.path.exists(json_path):
        print(f"Error: JSON file not found at {json_path}")
        return

    with open(json_path, 'r') as f:
        d = json.load(f)

    film_base = d['film_base']
    fb_r = film_base['r']['avg']
    fb_g = film_base['g']['avg']
    fb_b = film_base['b']['avg']
    
    t_base = parse_shutter(film_base['shutter']) * (film_base['iso'] / 100.0)

    targets = d['targets']
    
    # Sort targets by exposure time so curves are in order of exposure
    for t in targets:
        t['exp_time'] = parse_shutter(t['shutter']) * (t['iso'] / 100.0)
    targets = sorted(targets, key=lambda x: x['exp_time'])

    gs_keys = [f"gs{i}" for i in range(24)]

    # Plot normalized R, G, B for each target
    plt.style.use('dark_background')
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    fig.patch.set_facecolor('#181818')
    
    colors = ['#ff4444', '#ffaa00', '#ffff00', '#44ff44', '#00ffff', '#4444ff', '#ff00ff']
    
    for ax in axes:
        ax.set_facecolor('#121212')
        ax.grid(True, color='#2c2c2c', linestyle='--', linewidth=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#444444')
        ax.spines['bottom'].set_color('#444444')
        ax.tick_params(colors='#888888')

    channels = ['r', 'g', 'b']
    channel_names = ['Red Channel', 'Green Channel', 'Blue Channel']
    fb_vals = [fb_r, fb_g, fb_b]

    for c_idx, channel in enumerate(channels):
        ax = axes[c_idx]
        fb_val = fb_vals[c_idx]
        
        for t_idx, t in enumerate(targets):
            name = t['name']
            shutter = t['shutter']
            iso = t['iso']
            t_exp = t['exp_time']
            
            y_vals = []
            for k in gs_keys:
                p_val = t['patches'][k][channel]
                # Normalization by film base and scaling by exposure difference
                norm_val = (p_val / fb_val) * (t_base / t_exp)
                y_vals.append(norm_val)
                
            ax.plot(range(24), y_vals, marker='o', label=f"{name} ({shutter}, ISO {iso})", color=colors[t_idx % len(colors)], linewidth=1.5, markersize=4)
        
        ax.set_title(f"Normalized Grayscale Patches - {channel_names[c_idx]}", color='#ffffff', fontsize=12, pad=10)
        ax.set_ylabel("Normalized Transmittance", color='#ffffff', fontsize=10)
        if c_idx == 1:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True, facecolor='#1e1e1e', edgecolor='#444444', labelcolor='#ffffff')

    axes[2].set_xlabel("IT8 Grayscale Patch (gs0 = White/Dense, gs23 = Black/Clear)", color='#ffffff', fontsize=10)
    plt.xticks(range(24), [f"gs{i}" for i in range(24)], rotation=45, fontsize=8)
    
    plt.suptitle("Dynamic Range of Film Negative Grayscale Patches (Normalized by Film Base & Exposure Diff)", color='#ffffff', fontsize=14, y=0.98)
    plt.tight_layout()
    
    # Save a copy to dynamic_range_plot.png
    plt.savefig("dynamic_range_plot.png", dpi=150, facecolor=fig.get_facecolor(), bbox_inches='tight')
    print("Plot saved successfully to dynamic_range_plot.png")
    
    print("Launching interactive Matplotlib window...")
    plt.show()

if __name__ == "__main__":
    main()
