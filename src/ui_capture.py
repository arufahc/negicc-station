#!/usr/bin/env python3
"""
GTK3-based GUI application for the Negative Film Scanning Station.
Provides a modern dark theme interface to trigger captures, select shutter speeds,
display 16-bit linear C++ converted previews, and show metadata.
"""

import os
import sys
import threading
import time
import numpy as np
import json
import gi
import struct
import ctypes

# Require GTK3
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk, Gdk, GdkPixbuf, GLib
import cairo

from matplotlib.figure import Figure
from matplotlib.backends.backend_gtk3agg import FigureCanvasGTK3Agg as FigureCanvas

import atexit
import signal
import zipfile
import tarfile
import shutil
import negicc_station
from film_profiling import FilmProfile
import color_conversion
from target_selection import find_best_target_index
import auto_exposure

def read_arw_metadata(filepath):
    # Default fallback values
    shutter_speed = 0.125
    iso = 100
    try:
        with open(filepath, 'rb') as f:
            f.seek(0)
            byte_order = f.read(2)
            if byte_order == b'II':
                endian = '<'
            elif byte_order == b'MM':
                endian = '>'
            else:
                return shutter_speed, iso
                
            magic = struct.unpack(endian + 'H', f.read(2))[0]
            if magic != 42:
                return shutter_speed, iso
                
            f.seek(4)
            ifd0_offset = struct.unpack(endian + 'I', f.read(4))[0]
            
            # Helper to parse tags in an IFD offset
            def parse_ifd(offset):
                f.seek(offset)
                num_tags = struct.unpack(endian + 'H', f.read(2))[0]
                tags_dict = {}
                for i in range(num_tags):
                    f.seek(offset + 2 + i * 12)
                    tag_data = f.read(12)
                    if len(tag_data) < 12:
                        break
                    tag, fmt, count, val_offset = struct.unpack(endian + 'HHII', tag_data)
                    tags_dict[tag] = (fmt, count, val_offset)
                return tags_dict
                
            ifd0 = parse_ifd(ifd0_offset)
            
            # Find EXIF IFD pointer: Tag 34665 (0x8769)
            exif_ptr = ifd0.get(0x8769)
            if exif_ptr:
                exif_offset = exif_ptr[2]
                exif_ifd = parse_ifd(exif_offset)
                
                # Tag 33434 (0x829A): ExposureTime (RATIONAL = 5)
                exp_tag = exif_ifd.get(0x829a)
                if exp_tag:
                    fmt, count, offset = exp_tag
                    f.seek(offset)
                    num, den = struct.unpack(endian + 'II', f.read(8))
                    if den > 0:
                        shutter_speed = num / den
                        
                # Tag 34855 (0x8827): ISOSpeedRatings (SHORT = 3)
                iso_tag = exif_ifd.get(0x8827)
                if iso_tag:
                    fmt, count, val = iso_tag
                    if fmt == 3 and count == 1:
                        if endian == '<':
                            iso = val & 0xFFFF
                        else:
                            iso = (val >> 16) & 0xFFFF
                    elif fmt == 3 and count > 1:
                        f.seek(val)
                        iso = struct.unpack(endian + 'H', f.read(2))[0]
    except Exception as e:
        print(f"[Metadata] Warning: Failed to parse EXIF metadata from {filepath}: {e}", file=sys.stderr)
        
    return shutter_speed, iso

def unpack_archive_and_find_arws(archive_path, extract_dir):
    """Unpack archive and return list of absolute/relative file paths to .ARW files."""
    if os.path.exists(extract_dir):
        try:
            shutil.rmtree(extract_dir)
        except Exception:
            pass
    os.makedirs(extract_dir, exist_ok=True)
    
    # Check if zip
    if zipfile.is_zipfile(archive_path):
        with zipfile.ZipFile(archive_path, 'r') as z:
            z.extractall(extract_dir)
    # Check if tar
    elif tarfile.is_tarfile(archive_path):
        with tarfile.open(archive_path, 'r:*') as t:
            t.extractall(extract_dir)
    else:
        raise ValueError("Unsupported archive format. Supported formats: .zip, .tar, .tar.gz, .tgz, .tar.xz")
        
    # Find all .ARW files (case insensitive)
    arw_files = []
    for root, dirs, files in os.walk(extract_dir):
        for f in files:
            if f.lower().endswith('.arw'):
                arw_files.append(os.path.join(root, f))
                
    # Sort them alphabetically to keep them in consistent order (important for pixel shift order)
    arw_files.sort()
    return arw_files

def _register_cleanup_handlers():
    def cleanup():
        try:
            negicc_station.cleanup_temp_files()
        except Exception:
            pass
        try:
            extract_dir = os.path.join(os.getcwd(), "build", "tmp_load_archive")
            if os.path.exists(extract_dir):
                shutil.rmtree(extract_dir)
        except Exception:
            pass
    atexit.register(cleanup)
    
    def sig_handler(signum, frame):
        cleanup()
        sys.exit(128 + signum)
        
    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

_register_cleanup_handlers()

# Resolve path for local imports
src_dir = os.path.dirname(os.path.abspath(__file__))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# Helper functions for EXIF/TIFF orientation tags and numpy image transformations
def set_tiff_orientation_inplace(filepath, orientation_value):
    try:
        with open(filepath, 'r+b') as f:
            f.seek(0)
            byte_order = f.read(2)
            if byte_order == b'II': endian = '<'
            elif byte_order == b'MM': endian = '>'
            else: return False
            
            magic = struct.unpack(endian + 'H', f.read(2))[0]
            if magic != 42: return False
            
            f.seek(4)
            offset = struct.unpack(endian + 'I', f.read(4))[0]
            f.seek(offset)
            num_tags = struct.unpack(endian + 'H', f.read(2))[0]
            
            tags = []
            orientation_found = False
            
            for i in range(num_tags):
                tag_offset = offset + 2 + i * 12
                f.seek(tag_offset)
                tag_data = f.read(12)
                tag = struct.unpack(endian + 'H', tag_data[:2])[0]
                if tag == 274:
                    f.seek(tag_offset + 8)
                    f.write(struct.pack(endian + 'H', orientation_value))
                    orientation_found = True
                tags.append(tag_data)
                
            if orientation_found: return True
            
            f.seek(offset + 2 + num_tags * 12)
            next_ifd_offset = struct.unpack(endian + 'I', f.read(4))[0]
            
            new_tag_data = struct.pack(endian + 'HHII', 274, 3, 1, orientation_value)
            tags.append(new_tag_data)
            
            def get_tag_id(data): return struct.unpack(endian + 'H', data[:2])[0]
            tags.sort(key=get_tag_id)
            
            f.seek(0, 2)
            new_ifd_offset = f.tell()
            
            f.write(struct.pack(endian + 'H', len(tags)))
            for tag_data in tags:
                f.write(tag_data)
            f.write(struct.pack(endian + 'I', next_ifd_offset))
            
            f.seek(4)
            f.write(struct.pack(endian + 'I', new_ifd_offset))
            return True
    except Exception as e:
        print(f"Error setting TIFF orientation: {e}")
        return False

def get_exif_orientation(hflip, vflip, rot_cw):
    grid = np.array([[1, 2], [3, 4]])
    if rot_cw == 90: grid = np.rot90(grid, -1)
    elif rot_cw == 180: grid = np.rot90(grid, -2)
    elif rot_cw == 270: grid = np.rot90(grid, -3)
    if hflip: grid = np.fliplr(grid)
    if vflip: grid = np.flipud(grid)
    
    key = tuple(grid.flatten())
    mapping = {
        (1, 2, 3, 4): 1,
        (2, 1, 4, 3): 2,
        (4, 3, 2, 1): 3,
        (3, 4, 1, 2): 4,
        (1, 3, 2, 4): 5,
        (3, 1, 4, 2): 6,
        (4, 2, 3, 1): 7,
        (2, 4, 1, 3): 8
    }
    return mapping.get(key, 1)

def apply_transforms_numpy(img_array, hflip, vflip, rot_cw):
    if rot_cw == 90: img_array = np.rot90(img_array, -1)
    elif rot_cw == 180: img_array = np.rot90(img_array, -2)
    elif rot_cw == 270: img_array = np.rot90(img_array, -3)
    if hflip: img_array = np.fliplr(img_array)
    if vflip: img_array = np.flipud(img_array)
    return img_array

def map_raw_to_transformed_coords(x_raw, y_raw, w_raw, h_raw, hflip, vflip, rot_cw):
    x, y = x_raw, y_raw
    w, h = w_raw, h_raw
    if rot_cw == 90:
        x_new = h - 1 - y
        y_new = x
        w, h = h, w
        x, y = x_new, y_new
    elif rot_cw == 180:
        x_new = w - 1 - x
        y_new = h - 1 - y
        x, y = x_new, y_new
    elif rot_cw == 270:
        x_new = y
        y_new = w - 1 - x
        w, h = h, w
        x, y = x_new, y_new
    if hflip:
        x = w - 1 - x
    if vflip:
        y = h - 1 - y
    return x, y

def map_transformed_to_raw_coords(x_trans, y_trans, w_trans, h_trans, hflip, vflip, rot_cw):
    x, y = x_trans, y_trans
    w, h = w_trans, h_trans
    if hflip:
        x = w - 1 - x
    if vflip:
        y = h - 1 - y
    if rot_cw == 90:
        x_new = y
        y_new = w - 1 - x
        x, y = x_new, y_new
    elif rot_cw == 180:
        x_new = w - 1 - x
        y_new = h - 1 - y
        x, y = x_new, y_new
    elif rot_cw == 270:
        x_new = h - 1 - y
        y_new = x
        x, y = x_new, y_new
    return x, y

def map_transformed_rect_to_raw(rect, w_trans, h_trans, hflip, vflip, rot_cw):
    if rect is None:
        return None
    x1, y1, x2, y2 = rect
    rx1, ry1 = map_transformed_to_raw_coords(x1, y1, w_trans, h_trans, hflip, vflip, rot_cw)
    rx2, ry2 = map_transformed_to_raw_coords(x2, y2, w_trans, h_trans, hflip, vflip, rot_cw)
    return (min(rx1, rx2), min(ry1, ry2), max(rx1, rx2), max(ry1, ry2))

def map_raw_rect_to_transformed(rect, w_raw, h_raw, hflip, vflip, rot_cw):
    if rect is None:
        return None
    x1, y1, x2, y2 = rect
    tx1, ty1 = map_raw_to_transformed_coords(x1, y1, w_raw, h_raw, hflip, vflip, rot_cw)
    tx2, ty2 = map_raw_to_transformed_coords(x2, y2, w_raw, h_raw, hflip, vflip, rot_cw)
    return (min(tx1, tx2), min(ty1, ty2), max(tx1, tx2), max(ty1, ty2))

class HistogramCanvas(Gtk.Box):
    def __init__(self):
        super().__init__(orientation=Gtk.Orientation.VERTICAL)
        # Enforce 3:2 width-to-height ratio (2:3 height-to-width)
        self.figure = Figure(figsize=(3.0, 2.0), dpi=100)
        self.figure.patch.set_facecolor('#1e1e1e')
        self.canvas = FigureCanvas(self.figure)
        self.pack_start(self.canvas, True, True, 0)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_facecolor('#121212')
        self.ax.spines['top'].set_visible(False)
        self.ax.spines['right'].set_visible(False)
        self.ax.spines['left'].set_color('#444444')
        self.ax.spines['bottom'].set_color('#444444')
        self.ax.tick_params(colors='#888888', labelsize=7)
        self.ax.grid(True, color='#2c2c2c', linestyle='--', linewidth=0.5)
        self.figure.subplots_adjust(left=0.04, right=0.96, top=0.95, bottom=0.15)
        
        # Explicitly set canvas height request to 220px to match the fixed 360px sidebar (330px width content)
        self.canvas.set_size_request(-1, 220)

    def clear(self):
        self.ax.clear()
        self.ax.set_facecolor('#121212')
        self.ax.spines['top'].set_visible(False)
        self.ax.spines['right'].set_visible(False)
        self.ax.spines['left'].set_color('#444444')
        self.ax.spines['bottom'].set_color('#444444')
        self.ax.tick_params(colors='#888888', labelsize=7)
        self.ax.grid(True, color='#2c2c2c', linestyle='--', linewidth=0.5)
        self.ax.set_xlim(0, 16384)
        self.ax.set_ylim(0, 1.05)
        self.ax.set_yticks([])
        self.canvas.draw_idle()

    def plot_histogram(self, data, is_corrected, has_icc, show_overexposure=True):
        self.ax.clear()
        self.ax.set_facecolor('#121212')
        self.ax.spines['top'].set_visible(False)
        self.ax.spines['right'].set_visible(False)
        self.ax.spines['left'].set_color('#444444')
        self.ax.spines['bottom'].set_color('#444444')
        self.ax.tick_params(colors='#888888', labelsize=7)
        self.ax.grid(True, color='#2c2c2c', linestyle='--', linewidth=0.5)
        
        max_val = 65535 if (is_corrected and has_icc) else 16384
        
        if data is None or data.size == 0:
            self.ax.set_xlim(0, max_val)
            self.ax.set_ylim(0, 1.05)
            self.ax.set_yticks([])
            self.canvas.draw_idle()
            return
            
        # Decimate large data arrays for extremely fast plotting and percentile calculations
        H, W = data.shape[:2]
        total_pixels = H * W
        if total_pixels > 200000:
            step = int(np.sqrt(total_pixels / 200000))
            step = max(1, step)
            data_sampled = data[::step, ::step, :]
        else:
            data_sampled = data
            
        max_val = 65535 if (is_corrected and has_icc) else 16384
        
        bins = 256
        hist_r, _ = np.histogram(data_sampled[:, :, 0], bins=bins, range=(0, max_val))
        hist_g, _ = np.histogram(data_sampled[:, :, 1], bins=bins, range=(0, max_val))
        hist_b, _ = np.histogram(data_sampled[:, :, 2], bins=bins, range=(0, max_val))
        max_hist_val = max(hist_r.max(), hist_g.max(), hist_b.max(), 1)
        
        hist_r_norm = hist_r / max_hist_val
        hist_g_norm = hist_g / max_hist_val
        hist_b_norm = hist_b / max_hist_val
        
        bins_x = np.linspace(0, max_val, bins)
        
        # Plot channels with area fills
        self.ax.plot(bins_x, hist_r_norm, color='#ff6666', alpha=0.8, linewidth=1.2)
        self.ax.fill_between(bins_x, 0, hist_r_norm, color='#ff6666', alpha=0.12)
        
        self.ax.plot(bins_x, hist_g_norm, color='#66ff66', alpha=0.8, linewidth=1.2)
        self.ax.fill_between(bins_x, 0, hist_g_norm, color='#66ff66', alpha=0.12)
        
        self.ax.plot(bins_x, hist_b_norm, color='#66aaff', alpha=0.8, linewidth=1.2)
        self.ax.fill_between(bins_x, 0, hist_b_norm, color='#66aaff', alpha=0.12)
        
        if show_overexposure:
            ovr_val = 0.8 * max_val
            self.ax.axvline(ovr_val, color='#e74c3c', linestyle='-', alpha=0.8, linewidth=1.5)
            self.ax.text(ovr_val - (max_val * 0.015), 0.95, "Overexposure (80%)", color='#e74c3c', fontsize=7.5,
                         horizontalalignment='right', verticalalignment='top', rotation=90,
                         bbox=dict(boxstyle='round,pad=0.15', facecolor='#121212', alpha=0.6, edgecolor='none'))

        # Calculate percentiles and metrics excluding 5% borders
        H_s, W_s, C_s = data_sampled.shape
        h_border = int(H_s * 0.05)
        w_border = int(W_s * 0.05)
        cropped = data_sampled[h_border:H_s-h_border, w_border:W_s-w_border, :]
        
        p2_r = float(np.percentile(cropped[:, :, 0], 2))
        p98_r = float(np.percentile(cropped[:, :, 0], 98))
        p2_g = float(np.percentile(cropped[:, :, 1], 2))
        p98_g = float(np.percentile(cropped[:, :, 1], 98))
        p2_b = float(np.percentile(cropped[:, :, 2], 2))
        p98_b = float(np.percentile(cropped[:, :, 2], 98))
        
        dr_r = p98_r - p2_r
        dr_g = p98_g - p2_g
        dr_b = p98_b - p2_b
        avg_dr = (dr_r + dr_g + dr_b) / 3.0
        
        mean_r = float(np.mean(cropped[:, :, 0]))
        mean_g = float(np.mean(cropped[:, :, 1]))
        mean_b = float(np.mean(cropped[:, :, 2]))
        avg_mean = (mean_r + mean_g + mean_b) / 3.0
        
        # Plot percentile indicators
        colors = ['#ff6666', '#66ff66', '#66aaff']
        p2 = [p2_r, p2_g, p2_b]
        p98 = [p98_r, p98_g, p98_b]
        for i in range(3):
            self.ax.axvline(p2[i], color=colors[i], linestyle='--', alpha=0.6, linewidth=1.0)
            self.ax.axvline(p98[i], color=colors[i], linestyle='--', alpha=0.6, linewidth=1.0)
            
        # Plot cross markers for averages
        means = [mean_r, mean_g, mean_b]
        hists_norm = [hist_r_norm, hist_g_norm, hist_b_norm]
        channel_labels = ['R', 'G', 'B']
        sorted_indices = np.argsort(means)
        for rank, idx in enumerate(sorted_indices):
            m_val = means[idx]
            h_norm = hists_norm[idx]
            bin_idx = int(round(m_val / max_val * (len(h_norm) - 1)))
            bin_idx = max(0, min(len(h_norm) - 1, bin_idx))
            y_val = h_norm[bin_idx]
            
            self.ax.plot(m_val, y_val, marker='x', color=colors[idx], markersize=8, markeredgewidth=1.0)
            text_y = 0.5 + (rank * 0.12)
            self.ax.plot([m_val, m_val], [y_val, text_y], color=colors[idx], linestyle=':', alpha=0.5, linewidth=1.0)
            
            self.ax.text(m_val, text_y, f"{channel_labels[idx]}_avg: {int(m_val)}",
                         color=colors[idx], fontsize=8, fontweight='bold',
                         horizontalalignment='center', verticalalignment='center',
                         bbox=dict(boxstyle='round,pad=0.2', facecolor='#121212', alpha=0.85, edgecolor='none'))
                         
        self.ax.set_xlim(0, max_val)
        self.ax.set_ylim(0, 1.05)
        self.ax.set_yticks([])
        
        # Stats textbox
        text_str = (
            f"R: [2%:{int(p2_r)}, 98%:{int(p98_r)}] DR:{dr_r:.1f} Mean:{mean_r:.1f}\n"
            f"G: [2%:{int(p2_g)}, 98%:{int(p98_g)}] DR:{dr_g:.1f} Mean:{mean_g:.1f}\n"
            f"B: [2%:{int(p2_b)}, 98%:{int(p98_b)}] DR:{dr_b:.1f} Mean:{mean_b:.1f}\n"
            f"Avg DR: {avg_dr:.1f} | Avg Value: {avg_mean:.1f}"
        )
        props = dict(boxstyle='round', facecolor='#1e1e1e', alpha=0.8, edgecolor='#333333')
        self.ax.text(0.02, 0.98, text_str, transform=self.ax.transAxes, fontsize=8.5, color='#ffffff',
                     verticalalignment='top', bbox=props, family='monospace')
                     
        self.canvas.draw_idle()

class AEGraphCanvas(Gtk.Box):
    def __init__(self, select_callback=None):
        super().__init__(orientation=Gtk.Orientation.VERTICAL)
        self.select_callback = select_callback
        
        self.figure = Figure(figsize=(3.0, 1.8), dpi=100)
        self.figure.patch.set_facecolor('#1e1e1e')
        self.canvas = FigureCanvas(self.figure)
        self.pack_start(self.canvas, True, True, 0)
        
        self.ax = self.figure.add_subplot(111)
        self.ax.set_facecolor('#121212')
        self.ax.spines['top'].set_visible(False)
        self.ax.spines['right'].set_visible(False)
        self.ax.spines['left'].set_color('#444444')
        self.ax.spines['bottom'].set_color('#444444')
        self.ax.tick_params(colors='#888888', labelsize=7)
        self.ax.grid(True, color='#2c2c2c', linestyle='--', linewidth=0.5)
        
        self.figure.subplots_adjust(left=0.18, right=0.95, top=0.90, bottom=0.25)
        self.canvas.set_size_request(-1, 140)
        
        # State
        self.steps = []
        self.current_tooltip = None
        
        # Connect matplotlib events
        self.canvas.mpl_connect("motion_notify_event", self.on_motion)
        self.canvas.mpl_connect("button_press_event", self.on_click)

    def clear(self):
        self.steps = []
        self.canvas.set_tooltip_text(None)
        self.current_tooltip = None
        self.ax.clear()
        self.ax.set_facecolor('#121212')
        self.ax.spines['top'].set_visible(False)
        self.ax.spines['right'].set_visible(False)
        self.ax.spines['left'].set_color('#444444')
        self.ax.spines['bottom'].set_color('#444444')
        self.ax.tick_params(colors='#888888', labelsize=7)
        self.ax.grid(True, color='#2c2c2c', linestyle='--', linewidth=0.5)
        self.canvas.draw_idle()

    def add_step(self, ss, iso, dr_r, dr_g, dr_b, avg_dr, overexposed=None):
        try:
            shutter_idx = auto_exposure.SHUTTER_SPEEDS.index(ss)
        except ValueError:
            shutter_idx = 0
            
        step_data = {
            'shutter_idx': shutter_idx,
            'shutter': ss,
            'iso': iso,
            'dr_r': dr_r,
            'dr_g': dr_g,
            'dr_b': dr_b,
            'avg_dr': avg_dr,
            'overexposed': overexposed or [False, False, False]
        }
        self.steps.append(step_data)
        self.redraw()

    def redraw(self):
        self.ax.clear()
        self.ax.set_facecolor('#121212')
        self.ax.spines['top'].set_visible(False)
        self.ax.spines['right'].set_visible(False)
        self.ax.spines['left'].set_color('#444444')
        self.ax.spines['bottom'].set_color('#444444')
        self.ax.tick_params(colors='#888888', labelsize=7)
        self.ax.grid(True, color='#2c2c2c', linestyle='--', linewidth=0.5)
        
        if not self.steps:
            self.canvas.draw_idle()
            return
            
        x = np.arange(1, len(self.steps) + 1)
        
        for i, step in enumerate(self.steps):
            xi = x[i]
            r, g, b = step['dr_r'], step['dr_g'], step['dr_b']
            ymin = min(r, g, b)
            ymax = max(r, g, b)
            
            # Draw wick
            self.ax.plot([xi, xi], [ymin, ymax], color='#555555', linewidth=1.5, zorder=1)
            
            # Draw R, G, B markers
            over_r, over_g, over_b = step.get('overexposed', [False, False, False])
            marker_r = 'x' if over_r else '_'
            marker_g = 'x' if over_g else '_'
            marker_b = 'x' if over_b else '_'
            
            self.ax.plot(xi, r, marker=marker_r, color='#ff6666', markersize=10 if not over_r else 8, markeredgewidth=2.5 if over_r else 2.0, zorder=2)
            self.ax.plot(xi, g, marker=marker_g, color='#66ff66', markersize=10 if not over_g else 8, markeredgewidth=2.5 if over_g else 2.0, zorder=2)
            self.ax.plot(xi, b, marker=marker_b, color='#66aaff', markersize=10 if not over_b else 8, markeredgewidth=2.5 if over_b else 2.0, zorder=2)
            
            # Draw Avg DR point
            self.ax.plot(xi, step['avg_dr'], marker='o', color='#ffffff', markersize=4, zorder=3)
            
        self.ax.set_xticks(x)
        self.ax.set_xticklabels([step['shutter'] for step in self.steps], rotation=45, ha='right', fontsize=6.5)
        
        # Adjust Y limits
        all_vals = []
        for s in self.steps:
            all_vals.extend([s['dr_r'], s['dr_g'], s['dr_b']])
        if all_vals:
            ymin, ymax = min(all_vals), max(all_vals)
            padding = max(100.0, (ymax - ymin) * 0.1)
            self.ax.set_ylim(max(0, ymin - padding), ymax + padding)
            
        self.ax.set_xlim(0.5, len(self.steps) + 0.5)
        self.figure.subplots_adjust(left=0.12, right=0.95, top=0.90, bottom=0.25)
        self.canvas.draw_idle()

    def on_motion(self, event):
        if not self.steps or event.inaxes != self.ax:
            if self.current_tooltip is not None:
                self.canvas.set_tooltip_text(None)
                self.current_tooltip = None
            return
            
        x_mouse = event.xdata
        if x_mouse is None:
            if self.current_tooltip is not None:
                self.canvas.set_tooltip_text(None)
                self.current_tooltip = None
            return
            
        idx = int(round(x_mouse)) - 1
        if 0 <= idx < len(self.steps):
            step = self.steps[idx]
            if abs(x_mouse - (idx + 1)) < 0.4:
                r, g, b = step['dr_r'], step['dr_g'], step['dr_b']
                avg = step['avg_dr']
                
                over_r, over_g, over_b = step.get('overexposed', [False, False, False])
                r_txt = f"{r:.0f} (OVER!)" if over_r else f"{r:.0f}"
                g_txt = f"{g:.0f} (OVER!)" if over_g else f"{g:.0f}"
                b_txt = f"{b:.0f} (OVER!)" if over_b else f"{b:.0f}"
                
                text = (
                    f"Shutter: {step['shutter']}\n"
                    f"ISO: {step['iso']}\n"
                    f"DR Red:   {r_txt}\n"
                    f"DR Green: {g_txt}\n"
                    f"DR Blue:  {b_txt}\n"
                    f"DR Avg:   {avg:.0f}"
                )
                if self.current_tooltip != text:
                    self.canvas.set_tooltip_text(text)
                    self.current_tooltip = text
                return
                
        if self.current_tooltip is not None:
            self.canvas.set_tooltip_text(None)
            self.current_tooltip = None

    def on_click(self, event):
        if not self.steps or event.inaxes != self.ax:
            return
            
        x_mouse = event.xdata
        if x_mouse is None:
            return
            
        idx = int(round(x_mouse)) - 1
        if 0 <= idx < len(self.steps):
            if abs(x_mouse - (idx + 1)) < 0.4:
                step = self.steps[idx]
                if self.select_callback:
                    GLib.idle_add(self.select_callback, step['shutter'])

def get_target_transmittances(profile, target_idx):
    # returns list of (r, g, b) transmittance for gs0..gs23
    targets = profile.raw_data.get('targets', [])
    if target_idx >= len(targets):
        return []
    target = targets[target_idx]
    
    # film base exposure
    from film_profiling import parse_shutter_speed
    fb_shutter = getattr(profile, 'film_base_shutter', '1/8s')
    fb_iso = getattr(profile, 'film_base_iso', 100)
    fb_num, fb_den = parse_shutter_speed(fb_shutter)
    t_base = (fb_num / fb_den) * (fb_iso / 100.0)
    
    # target exposure
    tgt_shutter = target.get('shutter', '1/8s')
    tgt_iso = target.get('iso', 100)
    tgt_num, tgt_den = parse_shutter_speed(tgt_shutter)
    t_exp = (tgt_num / tgt_den) * (tgt_iso / 100.0)
    
    exposure_ratio = t_base / t_exp if t_exp > 0 else 1.0
    
    fb_r = profile.film_base.get('r_avg', 1.0)
    fb_g = profile.film_base.get('g_avg', 1.0)
    fb_b = profile.film_base.get('b_avg', 1.0)
    if fb_r <= 0: fb_r = 1.0
    if fb_g <= 0: fb_g = 1.0
    if fb_b <= 0: fb_b = 1.0
    
    patches = target.get('patches', {})
    res = []
    for i in range(24):
        key = f"gs{i}"
        if key in patches:
            p_val = patches[key]
            r_t = (p_val.get('r', 0.0) / fb_r) * exposure_ratio
            g_t = (p_val.get('g', 0.0) / fb_g) * exposure_ratio
            b_t = (p_val.get('b', 0.0) / fb_b) * exposure_ratio
            res.append((r_t, g_t, b_t))
        else:
            res.append((0.0, 0.0, 0.0))
    return res

class CalibrationTargetsDetailsWindow(Gtk.Window):
    def __init__(self, parent_app):
        super().__init__(title="Calibration Target Details & Dynamic Range")
        self.app = parent_app
        self.set_transient_for(parent_app)
        self.set_default_size(1150, 750)
        self.set_modal(False)
        self.set_destroy_with_parent(True)
        
        # Main layout: Horizontal split
        hbox = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=15)
        hbox.set_margin_top(10)
        hbox.set_margin_bottom(10)
        hbox.set_margin_start(10)
        hbox.set_margin_end(10)
        self.add(hbox)
        
        # Left pane: Table & Text info
        left_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        hbox.pack_start(left_box, False, False, 0)
        left_box.set_size_request(450, -1)
        
        lbl_table_title = Gtk.Label()
        lbl_table_title.set_markup("<b>Target List</b> (Click to select active target)")
        lbl_table_title.set_xalign(0.0)
        left_box.pack_start(lbl_table_title, False, False, 0)
        
        # TreeView ListStore
        # Columns: Active (str), Index (int), Name (str), Shutter/ISO (str), gs0 (str), gs23 (str)
        self.liststore = Gtk.ListStore(str, int, str, str, str, str)
        self.treeview = Gtk.TreeView(model=self.liststore)
        
        # Add columns
        r_indicator = Gtk.CellRendererText()
        col_ind = Gtk.TreeViewColumn("Active", r_indicator, text=0)
        col_ind.set_alignment(0.5)
        self.treeview.append_column(col_ind)
        
        r_name = Gtk.CellRendererText()
        col_name = Gtk.TreeViewColumn("Target Name", r_name, text=2)
        self.treeview.append_column(col_name)
        
        r_exp = Gtk.CellRendererText()
        col_exp = Gtk.TreeViewColumn("Exposure", r_exp, text=3)
        self.treeview.append_column(col_exp)
        
        r_gs0 = Gtk.CellRendererText()
        col_gs0 = Gtk.TreeViewColumn("gs0 (Densest) R/G/B T", r_gs0, text=4)
        self.treeview.append_column(col_gs0)
        
        r_gs23 = Gtk.CellRendererText()
        col_gs23 = Gtk.TreeViewColumn("gs23 (Lightest) R/G/B T", r_gs23, text=5)
        self.treeview.append_column(col_gs23)
        
        # Selection
        select = self.treeview.get_selection()
        select.connect("changed", self.on_details_tree_selection_changed)
        
        scroll = Gtk.ScrolledWindow()
        scroll.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        scroll.add(self.treeview)
        left_box.pack_start(scroll, True, True, 0)
        
        # Textual details frame
        frame = Gtk.Frame(label="Transmittance Summary")
        frame_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=5)
        frame_box.set_margin_top(8)
        frame_box.set_margin_bottom(8)
        frame_box.set_margin_start(8)
        frame_box.set_margin_end(8)
        frame.add(frame_box)
        left_box.pack_start(frame, False, False, 0)
        
        self.lbl_info = Gtk.Label()
        self.lbl_info.set_xalign(0.0)
        self.lbl_info.set_line_wrap(True)
        frame_box.pack_start(self.lbl_info, True, True, 0)
        
        # Right pane: Matplotlib Canvas
        right_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=5)
        hbox.pack_start(right_box, True, True, 0)
        
        self.figure = Figure(figsize=(7, 8), dpi=100)
        self.figure.patch.set_facecolor('#181818')
        self.canvas = FigureCanvas(self.figure)
        right_box.pack_start(self.canvas, True, True, 0)
        
        # Populate & draw initially
        self.populate_targets_table()
        self.update_plot()
        self.update_text_info()
        
        # Set selection in treeview to current active target
        self.select_active_in_treeview()
        
    def select_active_in_treeview(self):
        select = self.treeview.get_selection()
        # Temporarily block changed signal to avoid redundant updates
        select.handler_block_by_func(self.on_details_tree_selection_changed)
        for i, row in enumerate(self.liststore):
            if row[1] == self.app.selected_target_idx:
                select.select_path(Gtk.TreePath.new_from_indices([i]))
                break
        select.handler_unblock_by_func(self.on_details_tree_selection_changed)

    def populate_targets_table(self):
        self.liststore.clear()
        profile = self.app.profile
        if not profile or not hasattr(profile, 'raw_data'):
            return
            
        targets = profile.raw_data.get('targets', [])
        for idx, target in enumerate(targets):
            name = target.get('name', f"Target {idx}")
            shutter = target.get('shutter', '1/8s')
            iso = target.get('iso', 100)
            shutter_iso = f"{shutter} @ ISO {iso}"
            
            # Get gs0 and gs23 transmittances
            ts = get_target_transmittances(profile, idx)
            if ts and len(ts) == 24:
                gs0_r, gs0_g, gs0_b = ts[0]
                gs23_r, gs23_g, gs23_b = ts[23]
                
                gs0_str = f"{gs0_r:.3f} / {gs0_g:.3f} / {gs0_b:.3f}"
                gs23_str = f"{gs23_r:.3f} / {gs23_g:.3f} / {gs23_b:.3f}"
            else:
                gs0_str = "N/A"
                gs23_str = "N/A"
                
            active = "➔" if idx == self.app.selected_target_idx else ""
            self.liststore.append([active, idx, name, shutter_iso, gs0_str, gs23_str])

    def update_indicators(self):
        for row in self.liststore:
            row_idx = row[1]
            if row_idx == self.app.selected_target_idx:
                row[0] = "➔"
            else:
                row[0] = ""
                
    def on_details_tree_selection_changed(self, selection):
        model, treeiter = selection.get_selected()
        if treeiter is not None:
            idx = model[treeiter][1]
            name = model[treeiter][2]
            if idx != self.app.selected_target_idx:
                self.app.selected_target_idx = idx
                print(f"[Profile] Target selected via Details Window: Index {idx}, Name: {name}", file=sys.stdout)
                sys.stdout.flush()
                
                # Invalidate cache and update main capture preview
                self.app.capture_converted_rgb_cache = None
                self.app.capture_corr_hist_cache = None
                self.app.update_capture_preview()
                
                # Update main treeview selection to stay in sync
                main_select = self.app.target_treeview.get_selection()
                for i, row in enumerate(self.app.target_liststore):
                    if row[1] == idx:
                        main_select.select_path(Gtk.TreePath.new_from_indices([i]))
                        break
                
                # Redraw ourselves
                self.update_indicators()
                self.update_plot()
                self.update_text_info()

    def get_captured_film_transmittance_range(self):
        if self.app.raw_image is None or self.app.raw_linear_pixels is None:
            return None
            
        # Check cache (keyed on identity of raw image and active profile)
        cache_key = (id(self.app.raw_image), id(self.app.profile))
        if getattr(self, '_cached_range_key', None) == cache_key:
            if getattr(self, '_cached_range', None) is not None:
                return self._cached_range
            
        raw_image = self.app.raw_linear_pixels
        profile = self.app.profile
        
        h, w = raw_image.shape[:2]
        shorter_side = min(h, w)
        square_size = int(shorter_side * 2 / 3)
        y_start = (h - square_size) // 2
        x_start = (w - square_size) // 2
        
        # Subsample every 10th pixel for 100x fewer elements while maintaining statistical percentile accuracy
        center_square = raw_image[y_start:y_start+square_size:10, x_start:x_start+square_size:10]
        
        cc_img = center_square.astype(np.float32)
        if profile and hasattr(profile, 'crosstalk_matrix') and profile.crosstalk_matrix is not None:
            M = profile.crosstalk_matrix
            cc_img = np.dot(cc_img, M.T)
        cc_img = np.clip(cc_img, 0, 65535)
        
        p2_r = np.percentile(cc_img[..., 0], 2)
        p98_r = np.percentile(cc_img[..., 0], 98)
        p2_g = np.percentile(cc_img[..., 1], 2)
        p98_g = np.percentile(cc_img[..., 1], 98)
        p2_b = np.percentile(cc_img[..., 2], 2)
        p98_b = np.percentile(cc_img[..., 2], 98)
        
        t_scan = self.app.raw_image.shutter_speed
        iso_scan = self.app.raw_image.iso
        exposure_scan = t_scan * (iso_scan / 100.0)
        
        if self.app.film_base_img:
            t_base = self.app.film_base_img.shutter_speed
            iso_base = self.app.film_base_img.iso
        else:
            from film_profiling import parse_shutter_speed
            fb_shutter = getattr(profile, 'film_base_shutter', '1/8s')
            fb_iso = getattr(profile, 'film_base_iso', 100)
            fb_num, fb_den = parse_shutter_speed(fb_shutter)
            t_base = fb_num / fb_den
            iso_base = fb_iso
            
        exposure_base = t_base * (iso_base / 100.0)
        exposure_ratio = exposure_base / exposure_scan if exposure_scan > 0 else 1.0
        
        if self.app.film_base_rgb is not None:
            fb_r, fb_g, fb_b = self.app.film_base_rgb
        elif profile:
            fb_r = profile.film_base.get('r_avg', 1.0)
            fb_g = profile.film_base.get('g_avg', 1.0)
            fb_b = profile.film_base.get('b_avg', 1.0)
        else:
            fb_r = fb_g = fb_b = 1.0
            
        if fb_r <= 0: fb_r = 1.0
        if fb_g <= 0: fb_g = 1.0
        if fb_b <= 0: fb_b = 1.0
        
        t2_r = (p2_r / fb_r) * exposure_ratio
        t98_r = (p98_r / fb_r) * exposure_ratio
        t2_g = (p2_g / fb_g) * exposure_ratio
        t98_g = (p98_g / fb_g) * exposure_ratio
        t2_b = (p2_b / fb_b) * exposure_ratio
        t98_b = (p98_b / fb_b) * exposure_ratio
        
        res_dict = {
            'r': (t2_r, t98_r),
            'g': (t2_g, t98_g),
            'b': (t2_b, t98_b)
        }
        self._cached_range_key = cache_key
        self._cached_range = res_dict
        return res_dict

    def update_text_info(self):
        profile = self.app.profile
        if not profile:
            self.lbl_info.set_markup("<i>No calibration profile loaded</i>")
            return
            
        targets = profile.raw_data.get('targets', [])
        selected_idx = self.app.selected_target_idx
        
        info_text = f"<b>Active Profile:</b> {profile.film_name}\n"
        if selected_idx < len(targets):
            tgt = targets[selected_idx]
            info_text += f"<b>Active Target:</b> {tgt.get('name', 'N/A')}\n"
            
        film_range = self.get_captured_film_transmittance_range()
        if film_range:
            t2_r, t98_r = film_range['r']
            t2_g, t98_g = film_range['g']
            t2_b, t98_b = film_range['b']
            
            info_text += (
                f"\n<b>Captured Film Transmittance (2% - 98%):</b>\n"
                f"  Red Channel:   {t2_r:.3f} to {t98_r:.3f}\n"
                f"  Green Channel: {t2_g:.3f} to {t98_g:.3f}\n"
                f"  Blue Channel:  {t2_b:.3f} to {t98_b:.3f}\n"
            )
        else:
            info_text += "\n<i>No captured film image loaded to show percentiles.</i>"
            
        self.lbl_info.set_markup(info_text)

    def update_plot(self):
        self.figure.clear()
        
        profile = self.app.profile
        if not profile or not hasattr(profile, 'raw_data'):
            # Draw placeholder message
            ax = self.figure.add_subplot(111)
            ax.set_facecolor('#121212')
            ax.text(0.5, 0.5, "No calibration profile loaded", color='#888888', ha='center', va='center')
            ax.set_axis_off()
            self.canvas.draw()
            return
            
        # Create 3 subplots (Red, Green, Blue) sharing the x-axis
        axes = self.figure.subplots(3, 1, sharex=True)
        self.figure.patch.set_facecolor('#181818')
        
        for ax in axes:
            ax.set_facecolor('#121212')
            ax.grid(True, color='#2c2c2c', linestyle='--', linewidth=0.5)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color('#444444')
            ax.spines['bottom'].set_color('#444444')
            ax.tick_params(colors='#888888')
            
        channels = ['r', 'g', 'b']
        channel_names = ['Red Channel Transmittance', 'Green Channel Transmittance', 'Blue Channel Transmittance']
        channel_colors = ['#ff4444', '#44ff44', '#4444ff']
        
        targets = profile.raw_data.get('targets', [])
        selected_idx = self.app.selected_target_idx
        
        # Precompute transmittances for all targets
        target_ts = []
        for idx in range(len(targets)):
            target_ts.append(get_target_transmittances(profile, idx))
            
        # Draw target curves
        for c_idx, channel in enumerate(channels):
            ax = axes[c_idx]
            
            # 1. Draw non-selected targets as thin grey/translucent curves
            for idx, target in enumerate(targets):
                if idx == selected_idx:
                    continue
                ts = target_ts[idx]
                if not ts:
                    continue
                y_vals = [t[c_idx] for t in ts]
                ax.plot(range(24), y_vals, color='#555555', alpha=0.4, linewidth=1.0)
                
            # 2. Draw selected target as thick colorful curve
            if selected_idx < len(targets):
                ts = target_ts[selected_idx]
                if ts:
                    y_vals = [t[c_idx] for t in ts]
                    target = targets[selected_idx]
                    name = target.get('name', f"Target {selected_idx}")
                    shutter = target.get('shutter', 'Unknown')
                    iso = target.get('iso', 100)
                    ax.plot(range(24), y_vals, marker='o', label=f"{name} ({shutter}, ISO {iso}) [Active]", 
                            color=channel_colors[c_idx], linewidth=2.0, markersize=4)
                    ax.legend(loc='upper right', frameon=True, facecolor='#1e1e1e', edgecolor='#444444', labelcolor='#ffffff', fontsize=8)
            
            # 3. Draw captured film range if available
            film_range = self.get_captured_film_transmittance_range()
            if film_range:
                t2, t98 = film_range[channel]
                # Shade the region between t2 and t98
                ax.axhspan(t2, t98, color=channel_colors[c_idx], alpha=0.15)
                # Draw lines
                ax.axhline(t2, color=channel_colors[c_idx], linestyle='--', alpha=0.6, linewidth=1.0)
                ax.axhline(t98, color=channel_colors[c_idx], linestyle='--', alpha=0.6, linewidth=1.0)
                # Add text label at the edge of the line
                ax.text(23.5, t2, "Film p2", color=channel_colors[c_idx], alpha=0.8, fontsize=8, ha='right', va='bottom')
                ax.text(23.5, t98, "Film p98", color=channel_colors[c_idx], alpha=0.8, fontsize=8, ha='right', va='top')
                
            ax.set_title(channel_names[c_idx], color='#ffffff', fontsize=10, pad=5)
            ax.set_ylabel("Transmittance", color='#ffffff', fontsize=8)
            
        axes[2].set_xlabel("Grayscale Patch (gs0 = White/Dense, gs23 = Black/Clear)", color='#ffffff', fontsize=9)
        axes[2].set_xticks(range(24))
        axes[2].set_xticklabels([f"gs{i}" for i in range(24)], rotation=45, fontsize=7)
        self.figure.tight_layout()
        self.canvas.draw()
        
    def refresh_all(self):
        self.populate_targets_table()
        self.select_active_in_treeview()
        self.update_plot()
        self.update_text_info()

class ScanningAppWindow(Gtk.Window):
    def __init__(self):
        super().__init__(title="Sony Film Scanning Station")
        self.set_default_size(1400, 800)
        self.connect("destroy", self.on_destroy)

        # Force GTK dark theme
        settings = Gtk.Settings.get_default()
        settings.set_property("gtk-application-prefer-dark-theme", True)

        # Apply CSS styling
        self.apply_css()

        # Application State
        self.profile = None
        self.profile_filename = ""
        self.has_icc = False
        self.has_crosstalk = False
        
        self.raw_image = None
        self.raw_linear_pixels = None
        
        self.film_base_img = None
        self.film_base_raw_linear = None
        self.film_base_rgb = None
        self.last_save_folder = None
        self.capture_converted_rgb_cache = None
        self.capture_corr_hist_cache = None
        self.base_converted_rgb_cache = None
        self.ae_graph = None
        self.details_window = None
        
        self.camera_session = None
        self.is_connected = False
        self.is_connecting = False
        self.is_capturing = False
        
        self.gain = 1.0
        self.orientation = 0
        self.hflip = False
        self.vflip = False
        
        self.selected_target_idx = 0
        
        self.capture_rect_start = None
        self.capture_rect_end = None
        self.capture_rect_raw = None
        self.is_dragging_capture = False
        self.capture_preview_pixbuf = None
        self.scaled_capture_pixbuf = None
        
        self.base_rect_start = None
        self.base_rect_end = None
        self.base_rect_raw = None
        self.is_dragging_base = False
        self.base_preview_pixbuf = None
        self.scaled_base_pixbuf = None

        # Build UI layout
        self.init_ui()
        self.update_profile_dependencies()

        # Connect window keypress listener
        self.connect("key-press-event", self.on_key_press)

        # Initial background camera connection
        self.connect_camera()
        GLib.timeout_add_seconds(2, self.poll_camera_connection)

    def apply_css(self):
        css_provider = Gtk.CssProvider()
        css_provider.load_from_data(b"""
            .sidebar { background-color: #1e1e1e; padding: 15px; }
            .right-sidebar { background-color: #1e1e1e; padding: 15px 5px; }
            .preview-container { background-color: #121212; padding: 10px; }
            
            button {
                transition: background-image 0.1s ease-in-out, background-color 0.1s ease-in-out, box-shadow 0.1s ease-in-out;
            }
            
            .btn-action { 
                font-weight: bold; 
                padding: 8px 12px; 
                border-radius: 6px; 
                border: none;
            }
            
            .btn-green { 
                background-image: linear-gradient(to bottom, #2ea44f, #2c974b); 
                color: white; 
                box-shadow: 0 2px 4px rgba(0,0,0,0.3);
            }
            .btn-green:hover {
                background-image: linear-gradient(to bottom, #3bc262, #2ea44f);
                box-shadow: 0 4px 8px rgba(0,0,0,0.4);
            }
            .btn-green:active {
                background-image: linear-gradient(to bottom, #2c974b, #206a35);
                box-shadow: inset 0 2px 4px rgba(0,0,0,0.5);
            }
            
            .btn-yellow { 
                background-image: linear-gradient(to bottom, #b8860b, #996515); 
                color: white; 
                box-shadow: 0 2px 4px rgba(0,0,0,0.3);
            }
            .btn-yellow:hover {
                background-image: linear-gradient(to bottom, #d49b13, #b8860b);
                box-shadow: 0 4px 8px rgba(0,0,0,0.4);
            }
            .btn-yellow:active {
                background-image: linear-gradient(to bottom, #996515, #734b0f);
                box-shadow: inset 0 2px 4px rgba(0,0,0,0.5);
            }
            
            .btn-blue { 
                background-image: linear-gradient(to bottom, #1f77b4, #125c8d); 
                color: white; 
                box-shadow: 0 2px 4px rgba(0,0,0,0.3);
            }
            .btn-blue:hover {
                background-image: linear-gradient(to bottom, #3182bd, #1f77b4);
                box-shadow: 0 4px 8px rgba(0,0,0,0.4);
            }
            .btn-blue:active {
                background-image: linear-gradient(to bottom, #125c8d, #0b3d5e);
                box-shadow: inset 0 2px 4px rgba(0,0,0,0.5);
            }
            
            .meta-label { font-family: monospace; font-size: 11px; color: #b3b3b3; }
            .warning-label { color: #ffaa00; font-weight: bold; }
            .status-indicator { font-weight: bold; font-size: 14px; padding: 10px; background-color: #2b2b2b; border-radius: 5px; }
        """)
        Gtk.StyleContext.add_provider_for_screen(
            Gdk.Screen.get_default(), css_provider, Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION
        )

    def init_ui(self):
        main_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL)
        self.add(main_box)
        
        # ================= LEFT SIDEBAR =================
        left_sidebar = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        left_sidebar.get_style_context().add_class("sidebar")
        left_sidebar.set_size_request(210, -1)
        main_box.pack_start(left_sidebar, False, False, 0)
        
        # Camera status indicator
        self.lbl_camera_status = Gtk.Label()
        self.lbl_camera_status.set_markup("<span><span foreground='#e6a23c'>●</span> <b>Camera: Connecting...</b></span>")
        self.lbl_camera_status.get_style_context().add_class("status-indicator")
        self.lbl_camera_status.set_line_wrap(True)
        self.lbl_camera_status.set_width_chars(25)
        self.lbl_camera_status.set_max_width_chars(25)
        self.lbl_camera_status.set_xalign(0.0)
        left_sidebar.pack_start(self.lbl_camera_status, False, False, 10)
        
        # Profile Section
        left_sidebar.pack_start(Gtk.Label(label="<b>Calibration Profile</b>", use_markup=True), False, False, 2)
        self.btn_load = Gtk.Button(label="Load Profile...")
        self.btn_load.connect("clicked", self.on_load_profile)
        left_sidebar.pack_start(self.btn_load, False, False, 0)
        
        self.lbl_profile_info = Gtk.Label()
        self.lbl_profile_info.set_markup("<b>Profile:</b> None")
        self.lbl_profile_info.set_line_wrap(True)
        self.lbl_profile_info.set_width_chars(25)
        self.lbl_profile_info.set_max_width_chars(25)
        self.lbl_profile_info.set_xalign(0.0)
        self.lbl_profile_info.get_style_context().add_class("meta-label")
        left_sidebar.pack_start(self.lbl_profile_info, False, False, 2)
        
        btn_reset = Gtk.Button(label="Reset Profile")
        btn_reset.connect("clicked", self.on_reset_profile)
        left_sidebar.pack_start(btn_reset, False, False, 0)
        
        left_sidebar.pack_start(Gtk.Separator(), False, False, 5)
        
        # Capture Settings Section
        left_sidebar.pack_start(Gtk.Label(label="<b>Capture Mode</b>", use_markup=True), False, False, 0)
        self.mode_combo = Gtk.ComboBoxText()
        self.mode_combo.append("0", "Single Shot Capture")
        self.mode_combo.append("1", "Sony 4-Shot Pixel Shift")
        self.mode_combo.set_active(0)
        left_sidebar.pack_start(self.mode_combo, False, False, 0)

        left_sidebar.pack_start(Gtk.Label(label="<b>Shutter Speed</b>", use_markup=True), False, False, 0)
        self.shutter_combo = Gtk.ComboBoxText()
        for speed in auto_exposure.SHUTTER_SPEEDS:
            self.shutter_combo.append(speed, speed)
        self.shutter_combo.set_active(auto_exposure.SHUTTER_SPEEDS.index("1/8s"))
        left_sidebar.pack_start(self.shutter_combo, False, False, 0)
        
        self.ae_checkbox = Gtk.CheckButton(label="Auto Exposure")
        self.ae_checkbox.connect("toggled", self.on_ae_toggled)
        left_sidebar.pack_start(self.ae_checkbox, False, False, 5)
        
        left_sidebar.pack_start(Gtk.Separator(), False, False, 5)
        
        # Capture Action (Combined)
        self.btn_capture = Gtk.Button(label="Capture Image")
        self.btn_capture.get_style_context().add_class("btn-action")
        self.btn_capture.get_style_context().add_class("btn-green")
        self.btn_capture.connect("clicked", self.on_capture_clicked)
        left_sidebar.pack_start(self.btn_capture, False, False, 0)

        # Load Image/Archive Action
        self.btn_load_image = Gtk.Button(label="Load Image/Archive...")
        self.btn_load_image.get_style_context().add_class("btn-action")
        self.btn_load_image.get_style_context().add_class("btn-blue")
        self.btn_load_image.connect("clicked", self.on_load_image_clicked)
        left_sidebar.pack_start(self.btn_load_image, False, False, 5)

        # AE Search Graph (replaces Capture Log)
        self.ae_graph_frame = Gtk.Frame(label="AE Search Graph")
        self.ae_graph = AEGraphCanvas(select_callback=self.on_ae_graph_bar_selected)
        self.ae_graph_frame.add(self.ae_graph)
        left_sidebar.pack_start(self.ae_graph_frame, False, False, 5)

        # AE Steps output display frame
        self.ae_steps_frame = Gtk.Frame(label="Auto-Exposure Steps")
        self.ae_steps_frame.set_no_show_all(True)
        ae_steps_scroll = Gtk.ScrolledWindow()
        ae_steps_scroll.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        ae_steps_scroll.set_min_content_height(100)
        self.ae_steps_listbox = Gtk.ListBox()
        self.ae_steps_listbox.set_selection_mode(Gtk.SelectionMode.NONE)
        ae_steps_scroll.add(self.ae_steps_listbox)
        self.ae_steps_frame.add(ae_steps_scroll)
        left_sidebar.pack_start(self.ae_steps_frame, False, False, 5)
        


        # Spinner & status display
        status_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=5)
        self.spinner = Gtk.Spinner()
        status_box.pack_start(self.spinner, False, False, 0)
        self.lbl_status = Gtk.Label(label="Status: Connecting")
        self.lbl_status.set_line_wrap(True)
        self.lbl_status.set_width_chars(25)
        self.lbl_status.set_max_width_chars(25)
        self.lbl_status.set_xalign(0.0)
        status_box.pack_start(self.lbl_status, True, True, 0)
        left_sidebar.pack_end(status_box, False, False, 5)

        # ================= CENTER NOTEBOOK =================
        self.notebook = Gtk.Notebook()
        main_box.pack_start(self.notebook, True, True, 0)
        
        # Tab 0: Capture preview area
        self.capture_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        self.capture_tab_label = Gtk.Label(label="Capture (Base Needed)")
        self.capture_tab_label.get_style_context().add_class("warning-label")
        self.notebook.append_page(self.capture_box, self.capture_tab_label)
        self.init_capture_tab()
        
        # Tab 1: Film Base preview area
        self.base_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        self.base_tab_label = Gtk.Label(label="Film Base")
        self.notebook.append_page(self.base_box, self.base_tab_label)
        self.init_base_tab()

        # ================= RIGHT SIDEBAR =================
        right_sidebar = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        right_sidebar.get_style_context().add_class("right-sidebar")
        right_sidebar.set_size_request(360, -1)
        main_box.pack_start(right_sidebar, False, False, 0)
        
        lbl_hist_raw = Gtk.Label(label="RAW Linear (Uncorrected)")
        right_sidebar.pack_start(lbl_hist_raw, False, False, 0)
        self.hist_raw = HistogramCanvas()
        right_sidebar.pack_start(self.hist_raw, False, False, 0)
        
        lbl_hist_corr = Gtk.Label(label="Corrected Preview")
        right_sidebar.pack_start(lbl_hist_corr, False, False, 0)
        self.hist_corr = HistogramCanvas()
        right_sidebar.pack_start(self.hist_corr, False, False, 0)
        


        # Tab switch listener
        self.notebook.connect("switch-page", self.on_tab_changed)

    def init_capture_tab(self):
        # TOP TOOLBAR
        top_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=5)
        top_box.set_margin_top(5)
        top_box.set_margin_bottom(5)
        top_box.set_margin_start(5)
        top_box.set_margin_end(5)
        self.capture_box.pack_start(top_box, False, False, 0)
        
        self.btn_save_tiff = Gtk.Button(label="Save TIFF...")
        self.btn_save_tiff.get_style_context().add_class("btn-action")
        self.btn_save_tiff.set_sensitive(False)
        self.btn_save_tiff.connect("clicked", self.on_save_tiff)
        top_box.pack_start(self.btn_save_tiff, False, False, 5)
        
        self.btn_save_raw_capture = Gtk.Button(label="Save RAW...")
        self.btn_save_raw_capture.get_style_context().add_class("btn-action")
        self.btn_save_raw_capture.set_sensitive(False)
        self.btn_save_raw_capture.connect("clicked", self.on_save_raw_capture)
        top_box.pack_start(self.btn_save_raw_capture, False, False, 5)
        
        top_box.pack_start(Gtk.Separator(orientation=Gtk.Orientation.VERTICAL), False, False, 5)
        
        top_box.pack_start(Gtk.Label(label="Gain: "), False, False, 0)
        btn_g_down = Gtk.Button(label="-")
        btn_g_down.connect("clicked", lambda x: self.adj_gain(-0.10))
        top_box.pack_start(btn_g_down, False, False, 0)
        self.entry_gain = Gtk.Entry()
        self.entry_gain.set_editable(False)
        self.entry_gain.set_text("1.00")
        self.entry_gain.set_width_chars(5)
        self.entry_gain.connect("activate", self.on_gain_entry_activated)
        top_box.pack_start(self.entry_gain, False, False, 5)
        btn_g_up = Gtk.Button(label="+")
        btn_g_up.connect("clicked", lambda x: self.adj_gain(0.10))
        top_box.pack_start(btn_g_up, False, False, 0)
        
        btn_g_reset = Gtk.Button(label="1.0")
        btn_g_reset.connect("clicked", lambda x: self.reset_gain())
        top_box.pack_start(btn_g_reset, False, False, 5)
        
        top_box.pack_start(Gtk.Separator(orientation=Gtk.Orientation.VERTICAL), False, False, 5)
        
        for rot in [0, 90, 180, 270]:
            b = Gtk.Button(label=f"{rot}°")
            b.connect("clicked", self.on_set_rot, rot)
            top_box.pack_start(b, False, False, 0)
        
        self.btn_hf = Gtk.ToggleButton(label="H-Flip")
        self.btn_hf.connect("toggled", self.on_hflip)
        top_box.pack_start(self.btn_hf, False, False, 0)
        
        self.btn_vf = Gtk.ToggleButton(label="V-Flip")
        self.btn_vf.connect("toggled", self.on_vflip)
        top_box.pack_start(self.btn_vf, False, False, 0)
        
        # PREVIEW SCREEN AREA
        preview_event_box = Gtk.EventBox()
        preview_event_box.add_events(Gdk.EventMask.POINTER_MOTION_MASK | Gdk.EventMask.BUTTON_RELEASE_MASK | Gdk.EventMask.BUTTON_PRESS_MASK)
        preview_event_box.connect("button-press-event", self.on_capture_press)
        preview_event_box.connect("button-release-event", self.on_capture_release)
        preview_event_box.connect("motion-notify-event", self.on_capture_motion)
        
        self.capture_image_area = Gtk.DrawingArea()
        self.capture_image_area.connect("draw", self.on_draw_capture)
        self.capture_image_area.connect("size-allocate", self.on_capture_area_size_allocate)
        preview_event_box.add(self.capture_image_area)
        self.capture_box.pack_start(preview_event_box, True, True, 0)
        
        # BOTTOM: Target Table list
        target_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        
        tgt_header_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL)
        lbl_tgt_title = Gtk.Label()
        lbl_tgt_title.set_markup("<b>Calibration Targets</b>")
        lbl_tgt_title.set_xalign(0.0)
        tgt_header_box.pack_start(lbl_tgt_title, True, True, 0)
        
        self.btn_target_details = Gtk.Button(label="Details...")
        self.btn_target_details.connect("clicked", self.on_show_target_details)
        tgt_header_box.pack_end(self.btn_target_details, False, False, 0)
        
        target_box.pack_start(tgt_header_box, False, False, 5)
        
        self.target_liststore = Gtk.ListStore(str, int, str, str, str) # Active, Index, Target Name, gs0 T, gs23 T
        self.target_treeview = Gtk.TreeView(model=self.target_liststore)
        
        renderer_text = Gtk.CellRendererText()
        col_active = Gtk.TreeViewColumn("Active", renderer_text, text=0)
        col_active.set_alignment(0.5)
        col1 = Gtk.TreeViewColumn("Target Name", renderer_text, text=2)
        col_gs0 = Gtk.TreeViewColumn("gs0 (Densest) R/G/B T", renderer_text, text=3)
        col_gs23 = Gtk.TreeViewColumn("gs23 (Lightest) R/G/B T", renderer_text, text=4)
        self.target_treeview.append_column(col_active)
        self.target_treeview.append_column(col1)
        self.target_treeview.append_column(col_gs0)
        self.target_treeview.append_column(col_gs23)
        
        select = self.target_treeview.get_selection()
        select.connect("changed", self.on_target_selection_changed)
        
        scroll = Gtk.ScrolledWindow()
        scroll.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        scroll.set_min_content_height(100)
        scroll.add(self.target_treeview)
        target_box.pack_start(scroll, False, False, 0)
        
        self.capture_box.pack_end(target_box, False, False, 0)

    def init_base_tab(self):
        # TOP TOOLBAR
        top_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=10)
        top_box.set_margin_top(5)
        top_box.set_margin_bottom(5)
        top_box.set_margin_start(5)
        top_box.set_margin_end(5)
        self.base_box.pack_start(top_box, False, False, 0)
        
        btn_read = Gtk.Button(label="Read Film Base Values")
        btn_read.connect("clicked", self.on_read_film_base)
        top_box.pack_start(btn_read, False, False, 0)
        
        self.btn_save_raw_base = Gtk.Button(label="Save RAW...")
        self.btn_save_raw_base.get_style_context().add_class("btn-action")
        self.btn_save_raw_base.set_sensitive(False)
        self.btn_save_raw_base.connect("clicked", self.on_save_raw_base)
        top_box.pack_start(self.btn_save_raw_base, False, False, 0)
        
        self.lbl_base_vals = Gtk.Label(label="Raw: -- | Corr: --")
        top_box.pack_start(self.lbl_base_vals, False, False, 0)

        # PREVIEW AREA
        preview_event_box = Gtk.EventBox()
        preview_event_box.add_events(Gdk.EventMask.POINTER_MOTION_MASK | Gdk.EventMask.BUTTON_RELEASE_MASK | Gdk.EventMask.BUTTON_PRESS_MASK)
        preview_event_box.connect("button-press-event", self.on_base_press)
        preview_event_box.connect("button-release-event", self.on_base_release)
        preview_event_box.connect("motion-notify-event", self.on_base_motion)
        
        self.base_image_area = Gtk.DrawingArea()
        self.base_image_area.connect("draw", self.on_draw_base)
        self.base_image_area.connect("size-allocate", self.on_base_area_size_allocate)
        preview_event_box.add(self.base_image_area)
        self.base_box.pack_start(preview_event_box, True, True, 0)

    def poll_camera_connection(self):
        if self.is_connected:
            if not negicc_station.is_camera_connected():
                self.is_connected = False
                self.update_connection_ui(False, "Camera disconnected.")
        return True

    def connect_camera(self, on_success=None):
        if self.is_connecting:
            return
        if self.is_connected:
            if on_success:
                on_success()
            return
        self.is_connecting = True
        self.btn_capture.set_sensitive(False)
        self.lbl_camera_status.set_markup("<span><span foreground='#e6a23c'>●</span> <b>Camera: Connecting...</b></span>")
        
        def run():
            try:
                if self.camera_session is None:
                    self.camera_session = negicc_station.CameraSession()
                ok = self.camera_session.connect()
                if ok:
                    self.is_connected = True
                    def success_cb():
                        self.update_connection_ui(True, None)
                        if on_success:
                            on_success()
                    GLib.idle_add(success_cb)
                else:
                    self.is_connected = False
                    GLib.idle_add(self.update_connection_ui, False, "Connection failed.")
            except Exception as e:
                self.is_connected = False
                GLib.idle_add(self.update_connection_ui, False, str(e))
                
        thread = threading.Thread(target=run)
        thread.daemon = True
        thread.start()

    def update_connection_ui(self, connected, error_msg):
        self.is_connecting = False
        if connected:
            self.lbl_camera_status.set_markup("<span foreground='#44ff44'>●</span> <b>Camera: Connected</b>")
            self.update_capture_button_sensitivity()
            self.lbl_status.set_text("Status: Camera connected, ready.")
        else:
            self.lbl_camera_status.set_markup("<span foreground='#ff4444'>●</span> <b>Camera: Disconnected</b>")
            self.update_capture_button_sensitivity()
            self.spinner.stop()
            if error_msg:
                self.lbl_status.set_text(f"Status: Disconnected ({error_msg})")
            else:
                self.lbl_status.set_text("Status: Disconnected.")

    def adj_gain(self, delta):
        self.gain = max(0.1, self.gain + delta)
        self.entry_gain.set_text(f"{self.gain:.2f}")
        self.capture_converted_rgb_cache = None
        self.capture_corr_hist_cache = None
        self.update_capture_preview()

    def reset_gain(self):
        self.gain = 1.0
        self.entry_gain.set_text("1.00")
        self.capture_converted_rgb_cache = None
        self.capture_corr_hist_cache = None
        self.update_capture_preview()

    def on_gain_entry_activated(self, entry):
        text = entry.get_text()
        try:
            val = float(text)
            self.gain = max(0.1, val)
            entry.set_text(f"{self.gain:.2f}")
            self.capture_converted_rgb_cache = None
            self.capture_corr_hist_cache = None
            self.update_capture_preview()
        except ValueError:
            entry.set_text(f"{self.gain:.2f}")

    def on_key_press(self, widget, event):
        keyval = event.keyval
        
        if keyval in (Gdk.KEY_plus, Gdk.KEY_equal, Gdk.KEY_KP_Add):
            self.adj_gain(0.10)
            return True
        elif keyval in (Gdk.KEY_minus, Gdk.KEY_underscore, Gdk.KEY_KP_Subtract):
            self.adj_gain(-0.10)
            return True
            
        return False

    def on_set_rot(self, btn, rot):
        self.orientation = rot
        self.update_capture_preview()
        self.update_base_preview()

    def on_hflip(self, btn):
        self.hflip = btn.get_active()
        self.update_capture_preview()
        self.update_base_preview()

    def on_vflip(self, btn):
        self.vflip = btn.get_active()
        self.update_capture_preview()
        self.update_base_preview()

    def on_load_profile(self, widget):
        dialog = Gtk.FileChooserDialog(
            title="Load Profile JSON", parent=self,
            action=Gtk.FileChooserAction.OPEN
        )
        dialog.add_buttons(Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL, Gtk.STOCK_OPEN, Gtk.ResponseType.OK)
        
        # Add filter for json
        filter_json = Gtk.FileFilter()
        filter_json.set_name("JSON Profile")
        filter_json.add_pattern("*.json")
        dialog.add_filter(filter_json)

        if dialog.run() == Gtk.ResponseType.OK:
            filepath = dialog.get_filename()
            try:
                self.profile = FilmProfile(filepath)
                self.profile_filename = filepath
                
                self.has_icc = bool(getattr(self.profile, 'icc_profile_bytes', None))
                self.has_crosstalk = hasattr(self.profile, 'crosstalk_matrix') and self.profile.crosstalk_matrix is not None
                
                if hasattr(self.profile, 'film_base') and self.profile.film_base:
                    fb_r = self.profile.film_base.get('r_avg', 0.0)
                    fb_g = self.profile.film_base.get('g_avg', 0.0)
                    fb_b = self.profile.film_base.get('b_avg', 0.0)
                    if fb_r > 0 or fb_g > 0 or fb_b > 0:
                        # Only use profile fallback values if a custom film base image has not been captured/loaded yet
                        if self.film_base_img is None:
                            self.film_base_rgb = (fb_r, fb_g, fb_b)
                            # Averages in the profile are already crosstalk-corrected. 
                            # Print them directly as both Raw and Corr.
                            self.lbl_base_vals.set_text(f"Raw: {fb_r:.1f}, {fb_g:.1f}, {fb_b:.1f} | "
                                                        f"Corr: {fb_r:.1f}, {fb_g:.1f}, {fb_b:.1f}")
                
                # Clip name to 10 chars
                film_name = self.profile.film_name if self.profile.film_name else "Unknown"
                name_clipped = film_name[:10] + "..." if len(film_name) > 10 else film_name
                
                # Determine tags
                tags = []
                if self.has_icc: tags.append("ICC")
                if self.has_crosstalk: tags.append("XT")
                tags_str = f" [{'+'.join(tags)}]" if tags else ""
                
                self.btn_load.set_label(f"Load: {name_clipped}{tags_str}")
                self.lbl_profile_info.set_markup(
                    f"<b>Profile:</b> {film_name}\n"
                    f"<b>File:</b> {os.path.basename(filepath)}"
                )
                self.lbl_status.set_text("Status: Profile loaded.")
                
                targets_count = 0
                if hasattr(self.profile, 'targets'):
                    targets_count = len(self.profile.targets)
                elif hasattr(self.profile, 'raw_data') and 'targets' in self.profile.raw_data:
                    targets_count = len(self.profile.raw_data['targets'])
                
                print(f"[Profile] Loaded Film Profile: {self.profile.film_name} from {filepath}", file=sys.stdout)
                print(f"[Profile]   Has ICC curves/targets: {'Yes' if self.has_icc else 'No'}", file=sys.stdout)
                print(f"[Profile]   Has Crosstalk matrix: {'Yes' if self.has_crosstalk else 'No'}", file=sys.stdout)
                if self.has_crosstalk:
                    print(f"[Profile]   Crosstalk Matrix:", file=sys.stdout)
                    for row in self.profile.crosstalk_matrix:
                        print(f"      [ {row[0]:.6f}, {row[1]:.6f}, {row[2]:.6f} ]", file=sys.stdout)
                if hasattr(self.profile, 'film_base') and self.profile.film_base:
                    fb_r = self.profile.film_base.get('r_avg', 0.0)
                    fb_g = self.profile.film_base.get('g_avg', 0.0)
                    fb_b = self.profile.film_base.get('b_avg', 0.0)
                    print(f"[Profile]   Default Film Base RGB: R={fb_r:.1f}, G={fb_g:.1f}, B={fb_b:.1f}", file=sys.stdout)
                fb_shutter = getattr(self.profile, 'film_base_shutter', 'None')
                fb_iso = getattr(self.profile, 'film_base_iso', 100)
                print(f"[Profile]   Default Film Base Exposure: Shutter={fb_shutter}, ISO={fb_iso}", file=sys.stdout)
                print(f"[Profile]   Targets count loaded: {targets_count}", file=sys.stdout)
                sys.stdout.flush()
                
                # Update targets table dropdown
                self.target_liststore.clear()
                if hasattr(self.profile, 'raw_data') and 'targets' in self.profile.raw_data:
                    for idx, tgt in enumerate(self.profile.raw_data['targets']):
                        name = tgt.get('name', f"Target {idx}")
                        
                        # Get gs0 and gs23 transmittances
                        ts = get_target_transmittances(self.profile, idx)
                        if ts and len(ts) == 24:
                            gs0_r, gs0_g, gs0_b = ts[0]
                            gs23_r, gs23_g, gs23_b = ts[23]
                            gs0_str = f"{gs0_r:.3f} / {gs0_g:.3f} / {gs0_b:.3f}"
                            gs23_str = f"{gs23_r:.3f} / {gs23_g:.3f} / {gs23_b:.3f}"
                        else:
                            gs0_str = "N/A"
                            gs23_str = "N/A"
                            
                        active = "➔" if idx == self.selected_target_idx else ""
                        self.target_liststore.append([active, idx, name, gs0_str, gs23_str])
                        
                if self.raw_linear_pixels is not None and self.film_base_rgb is not None and len(self.target_liststore) > 0:
                    scan_shutter = self.raw_image.shutter_speed if self.raw_image else 0.125
                    scan_iso = self.raw_image.iso if self.raw_image else 100
                    base_shutter = self.film_base_img.shutter_speed if self.film_base_img else None
                    base_iso = self.film_base_img.iso if self.film_base_img else 100
                    best_idx, dist = find_best_target_index(
                        self.profile, self.raw_linear_pixels, self.film_base_rgb,
                        scan_shutter=scan_shutter, scan_iso=scan_iso,
                        base_shutter=base_shutter, base_iso=base_iso
                    )
                    self.selected_target_idx = best_idx
                    self.update_main_target_indicators()
                else:
                    self.selected_target_idx = 0
                    self.update_main_target_indicators()
                    
                if len(self.target_liststore) > 0:
                    select = self.target_treeview.get_selection()
                    select.select_path(Gtk.TreePath.new_from_indices([self.selected_target_idx]))
                    
                self.capture_converted_rgb_cache = None
                self.capture_corr_hist_cache = None
                self.base_converted_rgb_cache = None
                self.update_profile_dependencies()
                self.update_capture_preview()
                self.update_base_preview()
                
                if hasattr(self, 'details_window') and self.details_window is not None:
                    self.details_window.refresh_all()
            except Exception as e:
                self.lbl_profile_info.set_text(f"Error: {e}")
                self.lbl_status.set_text("Status: Failed to load profile.")
                self.profile = None
                self.has_icc = False
                self.has_crosstalk = False
                self.update_profile_dependencies()
                self.update_base_preview()
                
        dialog.destroy()

    def on_reset_profile(self, widget):
        self.profile = None
        self.profile_filename = ""
        self.has_icc = False
        self.has_crosstalk = False
        self.btn_load.set_label("Load Profile...")
        self.lbl_profile_info.set_markup("<b>Profile:</b> None")
        self.lbl_status.set_text("Status: Profile cleared.")
        self.target_liststore.clear()
        self.capture_converted_rgb_cache = None
        self.capture_corr_hist_cache = None
        self.base_converted_rgb_cache = None
        if not getattr(self, 'film_base_img', None):
            self.film_base_rgb = None
            self.lbl_base_vals.set_text("Raw: -- | Corr: --")
        self.update_profile_dependencies()
        self.update_capture_preview()
        self.update_base_preview()
        
        if hasattr(self, 'details_window') and self.details_window is not None:
            self.details_window.refresh_all()

    def on_load_image_clicked(self, widget):
        current_page = self.notebook.get_current_page()
        is_base = (current_page == 1)
        
        dialog = Gtk.FileChooserDialog(
            title="Load Image or Archive", parent=self,
            action=Gtk.FileChooserAction.OPEN
        )
        dialog.add_buttons(Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL, Gtk.STOCK_OPEN, Gtk.ResponseType.OK)
        
        # Add filters
        filter_all = Gtk.FileFilter()
        filter_all.set_name("All supported files (ARW, ZIP, TAR)")
        filter_all.add_pattern("*.arw")
        filter_all.add_pattern("*.ARW")
        filter_all.add_pattern("*.zip")
        filter_all.add_pattern("*.ZIP")
        filter_all.add_pattern("*.tar")
        filter_all.add_pattern("*.TAR")
        filter_all.add_pattern("*.tar.gz")
        filter_all.add_pattern("*.tgz")
        filter_all.add_pattern("*.tar.xz")
        dialog.add_filter(filter_all)
        
        filter_arw = Gtk.FileFilter()
        filter_arw.set_name("Sony RAW Image (*.ARW)")
        filter_arw.add_pattern("*.arw")
        filter_arw.add_pattern("*.ARW")
        dialog.add_filter(filter_arw)
        
        filter_archive = Gtk.FileFilter()
        filter_archive.set_name("Archive Files (*.zip, *.tar*)")
        filter_archive.add_pattern("*.zip")
        filter_archive.add_pattern("*.ZIP")
        filter_archive.add_pattern("*.tar")
        filter_archive.add_pattern("*.TAR")
        filter_archive.add_pattern("*.tar.gz")
        filter_archive.add_pattern("*.tgz")
        filter_archive.add_pattern("*.tar.xz")
        dialog.add_filter(filter_archive)

        if dialog.run() == Gtk.ResponseType.OK:
            filepath = dialog.get_filename()
            dialog.destroy()
            
            # Run loading and conversion in a background thread to keep UI responsive
            self.lbl_status.set_text("Loading offline image...")
            self.btn_capture.set_sensitive(False)
            self.btn_load_image.set_sensitive(False)
            self.spinner.start()
            
            def load_task():
                try:
                    # Check if archive
                    is_archive = False
                    lower_path = filepath.lower()
                    for ext in ['.zip', '.tar', '.tgz', '.tar.gz', '.tar.xz']:
                        if lower_path.endswith(ext):
                            is_archive = True
                            break
                            
                    if is_archive:
                        extract_dir = os.path.join(os.getcwd(), "build", "tmp_load_archive")
                        arw_files = unpack_archive_and_find_arws(filepath, extract_dir)
                        if not arw_files:
                            raise ValueError("No .ARW files found in the archive.")
                    else:
                        arw_files = [filepath]
                        
                    # Determine type
                    if len(arw_files) == 4:
                        cap_type = 1 # pixel shift
                    else:
                        cap_type = 0 # single shot
                        
                    # Read EXIF metadata from the first file to get real exposure settings
                    shutter_speed, iso = read_arw_metadata(arw_files[0])
                    
                    # Create CapturedImage object
                    loaded_img = negicc_station.CapturedImage(
                        type=cap_type,
                        shutter_speed=shutter_speed,
                        iso=iso,
                        filepaths=arw_files
                    )
                    
                    # Process the image in main thread/GLib idle
                    def success():
                        self.process_captured_image(loaded_img, is_base=is_base, capture_duration=0.0)
                        self.btn_load_image.set_sensitive(True)
                        
                    GLib.idle_add(success)
                    
                except Exception as e:
                    def failure(err_msg):
                        self.lbl_status.set_text(f"Error loading: {err_msg}")
                        self.btn_capture.set_sensitive(True)
                        self.btn_load_image.set_sensitive(True)
                        self.spinner.stop()
                        
                    GLib.idle_add(failure, str(e))
                    
            threading.Thread(target=load_task, daemon=True).start()
        else:
            dialog.destroy()

    def update_main_target_indicators(self):
        for row in self.target_liststore:
            row[0] = "➔" if row[1] == self.selected_target_idx else ""

    def on_target_selection_changed(self, selection):
        model, treeiter = selection.get_selected()
        if treeiter is not None:
            idx = model[treeiter][1]
            name = model[treeiter][2]
            if idx != self.selected_target_idx:
                self.selected_target_idx = idx
                self.update_main_target_indicators()
                self.capture_converted_rgb_cache = None
                self.capture_corr_hist_cache = None
                self.update_capture_preview()
                
                print(f"[Profile] Target selected: Index {idx}, Name: {name}", file=sys.stdout)
                sys.stdout.flush()
                
                if hasattr(self, 'details_window') and self.details_window is not None:
                    self.details_window.update_indicators()
                    self.details_window.select_active_in_treeview()
                    self.details_window.update_plot()
                    self.details_window.update_text_info()

    def on_show_target_details(self, btn):
        if hasattr(self, 'details_window') and self.details_window is not None:
            self.details_window.present()
            return
            
        self.details_window = CalibrationTargetsDetailsWindow(self)
        self.details_window.connect("destroy", self.on_details_window_destroyed)
        self.details_window.show_all()
        
    def on_details_window_destroyed(self, window):
        self.details_window = None

    def on_tab_changed(self, notebook, page, page_num):
        if page_num == 0:
            self.update_capture_histograms()
            self.btn_capture.set_label("Capture Image")
            self.btn_capture.get_style_context().remove_class("btn-yellow")
            self.btn_capture.get_style_context().add_class("btn-green")
            if hasattr(self, 'btn_load_image'):
                self.btn_load_image.set_label("Load Image/Archive...")
        else:
            self.update_base_histograms()
            self.btn_capture.set_label("Capture Film Base")
            self.btn_capture.get_style_context().remove_class("btn-green")
            self.btn_capture.get_style_context().add_class("btn-yellow")
            if hasattr(self, 'btn_load_image'):
                self.btn_load_image.set_label("Load Base/Archive...")
        self.update_capture_button_sensitivity()

    def on_ae_toggled(self, checkbox):
        self.shutter_combo.set_sensitive(not checkbox.get_active())

    def on_ae_graph_bar_selected(self, shutter_str):
        self._set_shutter_combo(shutter_str)

    def on_capture_clicked(self, widget):
        current_page = self.notebook.get_current_page()
        if current_page == 1:
            self.on_capture(is_base=True)
        else:
            self.on_capture(is_base=False)

    def on_capture(self, is_base=False):
        if not self.is_connected or not self.camera_session:
            self.lbl_status.set_text("Camera not connected. Reconnecting...")
            self.connect_camera(on_success=lambda: self.on_capture(is_base=is_base))
            return
            
        shutter_str = self.shutter_combo.get_active_text()
        is_ae = self.ae_checkbox.get_active()
        mode_id = int(self.mode_combo.get_active_id())
        
        self.lbl_status.set_text("Capturing...")
        self.clear_ae_steps()
        self.ae_graph.clear()
        self.ae_overexposed_states = {}
        
        self.is_capturing = True
        self.btn_capture.set_sensitive(False)
        self.btn_save_tiff.set_sensitive(False)
        self.btn_save_raw_capture.set_sensitive(False)
        self.btn_save_raw_base.set_sensitive(False)
        self.spinner.start()
        
        def task():
            try:
                final_shutter_str = shutter_str
                
                if is_ae:
                    GLib.idle_add(self.lbl_status.set_text, "Running Auto Exposure...")
                    GLib.idle_add(self.ae_steps_frame.show_all)
                    
                    def ae_local_capture(idx):
                        ss = auto_exposure.SHUTTER_SPEEDS[idx]
                        GLib.idle_add(self.lbl_status.set_text, f"AE: Capturing {ss}...")
                        num, den = auto_exposure.parse_shutter_speed(ss)
                        ae_img = self.camera_session.capture(type=0, shutter_num=num, shutter_den=den) # Single shot for AE search
                        if not ae_img:
                            raise RuntimeError("AE Capture returned Null. Camera capture or download failed.")
                        arr = ae_img.to_numpy(half=True)
                        arr_annotated = auto_exposure.AnnotatedArray(arr, iso=ae_img.iso)
                        
                        # Calculate overexposure flags
                        H, W = arr.shape[:2]
                        square_size = int(min(H, W) * 2 // 3)
                        y_start = (H - square_size) // 2
                        x_start = (W - square_size) // 2
                        cropped = arr[y_start:y_start+square_size, x_start:x_start+square_size, :]
                        
                        # Decimate cropped array to keep performance snappy
                        cH, cW = cropped.shape[:2]
                        total_c = cH * cW
                        if total_c > 200000:
                            step = int(np.sqrt(total_c / 200000))
                            step = max(1, step)
                            cropped_sampled = cropped[::step, ::step, :]
                        else:
                            cropped_sampled = cropped
                            
                        OVEREXPOSURE_THRESHOLD = 13107.2
                        over_flags = []
                        for c in range(3):
                            p95 = np.percentile(cropped_sampled[:, :, c], 95)
                            over_flags.append(bool(p95 > OVEREXPOSURE_THRESHOLD))
                            
                        self.ae_overexposed_states[ss] = over_flags
                        
                        ae_img.discard()
                        return arr_annotated
                    
                    def ae_progress(step_idx, ss, iso, ch_dr, avg_dr):
                        dr_r, dr_g, dr_b = ch_dr
                        over_flags = self.ae_overexposed_states.get(ss, [False, False, False])
                        GLib.idle_add(self.add_ae_step, ss, iso, dr_r, dr_g, dr_b, avg_dr, over_flags)
                    
                    opt_shutter, steps = auto_exposure.run_auto_exposure(
                        start_shutter_str=shutter_str,
                        capture_func=ae_local_capture,
                        progress_callback=ae_progress
                    )
                    final_shutter_str = opt_shutter
                    GLib.idle_add(self._set_shutter_combo, opt_shutter)
                else:
                    GLib.idle_add(self.ae_steps_frame.hide)
                
                GLib.idle_add(self.lbl_status.set_text, f"Capturing at {final_shutter_str}...")
                num, den = auto_exposure.parse_shutter_speed(final_shutter_str)
                
                cap_type = 0 if is_base else mode_id
                t_start = time.time()
                img = self.camera_session.capture(type=cap_type, shutter_num=num, shutter_den=den)
                t_dur = time.time() - t_start
                
                if not img:
                    GLib.idle_add(self.update_ui_failure, "C++ capture returned null")
                    return
                    
                GLib.idle_add(self.process_captured_image, img, is_base, t_dur)
            except Exception as e:
                import traceback
                print(f"Capture error: {e}", file=sys.stdout)
                traceback.print_exc(file=sys.stdout)
                sys.stdout.flush()
                GLib.idle_add(self.update_ui_failure, str(e))
                
        t = threading.Thread(target=task)
        t.daemon = True
        t.start()

    def _set_shutter_combo(self, shutter_str):
        if shutter_str in auto_exposure.SHUTTER_SPEEDS:
            self.shutter_combo.set_active(auto_exposure.SHUTTER_SPEEDS.index(shutter_str))

    def clear_ae_steps(self):
        for child in self.ae_steps_listbox.get_children():
            self.ae_steps_listbox.remove(child)

    def add_ae_step(self, ss, iso, dr_r, dr_g, dr_b, avg_dr, over_flags=None):
        row_label = Gtk.Label()
        
        # Color coding for R, G, B channels
        r_color = '#ff6666'
        g_color = '#66ff66'
        b_color = '#66aaff'
        
        # Suffix with (O) if overexposed
        over_r, over_g, over_b = over_flags if over_flags else [False, False, False]
        r_txt = f"R:{dr_r:.0f}" + ("(O)" if over_r else "")
        g_txt = f"G:{dr_g:.0f}" + ("(O)" if over_g else "")
        b_txt = f"B:{dr_b:.0f}" + ("(O)" if over_b else "")
        
        row_label.set_markup(
            f"<span size='small' font_family='monospace'>"
            f"Speed: <b>{ss}</b> (ISO {iso})\n"
            f"<span color='{r_color}'>{r_txt}</span> "
            f"<span color='{g_color}'>{g_txt}</span> "
            f"<span color='{b_color}'>{b_txt}</span>\n"
            f"<b>Avg: {avg_dr:.0f}</b>"
            f"</span>"
        )
        row_label.set_line_wrap(True)
        row_label.set_max_width_chars(25)
        row_label.set_xalign(0.0)
        row = Gtk.ListBoxRow()
        row.add(row_label)
        self.ae_steps_listbox.add(row)
        self.ae_steps_listbox.show_all()
        self.ae_graph.add_step(ss, iso, dr_r, dr_g, dr_b, avg_dr, over_flags)

    def process_captured_image(self, img_obj, is_base, capture_duration):
        self.lbl_status.set_text("Processing capture...")
        
        t_conv_start = time.time()
        raw_linear = img_obj.to_numpy(half=True)
        conv_duration = time.time() - t_conv_start
        
        if is_base:
            if self.film_base_img: self.film_base_img.discard()
            self.film_base_img = img_obj
            self.film_base_raw_linear = raw_linear
            self.base_converted_rgb_cache = None
            self.base_rect_start = None
            self.base_rect_end = None
            self.base_rect_raw = None
            
            self.update_base_preview()
            self.update_base_histograms()
            self.notebook.set_current_page(1)
            
            self.is_capturing = False
            self.update_capture_button_sensitivity()
            self.update_save_buttons_sensitivity()
            self.spinner.stop()
            self.lbl_status.set_text("Status: Film base updated.")
            
        else:
            if self.raw_image: self.raw_image.discard()
            self.raw_image = img_obj
            self.raw_linear_pixels = raw_linear
            self.capture_converted_rgb_cache = None
            self.capture_corr_hist_cache = None
            self.capture_rect_start = None
            self.capture_rect_end = None
            self.capture_rect_raw = None
            
            self.gain = 1.0
            self.entry_gain.set_text(f"{self.gain:.2f}")
            
            if self.film_base_rgb is not None and self.profile is not None and len(self.target_liststore) > 0:
                scan_shutter = img_obj.shutter_speed
                scan_iso = img_obj.iso
                base_shutter = self.film_base_img.shutter_speed if self.film_base_img else None
                base_iso = self.film_base_img.iso if self.film_base_img else 100
                best_idx, dist = find_best_target_index(
                    self.profile, raw_linear, self.film_base_rgb,
                    scan_shutter=scan_shutter, scan_iso=scan_iso,
                    base_shutter=base_shutter, base_iso=base_iso
                )
                self.selected_target_idx = best_idx
                self.update_main_target_indicators()
                
                select = self.target_treeview.get_selection()
                for i, row in enumerate(self.target_liststore):
                    if row[1] == best_idx:
                        select.select_path(Gtk.TreePath.new_from_indices([i]))
                        break
                        
                if hasattr(self, 'details_window') and self.details_window is not None:
                    self.details_window.refresh_all()
                        
            # Print captured metadata, capture duration, and preview conversion to stdout
            paths_str = ", ".join(img_obj.filepaths)
            h, w = raw_linear.shape[:2]
            print(f"[Capture] Filepath(s): {paths_str}", file=sys.stdout)
            print(f"[Capture] ISO: {img_obj.iso} | Shutter: {img_obj.shutter_speed:.4f}s | Dimensions: {w}x{h}", file=sys.stdout)
            print(f"[Capture] Capture Duration: {capture_duration:.3f}s | Conversion Duration: {conv_duration:.3f}s", file=sys.stdout)
            sys.stdout.flush()
            self.update_capture_preview()
            self.notebook.set_current_page(0)
            
            self.is_capturing = False
            self.update_capture_button_sensitivity()
            self.update_save_buttons_sensitivity()
            self.spinner.stop()
            self.lbl_status.set_text("Status: Image updated successfully.")

    def update_base_histograms(self):
        if self.film_base_raw_linear is None:
            self.hist_raw.clear()
            self.hist_corr.clear()
            return
            
        raw_d = self.film_base_raw_linear
        if hasattr(self, 'base_rect_raw') and self.base_rect_raw is not None:
            x1, y1, x2, y2 = self.base_rect_raw
            dh, dw = raw_d.shape[:2]
            x1, x2 = max(0, x1), min(dw, x2)
            y1, y2 = max(0, y1), min(dh, y2)
            if x2 > x1 and y2 > y1:
                raw_d = raw_d[y1:y2, x1:x2]
                
        self.hist_raw.plot_histogram(raw_d, is_corrected=False, has_icc=False, show_overexposure=True)
        
        if self.profile and self.has_crosstalk:
            corr_d = raw_d.astype(np.float32)
            corr_d = np.dot(corr_d, self.profile.crosstalk_matrix.T)
            corr_d = np.clip(corr_d, 0, 16384)
            self.hist_corr.plot_histogram(corr_d, is_corrected=True, has_icc=False, show_overexposure=False)
        else:
            self.hist_corr.clear()

    def on_read_film_base(self, btn):
        if self.film_base_raw_linear is None:
            return
            
        data = self.film_base_raw_linear
        if hasattr(self, 'base_rect_raw') and self.base_rect_raw is not None:
            x1, y1, x2, y2 = self.base_rect_raw
            dh, dw = data.shape[:2]
            x1, x2 = max(0, x1), min(dw, x2)
            y1, y2 = max(0, y1), min(dh, y2)
            if x2 > x1 and y2 > y1:
                data = data[y1:y2, x1:x2]
                
        data_f32 = data.astype(np.float32)
        rgb = np.mean(data_f32, axis=(0, 1), dtype=np.float32)
        
        corr_rgb = rgb
        if self.profile and self.has_crosstalk:
            matrix = np.array(self.profile.crosstalk_matrix, dtype=np.float32)
            corr_rgb = np.dot(rgb, matrix.T)
            
        # The film_base_rgb passed to target matching and conversions must be crosstalk-corrected.
        self.film_base_rgb = (float(corr_rgb[0]), float(corr_rgb[1]), float(corr_rgb[2]))
        
        self.update_profile_dependencies()
        
        self.lbl_base_vals.set_text(f"Raw: {rgb[0]:.1f}, {rgb[1]:.1f}, {rgb[2]:.1f} | "
                                    f"Corr: {corr_rgb[0]:.1f}, {corr_rgb[1]:.1f}, {corr_rgb[2]:.1f}")
                                    
        val_str = f"raw: {rgb[0]:.1f}, {rgb[1]:.1f}, {rgb[2]:.1f} | Corr: {corr_rgb[0]:.1f}, {corr_rgb[1]:.1f}, {corr_rgb[2]:.1f}"
        print(val_str, file=sys.stdout)
        sys.stdout.flush()
                                    
        if self.raw_image:
            self.capture_converted_rgb_cache = None
            self.capture_corr_hist_cache = None
            self.update_capture_preview()

    def refresh_capture_preview(self):
        if not hasattr(self, 'capture_preview_pixbuf') or self.capture_preview_pixbuf is None:
            self.scaled_capture_pixbuf = None
            return
            
        w_alloc = self.capture_image_area.get_allocated_width()
        h_alloc = self.capture_image_area.get_allocated_height()
        if w_alloc < 10 or h_alloc < 10:
            self.scaled_capture_pixbuf = None
            return
            
        w_img = self.capture_preview_pixbuf.get_width()
        h_img = self.capture_preview_pixbuf.get_height()
        
        scale = min(w_alloc / w_img, h_alloc / h_img)
        new_w = max(1, int(w_img * scale))
        new_h = max(1, int(h_img * scale))
        
        self.scaled_capture_pixbuf = self.capture_preview_pixbuf.scale_simple(
            new_w, new_h, GdkPixbuf.InterpType.BILINEAR
        )
        self.capture_image_area.queue_draw()

    def refresh_base_preview(self):
        if not hasattr(self, 'base_preview_pixbuf') or self.base_preview_pixbuf is None:
            self.scaled_base_pixbuf = None
            return
            
        w_alloc = self.base_image_area.get_allocated_width()
        h_alloc = self.base_image_area.get_allocated_height()
        if w_alloc < 10 or h_alloc < 10:
            self.scaled_base_pixbuf = None
            return
            
        w_img = self.base_preview_pixbuf.get_width()
        h_img = self.base_preview_pixbuf.get_height()
        
        scale = min(w_alloc / w_img, h_alloc / h_img)
        new_w = max(1, int(w_img * scale))
        new_h = max(1, int(h_img * scale))
        
        self.scaled_base_pixbuf = self.base_preview_pixbuf.scale_simple(
            new_w, new_h, GdkPixbuf.InterpType.BILINEAR
        )
        self.base_image_area.queue_draw()

    def on_capture_area_size_allocate(self, widget, allocation):
        self.refresh_capture_preview()

    def on_base_area_size_allocate(self, widget, allocation):
        self.refresh_base_preview()

    def update_capture_preview(self):
        if self.raw_linear_pixels is None:
            return
            
        img_array = None
        corr_hist_array = None
        
        if getattr(self, 'capture_converted_rgb_cache', None) is not None:
            img_array = self.capture_converted_rgb_cache
            corr_hist_array = self.capture_corr_hist_cache
        else:
            if self.profile:
                if self.has_icc:
                    prof_data = json.loads(json.dumps(self.profile.raw_data))
                    if 'targets' in prof_data and self.selected_target_idx < len(prof_data['targets']):
                        prof_data['targets'] = [prof_data['targets'][self.selected_target_idx]]
                        tgt = prof_data['targets'][0]
                        if 'icc_profile_base64' in tgt:
                            prof_data['icc_profile_base64'] = tgt['icc_profile_base64']
                            
                    temp_profile = FilmProfile(prof_data)
                    if temp_profile.icc_profile_bytes is None and getattr(self.profile, 'icc_profile_bytes', None):
                        temp_profile.icc_profile_bytes = self.profile.icc_profile_bytes
                        
                    import film_profiling
                    res = film_profiling.convert_raw_to_numpy(
                        img=self.raw_image, profile=temp_profile,
                        exposure_comp=self.gain, half=True, film_base_rgb=self.film_base_rgb,
                        film_base_img=self.film_base_img, pipeline="cuda"
                    )
                    img_array = res
                    corr_hist_array = res
                else:
                    raw = self.raw_linear_pixels.astype(np.float32)
                    if self.has_crosstalk:
                        raw = np.dot(raw, self.profile.crosstalk_matrix.T)
                    img_array = np.clip(raw * self.gain, 0, 16384).astype(np.uint16)
                    corr_hist_array = img_array
            else:
                raw = self.raw_linear_pixels.astype(np.float32) * self.gain
                img_array = np.clip(raw, 0, 16384).astype(np.uint16)
                corr_hist_array = img_array
                
            self.capture_converted_rgb_cache = img_array
            self.capture_corr_hist_cache = corr_hist_array
            
        # Convert to 8-bit strictly for display (GdkPixbuf requires 8-bit)
        assert img_array.dtype in (np.uint16, np.float32), f"Expected 16-bit image array for preview scaling, got {img_array.dtype}"
        if self.profile and self.has_icc:
            # Range is [0, 65535]
            img_8bit = (img_array / 256.0).astype(np.uint8)
        else:
            # Range is [0, 16384]
            img_8bit = (img_array / 64.0).astype(np.uint8)
            
        img_8bit = apply_transforms_numpy(img_8bit, self.hflip, self.vflip, self.orientation)
        
        h, w, c = img_8bit.shape
        img_8bit = np.ascontiguousarray(img_8bit)
        self.capture_preview_pixbuf = GdkPixbuf.Pixbuf.new_from_data(
            img_8bit.tobytes(), GdkPixbuf.Colorspace.RGB, False, 8, w, h, w * 3
        )
        self.capture_corr_hist_data = apply_transforms_numpy(corr_hist_array, self.hflip, self.vflip, self.orientation)
        self.capture_raw_hist_data = apply_transforms_numpy(self.raw_linear_pixels, self.hflip, self.vflip, self.orientation)
        
        self.refresh_capture_preview()
        if self.notebook.get_current_page() == 0:
            self.update_capture_histograms()

    def update_base_preview(self):
        if self.film_base_raw_linear is None:
            return
            
        if getattr(self, 'base_converted_rgb_cache', None) is not None:
            img_array = self.base_converted_rgb_cache
        else:
            if self.profile and self.has_crosstalk:
                raw = self.film_base_raw_linear.astype(np.float32)
                raw = np.dot(raw, self.profile.crosstalk_matrix.T)
                img_array = np.clip(raw, 0, 16384).astype(np.uint16)
            else:
                img_array = np.clip(self.film_base_raw_linear, 0, 16384).astype(np.uint16)
            self.base_converted_rgb_cache = img_array
            
        # Convert to 8-bit strictly for display (GdkPixbuf requires 8-bit)
        assert img_array.dtype in (np.uint16, np.float32), f"Expected 16-bit image array for preview scaling, got {img_array.dtype}"
        img_8bit = (img_array / 64.0).astype(np.uint8)
        img_8bit = apply_transforms_numpy(img_8bit, False, False, 0)
        img_8bit = np.ascontiguousarray(img_8bit)
        h, w, c = img_8bit.shape
        self.base_preview_pixbuf = GdkPixbuf.Pixbuf.new_from_data(
            img_8bit.tobytes(), GdkPixbuf.Colorspace.RGB, False, 8, w, h, w * 3
        )
        self.refresh_base_preview()

    def update_profile_dependencies(self):
        has_full_profile = self.profile is not None and self.has_crosstalk and self.has_icc
        
        # Update tab sensitivity
        base_page = self.notebook.get_nth_page(1)
        if base_page:
            base_page.set_sensitive(has_full_profile)
            
        self.update_capture_button_sensitivity()

    def update_capture_button_sensitivity(self):
        has_full_profile = self.profile is not None and self.has_crosstalk and self.has_icc
        
        # Determine capture button sensitivity based on connection and capturing state
        if getattr(self, 'is_capturing', False):
            self.btn_capture.set_sensitive(False)
        else:
            current_page = self.notebook.get_current_page()
            if current_page == 1: # Film Base Tab
                self.btn_capture.set_sensitive(has_full_profile)
            else: # Capture Tab
                self.btn_capture.set_sensitive(True)

        if hasattr(self, 'btn_load_image'):
            if getattr(self, 'is_capturing', False):
                self.btn_load_image.set_sensitive(False)
            else:
                self.btn_load_image.set_sensitive(True)
        
        # Update tab labels warnings/statuses
        if has_full_profile:
            if self.film_base_rgb is not None:
                self.base_tab_label.set_markup("<span foreground='#44ff44'>● Film Base (Calibrated)</span>")
                self.capture_tab_label.set_markup("<span>Capture</span>")
            else:
                self.base_tab_label.set_markup("<span foreground='#e6a23c'>● Film Base (Needed)</span>")
                self.capture_tab_label.set_markup("<span foreground='#e6a23c'>Capture (Base Needed)</span>")
        else:
            self.base_tab_label.set_markup("<span>Film Base</span>")
            self.capture_tab_label.set_markup("<span>Capture</span>")

    def update_save_buttons_sensitivity(self):
        has_raw = getattr(self, 'raw_image', None) is not None
        has_base = getattr(self, 'film_base_img', None) is not None
        
        if hasattr(self, 'btn_save_tiff'):
            self.btn_save_tiff.set_sensitive(has_raw)
        if hasattr(self, 'btn_save_raw_capture'):
            self.btn_save_raw_capture.set_sensitive(has_raw)
        if hasattr(self, 'btn_save_raw_base'):
            self.btn_save_raw_base.set_sensitive(has_base)

    def update_capture_histograms(self):
        if not hasattr(self, 'hist_raw') or self.hist_raw is None:
            return
        if not hasattr(self, 'capture_raw_hist_data') or self.capture_raw_hist_data is None:
            self.hist_raw.clear()
            self.hist_corr.clear()
            return
            
        raw_d = self.capture_raw_hist_data
        corr_d = self.capture_corr_hist_data
        
        if hasattr(self, 'capture_rect_raw') and self.capture_rect_raw is not None and self.raw_linear_pixels is not None:
            h_raw, w_raw = self.raw_linear_pixels.shape[:2]
            rect_trans = map_raw_rect_to_transformed(
                self.capture_rect_raw, w_raw, h_raw, self.hflip, self.vflip, self.orientation
            )
            if rect_trans is not None:
                x1_trans, y1_trans, x2_trans, y2_trans = rect_trans
                dh, dw = raw_d.shape[:2]
                x1 = max(0, x1_trans)
                x2 = min(dw, x2_trans)
                y1 = max(0, y1_trans)
                y2 = min(dh, y2_trans)
                if x2 > x1 and y2 > y1:
                    raw_d = raw_d[y1:y2, x1:x2]
                    corr_d = corr_d[y1:y2, x1:x2]
                
        self.hist_raw.plot_histogram(raw_d, is_corrected=False, has_icc=False, show_overexposure=True)
        self.hist_corr.plot_histogram(corr_d, is_corrected=True, has_icc=self.has_icc, show_overexposure=False)
        


    def on_draw_capture(self, widget, cr):
        if hasattr(self, 'scaled_capture_pixbuf') and self.scaled_capture_pixbuf is not None:
            w_alloc = widget.get_allocated_width()
            h_alloc = widget.get_allocated_height()
            w_img = self.scaled_capture_pixbuf.get_width()
            h_img = self.scaled_capture_pixbuf.get_height()
            off_x = (w_alloc - w_img) / 2
            off_y = (h_alloc - h_img) / 2
            
            cr.save()
            Gdk.cairo_set_source_pixbuf(cr, self.scaled_capture_pixbuf, off_x, off_y)
            cr.paint()
            cr.restore()
            
            # Draw selection border
            if self.is_dragging_capture and self.capture_rect_start and self.capture_rect_end:
                cr.set_source_rgba(0, 1, 0, 0.8)
                cr.set_line_width(2)
                x = min(self.capture_rect_start[0], self.capture_rect_end[0])
                y = min(self.capture_rect_start[1], self.capture_rect_end[1])
                w = abs(self.capture_rect_start[0] - self.capture_rect_end[0])
                h = abs(self.capture_rect_start[1] - self.capture_rect_end[1])
                cr.rectangle(x, y, w, h)
                cr.stroke()
            elif hasattr(self, 'capture_rect_raw') and self.capture_rect_raw is not None and self.capture_preview_pixbuf is not None and self.raw_linear_pixels is not None:
                h_raw, w_raw = self.raw_linear_pixels.shape[:2]
                rect_trans = map_raw_rect_to_transformed(
                    self.capture_rect_raw, w_raw, h_raw, self.hflip, self.vflip, self.orientation
                )
                if rect_trans is not None:
                    x1_trans, y1_trans, x2_trans, y2_trans = rect_trans
                    w_orig = self.capture_preview_pixbuf.get_width()
                    h_orig = self.capture_preview_pixbuf.get_height()
                    scale = min(w_alloc / w_orig, h_alloc / h_orig)
                    off_x = (w_alloc - w_orig * scale) / 2
                    off_y = (h_alloc - h_orig * scale) / 2
                    
                    cr.set_source_rgba(0, 1, 0, 0.8)
                    cr.set_line_width(2)
                    x = x1_trans * scale + off_x
                    y = y1_trans * scale + off_y
                    w = (x2_trans - x1_trans) * scale
                    h = (y2_trans - y1_trans) * scale
                    cr.rectangle(x, y, w, h)
                    cr.stroke()

            # Draw processing version overlay (top-left)
            if self.profile:
                if self.has_icc and self.film_base_rgb:
                    status_text = "Crosstalk & ICC Corrected (Positive)"
                elif self.has_icc:
                    status_text = "Crosstalk Corrected Linear (No Film Base)"
                else:
                    status_text = "Crosstalk Corrected Linear (No ICC)"
            else:
                status_text = "Linear Raw (No Profile)"

            cr.save()
            cr.select_font_face("Inter", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)
            cr.set_font_size(10)
            extents = cr.text_extents(status_text)
            box_w = extents.width + 16
            box_h = extents.height + 10
            cr.set_source_rgba(0.08, 0.08, 0.08, 0.75)
            cr.rectangle(10, 10, box_w, box_h)
            cr.fill()
            cr.set_source_rgb(0.9, 0.9, 0.9)
            cr.move_to(18, 10 + 5 + extents.height)
            cr.show_text(status_text)
            cr.restore()

            # Draw captured metadata overlay (top-right)
            if self.raw_image and self.raw_linear_pixels is not None:
                h_p, w_p = self.raw_linear_pixels.shape[:2]
                paths_str = ", ".join(self.raw_image.filepaths)
                lines = [
                    f"ISO {self.raw_image.iso} | {self.raw_image.shutter_speed:.4f}s | {w_p}x{h_p}",
                    paths_str
                ]
                
                cr.save()
                cr.select_font_face("Inter", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)
                cr.set_font_size(10)
                
                max_w = 0
                line_heights = []
                for line in lines:
                    ext = cr.text_extents(line)
                    if ext.width > max_w:
                        max_w = ext.width
                    line_heights.append(ext.height)
                
                box_w_m = max_w + 16
                spacing = 4
                total_text_height = sum(line_heights) + spacing * (len(lines) - 1)
                box_h_m = total_text_height + 10
                
                x_pos = w_alloc - box_w_m - 10
                cr.set_source_rgba(0.08, 0.08, 0.08, 0.75)
                cr.rectangle(x_pos, 10, box_w_m, box_h_m)
                cr.fill()
                
                cr.set_source_rgb(0.9, 0.9, 0.9)
                curr_y = 10 + 5
                for i, line in enumerate(lines):
                    curr_y += line_heights[i]
                    cr.move_to(x_pos + 8, curr_y)
                    cr.show_text(line)
                    curr_y += spacing
                    
                cr.restore()

    def on_draw_base(self, widget, cr):
        if hasattr(self, 'scaled_base_pixbuf') and self.scaled_base_pixbuf is not None:
            w_alloc = widget.get_allocated_width()
            h_alloc = widget.get_allocated_height()
            w_img = self.scaled_base_pixbuf.get_width()
            h_img = self.scaled_base_pixbuf.get_height()
            off_x = (w_alloc - w_img) / 2
            off_y = (h_alloc - h_img) / 2
            
            cr.save()
            Gdk.cairo_set_source_pixbuf(cr, self.scaled_base_pixbuf, off_x, off_y)
            cr.paint()
            cr.restore()
            
            if self.is_dragging_base and self.base_rect_start and self.base_rect_end:
                cr.set_source_rgba(0, 1, 0, 0.8)
                cr.set_line_width(2)
                x = min(self.base_rect_start[0], self.base_rect_end[0])
                y = min(self.base_rect_start[1], self.base_rect_end[1])
                w = abs(self.base_rect_start[0] - self.base_rect_end[0])
                h = abs(self.base_rect_start[1] - self.base_rect_end[1])
                cr.rectangle(x, y, w, h)
                cr.stroke()
            elif hasattr(self, 'base_rect_raw') and self.base_rect_raw is not None and self.base_preview_pixbuf is not None and self.film_base_raw_linear is not None:
                h_raw, w_raw = self.film_base_raw_linear.shape[:2]
                rect_trans = map_raw_rect_to_transformed(
                    self.base_rect_raw, w_raw, h_raw, False, False, 0
                )
                if rect_trans is not None:
                    x1_trans, y1_trans, x2_trans, y2_trans = rect_trans
                    w_orig = self.base_preview_pixbuf.get_width()
                    h_orig = self.base_preview_pixbuf.get_height()
                    scale = min(w_alloc / w_orig, h_alloc / h_orig)
                    off_x = (w_alloc - w_orig * scale) / 2
                    off_y = (h_alloc - h_orig * scale) / 2
                    
                    cr.set_source_rgba(0, 1, 0, 0.8)
                    cr.set_line_width(2)
                    x = x1_trans * scale + off_x
                    y = y1_trans * scale + off_y
                    w = (x2_trans - x1_trans) * scale
                    h = (y2_trans - y1_trans) * scale
                    cr.rectangle(x, y, w, h)
                    cr.stroke()

            # Draw processing version overlay (top-left)
            if self.profile and self.has_crosstalk:
                status_text = "Crosstalk Corrected Linear"
            else:
                status_text = "Linear Raw"

            cr.save()
            cr.select_font_face("Inter", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)
            cr.set_font_size(10)
            extents = cr.text_extents(status_text)
            box_w = extents.width + 16
            box_h = extents.height + 10
            cr.set_source_rgba(0.08, 0.08, 0.08, 0.75)
            cr.rectangle(10, 10, box_w, box_h)
            cr.fill()
            cr.set_source_rgb(0.9, 0.9, 0.9)
            cr.move_to(18, 10 + 5 + extents.height)
            cr.show_text(status_text)
            cr.restore()

            # Draw captured metadata overlay (top-right)
            if self.film_base_img and self.film_base_raw_linear is not None:
                h_p, w_p = self.film_base_raw_linear.shape[:2]
                paths_str = ", ".join(self.film_base_img.filepaths)
                lines = [
                    f"ISO {self.film_base_img.iso} | {self.film_base_img.shutter_speed:.4f}s | {w_p}x{h_p}",
                    paths_str
                ]
                
                cr.save()
                cr.select_font_face("Inter", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)
                cr.set_font_size(10)
                
                max_w = 0
                line_heights = []
                for line in lines:
                    ext = cr.text_extents(line)
                    if ext.width > max_w:
                        max_w = ext.width
                    line_heights.append(ext.height)
                
                box_w_m = max_w + 16
                spacing = 4
                total_text_height = sum(line_heights) + spacing * (len(lines) - 1)
                box_h_m = total_text_height + 10
                
                x_pos = w_alloc - box_w_m - 10
                cr.set_source_rgba(0.08, 0.08, 0.08, 0.75)
                cr.rectangle(x_pos, 10, box_w_m, box_h_m)
                cr.fill()
                
                cr.set_source_rgb(0.9, 0.9, 0.9)
                curr_y = 10 + 5
                for i, line in enumerate(lines):
                    curr_y += line_heights[i]
                    cr.move_to(x_pos + 8, curr_y)
                    cr.show_text(line)
                    curr_y += spacing
                    
                cr.restore()

    # Mouse events
    def on_base_press(self, w, e):
        if not hasattr(self, 'scaled_base_pixbuf') or self.scaled_base_pixbuf is None:
            return
        w_alloc = self.base_image_area.get_allocated_width()
        h_alloc = self.base_image_area.get_allocated_height()
        w_img = self.scaled_base_pixbuf.get_width()
        h_img = self.scaled_base_pixbuf.get_height()
        off_x = (w_alloc - w_img) / 2
        off_y = (h_alloc - h_img) / 2
        
        x_clamped = max(off_x, min(off_x + w_img, e.x))
        y_clamped = max(off_y, min(off_y + h_img, e.y))
        
        self.is_dragging_base = True
        self.base_rect_start = (x_clamped, y_clamped)
        self.base_rect_end = (x_clamped, y_clamped)
        self.base_rect_raw = None
        self.base_image_area.queue_draw()

    def on_base_motion(self, w, e):
        if self.is_dragging_base and hasattr(self, 'scaled_base_pixbuf') and self.scaled_base_pixbuf is not None:
            w_alloc = self.base_image_area.get_allocated_width()
            h_alloc = self.base_image_area.get_allocated_height()
            w_img = self.scaled_base_pixbuf.get_width()
            h_img = self.scaled_base_pixbuf.get_height()
            off_x = (w_alloc - w_img) / 2
            off_y = (h_alloc - h_img) / 2
            
            x_clamped = max(off_x, min(off_x + w_img, e.x))
            y_clamped = max(off_y, min(off_y + h_img, e.y))
            
            self.base_rect_end = (x_clamped, y_clamped)
            self.base_image_area.queue_draw()

    def on_base_release(self, w, e):
        if self.is_dragging_base:
            self.is_dragging_base = False
            if hasattr(self, 'scaled_base_pixbuf') and self.scaled_base_pixbuf is not None:
                w_alloc = self.base_image_area.get_allocated_width()
                h_alloc = self.base_image_area.get_allocated_height()
                w_img = self.scaled_base_pixbuf.get_width()
                h_img = self.scaled_base_pixbuf.get_height()
                off_x = (w_alloc - w_img) / 2
                off_y = (h_alloc - h_img) / 2
                
                x_clamped = max(off_x, min(off_x + w_img, e.x))
                y_clamped = max(off_y, min(off_y + h_img, e.y))
                self.base_rect_end = (x_clamped, y_clamped)
                
                w_orig = self.base_preview_pixbuf.get_width()
                h_orig = self.base_preview_pixbuf.get_height()
                scale = min(w_alloc / w_orig, h_alloc / h_orig)
                
                x1 = int((min(self.base_rect_start[0], self.base_rect_end[0]) - off_x) / scale)
                x2 = int((max(self.base_rect_start[0], self.base_rect_end[0]) - off_x) / scale)
                y1 = int((min(self.base_rect_start[1], self.base_rect_end[1]) - off_y) / scale)
                y2 = int((max(self.base_rect_start[1], self.base_rect_end[1]) - off_y) / scale)
                
                x1 = max(0, min(w_orig, x1))
                x2 = max(0, min(w_orig, x2))
                y1 = max(0, min(h_orig, y1))
                y2 = max(0, min(h_orig, y2))
                
                if x2 > x1 and y2 > y1:
                    self.base_rect_raw = map_transformed_rect_to_raw(
                        (x1, y1, x2, y2), w_orig, h_orig, False, False, 0
                    )
                else:
                    self.base_rect_raw = None
            else:
                self.base_rect_raw = None
                
            self.base_image_area.queue_draw()
            if self.notebook.get_current_page() == 1:
                self.update_base_histograms()

    def on_capture_press(self, w, e):
        if not hasattr(self, 'scaled_capture_pixbuf') or self.scaled_capture_pixbuf is None:
            return
        w_alloc = self.capture_image_area.get_allocated_width()
        h_alloc = self.capture_image_area.get_allocated_height()
        w_img = self.scaled_capture_pixbuf.get_width()
        h_img = self.scaled_capture_pixbuf.get_height()
        off_x = (w_alloc - w_img) / 2
        off_y = (h_alloc - h_img) / 2
        
        x_clamped = max(off_x, min(off_x + w_img, e.x))
        y_clamped = max(off_y, min(off_y + h_img, e.y))
        
        self.is_dragging_capture = True
        self.capture_rect_start = (x_clamped, y_clamped)
        self.capture_rect_end = (x_clamped, y_clamped)
        self.capture_rect_raw = None
        self.capture_image_area.queue_draw()

    def on_capture_motion(self, w, e):
        if self.is_dragging_capture and hasattr(self, 'scaled_capture_pixbuf') and self.scaled_capture_pixbuf is not None:
            w_alloc = self.capture_image_area.get_allocated_width()
            h_alloc = self.capture_image_area.get_allocated_height()
            w_img = self.scaled_capture_pixbuf.get_width()
            h_img = self.scaled_capture_pixbuf.get_height()
            off_x = (w_alloc - w_img) / 2
            off_y = (h_alloc - h_img) / 2
            
            x_clamped = max(off_x, min(off_x + w_img, e.x))
            y_clamped = max(off_y, min(off_y + h_img, e.y))
            
            self.capture_rect_end = (x_clamped, y_clamped)
            self.capture_image_area.queue_draw()

    def on_capture_release(self, w, e):
        if self.is_dragging_capture:
            self.is_dragging_capture = False
            if hasattr(self, 'scaled_capture_pixbuf') and self.scaled_capture_pixbuf is not None:
                w_alloc = self.capture_image_area.get_allocated_width()
                h_alloc = self.capture_image_area.get_allocated_height()
                w_img = self.scaled_capture_pixbuf.get_width()
                h_img = self.scaled_capture_pixbuf.get_height()
                off_x = (w_alloc - w_img) / 2
                off_y = (h_alloc - h_img) / 2
                
                x_clamped = max(off_x, min(off_x + w_img, e.x))
                y_clamped = max(off_y, min(off_y + h_img, e.y))
                self.capture_rect_end = (x_clamped, y_clamped)
                
                if self.capture_preview_pixbuf:
                    w_orig = self.capture_preview_pixbuf.get_width()
                    h_orig = self.capture_preview_pixbuf.get_height()
                    scale = min(w_alloc / w_orig, h_alloc / h_orig)
                    
                    x1 = int((min(self.capture_rect_start[0], self.capture_rect_end[0]) - off_x) / scale)
                    x2 = int((max(self.capture_rect_start[0], self.capture_rect_end[0]) - off_x) / scale)
                    y1 = int((min(self.capture_rect_start[1], self.capture_rect_end[1]) - off_y) / scale)
                    y2 = int((max(self.capture_rect_start[1], self.capture_rect_end[1]) - off_y) / scale)
                    
                    x1 = max(0, min(w_orig, x1))
                    x2 = max(0, min(w_orig, x2))
                    y1 = max(0, min(h_orig, y1))
                    y2 = max(0, min(h_orig, y2))
                    
                    if x2 > x1 and y2 > y1:
                        self.capture_rect_raw = map_transformed_rect_to_raw(
                            (x1, y1, x2, y2), w_orig, h_orig, self.hflip, self.vflip, self.orientation
                        )
                    else:
                        self.capture_rect_raw = None
                else:
                    self.capture_rect_raw = None
            else:
                self.capture_rect_raw = None
                
            self.capture_image_area.queue_draw()
            if self.notebook.get_current_page() == 0:
                self.update_capture_histograms()

    def suggest_filename(self, folder, film_name, file_type, extension):
        # Clean film name: replace spaces and non-alphanumeric chars with underscores
        import re
        clean_film = re.sub(r'[^a-zA-Z0-9]', '_', film_name)
        clean_film = re.sub(r'_+', '_', clean_film).strip('_')
        
        prefix = f"{clean_film}_{file_type}_"
        
        max_idx = 0
        if os.path.exists(folder):
            try:
                for f in os.listdir(folder):
                    if f.startswith(prefix) and (f.endswith(f".{extension}") or f.endswith(f".{extension.upper()}") or f.endswith(f".{extension.lower()}")):
                        ext_len = len(extension)
                        name_part = f[len(prefix):-ext_len-1]
                        if name_part.isdigit():
                            max_idx = max(max_idx, int(name_part))
            except Exception:
                pass
        
        next_idx = max_idx + 1
        return f"{prefix}{next_idx:04d}.{extension}"

    def on_save_tiff(self, btn):
        if not self.raw_image:
            return
        
        dialog = Gtk.FileChooserDialog(
            title="Save TIFF", parent=self, action=Gtk.FileChooserAction.SAVE
        )
        dialog.add_buttons(Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL, Gtk.STOCK_SAVE, Gtk.ResponseType.OK)
        dialog.set_do_overwrite_confirmation(True)
        
        film_name = "capture"
        if self.profile and hasattr(self.profile, 'film_name') and self.profile.film_name:
            film_name = self.profile.film_name
            
        file_type = "corrected"
        extension = "tiff"
        
        folder = None
        if self.last_save_folder and os.path.exists(self.last_save_folder):
            folder = self.last_save_folder
        elif self.raw_image.filepaths:
            folder = os.path.dirname(self.raw_image.filepaths[0])
        if not folder or not os.path.exists(folder):
            folder = os.getcwd()
            
        dialog.set_current_folder(folder)
        
        default_filename = self.suggest_filename(folder, film_name, file_type, extension)
        dialog.set_current_name(default_filename)
        
        def on_folder_changed(file_chooser):
            curr_folder = file_chooser.get_current_folder()
            if curr_folder and os.path.exists(curr_folder):
                name = self.suggest_filename(curr_folder, film_name, file_type, extension)
                file_chooser.set_current_name(name)
                
        dialog.connect("current-folder-changed", on_folder_changed)
        
        if dialog.run() == Gtk.ResponseType.OK:
            filepath = dialog.get_filename()
            dialog.destroy()
            
            self.lbl_status.set_text("Saving TIFF image...")
            self.btn_capture.set_sensitive(False)
            self.btn_save_tiff.set_sensitive(False)
            self.btn_save_raw_capture.set_sensitive(False)
            self.btn_save_raw_base.set_sensitive(False)
            self.spinner.start()
            
            def save_task():
                try:
                    t_start = time.time()
                    if self.film_base_rgb and self.profile and self.has_icc:
                        prof_data = json.loads(json.dumps(self.profile.raw_data))
                        if 'targets' in prof_data and self.selected_target_idx < len(prof_data['targets']):
                            prof_data['targets'] = [prof_data['targets'][self.selected_target_idx]]
                            tgt = prof_data['targets'][0]
                            if 'icc_profile_base64' in tgt:
                                prof_data['icc_profile_base64'] = tgt['icc_profile_base64']
                        temp_profile = FilmProfile(prof_data)
                        if temp_profile.icc_profile_bytes is None and getattr(self.profile, 'icc_profile_bytes', None):
                            temp_profile.icc_profile_bytes = self.profile.icc_profile_bytes
                            
                        import film_profiling
                        film_profiling.convert_raw_to_tiff(
                            img=self.raw_image, profile=temp_profile, output_path=filepath,
                            exposure_comp=self.gain, half=False, film_base_rgb=self.film_base_rgb,
                            film_base_img=self.film_base_img, pipeline="cuda"
                        )
                    else:
                        matrix = None
                        if self.profile and self.has_crosstalk:
                            matrix = [val for row in self.profile.crosstalk_matrix for val in row]
                        self.raw_image.write_tiff(filepath, half=False, crosstalk_matrix=matrix)
                        
                    tag_val = get_exif_orientation(self.hflip, self.vflip, self.orientation)
                    set_tiff_orientation_inplace(filepath, tag_val)
                    
                    t_dur = time.time() - t_start
                    GLib.idle_add(self.on_save_success, filepath, t_dur)
                except Exception as e:
                    import traceback
                    print(f"Save error: {e}", file=sys.stdout)
                    traceback.print_exc(file=sys.stdout)
                    sys.stdout.flush()
                    GLib.idle_add(self.update_ui_failure, f"Save Error: {e}")
            
            t = threading.Thread(target=save_task)
            t.daemon = True
            t.start()
        else:
            dialog.destroy()

    def on_save_raw_capture(self, btn):
        self.on_save_raw(self.raw_image, is_base=False)

    def on_save_raw_base(self, btn):
        self.on_save_raw(self.film_base_img, is_base=True)

    def on_save_raw(self, img_obj, is_base):
        if not img_obj:
            return
        
        filepaths = img_obj.filepaths
        if not filepaths:
            return
            
        dialog = Gtk.FileChooserDialog(
            title="Save RAW", parent=self, action=Gtk.FileChooserAction.SAVE
        )
        dialog.add_buttons(Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL, Gtk.STOCK_SAVE, Gtk.ResponseType.OK)
        dialog.set_do_overwrite_confirmation(True)
        
        film_name = "capture"
        if self.profile and hasattr(self.profile, 'film_name') and self.profile.film_name:
            film_name = self.profile.film_name
            
        file_type = "base" if is_base else "raw"
        extension = "zip" if len(filepaths) > 1 else "ARW"
        
        if len(filepaths) == 1:
            filter_arw = Gtk.FileFilter()
            filter_arw.set_name("Sony RAW Image (*.ARW)")
            filter_arw.add_pattern("*.arw")
            filter_arw.add_pattern("*.ARW")
            dialog.add_filter(filter_arw)
        else:
            filter_zip = Gtk.FileFilter()
            filter_zip.set_name("ZIP Archive (*.zip)")
            filter_zip.add_pattern("*.zip")
            dialog.add_filter(filter_zip)
            
        folder = None
        if self.last_save_folder and os.path.exists(self.last_save_folder):
            folder = self.last_save_folder
        elif filepaths:
            folder = os.path.dirname(filepaths[0])
        if not folder or not os.path.exists(folder):
            folder = os.getcwd()
            
        dialog.set_current_folder(folder)
        
        default_filename = self.suggest_filename(folder, film_name, file_type, extension)
        dialog.set_current_name(default_filename)
        
        def on_folder_changed(file_chooser):
            curr_folder = file_chooser.get_current_folder()
            if curr_folder and os.path.exists(curr_folder):
                name = self.suggest_filename(curr_folder, film_name, file_type, extension)
                file_chooser.set_current_name(name)
                
        dialog.connect("current-folder-changed", on_folder_changed)
        
        if dialog.run() == Gtk.ResponseType.OK:
            dest_path = dialog.get_filename()
            dialog.destroy()
            
            self.lbl_status.set_text("Saving RAW image...")
            self.btn_capture.set_sensitive(False)
            self.btn_save_tiff.set_sensitive(False)
            self.btn_save_raw_capture.set_sensitive(False)
            self.btn_save_raw_base.set_sensitive(False)
            self.spinner.start()
            
            def save_task():
                try:
                    t_start = time.time()
                    if len(filepaths) == 1:
                        # Copy single ARW
                        shutil.copy2(filepaths[0], dest_path)
                    else:
                        # Create ZIP archive
                        with zipfile.ZipFile(dest_path, 'w', zipfile.ZIP_DEFLATED) as z:
                            for path in filepaths:
                                z.write(path, os.path.basename(path))
                                
                    t_dur = time.time() - t_start
                    GLib.idle_add(self.on_save_raw_success, dest_path, t_dur)
                except Exception as e:
                    import traceback
                    print(f"Save RAW error: {e}", file=sys.stdout)
                    traceback.print_exc(file=sys.stdout)
                    sys.stdout.flush()
                    GLib.idle_add(self.update_ui_failure, f"Save RAW Error: {e}")
            
            t = threading.Thread(target=save_task)
            t.daemon = True
            t.start()
        else:
            dialog.destroy()

    def on_save_raw_success(self, filepath, duration):
        self.spinner.stop()
        self.is_capturing = False
        self.last_save_folder = os.path.dirname(filepath)
        self.update_capture_button_sensitivity()
        self.update_save_buttons_sensitivity()
        self.lbl_status.set_text(f"Status: Saved RAW to {os.path.basename(filepath)}.")
        print(f"[Save RAW] RAW image saved successfully to: {filepath} | Time taken: {duration:.2f}s", file=sys.stdout)
        sys.stdout.flush()

    def on_save_success(self, filepath, duration):
        self.spinner.stop()
        self.is_capturing = False
        self.last_save_folder = os.path.dirname(filepath)
        self.update_capture_button_sensitivity()
        self.update_save_buttons_sensitivity()
        self.lbl_status.set_text(f"Status: Saved {os.path.basename(filepath)}.")
        print(f"[Save TIFF] Image saved successfully to: {filepath} | Time taken: {duration:.2f}s", file=sys.stdout)
        sys.stdout.flush()

    def update_ui_failure(self, error_msg):
        self.spinner.stop()
        self.is_capturing = False
        self.update_capture_button_sensitivity()
        self.mode_combo.set_sensitive(True)
        self.shutter_combo.set_sensitive(not self.ae_checkbox.get_active())
        self.ae_checkbox.set_sensitive(True)
        self.update_save_buttons_sensitivity()
        self.lbl_status.set_text(f"Status: Capture failed.")
        
        dialog = Gtk.MessageDialog(
            transient_for=self,
            flags=0,
            message_type=Gtk.MessageType.ERROR,
            buttons=Gtk.ButtonsType.OK,
            text="Capture Error"
        )
        dialog.format_secondary_text(error_msg if error_msg else "An unknown error occurred during image capture.")
        dialog.run()
        dialog.destroy()

    def on_window_resized(self, widget, allocation):
        if self.capture_preview_pixbuf:
            self.refresh_capture_preview()
        if self.base_preview_pixbuf:
            self.refresh_base_preview()

    def on_destroy(self, widget):
        if self.raw_image:
            try:
                self.raw_image.discard()
            except Exception:
                pass
        if self.film_base_img:
            try:
                self.film_base_img.discard()
            except Exception:
                pass
        if self.camera_session:
            try:
                self.camera_session.close()
            except Exception:
                pass
        Gtk.main_quit()

def main():
    # Preload the Sony CrSDK shared library from the virtual environment
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    lib_path = os.path.join(project_dir, 'venv/bin/libCr_Core.so')
    if os.path.exists(lib_path):
        import ctypes
        ctypes.CDLL(lib_path)

    app = ScanningAppWindow()
    app.show_all()
    Gtk.main()

if __name__ == '__main__':
    main()
