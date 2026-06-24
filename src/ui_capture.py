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

from matplotlib.figure import Figure
from matplotlib.backends.backend_gtk3agg import FigureCanvasGTK3Agg as FigureCanvas

import negicc_station
from film_profiling import FilmProfile
import color_conversion
from target_selection import find_best_target_index
import auto_exposure

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
    row_pos = 1 # Top
    col_pos = 3 # Left
    if hflip: col_pos = 4 if col_pos == 3 else 3
    if vflip: row_pos = 2 if row_pos == 1 else 1
    for _ in range(rot_cw // 90):
        new_row = {1:4, 4:2, 2:3, 3:1}[row_pos]
        new_col = {1:4, 4:2, 2:3, 3:1}[col_pos]
        row_pos, col_pos = new_row, new_col
    tag_map = {
        (1, 3): 1, (1, 4): 2, (2, 4): 3, (2, 3): 4,
        (3, 1): 5, (4, 1): 6, (4, 2): 7, (3, 2): 8
    }
    return tag_map.get((row_pos, col_pos), 1)

def apply_transforms_numpy(img_array, hflip, vflip, rot_cw):
    if hflip: img_array = np.fliplr(img_array)
    if vflip: img_array = np.flipud(img_array)
    if rot_cw == 90: img_array = np.rot90(img_array, -1)
    elif rot_cw == 180: img_array = np.rot90(img_array, -2)
    elif rot_cw == 270: img_array = np.rot90(img_array, -3)
    return img_array

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
        self.figure.tight_layout()
        
        # Connect size-allocate to dynamically force height as a 2:3 ratio of width
        self.canvas.connect("size-allocate", self.on_size_allocate)

    def on_size_allocate(self, widget, allocation):
        # Enforce 2:3 height-to-width ratio
        target_height = int(allocation.width * 2 // 3)
        target_height = max(100, target_height)
        if widget.get_size_request()[1] != target_height:
            widget.set_size_request(-1, target_height)

    def clear(self):
        self.ax.clear()
        self.ax.set_facecolor('#121212')
        self.ax.grid(True, color='#2c2c2c', linestyle='--', linewidth=0.5)
        self.canvas.draw_idle()

    def plot_histogram(self, data, is_corrected, has_icc, show_overexposure=True):
        self.ax.clear()
        self.ax.grid(True, color='#2c2c2c', linestyle='--', linewidth=0.5)
        
        if data is None or data.size == 0:
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
        
        self.camera_session = None
        self.is_connected = False
        self.is_connecting = False
        
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

        # Connect window keypress listener
        self.connect("key-press-event", self.on_key_press)

        # Initial background camera connection
        self.connect_camera()
        GLib.timeout_add_seconds(2, self.poll_camera_connection)

    def apply_css(self):
        css_provider = Gtk.CssProvider()
        css_provider.load_from_data(b"""
            .sidebar { background-color: #1e1e1e; padding: 15px; }
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
        left_sidebar.set_size_request(260, -1)
        main_box.pack_start(left_sidebar, False, False, 0)
        
        # Camera status indicator
        self.lbl_camera_status = Gtk.Label()
        self.lbl_camera_status.set_markup("<span><span foreground='#e6a23c'>●</span> <b>Camera: Connecting...</b></span>")
        self.lbl_camera_status.get_style_context().add_class("status-indicator")
        left_sidebar.pack_start(self.lbl_camera_status, False, False, 10)
        
        # Profile Section
        left_sidebar.pack_start(Gtk.Label(label="<b>Calibration Profile</b>", use_markup=True), False, False, 2)
        btn_load = Gtk.Button(label="Load Profile...")
        btn_load.connect("clicked", self.on_load_profile)
        left_sidebar.pack_start(btn_load, False, False, 0)
        
        self.lbl_profile_info = Gtk.Label(label="No profile loaded.")
        self.lbl_profile_info.set_line_wrap(True)
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
        
        # Capture Actions
        self.btn_cap_base = Gtk.Button(label="Capture Film Base")
        self.btn_cap_base.get_style_context().add_class("btn-action")
        self.btn_cap_base.get_style_context().add_class("btn-yellow")
        self.btn_cap_base.connect("clicked", self.on_capture_base_clicked)
        left_sidebar.pack_start(self.btn_cap_base, False, False, 0)
        
        self.btn_cap_img = Gtk.Button(label="Capture Image")
        self.btn_cap_img.get_style_context().add_class("btn-action")
        self.btn_cap_img.get_style_context().add_class("btn-green")
        self.btn_cap_img.connect("clicked", self.on_capture_image_clicked)
        left_sidebar.pack_start(self.btn_cap_img, False, False, 0)

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
        
        # Image Metadata Box
        meta_frame = Gtk.Frame(label="Captured Metadata")
        self.meta_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=5)
        self.meta_box.set_border_width(8)
        meta_frame.add(self.meta_box)
        left_sidebar.pack_start(meta_frame, True, True, 5)
        
        self.iso_label = Gtk.Label(label="ISO: --")
        self.iso_label.set_xalign(0.0)
        self.iso_label.get_style_context().add_class("meta-label")
        self.meta_box.pack_start(self.iso_label, False, False, 0)
        
        self.shutter_label = Gtk.Label(label="Shutter Speed: --")
        self.shutter_label.set_xalign(0.0)
        self.shutter_label.get_style_context().add_class("meta-label")
        self.meta_box.pack_start(self.shutter_label, False, False, 0)
        
        self.size_label = Gtk.Label(label="Dimensions: --")
        self.size_label.set_xalign(0.0)
        self.size_label.get_style_context().add_class("meta-label")
        self.meta_box.pack_start(self.size_label, False, False, 0)
        
        self.files_label = Gtk.Label(label="RAW Path(s): --")
        self.files_label.set_xalign(0.0)
        self.files_label.set_line_wrap(True)
        self.files_label.get_style_context().add_class("meta-label")
        self.meta_box.pack_start(self.files_label, False, False, 0)
        
        self.capture_time_label = Gtk.Label(label="Capture Duration: --")
        self.capture_time_label.set_xalign(0.0)
        self.capture_time_label.get_style_context().add_class("meta-label")
        self.meta_box.pack_start(self.capture_time_label, False, False, 0)
        
        self.convert_time_label = Gtk.Label(label="Conversion Duration: --")
        self.convert_time_label.set_xalign(0.0)
        self.convert_time_label.get_style_context().add_class("meta-label")
        self.meta_box.pack_start(self.convert_time_label, False, False, 0)

        # Spinner & status display
        status_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=5)
        self.spinner = Gtk.Spinner()
        status_box.pack_start(self.spinner, False, False, 0)
        self.lbl_status = Gtk.Label(label="Status: Connecting")
        self.lbl_status.set_line_wrap(True)
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
        right_sidebar.get_style_context().add_class("sidebar")
        right_sidebar.set_size_request(440, -1)
        main_box.pack_start(right_sidebar, False, False, 0)
        
        lbl_hist_raw = Gtk.Label(label="RAW Linear (Uncorrected)")
        right_sidebar.pack_start(lbl_hist_raw, False, False, 0)
        self.hist_raw = HistogramCanvas()
        right_sidebar.pack_start(self.hist_raw, False, False, 0)
        
        lbl_hist_corr = Gtk.Label(label="Corrected Preview")
        right_sidebar.pack_start(lbl_hist_corr, False, False, 0)
        self.hist_corr = HistogramCanvas()
        right_sidebar.pack_start(self.hist_corr, False, False, 0)
        
        # Dynamic Range per channel display
        self.lbl_dr = Gtk.Label()
        self.lbl_dr.set_use_markup(True)
        self.lbl_dr.set_xalign(0.0)
        self.lbl_dr.set_line_wrap(True)
        self.lbl_dr.get_style_context().add_class("meta-label")
        right_sidebar.pack_start(self.lbl_dr, False, False, 0)

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
        
        top_box.pack_start(Gtk.Separator(orientation=Gtk.Orientation.VERTICAL), False, False, 5)
        
        top_box.pack_start(Gtk.Label(label="Gain: "), False, False, 0)
        btn_g_down = Gtk.Button(label="-")
        btn_g_down.connect("clicked", lambda x: self.adj_gain(-0.10))
        top_box.pack_start(btn_g_down, False, False, 0)
        self.entry_gain = Gtk.Entry()
        self.entry_gain.set_text("1.00")
        self.entry_gain.set_width_chars(5)
        self.entry_gain.connect("activate", self.on_gain_entry_activated)
        top_box.pack_start(self.entry_gain, False, False, 5)
        btn_g_up = Gtk.Button(label="+")
        btn_g_up.connect("clicked", lambda x: self.adj_gain(0.10))
        top_box.pack_start(btn_g_up, False, False, 0)
        
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
        target_box.pack_start(Gtk.Label(label="Select Profile Target:"), False, False, 5)
        
        self.target_liststore = Gtk.ListStore(int, str, str) # Index, Target Name, Mid-grey Match Distance
        self.target_treeview = Gtk.TreeView(model=self.target_liststore)
        
        renderer_text = Gtk.CellRendererText()
        col1 = Gtk.TreeViewColumn("Target Name", renderer_text, text=1)
        col2 = Gtk.TreeViewColumn("Mid-Grey Distance", renderer_text, text=2)
        self.target_treeview.append_column(col1)
        self.target_treeview.append_column(col2)
        
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
        
        self.lbl_base_vals = Gtk.Label(label="Raw: -- | Corr: --")
        top_box.pack_start(self.lbl_base_vals, False, False, 0)

        # PREVIEW AREA
        preview_event_box = Gtk.EventBox()
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
        elif not self.is_connecting:
            if negicc_station.is_camera_connected():
                self.connect_camera()
        return True

    def connect_camera(self):
        if self.is_connecting or self.is_connected:
            return
        self.is_connecting = True
        self.btn_cap_img.set_sensitive(False)
        self.btn_cap_base.set_sensitive(False)
        self.lbl_camera_status.set_markup("<span><span foreground='#e6a23c'>●</span> <b>Camera: Connecting...</b></span>")
        
        def run():
            try:
                if self.camera_session is None:
                    self.camera_session = negicc_station.CameraSession()
                ok = self.camera_session.connect()
                if ok:
                    self.is_connected = True
                    GLib.idle_add(self.update_connection_ui, True, None)
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
            self.btn_cap_img.set_sensitive(True)
            self.btn_cap_base.set_sensitive(True)
            self.lbl_status.set_text("Status: Camera connected, ready.")
        else:
            self.lbl_camera_status.set_markup("<span foreground='#ff4444'>●</span> <b>Camera: Disconnected</b>")
            self.btn_cap_img.set_sensitive(False)
            self.btn_cap_base.set_sensitive(False)
            if error_msg:
                self.lbl_status.set_text(f"Status: Disconnected ({error_msg})")
            else:
                self.lbl_status.set_text("Status: Disconnected.")

    def adj_gain(self, delta):
        self.gain = max(0.1, self.gain + delta)
        self.entry_gain.set_text(f"{self.gain:.2f}")
        self.update_capture_preview()

    def on_gain_entry_activated(self, entry):
        text = entry.get_text()
        try:
            val = float(text)
            self.gain = max(0.1, val)
            entry.set_text(f"{self.gain:.2f}")
            self.update_capture_preview()
        except ValueError:
            entry.set_text(f"{self.gain:.2f}")

    def on_key_press(self, widget, event):
        keyval = event.keyval
        
        # Don't intercept gain keys if typing in entry
        focus_widget = self.get_focus()
        if focus_widget == self.entry_gain:
            return False
            
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

    def on_hflip(self, btn):
        self.hflip = btn.get_active()
        self.update_capture_preview()

    def on_vflip(self, btn):
        self.vflip = btn.get_active()
        self.update_capture_preview()

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
                
                status_txt = f"Loaded: {self.profile.film_name}\n"
                if self.has_icc: status_txt += "• Has ICC curves/targets\n"
                if self.has_crosstalk: status_txt += "• Has Crosstalk matrix"
                
                self.lbl_profile_info.set_text(status_txt)
                self.lbl_status.set_text(f"Status: Profile loaded: {os.path.basename(filepath)}")
                
                # Update targets table dropdown
                self.target_liststore.clear()
                if hasattr(self.profile, 'raw_data') and 'targets' in self.profile.raw_data:
                    for idx, tgt in enumerate(self.profile.raw_data['targets']):
                        name = tgt.get('name', f"Target {idx}")
                        self.target_liststore.append([idx, name, "--"])
                        
                self.update_capture_preview()
            except Exception as e:
                self.lbl_profile_info.set_text(f"Error: {e}")
                self.lbl_status.set_text("Status: Failed to load profile.")
                self.profile = None
                
        dialog.destroy()

    def on_reset_profile(self, widget):
        self.profile = None
        self.profile_filename = ""
        self.has_icc = False
        self.has_crosstalk = False
        self.lbl_profile_info.set_text("No profile loaded.")
        self.lbl_status.set_text("Status: Profile cleared.")
        self.target_liststore.clear()
        self.update_capture_preview()

    def on_target_selection_changed(self, selection):
        model, treeiter = selection.get_selected()
        if treeiter is not None:
            idx = model[treeiter][0]
            if idx != self.selected_target_idx:
                self.selected_target_idx = idx
                self.update_capture_preview()

    def on_tab_changed(self, notebook, page, page_num):
        if page_num == 0:
            self.update_capture_histograms()
        else:
            self.update_base_histograms()

    def on_ae_toggled(self, checkbox):
        self.shutter_combo.set_sensitive(not checkbox.get_active())

    def on_capture_image_clicked(self, widget):
        self.on_capture(is_base=False)

    def on_capture_base_clicked(self, widget):
        self.on_capture(is_base=True)

    def on_capture(self, is_base=False):
        if not self.is_connected or not self.camera_session:
            self.lbl_status.set_text("Error: Camera not connected.")
            return
            
        shutter_str = self.shutter_combo.get_active_text()
        is_ae = self.ae_checkbox.get_active()
        mode_id = int(self.mode_combo.get_active_id())
        
        self.lbl_status.set_text("Capturing...")
        self.clear_ae_steps()
        
        self.btn_cap_img.set_sensitive(False)
        self.btn_cap_base.set_sensitive(False)
        self.btn_save_tiff.set_sensitive(False)
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
                        arr = ae_img.to_numpy(half=True)
                        ae_img.discard()
                        return arr
                    
                    def ae_progress(step_idx, ss, ch_dr, avg_dr):
                        dr_r, dr_g, dr_b = ch_dr
                        GLib.idle_add(self.add_ae_step, ss, dr_r, dr_g, dr_b, avg_dr)
                    
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

    def add_ae_step(self, ss, dr_r, dr_g, dr_b, avg_dr):
        def fmt_dr(v):
            return "<span foreground='#ff6666'>OVR</span>" if v < 0 else f"{v:.0f}"
            
        row_label = Gtk.Label()
        row_label.set_markup(
            f"<span size='small' font_family='monospace'>"
            f"Speed: <b>{ss}</b>\n"
            f"DR: R:{fmt_dr(dr_r)} G:{fmt_dr(dr_g)} B:{fmt_dr(dr_b)} | <b>Avg:{fmt_dr(avg_dr)}</b>"
            f"</span>"
        )
        row_label.set_xalign(0.0)
        row = Gtk.ListBoxRow()
        row.add(row_label)
        self.ae_steps_listbox.add(row)
        self.ae_steps_listbox.show_all()

    def process_captured_image(self, img_obj, is_base, capture_duration):
        self.lbl_status.set_text("Processing capture...")
        
        t_conv_start = time.time()
        raw_linear = img_obj.to_numpy(half=True)
        conv_duration = time.time() - t_conv_start
        
        if is_base:
            if self.film_base_img: self.film_base_img.discard()
            self.film_base_img = img_obj
            self.film_base_raw_linear = raw_linear
            self.base_rect_start = None
            self.base_rect_end = None
            self.base_rect_raw = None
            
            raw = np.clip(raw_linear, 0, 16384)
            img_array = (raw / 16384.0 * 255).astype(np.uint8)
            img_array = np.ascontiguousarray(img_array)
            h, w, c = img_array.shape
            self.base_preview_pixbuf = GdkPixbuf.Pixbuf.new_from_data(
                img_array.tobytes(), GdkPixbuf.Colorspace.RGB, False, 8, w, h, w * 3
            )
            
            self.refresh_base_preview()
            self.update_base_histograms()
            self.notebook.set_current_page(1)
            
            self.btn_cap_img.set_sensitive(self.is_connected)
            self.btn_cap_base.set_sensitive(self.is_connected)
            self.btn_save_tiff.set_sensitive(self.raw_image is not None)
            self.spinner.stop()
            self.lbl_status.set_text("Status: Film base updated.")
            
        else:
            if self.raw_image: self.raw_image.discard()
            self.raw_image = img_obj
            self.raw_linear_pixels = raw_linear
            self.capture_rect_start = None
            self.capture_rect_end = None
            self.capture_rect_raw = None
            
            self.gain = 1.0
            self.entry_gain.set_text(f"{self.gain:.2f}")
            
            if self.film_base_rgb is not None and self.profile is not None and len(self.target_liststore) > 0:
                best_idx, dist = find_best_target_index(self.profile, raw_linear, self.film_base_rgb)
                self.selected_target_idx = best_idx
                
                for row in self.target_liststore:
                    row[2] = f"{dist:.2f}" if row[0] == best_idx else "--"
                    
                select = self.target_treeview.get_selection()
                for i, row in enumerate(self.target_liststore):
                    if row[0] == best_idx:
                        select.select_path(Gtk.TreePath.new_from_indices([i]))
                        break
                        
            # Update labels
            self.iso_label.set_text(f"ISO: {img_obj.iso}")
            self.shutter_label.set_text(f"Shutter Speed: {img_obj.shutter_speed:.4f}s")
            h, w = raw_linear.shape[:2]
            self.size_label.set_text(f"Dimensions: {w} x {h} (Half-size)")
            self.files_label.set_text("RAW Filepath(s):\n" + "\n".join(img_obj.filepaths))
            self.capture_time_label.set_text(f"Capture Duration: {capture_duration:.3f}s")
            self.convert_time_label.set_text(f"Conversion Duration: {conv_duration:.3f}s")
            
            self.update_capture_preview()
            self.notebook.set_current_page(0)
            
            self.btn_cap_img.set_sensitive(self.is_connected)
            self.btn_cap_base.set_sensitive(self.is_connected)
            self.btn_save_tiff.set_sensitive(True)
            self.spinner.stop()
            self.lbl_status.set_text("Status: Image updated successfully.")

    def update_base_histograms(self):
        if self.film_base_raw_linear is None:
            self.hist_raw.clear()
            self.hist_corr.clear()
            return
            
        data = self.film_base_raw_linear
        if hasattr(self, 'base_rect_raw') and self.base_rect_raw is not None:
            x1, y1, x2, y2 = self.base_rect_raw
            dh, dw = data.shape[:2]
            x1, x2 = max(0, x1), min(dw, x2)
            y1, y2 = max(0, y1), min(dh, y2)
            if x2 > x1 and y2 > y1:
                data = data[y1:y2, x1:x2]
                
        self.hist_raw.plot_histogram(data, is_corrected=False, has_icc=False, show_overexposure=True)
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
                
        rgb = np.mean(data, axis=(0, 1))
        self.film_base_rgb = (rgb[0], rgb[1], rgb[2])
        
        self.base_tab_label.set_markup("<span foreground='green'>Film Base</span>")
        self.capture_tab_label.set_markup("<span>Capture</span>")
        
        corr_rgb = rgb
        if self.has_crosstalk:
            corr_rgb = np.dot(rgb, self.profile.crosstalk_matrix.T)
            
        self.lbl_base_vals.set_text(f"Raw: {rgb[0]:.1f}, {rgb[1]:.1f}, {rgb[2]:.1f} | "
                                    f"Corr: {corr_rgb[0]:.1f}, {corr_rgb[1]:.1f}, {corr_rgb[2]:.1f}")
                                    
        if self.raw_image:
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
        
        if self.profile:
            if self.has_icc and self.film_base_rgb:
                prof_data = json.loads(json.dumps(self.profile.raw_data))
                if 'targets' in prof_data and self.selected_target_idx < len(prof_data['targets']):
                    prof_data['targets'] = [prof_data['targets'][self.selected_target_idx]]
                    tgt = prof_data['targets'][0]
                    if 'icc_profile_base64' in tgt:
                        prof_data['icc_profile_base64'] = tgt['icc_profile_base64']
                        
                temp_profile = FilmProfile(prof_data)
                if getattr(self.profile, 'icc_profile_bytes', None):
                    temp_profile.icc_profile_bytes = self.profile.icc_profile_bytes
                    
                res = color_conversion.convert_raw_to_tiff(
                    img=self.raw_image, profile=temp_profile, output_path="",
                    exposure_comp=self.gain, half=True, film_base_rgb=self.film_base_rgb
                )
                img_array = res
                corr_hist_array = res
            else:
                raw = self.raw_linear_pixels.astype(np.float32)
                if self.has_crosstalk:
                    raw = np.dot(raw, self.profile.crosstalk_matrix.T)
                raw = np.clip(raw * self.gain, 0, 16384)
                img_array = (raw / 16384.0 * 255).astype(np.uint8)
                corr_hist_array = raw
        else:
            raw = self.raw_linear_pixels.astype(np.float32) * self.gain
            raw_c = np.clip(raw, 0, 16384)
            img_array = (raw_c / 16384.0 * 255).astype(np.uint8)
            corr_hist_array = raw_c
            
        img_array = apply_transforms_numpy(img_array, self.hflip, self.vflip, self.orientation)
        
        h, w, c = img_array.shape
        img_array = np.ascontiguousarray(img_array)
        self.capture_preview_pixbuf = GdkPixbuf.Pixbuf.new_from_data(
            img_array.tobytes(), GdkPixbuf.Colorspace.RGB, False, 8, w, h, w * 3
        )
        self.capture_corr_hist_data = apply_transforms_numpy(corr_hist_array, self.hflip, self.vflip, self.orientation)
        self.capture_raw_hist_data = apply_transforms_numpy(self.raw_linear_pixels, self.hflip, self.vflip, self.orientation)
        
        self.refresh_capture_preview()
        if self.notebook.get_current_page() == 0:
            self.update_capture_histograms()

    def update_capture_histograms(self):
        if not hasattr(self, 'hist_raw') or self.hist_raw is None:
            return
        if not hasattr(self, 'capture_raw_hist_data') or self.capture_raw_hist_data is None:
            self.hist_raw.clear()
            self.hist_corr.clear()
            return
            
        raw_d = self.capture_raw_hist_data
        corr_d = self.capture_corr_hist_data
        
        if hasattr(self, 'capture_rect_raw') and self.capture_rect_raw is not None:
            x1, y1, x2, y2 = self.capture_rect_raw
            dh, dw = raw_d.shape[:2]
            x1, x2 = max(0, x1), min(dw, x2)
            y1, y2 = max(0, y1), min(dh, y2)
            if x2 > x1 and y2 > y1:
                raw_d = raw_d[y1:y2, x1:x2]
                corr_d = corr_d[y1:y2, x1:x2]
                
        self.hist_raw.plot_histogram(raw_d, is_corrected=False, has_icc=False, show_overexposure=True)
        self.hist_corr.plot_histogram(corr_d, is_corrected=True, has_icc=self.has_icc, show_overexposure=False)
        
        self._update_dr_label(raw_d)

    def _update_dr_label(self, raw_data):
        if not hasattr(self, 'lbl_dr') or self.lbl_dr is None:
            return
        if raw_data is None or raw_data.size == 0:
            self.lbl_dr.set_text("")
            return
        
        # Decimate if raw_data is large to ensure snappy UI responsiveness
        H, W = raw_data.shape[:2]
        total_pixels = H * W
        if total_pixels > 200000:
            step = int(np.sqrt(total_pixels / 200000))
            step = max(1, step)
            raw_data_sampled = raw_data[::step, ::step, :]
        else:
            raw_data_sampled = raw_data
            
        avg_dr, (dr_r, dr_g, dr_b) = auto_exposure.calculate_dynamic_range(raw_data_sampled)
        
        def fmt(v):
            return "<span foreground='#ff6666'><b>Overexposed</b></span>" if v < 0 else f"{v:.0f}"
        
        self.lbl_dr.set_markup(
            f"<b>Dynamic Range</b> (p5-p95)\n"
            f"  R: {fmt(dr_r)}  G: {fmt(dr_g)}  B: {fmt(dr_b)}\n"
            f"  Avg: {fmt(avg_dr)}"
        )

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
            elif hasattr(self, 'capture_rect_raw') and self.capture_rect_raw is not None and self.capture_preview_pixbuf is not None:
                x1_raw, y1_raw, x2_raw, y2_raw = self.capture_rect_raw
                w_orig = self.capture_preview_pixbuf.get_width()
                h_orig = self.capture_preview_pixbuf.get_height()
                scale = min(w_alloc / w_orig, h_alloc / h_orig)
                off_x = (w_alloc - w_orig * scale) / 2
                off_y = (h_alloc - h_orig * scale) / 2
                
                cr.set_source_rgba(0, 1, 0, 0.8)
                cr.set_line_width(2)
                x = x1_raw * scale + off_x
                y = y1_raw * scale + off_y
                w = (x2_raw - x1_raw) * scale
                h = (y2_raw - y1_raw) * scale
                cr.rectangle(x, y, w, h)
                cr.stroke()

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
            elif hasattr(self, 'base_rect_raw') and self.base_rect_raw is not None and self.base_preview_pixbuf is not None:
                x1_raw, y1_raw, x2_raw, y2_raw = self.base_rect_raw
                w_orig = self.base_preview_pixbuf.get_width()
                h_orig = self.base_preview_pixbuf.get_height()
                scale = min(w_alloc / w_orig, h_alloc / h_orig)
                off_x = (w_alloc - w_orig * scale) / 2
                off_y = (h_alloc - h_orig * scale) / 2
                
                cr.set_source_rgba(0, 1, 0, 0.8)
                cr.set_line_width(2)
                x = x1_raw * scale + off_x
                y = y1_raw * scale + off_y
                w = (x2_raw - x1_raw) * scale
                h = (y2_raw - y1_raw) * scale
                cr.rectangle(x, y, w, h)
                cr.stroke()

    # Mouse events
    def on_base_press(self, w, e):
        self.is_dragging_base = True
        self.base_rect_start = (e.x, e.y)
        self.base_rect_end = (e.x, e.y)
        self.base_rect_raw = None
        self.base_image_area.queue_draw()

    def on_base_motion(self, w, e):
        if self.is_dragging_base:
            self.base_rect_end = (e.x, e.y)
            self.base_image_area.queue_draw()

    def on_base_release(self, w, e):
        if self.is_dragging_base:
            self.is_dragging_base = False
            self.base_rect_end = (e.x, e.y)
            
            if self.base_preview_pixbuf:
                w_alloc = self.base_image_area.get_allocated_width()
                h_alloc = self.base_image_area.get_allocated_height()
                w_img = self.base_preview_pixbuf.get_width()
                h_img = self.base_preview_pixbuf.get_height()
                scale = min(w_alloc / w_img, h_alloc / h_img)
                off_x = (w_alloc - w_img * scale) / 2
                off_y = (h_alloc - h_img * scale) / 2
                
                x1 = int((min(self.base_rect_start[0], self.base_rect_end[0]) - off_x) / scale)
                x2 = int((max(self.base_rect_start[0], self.base_rect_end[0]) - off_x) / scale)
                y1 = int((min(self.base_rect_start[1], self.base_rect_end[1]) - off_y) / scale)
                y2 = int((max(self.base_rect_start[1], self.base_rect_end[1]) - off_y) / scale)
                
                if x2 > x1 and y2 > y1:
                    self.base_rect_raw = (x1, y1, x2, y2)
                else:
                    self.base_rect_raw = None
            else:
                self.base_rect_raw = None
                
            self.base_image_area.queue_draw()
            if self.notebook.get_current_page() == 1:
                self.update_base_histograms()

    def on_capture_press(self, w, e):
        self.is_dragging_capture = True
        self.capture_rect_start = (e.x, e.y)
        self.capture_rect_end = (e.x, e.y)
        self.capture_rect_raw = None
        self.capture_image_area.queue_draw()

    def on_capture_motion(self, w, e):
        if self.is_dragging_capture:
            self.capture_rect_end = (e.x, e.y)
            self.capture_image_area.queue_draw()

    def on_capture_release(self, w, e):
        if self.is_dragging_capture:
            self.is_dragging_capture = False
            self.capture_rect_end = (e.x, e.y)
            
            if self.capture_preview_pixbuf:
                w_alloc = self.capture_image_area.get_allocated_width()
                h_alloc = self.capture_image_area.get_allocated_height()
                w_img = self.capture_preview_pixbuf.get_width()
                h_img = self.capture_preview_pixbuf.get_height()
                scale = min(w_alloc / w_img, h_alloc / h_img)
                off_x = (w_alloc - w_img * scale) / 2
                off_y = (h_alloc - h_img * scale) / 2
                
                x1 = int((min(self.capture_rect_start[0], self.capture_rect_end[0]) - off_x) / scale)
                x2 = int((max(self.capture_rect_start[0], self.capture_rect_end[0]) - off_x) / scale)
                y1 = int((min(self.capture_rect_start[1], self.capture_rect_end[1]) - off_y) / scale)
                y2 = int((max(self.capture_rect_start[1], self.capture_rect_end[1]) - off_y) / scale)
                
                if x2 > x1 and y2 > y1:
                    self.capture_rect_raw = (x1, y1, x2, y2)
                else:
                    self.capture_rect_raw = None
            else:
                self.capture_rect_raw = None
                
            self.capture_image_area.queue_draw()
            if self.notebook.get_current_page() == 0:
                self.update_capture_histograms()

    def on_save_tiff(self, btn):
        if not self.raw_image:
            return
        
        dialog = Gtk.FileChooserDialog(
            title="Save TIFF", parent=self, action=Gtk.FileChooserAction.SAVE
        )
        dialog.add_buttons(Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL, Gtk.STOCK_SAVE, Gtk.ResponseType.OK)
        dialog.set_do_overwrite_confirmation(True)
        
        raw_paths = self.raw_image.filepaths
        if raw_paths:
            base = os.path.splitext(os.path.basename(raw_paths[0]))[0]
            if len(raw_paths) == 4:
                base += "_merged"
            default_filename = f"{base}.tiff"
        else:
            default_filename = "capture.tiff"
        dialog.set_current_name(default_filename)
        
        if dialog.run() == Gtk.ResponseType.OK:
            filepath = dialog.get_filename()
            dialog.destroy()
            
            self.lbl_status.set_text("Saving TIFF image...")
            self.btn_cap_img.set_sensitive(False)
            self.btn_cap_base.set_sensitive(False)
            self.btn_save_tiff.set_sensitive(False)
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
                        if getattr(self.profile, 'icc_profile_bytes', None):
                            temp_profile.icc_profile_bytes = self.profile.icc_profile_bytes
                            
                        color_conversion.convert_raw_to_tiff(
                            img=self.raw_image, profile=temp_profile, output_path=filepath,
                            exposure_comp=self.gain, half=False, film_base_rgb=self.film_base_rgb
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

    def on_save_success(self, filepath, duration):
        self.spinner.stop()
        self.btn_cap_img.set_sensitive(self.is_connected)
        self.btn_cap_base.set_sensitive(self.is_connected)
        self.btn_save_tiff.set_sensitive(True)
        self.lbl_status.set_text(f"Status: Saved {os.path.basename(filepath)} in {duration:.2f}s")
        
        dialog = Gtk.MessageDialog(
            transient_for=self,
            flags=0,
            message_type=Gtk.MessageType.INFO,
            buttons=Gtk.ButtonsType.OK,
            text="Save Successful"
        )
        dialog.format_secondary_text(f"Image saved successfully to:\n{filepath}\n\nTime taken: {duration:.2f}s")
        dialog.run()
        dialog.destroy()

    def update_ui_failure(self, error_msg):
        self.spinner.stop()
        self.btn_cap_img.set_sensitive(self.is_connected)
        self.btn_cap_base.set_sensitive(self.is_connected)
        self.mode_combo.set_sensitive(True)
        self.shutter_combo.set_sensitive(not self.ae_checkbox.get_active())
        self.ae_checkbox.set_sensitive(True)
        self.btn_save_tiff.set_sensitive(self.raw_image is not None)
        self.lbl_status.set_text(f"Status: Capture failed. See terminal.")

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
