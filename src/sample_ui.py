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

# Require GTK3
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk, Gdk, GdkPixbuf, GLib
from matplotlib.figure import Figure
from matplotlib.backends.backend_gtk3agg import FigureCanvasGTK3Agg as FigureCanvas

import negicc_station

# Add project src directory to path to ensure auto_exposure can be loaded
src_dir = os.path.dirname(os.path.abspath(__file__))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

import auto_exposure

# The 55 standard shutter speeds supported by the Sony A7R4
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

def compute_hist_and_percentiles(arr):
    # Calculate 256-bin normalized histograms in range [0, 16384]
    bins = 256
    hist_r, _ = np.histogram(arr[:, :, 0], bins=bins, range=(0, 16384))
    hist_g, _ = np.histogram(arr[:, :, 1], bins=bins, range=(0, 16384))
    hist_b, _ = np.histogram(arr[:, :, 2], bins=bins, range=(0, 16384))
    max_val = max(hist_r.max(), hist_g.max(), hist_b.max(), 1)
    
    hist_r_norm = hist_r / max_val
    hist_g_norm = hist_g / max_val
    hist_b_norm = hist_b / max_val

    # Exclude 5% borders for percentiles and averages
    H, W, C = arr.shape
    h_border = int(H * 0.05)
    w_border = int(W * 0.05)
    cropped = arr[h_border:H-h_border, w_border:W-w_border, :]

    # Calculate 2nd and 98th percentiles
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

    # Calculate averages (means)
    mean_r = float(np.mean(cropped[:, :, 0]))
    mean_g = float(np.mean(cropped[:, :, 1]))
    mean_b = float(np.mean(cropped[:, :, 2]))
    avg_mean = (mean_r + mean_g + mean_b) / 3.0

    return (hist_r_norm, hist_g_norm, hist_b_norm), (p2_r, p2_g, p2_b), (p98_r, p98_g, p98_b), (dr_r, dr_g, dr_b, avg_dr), (mean_r, mean_g, mean_b, avg_mean)

def draw_matplotlib_histogram(ax, hists, p2, p98, dr_metrics=None, mean_metrics=None, show_overexposure=True):
    ax.clear()
    ax.grid(True, color='#2c2c2c', linestyle='--', linewidth=0.5)
    
    if hists is None:
        ax.figure.canvas.draw_idle()
        return
        
    hist_r_norm, hist_g_norm, hist_b_norm = hists
    bins_x = np.linspace(0, 16384, len(hist_r_norm))
    
    # Plot channels with area fills
    ax.plot(bins_x, hist_r_norm, color='#ff6666', alpha=0.8, linewidth=1.2)
    ax.fill_between(bins_x, 0, hist_r_norm, color='#ff6666', alpha=0.12)
    
    ax.plot(bins_x, hist_g_norm, color='#66ff66', alpha=0.8, linewidth=1.2)
    ax.fill_between(bins_x, 0, hist_g_norm, color='#66ff66', alpha=0.12)
    
    ax.plot(bins_x, hist_b_norm, color='#66aaff', alpha=0.8, linewidth=1.2)
    ax.fill_between(bins_x, 0, hist_b_norm, color='#66aaff', alpha=0.12)
    
    # Plot 80% overexposure vertical bar of 16384 (0.8 * 16384 = 13107.2)
    if show_overexposure:
        ax.axvline(13107.2, color='#e74c3c', linestyle='-', alpha=0.8, linewidth=1.5)
        # Add textual label next to the line since we don't have a legend
        ax.text(13107.2 - 200, 0.95, "Overexposure (80%)", color='#e74c3c', fontsize=7.5,
                horizontalalignment='right', verticalalignment='top', rotation=90,
                bbox=dict(boxstyle='round,pad=0.15', facecolor='#121212', alpha=0.6, edgecolor='none'))
    
    # Plot percentile indicators
    if p2 is not None and p98 is not None:
        colors = ['#ff6666', '#66ff66', '#66aaff']
        for i in range(3):
            ax.axvline(p2[i], color=colors[i], linestyle='--', alpha=0.6, linewidth=1.0)
            ax.axvline(p98[i], color=colors[i], linestyle='--', alpha=0.6, linewidth=1.0)
            
    # Plot a cross marker ('x') for the channel averages and display their values
    if mean_metrics is not None:
        mean_r, mean_g, mean_b, _ = mean_metrics
        means = [mean_r, mean_g, mean_b]
        hists_norm = [hist_r_norm, hist_g_norm, hist_b_norm]
        colors = ['#ff6666', '#66ff66', '#66aaff']
        channel_labels = ['R', 'G', 'B']

        # Sort indices by mean values to stack their labels vertically without overlap
        sorted_indices = np.argsort(means)
        for rank, idx in enumerate(sorted_indices):
            m_val = means[idx]
            h_norm = hists_norm[idx]
            # Map mean value to corresponding bin index
            bin_idx = int(round(m_val / 16384.0 * (len(h_norm) - 1)))
            bin_idx = max(0, min(len(h_norm) - 1, bin_idx))
            y_val = h_norm[bin_idx]

            # Draw cross marker ('x') at (mean_val, curve_height)
            ax.plot(m_val, y_val, marker='x', color=colors[idx], markersize=8, markeredgewidth=1.0)

            # Vertically stagger labels to prevent overlaps (stacking between 0.50 and 0.74)
            text_y = 0.5 + (rank * 0.12)

            # Draw a faint vertical leader line connecting the cross marker to the label
            ax.plot([m_val, m_val], [y_val, text_y], color=colors[idx], linestyle=':', alpha=0.5, linewidth=1.0)

            # Draw value text centered horizontally at the mean value
            ax.text(m_val, text_y, f"{channel_labels[idx]}_avg: {int(m_val)}",
                    color=colors[idx], fontsize=8, fontweight='bold',
                    horizontalalignment='center', verticalalignment='center',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='#121212', alpha=0.85, edgecolor='none'))

    ax.set_xlim(0, 16384)
    ax.set_ylim(0, 1.05)

    # Put the R, G, B and dynamic range values inside the graph using a textbox
    if p2 is not None and p98 is not None and dr_metrics is not None:
        dr_r, dr_g, dr_b, avg_dr = dr_metrics
        mean_r, mean_g, mean_b, avg_mean = mean_metrics if mean_metrics is not None else (0, 0, 0, 0)
        text_str = (
            f"R: [2%:{int(p2[0])}, 98%:{int(p98[0])}] DR:{dr_r:.1f} Mean:{mean_r:.1f}\n"
            f"G: [2%:{int(p2[1])}, 98%:{int(p98[1])}] DR:{dr_g:.1f} Mean:{mean_g:.1f}\n"
            f"B: [2%:{int(p2[2])}, 98%:{int(p98[2])}] DR:{dr_b:.1f} Mean:{mean_b:.1f}\n"
            f"Avg DR: {avg_dr:.1f} | Avg Value: {avg_mean:.1f}"
        )
        props = dict(boxstyle='round', facecolor='#1e1e1e', alpha=0.8, edgecolor='#333333')
        ax.text(0.02, 0.98, text_str, transform=ax.transAxes, fontsize=8.5, color='#ffffff',
                verticalalignment='top', bbox=props, family='monospace')

    # Force redraw of the matplotlib canvas
    ax.figure.canvas.draw_idle()

class ScanningAppWindow(Gtk.Window):
    def __init__(self):
        super().__init__(title="Sony Film Scanning Station")
        self.set_default_size(1100, 750)
        self.connect("destroy", self.on_destroy)

        # Force GTK dark theme for premium aesthetics
        settings = Gtk.Settings.get_default()
        settings.set_property("gtk-application-prefer-dark-theme", True)

        # Apply custom CSS styling for capture button and spacing
        css_provider = Gtk.CssProvider()
        css_provider.load_from_data(b"""
            .capture-btn {
                background-image: linear-gradient(to bottom, #2ea44f, #2c974b);
                color: white;
                text-shadow: 0 1px 0 rgba(0,0,0,0.2);
                font-weight: bold;
                border: 1px solid rgba(27,31,35,0.15);
                border-radius: 6px;
                padding: 10px;
            }
            .capture-btn:hover {
                background-image: linear-gradient(to bottom, #30bc5a, #2ea44f);
            }
            .capture-btn:disabled, .capture-btn:insensitive {
                background-image: none;
                background-color: #444444;
                color: #888888;
                border-color: #2c2c2c;
            }
            .sidebar {
                background-color: #1e1e1e;
                border-right: 1px solid #333333;
                padding: 15px;
            }
            .right-sidebar {
                background-color: #1e1e1e;
                border-left: 1px solid #333333;
                padding: 15px;
            }
            .preview-container {
                background-color: #121212;
                padding: 15px;
            }
            .meta-label {
                font-family: monospace;
                font-size: 11px;
                color: #b3b3b3;
            }
        """)
        Gtk.StyleContext.add_provider_for_screen(
            Gdk.Screen.get_default(),
            css_provider,
            Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION
        )

        # Layout containers
        main_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=0)
        self.add(main_box)

        # =====================================================================
        # LEFT PANEL: Controls & Metadata
        # =====================================================================
        sidebar_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=15)
        sidebar_box.get_style_context().add_class("sidebar")
        sidebar_box.set_size_request(320, -1)
        main_box.pack_start(sidebar_box, False, False, 0)

        # Header Title
        title_label = Gtk.Label()
        title_label.set_markup("<span size='large' weight='bold'>Negative Scanner</span>")
        title_label.set_xalign(0.0)
        title_label.set_yalign(0.5)
        sidebar_box.pack_start(title_label, False, False, 5)

        # Camera status indicator label
        self.camera_status_label = Gtk.Label()
        self.camera_status_label.set_markup("<span><span foreground='#e6a23c'>●</span> Camera: Connecting...</span>")
        self.camera_status_label.set_xalign(0.0)
        sidebar_box.pack_start(self.camera_status_label, False, False, 2)

        # Section: Configuration
        config_frame = Gtk.Frame(label="Capture Settings")
        config_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        config_box.set_border_width(10)
        config_frame.add(config_box)
        sidebar_box.pack_start(config_frame, False, False, 5)

        # Section: Auto-Exposure Steps
        self.ae_steps_frame = Gtk.Frame(label="Auto-Exposure Steps")
        self.ae_steps_frame.set_no_show_all(True)
        
        ae_steps_scroll = Gtk.ScrolledWindow()
        ae_steps_scroll.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        ae_steps_scroll.set_min_content_height(150)
        
        self.ae_steps_listbox = Gtk.ListBox()
        self.ae_steps_listbox.set_selection_mode(Gtk.SelectionMode.NONE)
        ae_steps_scroll.add(self.ae_steps_listbox)
        self.ae_steps_frame.add(ae_steps_scroll)
        sidebar_box.pack_start(self.ae_steps_frame, False, False, 5)

        # Capture Mode Dropdown
        mode_label = Gtk.Label(label="Capture Mode:")
        mode_label.set_xalign(0.0)
        mode_label.set_yalign(0.5)
        config_box.pack_start(mode_label, False, False, 0)

        self.mode_combo = Gtk.ComboBoxText()
        self.mode_combo.append("0", "Single Shot Capture")
        self.mode_combo.append("1", "Sony 4-Shot Pixel Shift")
        self.mode_combo.set_active(0)
        config_box.pack_start(self.mode_combo, False, False, 0)

        # Shutter Speed Dropdown
        shutter_label = Gtk.Label(label="Shutter Speed:")
        shutter_label.set_xalign(0.0)
        shutter_label.set_yalign(0.5)
        config_box.pack_start(shutter_label, False, False, 0)

        self.shutter_combo = Gtk.ComboBoxText()
        for speed in SHUTTER_SPEEDS:
            self.shutter_combo.append(speed, speed)
        # Default to 1/8s
        self.shutter_combo.set_active(SHUTTER_SPEEDS.index("1/8s"))
        config_box.pack_start(self.shutter_combo, False, False, 0)

        # Auto-Exposure Checkbox
        self.ae_checkbox = Gtk.CheckButton(label="Auto Exposure")
        self.ae_checkbox.connect("toggled", self.on_ae_toggled)
        config_box.pack_start(self.ae_checkbox, False, False, 5)

        # Action: Capture Button & Spinner
        self.capture_button = Gtk.Button(label="CAPTURE IMAGE")
        self.capture_button.get_style_context().add_class("capture-btn")
        self.capture_button.set_sensitive(False)
        self.capture_button.connect("clicked", self.on_capture_clicked)
        sidebar_box.pack_start(self.capture_button, False, False, 10)

        # Action: Save to TIFF Button
        self.btn_save_tiff = Gtk.Button(label="SAVE TO TIFF...")
        self.btn_save_tiff.get_style_context().add_class("capture-btn")
        self.btn_save_tiff.set_sensitive(False)
        self.btn_save_tiff.connect("clicked", self.on_save_tiff_clicked)
        sidebar_box.pack_start(self.btn_save_tiff, False, False, 5)

        # Spinner & Status Info
        status_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=10)
        self.spinner = Gtk.Spinner()
        status_box.pack_start(self.spinner, False, False, 0)
        self.status_label = Gtk.Label(label="Status: Idle")
        self.status_label.set_xalign(0.0)
        self.status_label.set_yalign(0.5)
        status_box.pack_start(self.status_label, True, True, 0)
        sidebar_box.pack_start(status_box, False, False, 0)

        # Section: Crosstalk Correction
        cc_frame = Gtk.Frame(label="Crosstalk Correction")
        cc_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        cc_box.set_border_width(10)
        cc_frame.add(cc_box)
        sidebar_box.pack_start(cc_frame, False, False, 5)

        self.btn_load_profile = Gtk.Button(label="Load Profile...")
        self.btn_load_profile.connect("clicked", self.on_load_profile_clicked)
        cc_box.pack_start(self.btn_load_profile, False, False, 0)

        self.lbl_profile_status = Gtk.Label(label="Profile: None")
        self.lbl_profile_status.set_xalign(0.0)
        self.lbl_profile_status.get_style_context().add_class("meta-label")
        cc_box.pack_start(self.lbl_profile_status, False, False, 0)

        self.cc_checkbox = Gtk.CheckButton(label="Apply Correction")
        self.cc_checkbox.set_active(False)
        self.cc_checkbox.set_sensitive(False)
        self.cc_handler_id = self.cc_checkbox.connect("toggled", self.on_cc_toggled)
        cc_box.pack_start(self.cc_checkbox, False, False, 5)

        # Section: Metadata Display
        meta_frame = Gtk.Frame(label="Image Metadata")
        self.meta_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        self.meta_box.set_border_width(10)
        meta_frame.add(self.meta_box)
        sidebar_box.pack_start(meta_frame, True, True, 5)

        self.iso_label = Gtk.Label(label="ISO: --")
        self.iso_label.set_xalign(0.0)
        self.iso_label.set_yalign(0.5)
        self.iso_label.get_style_context().add_class("meta-label")
        self.meta_box.pack_start(self.iso_label, False, False, 0)

        self.shutter_label = Gtk.Label(label="Shutter Speed: --")
        self.shutter_label.set_xalign(0.0)
        self.shutter_label.set_yalign(0.5)
        self.shutter_label.get_style_context().add_class("meta-label")
        self.meta_box.pack_start(self.shutter_label, False, False, 0)

        self.size_label = Gtk.Label(label="Dimensions: --")
        self.size_label.set_xalign(0.0)
        self.size_label.set_yalign(0.5)
        self.size_label.get_style_context().add_class("meta-label")
        self.meta_box.pack_start(self.size_label, False, False, 0)

        self.files_label = Gtk.Label(label="RAW Filepath(s): --")
        self.files_label.set_xalign(0.0)
        self.files_label.set_yalign(0.5)
        self.files_label.set_line_wrap(True)
        self.files_label.get_style_context().add_class("meta-label")
        self.meta_box.pack_start(self.files_label, False, False, 0)

        # Timing / Debug Labels
        self.capture_time_label = Gtk.Label(label="Capture Duration: --")
        self.capture_time_label.set_xalign(0.0)
        self.capture_time_label.set_yalign(0.5)
        self.capture_time_label.get_style_context().add_class("meta-label")
        self.meta_box.pack_start(self.capture_time_label, False, False, 0)

        self.convert_time_label = Gtk.Label(label="Conversion Duration: --")
        self.convert_time_label.set_xalign(0.0)
        self.convert_time_label.set_yalign(0.5)
        self.convert_time_label.get_style_context().add_class("meta-label")
        self.meta_box.pack_start(self.convert_time_label, False, False, 0)

        # =====================================================================
        # CENTER PANEL: Preview Canvas (using Stack)
        # =====================================================================
        self.preview_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        self.preview_box.get_style_context().add_class("preview-container")
        main_box.pack_start(self.preview_box, True, True, 0)

        self.preview_stack = Gtk.Stack()
        self.preview_stack.set_transition_type(Gtk.StackTransitionType.CROSSFADE)
        self.preview_stack.set_transition_duration(200)
        self.preview_box.pack_start(self.preview_stack, True, True, 0)

        # Initial Placeholder Label
        self.placeholder_label = Gtk.Label()
        self.placeholder_label.set_markup("<span size='large' foreground='#666666'>No Image Captured\n\nConfigure settings and click CAPTURE to display preview.</span>")
        self.placeholder_label.set_justify(Gtk.Justification.CENTER)
        self.preview_stack.add_named(self.placeholder_label, "placeholder")

        # Image DrawingArea for preview & selection (initially hidden inside Stack)
        self.image_view = Gtk.DrawingArea()
        self.image_view.set_can_focus(True)
        self.image_view.add_events(
            Gdk.EventMask.BUTTON_PRESS_MASK |
            Gdk.EventMask.BUTTON_RELEASE_MASK |
            Gdk.EventMask.POINTER_MOTION_MASK |
            Gdk.EventMask.BUTTON_MOTION_MASK
        )
        self.image_view.connect("draw", self.on_draw_image_view)
        self.image_view.connect("button-press-event", self.on_image_button_press)
        self.image_view.connect("button-release-event", self.on_image_button_release)
        self.image_view.connect("motion-notify-event", self.on_image_motion_notify)
        self.preview_stack.add_named(self.image_view, "preview")

        # Start by showing placeholder in center panel
        self.preview_stack.set_visible_child_name("placeholder")

        # =====================================================================
        # RIGHT PANEL: Histograms & Metrics (using Stack)
        # =====================================================================
        self.right_panel_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        self.right_panel_box.get_style_context().add_class("right-sidebar")
        self.right_panel_box.set_size_request(320, -1)
        main_box.pack_start(self.right_panel_box, False, False, 0)

        self.right_stack = Gtk.Stack()
        self.right_stack.set_transition_type(Gtk.StackTransitionType.CROSSFADE)
        self.right_stack.set_transition_duration(200)
        self.right_panel_box.pack_start(self.right_stack, True, True, 0)

        # Initial Right Placeholder Label
        self.right_placeholder_label = Gtk.Label()
        self.right_placeholder_label.set_markup("<span size='medium' foreground='#666666'>Capture an image or load a profile\nto see histograms.</span>")
        self.right_placeholder_label.set_justify(Gtk.Justification.CENTER)
        self.right_stack.add_named(self.right_placeholder_label, "placeholder")

        # Results metrics & Histogram drawing box (initially hidden inside Stack)
        self.results_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        self.right_stack.add_named(self.results_box, "results")

        # Selection region status label
        self.selection_status_label = Gtk.Label()
        self.selection_status_label.set_use_markup(True)
        self.selection_status_label.set_markup("<b>Selection:</b> Full Image")
        self.selection_status_label.set_xalign(0.0)
        self.selection_status_label.get_style_context().add_class("meta-label")
        self.results_box.pack_start(self.selection_status_label, False, False, 0)

        # Dynamic Range display label
        self.dr_label = Gtk.Label()
        self.dr_label.set_use_markup(True)
        self.dr_label.set_xalign(0.0)
        self.dr_label.get_style_context().add_class("meta-label")
        # self.results_box.pack_start(self.dr_label, False, False, 0)

        # VBox to hold the two histograms vertically
        self.histograms_vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12)
        self.results_box.pack_start(self.histograms_vbox, True, True, 0)

        # RAW (Uncorrected) Box
        raw_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        self.histograms_vbox.pack_start(raw_box, True, True, 0)

        raw_title = Gtk.Label()
        raw_title.set_markup("<b>Linearized RAW (Uncorrected)</b>")
        raw_title.set_xalign(0.0)
        raw_box.pack_start(raw_title, False, False, 0)

        # RAW Matplotlib Canvas Setup
        self.raw_fig = Figure(figsize=(3, 1.8), dpi=100)
        self.raw_fig.patch.set_facecolor('#1e1e1e')
        self.raw_canvas = FigureCanvas(self.raw_fig)
        self.raw_canvas.set_size_request(-1, 150)
        self.raw_ax = self.raw_fig.add_subplot(111)
        self.raw_ax.set_facecolor('#121212')
        self.raw_ax.spines['top'].set_visible(False)
        self.raw_ax.spines['right'].set_visible(False)
        self.raw_ax.spines['left'].set_color('#444444')
        self.raw_ax.spines['bottom'].set_color('#444444')
        self.raw_ax.tick_params(colors='#888888', labelsize=7)
        self.raw_ax.grid(True, color='#2c2c2c', linestyle='--', linewidth=0.5)
        self.raw_fig.tight_layout()
        raw_box.pack_start(self.raw_canvas, True, True, 0)

        self.percentiles_label_raw = Gtk.Label()
        self.percentiles_label_raw.set_use_markup(True)
        self.percentiles_label_raw.set_xalign(0.0)
        self.percentiles_label_raw.get_style_context().add_class("meta-label")
        # raw_box.pack_start(self.percentiles_label_raw, False, False, 2)

        # Crosstalk Corrected Box
        cc_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        self.histograms_vbox.pack_start(cc_box, True, True, 0)

        cc_title = Gtk.Label()
        cc_title.set_markup("<b>Crosstalk Corrected</b>")
        cc_title.set_xalign(0.0)
        cc_box.pack_start(cc_title, False, False, 0)

        # CC Matplotlib Canvas Setup
        self.cc_fig = Figure(figsize=(3, 1.8), dpi=100)
        self.cc_fig.patch.set_facecolor('#1e1e1e')
        self.cc_canvas = FigureCanvas(self.cc_fig)
        self.cc_canvas.set_size_request(-1, 150)
        self.cc_ax = self.cc_fig.add_subplot(111)
        self.cc_ax.set_facecolor('#121212')
        self.cc_ax.spines['top'].set_visible(False)
        self.cc_ax.spines['right'].set_visible(False)
        self.cc_ax.spines['left'].set_color('#444444')
        self.cc_ax.spines['bottom'].set_color('#444444')
        self.cc_ax.tick_params(colors='#888888', labelsize=7)
        self.cc_ax.grid(True, color='#2c2c2c', linestyle='--', linewidth=0.5)
        self.cc_fig.tight_layout()
        cc_box.pack_start(self.cc_canvas, True, True, 0)

        self.percentiles_label_cc = Gtk.Label()
        self.percentiles_label_cc.set_use_markup(True)
        self.percentiles_label_cc.set_xalign(0.0)
        self.percentiles_label_cc.get_style_context().add_class("meta-label")
        # cc_box.pack_start(self.percentiles_label_cc, False, False, 2)

        # Start by showing placeholder in right panel
        self.right_stack.set_visible_child_name("placeholder")

        # Initialize selection and array storage variables
        self.scaled_pixbuf = None
        self.is_dragging = False
        self.selection_start = None
        self.selection_end = None
        self.normalized_selection = None
        self.last_arr_raw = None
        self.last_arr_cc = None
        self.img_x_offset = 0
        self.img_y_offset = 0

        # Initialize histogram normalization variables
        self.hist_r_raw_norm = None
        self.hist_g_raw_norm = None
        self.hist_b_raw_norm = None
        self.p2_raw = None
        self.p98_raw = None

        self.hist_r_cc_norm = None
        self.hist_g_cc_norm = None
        self.hist_b_cc_norm = None
        self.p2_cc = None
        self.p98_cc = None

        # Keep reference to the full-size decoded pixbuf to resize on window changes
        self.current_pixbuf = None
        self.connect("size-allocate", self.on_window_resized)

        # Crosstalk Correction Profile data
        self.last_captured_image = None
        self.correction_matrix = None

        self.show_all()

        # Camera session and auto-connect
        self.camera_session = None
        self.is_connected = False
        self.is_connecting = False
        self.connect_camera()
        GLib.timeout_add_seconds(2, self.poll_camera_connection)

    def poll_camera_connection(self):
        if self.is_connected:
            # Check if camera was unplugged
            if not negicc_station.is_camera_connected():
                self.is_connected = False
                self.update_connection_ui(False, "Camera unplugged.")
        elif not self.is_connecting:
            # Check if camera was plugged in
            if negicc_station.is_camera_connected():
                self.connect_camera()
        return True

    def connect_camera(self):
        if self.is_connecting or self.is_connected:
            return
        self.is_connecting = True
        # Run connection in background thread to avoid freezing GTK UI
        self.capture_button.set_sensitive(False)
        self.camera_status_label.set_markup("<span><span foreground='#e6a23c'>●</span> Camera: Connecting...</span>")
        
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
                    GLib.idle_add(self.update_connection_ui, False, "Failed to connect to camera.")
            except Exception as e:
                self.is_connected = False
                GLib.idle_add(self.update_connection_ui, False, str(e))
                
        thread = threading.Thread(target=run)
        thread.daemon = True
        thread.start()

    def update_connection_ui(self, connected, error_msg):
        self.is_connecting = False
        if connected:
            self.camera_status_label.set_markup("<span foreground='#44ff44'>●</span> <b>Camera: Connected</b>")
            self.capture_button.set_sensitive(True)
            self.status_label.set_text("Status: Camera connected, ready.")
        else:
            self.camera_status_label.set_markup("<span foreground='#ff4444'>●</span> <b>Camera: Disconnected</b>")
            self.capture_button.set_sensitive(False)
            if error_msg:
                self.status_label.set_text(f"Status: Connection failed ({error_msg})")
            else:
                self.status_label.set_text("Status: Camera disconnected.")

    def on_capture_clicked(self, widget):
        # Read UI values on main thread (thread-safe)
        mode_id = int(self.mode_combo.get_active_id())
        shutter_str = self.shutter_combo.get_active_text()
        is_ae = self.ae_checkbox.get_active()
        cc_active = self.cc_checkbox.get_active()

        # Disable controls during capture thread run
        self.capture_button.set_sensitive(False)
        self.mode_combo.set_sensitive(False)
        self.shutter_combo.set_sensitive(False)
        self.ae_checkbox.set_sensitive(False)
        self.btn_save_tiff.set_sensitive(False)
        self.spinner.start()
        self.status_label.set_text("Status: Tethering and capturing...")

        # Run capture in background thread to keep GTK UI responsive
        capture_thread = threading.Thread(
            target=self.background_capture_and_convert,
            args=(mode_id, shutter_str, is_ae, cc_active)
        )
        capture_thread.daemon = True
        capture_thread.start()

    def background_capture_and_convert(self, mode_id, start_shutter_str, is_ae, cc_active):
        # Clean up the previous image if it exists
        if self.last_captured_image is not None:
            try:
                self.last_captured_image.discard()
            except Exception as e:
                print(f"Error discarding previous image: {e}")
            self.last_captured_image = None

        if not self.is_connected or self.camera_session is None:
            GLib.idle_add(self.update_ui_failure, "Camera is not connected.")
            return

        try:
            final_shutter_str = start_shutter_str

            if is_ae:
                # 1. Clear search steps and show frame
                GLib.idle_add(self.clear_ae_steps)
                GLib.idle_add(self.ae_steps_frame.show_all)

                # 2. Define capture callback for search
                def ae_local_capture(idx):
                    shutter_s = auto_exposure.SHUTTER_SPEEDS[idx]
                    GLib.idle_add(self.status_label.set_text, f"AE Search: Capturing {shutter_s}...")
                    num, den = parse_shutter_speed(shutter_s)
                    img = self.camera_session.capture(type=0, shutter_num=num, shutter_den=den) # Single-shot
                    arr = img.to_numpy(half=True)
                    img.discard()
                    return arr

                # 3. Define progress callback
                def ae_progress(idx, shutter_s, dr_channels, avg_dr):
                    dr_r, dr_g, dr_b = dr_channels
                    GLib.idle_add(self.add_ae_step_to_listbox, idx, shutter_s, dr_r, dr_g, dr_b, avg_dr)

                # 4. Run auto-exposure search
                GLib.idle_add(self.status_label.set_text, "AE Search: Finding optimal exposure...")
                opt_shutter, steps = auto_exposure.run_auto_exposure(
                    start_shutter_str=start_shutter_str,
                    capture_func=ae_local_capture,
                    progress_callback=ae_progress
                )

                final_shutter_str = opt_shutter
                GLib.idle_add(self.set_shutter_speed_active, opt_shutter)
            else:
                # Hide the AE steps frame if not auto-exposure
                GLib.idle_add(self.ae_steps_frame.hide)

            # Get current crosstalk matrix if active and loaded
            matrix = None
            if self.correction_matrix is not None:
                matrix = [val for row in self.correction_matrix for val in row]

            # Take the final shot
            GLib.idle_add(self.status_label.set_text, f"Status: Capturing final image at {final_shutter_str}...")
            shutter_num, shutter_den = parse_shutter_speed(final_shutter_str)

            t_cap_start = time.time()
            img = self.camera_session.capture(type=mode_id, shutter_num=shutter_num, shutter_den=shutter_den)
            self.last_captured_image = img
            t_cap_duration = time.time() - t_cap_start

            # Fetch metadata
            iso = img.iso
            shutter_sec = img.shutter_speed
            paths = img.filepaths

            # Convert to half-size numpy array for fast screen preview
            t_conv_start = time.time()
            arr_raw = img.to_numpy(half=True, crosstalk_matrix=None)
            if matrix is not None:
                matrix_3x3 = np.array(matrix).reshape(3, 3)
                arr_cc = np.clip(np.dot(arr_raw.astype(np.float32), matrix_3x3.T), 0, 65535).astype(np.uint16)
            else:
                arr_cc = arr_raw
            t_conv_duration = time.time() - t_conv_start

            self.last_arr_raw = arr_raw
            self.last_arr_cc = arr_cc

            # Crop if there is a normalized selection
            arr_raw_crop, arr_cc_crop = self.get_current_crop(arr_raw, arr_cc)

            # Calculate metrics for both
            hists_raw, p2_raw, p98_raw, dr_metrics_raw, mean_metrics_raw = compute_hist_and_percentiles(arr_raw_crop)
            hists_cc, p2_cc, p98_cc, dr_metrics_cc, mean_metrics_cc = compute_hist_and_percentiles(arr_cc_crop)

            arr_display = arr_cc if cc_active else arr_raw
            height, width, channels = arr_display.shape
            arr_8bit = (arr_display >> 8).astype(np.uint8)
            raw_bytes = arr_8bit.tobytes()

            # Schedule UI updates back onto the GTK main thread safely
            GLib.idle_add(
                self.update_ui_success_with_metrics,
                raw_bytes, width, height, iso, shutter_sec, paths,
                t_cap_duration, t_conv_duration,
                hists_raw, p2_raw, p98_raw, dr_metrics_raw, mean_metrics_raw,
                hists_cc, p2_cc, p98_cc, dr_metrics_cc, mean_metrics_cc
            )
        except Exception as e:
            GLib.idle_add(self.update_ui_failure, str(e))

    def update_ui_success(self, raw_bytes, width, height, iso, shutter_sec, paths, t_cap_duration, t_conv_duration):
        # Stop spinner and enable UI controls
        self.spinner.stop()
        self.capture_button.set_sensitive(self.is_connected)
        self.mode_combo.set_sensitive(True)
        self.shutter_combo.set_sensitive(not self.ae_checkbox.get_active())
        self.ae_checkbox.set_sensitive(True)
        self.btn_save_tiff.set_sensitive(True)
        self.status_label.set_text("Status: Success!")

        # Update metadata panel
        self.iso_label.set_text(f"ISO: {iso}")
        self.shutter_label.set_text(f"Shutter Speed: {shutter_sec:.4f}s")
        self.size_label.set_text(f"Dimensions: {width} x {height} (Half-size)")
        self.files_label.set_text(f"RAW Filepath(s):\n" + "\n".join(paths))
        self.capture_time_label.set_text(f"Capture Duration: {t_cap_duration:.3f}s")
        self.convert_time_label.set_text(f"Conversion Duration: {t_conv_duration:.3f}s")

        t_render_start = time.time()
        # Create Pixbuf from raw bytes safely using GLib.Bytes
        glib_bytes = GLib.Bytes.new(raw_bytes)
        self.current_pixbuf = GdkPixbuf.Pixbuf.new_from_bytes(
            glib_bytes,
            GdkPixbuf.Colorspace.RGB,
            False,  # Has alpha
            8,      # Bits per sample
            width,
            height,
            width * 3  # Rowstride
        )

        # Show image preview inside preview_stack
        self.preview_stack.set_visible_child_name("preview")

        # Update preview canvas image
        self.refresh_preview_image()
        t_render_duration = time.time() - t_render_start

        # Print detailed timing information to stdout
        print("\n=== Capture & Processing Timing (seconds) ===")
        print(f"  Capture & Transfer:   {t_cap_duration:.3f}s")
        print(f"  Linear Conversion:    {t_conv_duration:.3f}s")
        print(f"  UI Render & Display:  {t_render_duration:.3f}s")
        print(f"  Total pipeline:       {t_cap_duration + t_conv_duration + t_render_duration:.3f}s")
        print("=============================================")

    def update_ui_success_with_metrics(self, raw_bytes, width, height, iso, shutter_sec, paths, t_cap_duration, t_conv_duration,
                                       hists_raw, p2_raw, p98_raw, dr_metrics_raw, mean_metrics_raw,
                                       hists_cc, p2_cc, p98_cc, dr_metrics_cc, mean_metrics_cc):
        self.update_ui_success(raw_bytes, width, height, iso, shutter_sec, paths, t_cap_duration, t_conv_duration)
        self.update_metrics_and_histograms(hists_raw, p2_raw, p98_raw, dr_metrics_raw, mean_metrics_raw,
                                           hists_cc, p2_cc, p98_cc, dr_metrics_cc, mean_metrics_cc)

    def update_metrics_and_histograms(self, hists_raw, p2_raw, p98_raw, dr_metrics_raw, mean_metrics_raw,
                                      hists_cc, p2_cc, p98_cc, dr_metrics_cc, mean_metrics_cc):
        # Unpack raw metrics
        self.hist_r_raw_norm, self.hist_g_raw_norm, self.hist_b_raw_norm = hists_raw
        self.p2_raw = p2_raw
        self.p98_raw = p98_raw
        
        # Unpack cc metrics
        self.hist_r_cc_norm, self.hist_g_cc_norm, self.hist_b_cc_norm = hists_cc
        self.p2_cc = p2_cc
        self.p98_cc = p98_cc

        # Show results inside stack
        self.right_stack.set_visible_child_name("results")

        # Plot histograms using matplotlib
        draw_matplotlib_histogram(self.raw_ax, hists_raw, p2_raw, p98_raw, dr_metrics_raw, mean_metrics_raw)
        draw_matplotlib_histogram(self.cc_ax, hists_cc, p2_cc, p98_cc, dr_metrics_cc, mean_metrics_cc, show_overexposure=False)

    def update_ui_failure(self, error_msg):
        self.spinner.stop()
        self.capture_button.set_sensitive(self.is_connected)
        self.mode_combo.set_sensitive(True)
        self.shutter_combo.set_sensitive(not self.ae_checkbox.get_active())
        self.ae_checkbox.set_sensitive(True)
        self.btn_save_tiff.set_sensitive(self.last_captured_image is not None)
        self.status_label.set_text("Status: Failed!")

        # Show error dialog
        dialog = Gtk.MessageDialog(
            transient_for=self,
            flags=0,
            message_type=Gtk.MessageType.ERROR,
            buttons=Gtk.ButtonsType.OK,
            text="Capture Error"
        )
        dialog.format_secondary_text(error_msg)
        dialog.run()
        dialog.destroy()

    def on_ae_toggled(self, button):
        is_active = button.get_active()
        self.shutter_combo.set_sensitive(not is_active)

    def clear_ae_steps(self):
        for child in self.ae_steps_listbox.get_children():
            self.ae_steps_listbox.remove(child)

    def add_ae_step_to_listbox(self, idx, shutter_str, dr_r, dr_g, dr_b, avg_dr):
        row_label = Gtk.Label()
        row_label.set_markup(
            f"<span size='small' font_family='monospace'>"
            f"Step {len(self.ae_steps_listbox.get_children()) + 1}: <b>{shutter_str}</b>\n"
            f"  DR: R:{dr_r:.1f} G:{dr_g:.1f} B:{dr_b:.1f} | <b>Avg:{avg_dr:.1f}</b>"
            f"</span>"
        )
        row_label.set_xalign(0.0)
        row_label.set_padding(4, 4)

        row = Gtk.ListBoxRow()
        row.add(row_label)
        self.ae_steps_listbox.add(row)
        self.ae_steps_listbox.show_all()

    def set_shutter_speed_active(self, shutter_str):
        if shutter_str in SHUTTER_SPEEDS:
            self.shutter_combo.set_active(SHUTTER_SPEEDS.index(shutter_str))



    def refresh_preview_image(self):
        if not self.current_pixbuf:
            return

        # Calculate max size based on current preview pane size (with margins)
        alloc = self.preview_box.get_allocation()
        max_w = max(100, alloc.width - 30)
        max_h = max(100, alloc.height - 30)

        w = self.current_pixbuf.get_width()
        h = self.current_pixbuf.get_height()

        scale = min(max_w / w, max_h / h)
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))

        self.scaled_pixbuf = self.current_pixbuf.scale_simple(new_w, new_h, GdkPixbuf.InterpType.BILINEAR)
        self.image_view.queue_draw()

    def on_window_resized(self, widget, allocation):
        # Dynamically scale preview on window resizing
        if self.current_pixbuf:
            self.refresh_preview_image()

    def on_destroy(self, widget):
        if self.last_captured_image:
            try:
                self.last_captured_image.discard()
            except Exception:
                pass
        if hasattr(self, 'camera_session') and self.camera_session:
            try:
                self.camera_session.close()
            except Exception:
                pass
        Gtk.main_quit()

    def on_cc_toggled(self, button):
        if self.last_captured_image:
            self.update_preview_from_last_captured()

    def update_preview_from_last_captured(self):
        if not self.last_captured_image:
            return

        self.status_label.set_text("Status: Re-processing image...")
        self.capture_button.set_sensitive(False)
        self.btn_save_tiff.set_sensitive(False)
        self.spinner.start()

        def run():
            try:
                matrix = None
                if self.correction_matrix is not None:
                    matrix = [val for row in self.correction_matrix for val in row]

                t_conv_start = time.time()
                arr_raw = self.last_captured_image.to_numpy(half=True, crosstalk_matrix=None)
                if matrix is not None:
                    matrix_3x3 = np.array(matrix).reshape(3, 3)
                    arr_cc = np.clip(np.dot(arr_raw.astype(np.float32), matrix_3x3.T), 0, 65535).astype(np.uint16)
                else:
                    arr_cc = arr_raw
                t_conv_duration = time.time() - t_conv_start

                self.last_arr_raw = arr_raw
                self.last_arr_cc = arr_cc

                # Crop if there is a normalized selection
                arr_raw_crop, arr_cc_crop = self.get_current_crop(arr_raw, arr_cc)

                # Calculate metrics for both
                hists_raw, p2_raw, p98_raw, dr_metrics_raw, mean_metrics_raw = compute_hist_and_percentiles(arr_raw_crop)
                hists_cc, p2_cc, p98_cc, dr_metrics_cc, mean_metrics_cc = compute_hist_and_percentiles(arr_cc_crop)

                arr_display = arr_cc if self.cc_checkbox.get_active() else arr_raw
                height, width, channels = arr_display.shape
                arr_8bit = (arr_display >> 8).astype(np.uint8)
                raw_bytes = arr_8bit.tobytes()

                GLib.idle_add(
                    self.update_ui_success_with_metrics_no_cap,
                    raw_bytes, width, height, t_conv_duration,
                    hists_raw, p2_raw, p98_raw, dr_metrics_raw, mean_metrics_raw,
                    hists_cc, p2_cc, p98_cc, dr_metrics_cc, mean_metrics_cc
                )
            except Exception as e:
                GLib.idle_add(self.update_ui_failure, str(e))

        thread = threading.Thread(target=run)
        thread.daemon = True
        thread.start()

    def update_ui_success_with_metrics_no_cap(self, raw_bytes, width, height, t_conv_duration,
                                              hists_raw, p2_raw, p98_raw, dr_metrics_raw, mean_metrics_raw,
                                              hists_cc, p2_cc, p98_cc, dr_metrics_cc, mean_metrics_cc):
        # Stop spinner and enable UI controls
        self.spinner.stop()
        self.capture_button.set_sensitive(self.is_connected)
        self.mode_combo.set_sensitive(True)
        self.shutter_combo.set_sensitive(not self.ae_checkbox.get_active())
        self.ae_checkbox.set_sensitive(True)
        self.btn_save_tiff.set_sensitive(True)
        self.status_label.set_text("Status: Reprocessed successfully!")

        # Update metadata panel
        self.size_label.set_text(f"Dimensions: {width} x {height} (Half-size)")
        self.convert_time_label.set_text(f"Conversion Duration: {t_conv_duration:.3f}s")

        # Create Pixbuf
        glib_bytes = GLib.Bytes.new(raw_bytes)
        self.current_pixbuf = GdkPixbuf.Pixbuf.new_from_bytes(
            glib_bytes,
            GdkPixbuf.Colorspace.RGB,
            False,
            8,
            width,
            height,
            width * 3
        )
        # Show image preview inside preview_stack
        self.preview_stack.set_visible_child_name("preview")
        self.refresh_preview_image()

        self.update_metrics_and_histograms(hists_raw, p2_raw, p98_raw, dr_metrics_raw, mean_metrics_raw,
                                           hists_cc, p2_cc, p98_cc, dr_metrics_cc, mean_metrics_cc)

    def get_current_crop(self, arr_raw, arr_cc):
        if self.normalized_selection is not None:
            nx1, ny1, nx2, ny2 = self.normalized_selection
            h_raw, w_raw, _ = arr_raw.shape
            crop_x1 = int(nx1 * w_raw)
            crop_x2 = int(nx2 * w_raw)
            crop_y1 = int(ny1 * h_raw)
            crop_y2 = int(ny2 * h_raw)
            if crop_x2 > crop_x1 and crop_y2 > crop_y1:
                return arr_raw[crop_y1:crop_y2, crop_x1:crop_x2], arr_cc[crop_y1:crop_y2, crop_x1:crop_x2]
        return arr_raw, arr_cc

    def update_preview_histograms_only(self):
        if self.last_arr_raw is None or self.last_arr_cc is None:
            return

        def run():
            try:
                arr_raw = self.last_arr_raw
                arr_cc = self.last_arr_cc
                
                arr_raw_crop, arr_cc_crop = self.get_current_crop(arr_raw, arr_cc)

                hists_raw, p2_raw, p98_raw, dr_metrics_raw, mean_metrics_raw = compute_hist_and_percentiles(arr_raw_crop)
                hists_cc, p2_cc, p98_cc, dr_metrics_cc, mean_metrics_cc = compute_hist_and_percentiles(arr_cc_crop)

                GLib.idle_add(
                    self.update_metrics_and_histograms_only_success,
                    hists_raw, p2_raw, p98_raw, dr_metrics_raw, mean_metrics_raw,
                    hists_cc, p2_cc, p98_cc, dr_metrics_cc, mean_metrics_cc
                )
            except Exception as e:
                print(f"Error updating histograms of selection: {e}")

        thread = threading.Thread(target=run)
        thread.daemon = True
        thread.start()

    def update_metrics_and_histograms_only_success(self, hists_raw, p2_raw, p98_raw, dr_metrics_raw, mean_metrics_raw,
                                                 hists_cc, p2_cc, p98_cc, dr_metrics_cc, mean_metrics_cc):
        self.update_metrics_and_histograms(hists_raw, p2_raw, p98_raw, dr_metrics_raw, mean_metrics_raw,
                                           hists_cc, p2_cc, p98_cc, dr_metrics_cc, mean_metrics_cc)

    def on_draw_image_view(self, widget, cr):
        if not self.scaled_pixbuf:
            return False

        alloc = widget.get_allocation()
        img_w = self.scaled_pixbuf.get_width()
        img_h = self.scaled_pixbuf.get_height()

        self.img_x_offset = max(0, (alloc.width - img_w) // 2)
        self.img_y_offset = max(0, (alloc.height - img_h) // 2)

        # Draw centered image preview
        Gdk.cairo_set_source_pixbuf(cr, self.scaled_pixbuf, self.img_x_offset, self.img_y_offset)
        cr.paint()

        # Draw selection rectangle
        if self.is_dragging and self.selection_start and self.selection_end:
            x1, y1 = self.selection_start
            x2, y2 = self.selection_end
            x_min, x_max = min(x1, x2), max(x1, x2)
            y_min, y_max = min(y1, y2), max(y1, y2)

            cr.set_source_rgba(0.2, 0.6, 1.0, 0.15)
            cr.rectangle(x_min, y_min, x_max - x_min, y_max - y_min)
            cr.fill_preserve()

            cr.set_source_rgba(0.2, 0.6, 1.0, 0.8)
            cr.set_line_width(1.5)
            cr.set_dash([4.0, 4.0], 0)
            cr.stroke()
            cr.set_dash([], 0)

        elif self.normalized_selection is not None:
            nx1, ny1, nx2, ny2 = self.normalized_selection
            x_min = int(nx1 * img_w) + self.img_x_offset
            y_min = int(ny1 * img_h) + self.img_y_offset
            x_max = int(nx2 * img_w) + self.img_x_offset
            y_max = int(ny2 * img_h) + self.img_y_offset

            cr.set_source_rgba(0.2, 0.6, 1.0, 0.15)
            cr.rectangle(x_min, y_min, x_max - x_min, y_max - y_min)
            cr.fill_preserve()

            cr.set_source_rgba(0.2, 0.6, 1.0, 0.8)
            cr.set_line_width(1.5)
            cr.set_dash([4.0, 4.0], 0)
            cr.stroke()
            cr.set_dash([], 0)

        return True

    def on_image_button_press(self, widget, event):
        if not self.scaled_pixbuf:
            return False

        if event.button == 1:  # Left mouse button
            img_w = self.scaled_pixbuf.get_width()
            img_h = self.scaled_pixbuf.get_height()
            
            # Bound start coordinates within image rectangle
            x = max(self.img_x_offset, min(event.x, self.img_x_offset + img_w))
            y = max(self.img_y_offset, min(event.y, self.img_y_offset + img_h))

            self.is_dragging = True
            self.selection_start = (x, y)
            self.selection_end = (x, y)
            widget.queue_draw()
            return True
        return False

    def on_image_motion_notify(self, widget, event):
        if self.is_dragging and self.selection_start:
            img_w = self.scaled_pixbuf.get_width()
            img_h = self.scaled_pixbuf.get_height()

            # Bound current coordinates within image rectangle
            x = max(self.img_x_offset, min(event.x, self.img_x_offset + img_w))
            y = max(self.img_y_offset, min(event.y, self.img_y_offset + img_h))

            self.selection_end = (x, y)
            widget.queue_draw()
            return True
        return False

    def on_image_button_release(self, widget, event):
        if event.button == 1 and self.is_dragging:
            self.is_dragging = False
            if self.selection_start and self.selection_end:
                x1, y1 = self.selection_start
                x2, y2 = self.selection_end

                if abs(x2 - x1) > 5 and abs(y2 - y1) > 5:
                    img_w = self.scaled_pixbuf.get_width()
                    img_h = self.scaled_pixbuf.get_height()

                    img_x1 = max(0, min(x1, x2) - self.img_x_offset)
                    img_x2 = min(img_w, max(x1, x2) - self.img_x_offset)
                    img_y1 = max(0, min(y1, y2) - self.img_y_offset)
                    img_y2 = min(img_h, max(y1, y2) - self.img_y_offset)

                    self.normalized_selection = (
                        img_x1 / img_w,
                        img_y1 / img_h,
                        img_x2 / img_w,
                        img_y2 / img_h
                    )
                    self.selection_status_label.set_markup(
                        f"<b>Selection:</b> Region ({int(img_x1)}, {int(img_y1)}) to ({int(img_x2)}, {int(img_y2)})"
                    )
                else:
                    self.normalized_selection = None
                    self.selection_status_label.set_markup("<b>Selection:</b> Full Image")

                # Update histograms immediately
                self.update_preview_histograms_only()

            widget.queue_draw()
            return True
        return False

    def on_load_profile_clicked(self, button):
        dialog = Gtk.FileChooserDialog(
            title="Load Calibration Profile",
            parent=self,
            action=Gtk.FileChooserAction.OPEN
        )
        dialog.add_buttons(
            Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL,
            Gtk.STOCK_OPEN, Gtk.ResponseType.OK
        )

        # Add filter for json files
        filter_json = Gtk.FileFilter()
        filter_json.set_name("JSON files")
        filter_json.add_mime_type("application/json")
        filter_json.add_pattern("*.json")
        dialog.add_filter(filter_json)

        response = dialog.run()
        if response == Gtk.ResponseType.OK:
            filepath = dialog.get_filename()
            try:
                with open(filepath, 'r') as f:
                    profile = json.load(f)

                # Check for correction matrix
                if "crosstalk_correction_matrix" in profile:
                    self.correction_matrix = profile["crosstalk_correction_matrix"]
                    filename = os.path.basename(filepath)
                    self.lbl_profile_status.set_text(f"Profile: {filename}")

                    self.cc_checkbox.set_sensitive(True)
                    self.cc_checkbox.handler_block(self.cc_handler_id)
                    self.cc_checkbox.set_active(True)
                    self.cc_checkbox.handler_unblock(self.cc_handler_id)

                    # Update preview if we have a captured image
                    if self.last_captured_image:
                        self.update_preview_from_last_captured()
                    else:
                        self.status_label.set_text(f"Status: Profile loaded: {filename}")
                else:
                    self.status_label.set_text("Status: Invalid profile (missing matrix)")
                    self.show_error_dialog("Invalid Profile", "The loaded JSON file does not contain a 'crosstalk_correction_matrix'.")
            except Exception as e:
                self.status_label.set_text(f"Status: Error loading profile: {str(e)}")
                self.show_error_dialog("Load Error", f"Failed to parse calibration profile:\n{str(e)}")

        dialog.destroy()

    def show_error_dialog(self, title, message):
        dialog = Gtk.MessageDialog(
            transient_for=self,
            flags=0,
            message_type=Gtk.MessageType.ERROR,
            buttons=Gtk.ButtonsType.OK,
            text=title
        )
        dialog.format_secondary_text(message)
        dialog.run()
        dialog.destroy()

    def on_save_tiff_clicked(self, button):
        if not self.last_captured_image:
            return

        dialog = Gtk.FileChooserDialog(
            title="Save Image to TIFF",
            parent=self,
            action=Gtk.FileChooserAction.SAVE
        )
        dialog.add_buttons(
            Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL,
            Gtk.STOCK_SAVE, Gtk.ResponseType.OK
        )
        dialog.set_do_overwrite_confirmation(True)

        raw_paths = self.last_captured_image.filepaths
        if raw_paths:
            base = os.path.splitext(os.path.basename(raw_paths[0]))[0]
            if len(raw_paths) == 4:
                base += "_merged"
            default_filename = f"{base}.tiff"
        else:
            default_filename = "captured_image.tiff"
        dialog.set_current_name(default_filename)

        filter_tiff = Gtk.FileFilter()
        filter_tiff.set_name("TIFF images")
        filter_tiff.add_pattern("*.tiff")
        filter_tiff.add_pattern("*.tif")
        dialog.add_filter(filter_tiff)

        response = dialog.run()
        if response == Gtk.ResponseType.OK:
            filepath = dialog.get_filename()
            dialog.destroy()

            # Save in a background thread to keep UI responsive
            self.capture_button.set_sensitive(False)
            self.btn_save_tiff.set_sensitive(False)
            self.spinner.start()
            self.status_label.set_text("Status: Saving TIFF image...")

            # Get current crosstalk matrix if checkbox is checked
            matrix = None
            if self.cc_checkbox.get_active() and self.correction_matrix is not None:
                matrix = [val for row in self.correction_matrix for val in row]

            def save_thread():
                try:
                    t_start = time.time()
                    # Write TIFF using full resolution (half=False)
                    success = self.last_captured_image.write_tiff(
                        filepath,
                        half=False,
                        crosstalk_matrix=matrix
                    )
                    t_dur = time.time() - t_start
                    if success:
                        GLib.idle_add(self.on_save_tiff_success, filepath, t_dur)
                    else:
                        GLib.idle_add(self.update_ui_failure, "C++ write_tiff returned false.")
                except Exception as e:
                    GLib.idle_add(self.update_ui_failure, f"Error saving TIFF: {str(e)}")

            thread = threading.Thread(target=save_thread)
            thread.daemon = True
            thread.start()
        else:
            dialog.destroy()

    def on_save_tiff_success(self, filepath, duration):
        self.spinner.stop()
        self.capture_button.set_sensitive(self.is_connected)
        self.btn_save_tiff.set_sensitive(True)
        self.mode_combo.set_sensitive(True)
        self.shutter_combo.set_sensitive(not self.ae_checkbox.get_active())
        self.ae_checkbox.set_sensitive(True)

        filename = os.path.basename(filepath)
        self.status_label.set_text(f"Status: Saved {filename} in {duration:.2f}s")

        # Show success dialog
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

    def on_window_resized(self, widget, allocation):
        # Dynamically scale preview on window resizing
        if self.current_pixbuf:
            self.refresh_preview_image()

def main():
    # Preload the Sony CrSDK shared library from the virtual environment
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    lib_path = os.path.join(project_dir, 'venv/bin/libCr_Core.so')
    if os.path.exists(lib_path):
        import ctypes
        ctypes.CDLL(lib_path)

    # Launch GUI
    app = ScanningAppWindow()
    Gtk.main()

if __name__ == '__main__':
    main()
