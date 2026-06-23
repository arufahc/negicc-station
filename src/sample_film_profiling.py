#!/usr/bin/env python3
"""
GTK3-based GUI application for Negative Film Profiling.
Supports loading a crosstalk calibration profile, capturing an IT8 target
and film base, and displaying the crosstalk-corrected previews and histograms.
"""

import os
import sys
import threading
import time
import numpy as np
import gi

# Require GTK3
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk, Gdk, GdkPixbuf, GLib
from matplotlib.figure import Figure
from matplotlib.backends.backend_gtk3agg import FigureCanvasGTK3Agg as FigureCanvas

# Ensure the project src directory is in path
project_dir = os.path.dirname(os.path.abspath(__file__))
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)

import negicc_station
import auto_exposure
import crosstalk_calibration


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
    if H - 2 * h_border > 2 and W - 2 * w_border > 2:
        cropped = arr[h_border:H-h_border, w_border:W-w_border, :]
    else:
        cropped = arr

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

    ax.figure.canvas.draw_idle()


class FilmProfilingAppWindow(Gtk.Window):
    def __init__(self):
        super().__init__(title="Negative Film Profiling Station")
        self.set_default_size(1280, 800)
        self.connect("destroy", self.on_destroy)

        # Apply custom dark style
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
            .tool-btn {
                background-color: #2c2c2c;
                color: #e0e0e0;
                border: 1px solid #444444;
                border-radius: 4px;
                padding: 6px 12px;
                font-weight: bold;
            }
            .tool-btn:hover {
                background-color: #3d3d3d;
            }
            .tool-btn:disabled {
                color: #666666;
                background-color: #1e1e1e;
                border-color: #2a2a2a;
            }
            .capture-btn:hover {
                background-image: linear-gradient(to bottom, #30bc5a, #2ea44f);
            }
            .capture-btn:disabled {
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

        # App state variables
        self.calib = None
        self.camera_session = None
        self.is_connected = False
        self.is_connecting = False

        # IT8 mask parameters
        self.it8_mask_active = False
        self.it8_scale = 1.0
        self.it8_dx = 0.0
        self.it8_dy = 0.0

        # Target (IT8) tab state
        self.arr_raw_target = None
        self.arr_cc_target = None
        self.current_pixbuf_target = None
        self.scaled_pixbuf_target = None
        self.normalized_selection_target = None
        self.is_dragging_target = False
        self.selection_start_target = None
        self.selection_end_target = None
        self.img_x_offset_target = 0
        self.img_y_offset_target = 0

        # Film Base tab state
        self.arr_raw_base = None
        self.arr_cc_base = None
        self.current_pixbuf_base = None
        self.scaled_pixbuf_base = None
        self.normalized_selection_base = None
        self.is_dragging_base = False
        self.selection_start_base = None
        self.selection_end_base = None
        self.img_x_offset_base = 0
        self.img_y_offset_base = 0

        # Base Layout
        main_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=0)
        self.add(main_box)

        # =====================================================================
        # LEFT SIDEBAR
        # =====================================================================
        sidebar_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12)
        sidebar_box.get_style_context().add_class("sidebar")
        sidebar_box.set_size_request(300, -1)
        main_box.pack_start(sidebar_box, False, False, 0)

        # App Title
        title_lbl = Gtk.Label()
        title_lbl.set_markup("<span size='large' weight='bold'>Film Profiling Tool</span>")
        title_lbl.set_xalign(0.0)
        sidebar_box.pack_start(title_lbl, False, False, 5)

        # Connection Indicator
        self.camera_status_label = Gtk.Label()
        self.camera_status_label.set_markup("<span><span foreground='#e6a23c'>●</span> Camera: Connecting...</span>")
        self.camera_status_label.set_xalign(0.0)
        sidebar_box.pack_start(self.camera_status_label, False, False, 5)

        # Profile Load Panel
        profile_frame = Gtk.Frame(label="Crosstalk Calibration")
        profile_vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        profile_vbox.set_border_width(8)
        profile_frame.add(profile_vbox)
        sidebar_box.pack_start(profile_frame, False, False, 5)

        self.load_profile_btn = Gtk.Button(label="LOAD CROSSTALK PROFILE")
        self.load_profile_btn.connect("clicked", self.on_load_profile_clicked)
        profile_vbox.pack_start(self.load_profile_btn, False, False, 5)

        self.lbl_profile_status = Gtk.Label(label="No crosstalk profile loaded.")
        self.lbl_profile_status.set_xalign(0.0)
        self.lbl_profile_status.get_style_context().add_class("meta-label")
        profile_vbox.pack_start(self.lbl_profile_status, False, False, 5)

        # Capture Settings Frame
        settings_frame = Gtk.Frame(label="Capture Settings")
        settings_vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        settings_vbox.set_border_width(8)
        settings_frame.add(settings_vbox)
        sidebar_box.pack_start(settings_frame, False, False, 5)

        # ISO selector
        iso_hbox = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=10)
        iso_lbl = Gtk.Label(label="ISO:")
        iso_lbl.set_xalign(0.0)
        iso_hbox.pack_start(iso_lbl, False, False, 0)
        self.iso_combo = Gtk.ComboBoxText()
        for iso_val in ["100", "200", "400", "800", "1600"]:
            self.iso_combo.append(iso_val, iso_val)
        self.iso_combo.set_active(0)
        iso_hbox.pack_end(self.iso_combo, True, True, 0)
        settings_vbox.pack_start(iso_hbox, False, False, 2)

        # Shutter selector
        shutter_lbl = Gtk.Label(label="Shutter Speed (if manual):")
        shutter_lbl.set_xalign(0.0)
        settings_vbox.pack_start(shutter_lbl, False, False, 0)

        self.shutter_combo = Gtk.ComboBoxText()
        for s in auto_exposure.SHUTTER_SPEEDS:
            self.shutter_combo.append(s, s)
        self.shutter_combo.set_active(auto_exposure.SHUTTER_SPEEDS.index("1/8s"))
        settings_vbox.pack_start(self.shutter_combo, False, False, 2)

        # AE Checkbox
        self.ae_checkbox = Gtk.CheckButton(label="Auto Exposure")
        self.ae_checkbox.set_active(False)
        self.ae_checkbox.connect("toggled", self.on_ae_toggled)
        settings_vbox.pack_start(self.ae_checkbox, False, False, 5)

        # Actions Panel
        self.capture_it8_btn = Gtk.Button(label="CAPTURE IT8 TARGET")
        self.capture_it8_btn.get_style_context().add_class("capture-btn")
        self.capture_it8_btn.set_sensitive(False)
        self.capture_it8_btn.connect("clicked", self.on_capture_clicked, True)
        sidebar_box.pack_start(self.capture_it8_btn, False, False, 5)

        self.capture_base_btn = Gtk.Button(label="CAPTURE FILM BASE")
        self.capture_base_btn.get_style_context().add_class("capture-btn")
        self.capture_base_btn.set_sensitive(False)
        self.capture_base_btn.connect("clicked", self.on_capture_clicked, False)
        sidebar_box.pack_start(self.capture_base_btn, False, False, 5)

        # AE progress frame
        ae_frame = Gtk.Frame(label="Auto-Exposure Progress")
        ae_scroll = Gtk.ScrolledWindow()
        ae_scroll.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        ae_scroll.set_min_content_height(160)
        self.ae_steps_listbox = Gtk.ListBox()
        self.ae_steps_listbox.set_selection_mode(Gtk.SelectionMode.NONE)
        ae_scroll.add(self.ae_steps_listbox)
        ae_frame.add(ae_scroll)
        sidebar_box.pack_start(ae_frame, True, True, 5)

        # Status area
        status_hbox = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        self.spinner = Gtk.Spinner()
        status_hbox.pack_start(self.spinner, False, False, 0)
        self.status_lbl = Gtk.Label(label="Status: Waiting for profile...")
        self.status_lbl.set_xalign(0.0)
        status_hbox.pack_start(self.status_lbl, True, True, 0)
        sidebar_box.pack_start(status_hbox, False, False, 5)

        # =====================================================================
        # CENTER NOTEBOOK
        # =====================================================================
        self.notebook = Gtk.Notebook()
        main_box.pack_start(self.notebook, True, True, 0)

        # Page 1: Target (IT8) Tab
        self.target_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=5)
        self.target_box.get_style_context().add_class("preview-container")

        # Target Toolbar
        target_tb = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        self.target_box.pack_start(target_tb, False, False, 5)

        self.btn_rotate_target = Gtk.Button(label="Rotate 90°")
        self.btn_rotate_target.get_style_context().add_class("tool-btn")
        self.btn_rotate_target.set_sensitive(False)
        self.btn_rotate_target.connect("clicked", lambda w: self.rotate_active_tab())
        target_tb.pack_start(self.btn_rotate_target, False, False, 0)

        self.btn_hflip_target = Gtk.Button(label="H-Flip")
        self.btn_hflip_target.get_style_context().add_class("tool-btn")
        self.btn_hflip_target.set_sensitive(False)
        self.btn_hflip_target.connect("clicked", lambda w: self.hflip_active_tab())
        target_tb.pack_start(self.btn_hflip_target, False, False, 0)

        self.btn_vflip_target = Gtk.Button(label="V-Flip")
        self.btn_vflip_target.get_style_context().add_class("tool-btn")
        self.btn_vflip_target.set_sensitive(False)
        self.btn_vflip_target.connect("clicked", lambda w: self.vflip_active_tab())
        target_tb.pack_start(self.btn_vflip_target, False, False, 0)

        self.btn_crop_target = Gtk.Button(label="Crop to Selection")
        self.btn_crop_target.get_style_context().add_class("tool-btn")
        self.btn_crop_target.set_sensitive(False)
        self.btn_crop_target.connect("clicked", lambda w: self.crop_active_tab())
        target_tb.pack_start(self.btn_crop_target, False, False, 0)

        self.btn_layer_it8 = Gtk.Button(label="Layer IT8 Mask")
        self.btn_layer_it8.get_style_context().add_class("tool-btn")
        self.btn_layer_it8.set_sensitive(False)
        self.btn_layer_it8.connect("clicked", self.on_layer_it8_clicked)
        target_tb.pack_start(self.btn_layer_it8, False, False, 0)

        self.btn_read_it8 = Gtk.Button(label="Read Mask Values")
        self.btn_read_it8.get_style_context().add_class("tool-btn")
        self.btn_read_it8.set_sensitive(False)
        self.btn_read_it8.connect("clicked", lambda w: self.read_it8_values())
        target_tb.pack_start(self.btn_read_it8, False, False, 0)

        self.target_stack = Gtk.Stack()
        self.target_stack.set_transition_type(Gtk.StackTransitionType.CROSSFADE)
        self.target_stack.set_transition_duration(150)
        self.target_box.pack_start(self.target_stack, True, True, 0)

        target_placeholder = Gtk.Label()
        target_placeholder.set_markup(
            "<span size='large' foreground='#777777'>No IT8 Image Captured\n\n"
            "Please load a crosstalk calibration profile,\nthen click CAPTURE IT8.</span>"
        )
        target_placeholder.set_justify(Gtk.Justification.CENTER)
        self.target_stack.add_named(target_placeholder, "placeholder")

        self.image_view_target = Gtk.DrawingArea()
        self.image_view_target.set_can_focus(True)
        self.image_view_target.add_events(
            Gdk.EventMask.BUTTON_PRESS_MASK |
            Gdk.EventMask.BUTTON_RELEASE_MASK |
            Gdk.EventMask.POINTER_MOTION_MASK |
            Gdk.EventMask.BUTTON_MOTION_MASK
        )
        self.image_view_target.connect("draw", self.on_draw_image_view_target)
        self.image_view_target.connect("button-press-event", self.on_image_button_press_target)
        self.image_view_target.connect("button-release-event", self.on_image_button_release_target)
        self.image_view_target.connect("motion-notify-event", self.on_image_motion_notify_target)
        self.target_stack.add_named(self.image_view_target, "preview")
        self.target_stack.set_visible_child_name("placeholder")

        # Store model for IT8 patch values: Patch (str), R (int), G (int), B (int), R_std (float), G_std (float), B_std (float)
        self.it8_store = Gtk.ListStore(str, int, int, int, float, float, float)
        self.it8_treeview = Gtk.TreeView(model=self.it8_store)
        
        # Add columns to TreeView
        cols = [
            ("Patch", 0, False),
            ("R (Linear Avg)", 1, False),
            ("G (Linear Avg)", 2, False),
            ("B (Linear Avg)", 3, False),
            ("R (Std Dev)", 4, True),
            ("G (Std Dev)", 5, True),
            ("B (Std Dev)", 6, True)
        ]
        for col_title, col_idx, is_float in cols:
            renderer = Gtk.CellRendererText()
            if col_idx > 0:
                renderer.set_property("xalign", 1.0)
            col = Gtk.TreeViewColumn(col_title, renderer)
            if is_float:
                col.set_cell_data_func(renderer, lambda col, cell, model, iter, idx=col_idx: cell.set_property("text", f"{model.get_value(iter, idx):.2f}"))
            else:
                col.add_attribute(renderer, "text", col_idx)
            if col_idx > 0:
                col.set_alignment(1.0)
            col.set_sort_column_id(col_idx)
            self.it8_treeview.append_column(col)

        self.it8_scroll = Gtk.ScrolledWindow()
        self.it8_scroll.set_policy(Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.AUTOMATIC)
        self.it8_scroll.set_min_content_height(180)
        self.it8_scroll.add(self.it8_treeview)
        
        self.it8_frame = Gtk.Frame(label="Read IT8 Patch Values")
        self.it8_frame.add(self.it8_scroll)
        self.target_box.pack_start(self.it8_frame, False, False, 5)

        self.lbl_target_tab = Gtk.Label(label="Target (IT8)")
        self.lbl_target_tab.set_use_markup(True)
        self.notebook.append_page(self.target_box, self.lbl_target_tab)

        # Page 2: Film Base Tab
        self.base_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=5)
        self.base_box.get_style_context().add_class("preview-container")

        # Base Toolbar
        base_tb = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        self.base_box.pack_start(base_tb, False, False, 5)

        self.btn_rotate_base = Gtk.Button(label="Rotate 90°")
        self.btn_rotate_base.get_style_context().add_class("tool-btn")
        self.btn_rotate_base.set_sensitive(False)
        self.btn_rotate_base.connect("clicked", lambda w: self.rotate_active_tab())
        base_tb.pack_start(self.btn_rotate_base, False, False, 0)

        self.btn_hflip_base = Gtk.Button(label="H-Flip")
        self.btn_hflip_base.get_style_context().add_class("tool-btn")
        self.btn_hflip_base.set_sensitive(False)
        self.btn_hflip_base.connect("clicked", lambda w: self.hflip_active_tab())
        base_tb.pack_start(self.btn_hflip_base, False, False, 0)

        self.btn_vflip_base = Gtk.Button(label="V-Flip")
        self.btn_vflip_base.get_style_context().add_class("tool-btn")
        self.btn_vflip_base.set_sensitive(False)
        self.btn_vflip_base.connect("clicked", lambda w: self.vflip_active_tab())
        base_tb.pack_start(self.btn_vflip_base, False, False, 0)

        self.btn_crop_base = Gtk.Button(label="Crop to Selection")
        self.btn_crop_base.get_style_context().add_class("tool-btn")
        self.btn_crop_base.set_sensitive(False)
        self.btn_crop_base.connect("clicked", lambda w: self.crop_active_tab())
        base_tb.pack_start(self.btn_crop_base, False, False, 0)

        self.base_stack = Gtk.Stack()
        self.base_stack.set_transition_type(Gtk.StackTransitionType.CROSSFADE)
        self.base_stack.set_transition_duration(150)
        self.base_box.pack_start(self.base_stack, True, True, 0)

        base_placeholder = Gtk.Label()
        base_placeholder.set_markup(
            "<span size='large' foreground='#777777'>No Film Base Captured\n\n"
            "Please load a crosstalk calibration profile,\nthen click CAPTURE FILM BASE.</span>"
        )
        base_placeholder.set_justify(Gtk.Justification.CENTER)
        self.base_stack.add_named(base_placeholder, "placeholder")

        self.image_view_base = Gtk.DrawingArea()
        self.image_view_base.set_can_focus(True)
        self.image_view_base.add_events(
            Gdk.EventMask.BUTTON_PRESS_MASK |
            Gdk.EventMask.BUTTON_RELEASE_MASK |
            Gdk.EventMask.POINTER_MOTION_MASK |
            Gdk.EventMask.BUTTON_MOTION_MASK
        )
        self.image_view_base.connect("draw", self.on_draw_image_view_base)
        self.image_view_base.connect("button-press-event", self.on_image_button_press_base)
        self.image_view_base.connect("button-release-event", self.on_image_button_release_base)
        self.image_view_base.connect("motion-notify-event", self.on_image_motion_notify_base)
        self.base_stack.add_named(self.image_view_base, "preview")
        self.base_stack.set_visible_child_name("placeholder")

        self.notebook.append_page(self.base_box, Gtk.Label(label="Film Base"))

        # =====================================================================
        # RIGHT SIDEBAR (Histograms)
        # =====================================================================
        right_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        right_box.get_style_context().add_class("right-sidebar")
        right_box.set_size_request(320, -1)
        main_box.pack_start(right_box, False, False, 0)

        self.right_stack = Gtk.Stack()
        self.right_stack.set_transition_type(Gtk.StackTransitionType.CROSSFADE)
        self.right_stack.set_transition_duration(150)
        right_box.pack_start(self.right_stack, True, True, 0)

        # Right placeholder
        right_placeholder = Gtk.Label()
        right_placeholder.set_markup(
            "<span size='medium' foreground='#666666'>Capture and display images\nto see histograms.</span>"
        )
        right_placeholder.set_justify(Gtk.Justification.CENTER)
        self.right_stack.add_named(right_placeholder, "placeholder")

        # Results VBox
        self.results_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        self.right_stack.add_named(self.results_box, "results")

        self.selection_status_label = Gtk.Label()
        self.selection_status_label.set_use_markup(True)
        self.selection_status_label.set_markup("<b>Selection:</b> Full Image")
        self.selection_status_label.set_xalign(0.0)
        self.selection_status_label.get_style_context().add_class("meta-label")
        self.results_box.pack_start(self.selection_status_label, False, False, 0)

        # RAW Histogram
        raw_hist_vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        self.results_box.pack_start(raw_hist_vbox, True, True, 0)

        raw_title = Gtk.Label()
        raw_title.set_markup("<b>Linearized RAW (Uncorrected)</b>")
        raw_title.set_xalign(0.0)
        raw_hist_vbox.pack_start(raw_title, False, False, 0)

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
        raw_hist_vbox.pack_start(self.raw_canvas, True, True, 0)

        # Crosstalk Corrected Histogram
        cc_hist_vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        self.results_box.pack_start(cc_hist_vbox, True, True, 0)

        cc_title = Gtk.Label()
        cc_title.set_markup("<b>Crosstalk Corrected</b>")
        cc_title.set_xalign(0.0)
        cc_hist_vbox.pack_start(cc_title, False, False, 0)

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
        cc_hist_vbox.pack_start(self.cc_canvas, True, True, 0)

        self.show_all()

        # Connect signals after widgets are fully constructed
        self.notebook.connect("switch-page", self.on_notebook_page_changed)
        self.connect("size-allocate", self.on_window_resized)
        self.connect("key-press-event", self.on_key_press)

        # Camera polling initialization
        self.connect_camera()
        GLib.timeout_add_seconds(2, self.poll_camera_connection)

    # =====================================================================
    # CAMERA CONNECTION LOGIC
    # =====================================================================
    def poll_camera_connection(self):
        if self.is_connected:
            if not negicc_station.is_camera_connected():
                self.is_connected = False
                self.update_connection_ui(False, "Camera unplugged.")
                if self.camera_session:
                    try:
                        self.camera_session.close()
                    except Exception:
                        pass
                    self.camera_session = None
        elif not self.is_connecting:
            if negicc_station.is_camera_connected():
                self.connect_camera()
        return True

    def connect_camera(self):
        if self.is_connecting or self.is_connected:
            return
        self.is_connecting = True
        self.set_controls_sensitive(False)
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
            self.set_controls_sensitive(True)
            self.status_lbl.set_text("Status: Camera connected.")
        else:
            self.camera_status_label.set_markup("<span foreground='#ff4444'>●</span> <b>Camera: Disconnected</b>")
            self.set_controls_sensitive(False)
            if error_msg:
                self.status_lbl.set_text(f"Status: Connection failed ({error_msg})")
            else:
                self.status_lbl.set_text("Status: Camera disconnected.")

    def set_controls_sensitive(self, sensitive):
        self.load_profile_btn.set_sensitive(sensitive)
        self.shutter_combo.set_sensitive(sensitive and not self.ae_checkbox.get_active())
        self.ae_checkbox.set_sensitive(sensitive)

        has_profile = self.calib is not None
        self.capture_it8_btn.set_sensitive(sensitive and has_profile)
        self.capture_base_btn.set_sensitive(sensitive and has_profile)
        self.update_toolbar_sensitivities()

    def on_ae_toggled(self, button):
        self.shutter_combo.set_sensitive(not button.get_active())

    # =====================================================================
    # LOAD PROFILE LOGIC
    # =====================================================================
    def on_load_profile_clicked(self, widget):
        dialog = Gtk.FileChooserDialog(
            title="Load Crosstalk Calibration Profile",
            parent=self,
            action=Gtk.FileChooserAction.OPEN
        )
        dialog.add_buttons(
            Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL,
            Gtk.STOCK_OPEN, Gtk.ResponseType.OK
        )
        
        filter_json = Gtk.FileFilter()
        filter_json.set_name("JSON files")
        filter_json.add_mime_type("application/json")
        filter_json.add_pattern("*.json")
        dialog.add_filter(filter_json)

        response = dialog.run()
        if response == Gtk.ResponseType.OK:
            filepath = dialog.get_filename()
            try:
                self.calib = crosstalk_calibration.CrosstalkCalibration.load(filepath)
                self.lbl_profile_status.set_text(
                    f"Profile loaded successfully.\nCamera: {self.calib.camera_model}"
                )
                self.status_lbl.set_text(f"Loaded: {os.path.basename(filepath)}")
                # Re-check control sensitivities
                self.set_controls_sensitive(self.is_connected)
            except Exception as e:
                self.status_lbl.set_text(f"Error loading profile: {str(e)}")

        dialog.destroy()

    # =====================================================================
    # MOUSE EVENT HANDLERS (SEPARATE FOR TARGET AND FILM BASE)
    # =====================================================================
    def get_active_crop(self, arr_raw, arr_cc, selection):
        if selection is not None:
            nx1, ny1, nx2, ny2 = selection
            h, w, _ = arr_raw.shape
            x1, x2 = int(nx1 * w), int(nx2 * w)
            y1, y2 = int(ny1 * h), int(ny2 * h)
            if x2 > x1 and y2 > y1:
                return arr_raw[y1:y2, x1:x2], arr_cc[y1:y2, x1:x2]
        return arr_raw, arr_cc

    def update_histograms(self):
        if not hasattr(self, 'right_stack') or self.right_stack is None:
            return
        page_num = self.notebook.get_current_page()
        if page_num == 0:
            arr_raw = self.arr_raw_target
            arr_cc = self.arr_cc_target
            selection = self.normalized_selection_target
            scaled_pixbuf = self.scaled_pixbuf_target
        else:
            arr_raw = self.arr_raw_base
            arr_cc = self.arr_cc_base
            selection = self.normalized_selection_base
            scaled_pixbuf = self.scaled_pixbuf_base

        if arr_raw is None or arr_cc is None:
            self.right_stack.set_visible_child_name("placeholder")
            return

        self.right_stack.set_visible_child_name("results")

        if selection is not None and scaled_pixbuf:
            img_w = scaled_pixbuf.get_width()
            img_h = scaled_pixbuf.get_height()
            nx1, ny1, nx2, ny2 = selection
            self.selection_status_label.set_markup(
                f"<b>Selection:</b> Region ({int(nx1*img_w)}, {int(ny1*img_h)}) to "
                f"({int(nx2*img_w)}, {int(ny2*img_h)})"
            )
        else:
            self.selection_status_label.set_markup("<b>Selection:</b> Full Image")

        def run():
            try:
                arr_raw_crop, arr_cc_crop = self.get_active_crop(arr_raw, arr_cc, selection)
                hists_raw, p2_raw, p98_raw, dr_raw, mean_raw = compute_hist_and_percentiles(arr_raw_crop)
                hists_cc, p2_cc, p98_cc, dr_cc, mean_cc = compute_hist_and_percentiles(arr_cc_crop)

                GLib.idle_add(self.draw_hists_main_thread, hists_raw, p2_raw, p98_raw, dr_raw, mean_raw,
                              hists_cc, p2_cc, p98_cc, dr_cc, mean_cc)
            except Exception as e:
                print(f"Error drawing histograms: {e}")

        t = threading.Thread(target=run)
        t.daemon = True
        t.start()

    def draw_hists_main_thread(self, hists_raw, p2_raw, p98_raw, dr_raw, mean_raw,
                               hists_cc, p2_cc, p98_cc, dr_cc, mean_cc):
        draw_matplotlib_histogram(self.raw_ax, hists_raw, p2_raw, p98_raw, dr_metrics=dr_raw, mean_metrics=mean_raw, show_overexposure=True)
        self.raw_canvas.draw_idle()

        draw_matplotlib_histogram(self.cc_ax, hists_cc, p2_cc, p98_cc, dr_metrics=dr_cc, mean_metrics=mean_cc, show_overexposure=True)
        self.cc_canvas.draw_idle()

    def on_notebook_page_changed(self, notebook, page, page_num):
        self.update_histograms()

    # Target event wrappers
    def on_draw_image_view_target(self, widget, cr):
        return self.draw_image_preview(cr, self.scaled_pixbuf_target, self.is_dragging_target, 
                                       self.selection_start_target, self.selection_end_target,
                                       self.normalized_selection_target, 0)

    def on_image_button_press_target(self, widget, event):
        if not self.scaled_pixbuf_target:
            return False
        if event.button == 1:
            img_w = self.scaled_pixbuf_target.get_width()
            img_h = self.scaled_pixbuf_target.get_height()
            x = max(self.img_x_offset_target, min(event.x, self.img_x_offset_target + img_w))
            y = max(self.img_y_offset_target, min(event.y, self.img_y_offset_target + img_h))
            self.is_dragging_target = True
            self.selection_start_target = (x, y)
            self.selection_end_target = (x, y)
            widget.queue_draw()
            return True
        return False

    def on_image_motion_notify_target(self, widget, event):
        if self.is_dragging_target and self.selection_start_target:
            img_w = self.scaled_pixbuf_target.get_width()
            img_h = self.scaled_pixbuf_target.get_height()
            x = max(self.img_x_offset_target, min(event.x, self.img_x_offset_target + img_w))
            y = max(self.img_y_offset_target, min(event.y, self.img_y_offset_target + img_h))
            self.selection_end_target = (x, y)
            widget.queue_draw()
            return True
        return False

    def on_image_button_release_target(self, widget, event):
        if event.button == 1 and self.is_dragging_target:
            self.is_dragging_target = False
            if self.selection_start_target and self.selection_end_target:
                x1, y1 = self.selection_start_target
                x2, y2 = self.selection_end_target
                if abs(x2 - x1) > 5 and abs(y2 - y1) > 5:
                    img_w = self.scaled_pixbuf_target.get_width()
                    img_h = self.scaled_pixbuf_target.get_height()
                    img_x1 = max(0, min(x1, x2) - self.img_x_offset_target)
                    img_x2 = min(img_w, max(x1, x2) - self.img_x_offset_target)
                    img_y1 = max(0, min(y1, y2) - self.img_y_offset_target)
                    img_y2 = min(img_h, max(y1, y2) - self.img_y_offset_target)

                    self.normalized_selection_target = (
                        img_x1 / img_w,
                        img_y1 / img_h,
                        img_x2 / img_w,
                        img_y2 / img_h
                    )
                else:
                    self.normalized_selection_target = None
                self.update_histograms()
                self.update_toolbar_sensitivities()
            widget.queue_draw()
            return True
        return False

    # Film base event wrappers
    def on_draw_image_view_base(self, widget, cr):
        return self.draw_image_preview(cr, self.scaled_pixbuf_base, self.is_dragging_base, 
                                       self.selection_start_base, self.selection_end_base,
                                       self.normalized_selection_base, 1)

    def on_image_button_press_base(self, widget, event):
        if not self.scaled_pixbuf_base:
            return False
        if event.button == 1:
            img_w = self.scaled_pixbuf_base.get_width()
            img_h = self.scaled_pixbuf_base.get_height()
            x = max(self.img_x_offset_base, min(event.x, self.img_x_offset_base + img_w))
            y = max(self.img_y_offset_base, min(event.y, self.img_y_offset_base + img_h))
            self.is_dragging_base = True
            self.selection_start_base = (x, y)
            self.selection_end_base = (x, y)
            widget.queue_draw()
            return True
        return False

    def on_image_motion_notify_base(self, widget, event):
        if self.is_dragging_base and self.selection_start_base:
            img_w = self.scaled_pixbuf_base.get_width()
            img_h = self.scaled_pixbuf_base.get_height()
            x = max(self.img_x_offset_base, min(event.x, self.img_x_offset_base + img_w))
            y = max(self.img_y_offset_base, min(event.y, self.img_y_offset_base + img_h))
            self.selection_end_base = (x, y)
            widget.queue_draw()
            return True
        return False

    def on_image_button_release_base(self, widget, event):
        if event.button == 1 and self.is_dragging_base:
            self.is_dragging_base = False
            if self.selection_start_base and self.selection_end_base:
                x1, y1 = self.selection_start_base
                x2, y2 = self.selection_end_base
                if abs(x2 - x1) > 5 and abs(y2 - y1) > 5:
                    img_w = self.scaled_pixbuf_base.get_width()
                    img_h = self.scaled_pixbuf_base.get_height()
                    img_x1 = max(0, min(x1, x2) - self.img_x_offset_base)
                    img_x2 = min(img_w, max(x1, x2) - self.img_x_offset_base)
                    img_y1 = max(0, min(y1, y2) - self.img_y_offset_base)
                    img_y2 = min(img_h, max(y1, y2) - self.img_y_offset_base)

                    self.normalized_selection_base = (
                        img_x1 / img_w,
                        img_y1 / img_h,
                        img_x2 / img_w,
                        img_y2 / img_h
                    )
                else:
                    self.normalized_selection_base = None
                self.update_histograms()
                self.update_toolbar_sensitivities()
            widget.queue_draw()
            return True
        return False

    def draw_image_preview(self, cr, scaled_pixbuf, is_dragging, selection_start, selection_end, normalized_selection, page):
        if not scaled_pixbuf:
            return False

        alloc = self.image_view_target.get_allocation() if page == 0 else self.image_view_base.get_allocation()
        img_w = scaled_pixbuf.get_width()
        img_h = scaled_pixbuf.get_height()

        x_offset = max(0, (alloc.width - img_w) // 2)
        y_offset = max(0, (alloc.height - img_h) // 2)

        if page == 0:
            self.img_x_offset_target = x_offset
            self.img_y_offset_target = y_offset
        else:
            self.img_x_offset_base = x_offset
            self.img_y_offset_base = y_offset

        Gdk.cairo_set_source_pixbuf(cr, scaled_pixbuf, x_offset, y_offset)
        cr.paint()

        # Draw selection border
        if is_dragging and selection_start and selection_end:
            x1, y1 = selection_start
            x2, y2 = selection_end
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

        elif normalized_selection is not None:
            nx1, ny1, nx2, ny2 = normalized_selection
            x_min = int(nx1 * img_w) + x_offset
            y_min = int(ny1 * img_h) + y_offset
            x_max = int(nx2 * img_w) + x_offset
            y_max = int(ny2 * img_h) + y_offset

            cr.set_source_rgba(0.2, 0.6, 1.0, 0.15)
            cr.rectangle(x_min, y_min, x_max - x_min, y_max - y_min)
            cr.fill_preserve()

            cr.set_source_rgba(0.2, 0.6, 1.0, 0.8)
            cr.set_line_width(1.5)
            cr.set_dash([4.0, 4.0], 0)
            cr.stroke()
            cr.set_dash([], 0)

        # Draw IT8 mask grid on Target tab if active
        if page == 0 and self.it8_mask_active:
            boxes = self.get_it8_boxes()
            cr.set_source_rgba(0.0, 1.0, 0.3, 0.85)  # vibrant green
            cr.set_line_width(1.0)
            for patch, (bx, by, bw, bh) in boxes.items():
                px = int(bx * img_w) + x_offset
                py = int(by * img_h) + y_offset
                pw = int(bw * img_w)
                ph = int(bh * img_h)
                cr.rectangle(px, py, pw, ph)
                cr.stroke()

        return True

    # =====================================================================
    # CAPTURE FLOW AND BACKGROUND OPERATIONS
    # =====================================================================
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

    def clear_ae_steps(self):
        for child in self.ae_steps_listbox.get_children():
            self.ae_steps_listbox.remove(child)

    def on_capture_clicked(self, widget, is_target):
        if not self.is_connected or self.camera_session is None:
            self.status_lbl.set_text("Status: Camera not connected.")
            return

        self.set_controls_sensitive(False)
        self.clear_ae_steps()
        self.spinner.start()

        start_speed = self.shutter_combo.get_active_text()
        iso_str = self.iso_combo.get_active_text()
        iso = int(iso_str)
        is_ae = self.ae_checkbox.get_active()

        self.status_lbl.set_text("Status: Starting capture...")

        def thread_func():
            session = self.camera_session
            try:
                optimal_speed = start_speed
                if is_ae:
                    GLib.idle_add(self.status_lbl.set_text, "Status: Running Auto-Exposure...")
                    
                    def ae_progress_callback(idx, shutter_str, dr_channels, avg_dr):
                        dr_r, dr_g, dr_b = dr_channels
                        GLib.idle_add(self.add_ae_step_to_listbox, idx, shutter_str, dr_r, dr_g, dr_b, avg_dr)

                    def ae_capture_func(idx):
                        shutter_str = auto_exposure.SHUTTER_SPEEDS[idx]
                        return auto_exposure.capture_exposure_frame(shutter_str, half=True, session=session)

                    optimal_speed, _ = auto_exposure.run_auto_exposure(
                        start_shutter_str=start_speed,
                        capture_func=ae_capture_func,
                        progress_callback=ae_progress_callback,
                        channel='ALL'
                    )

                GLib.idle_add(self.status_lbl.set_text, f"Status: Capturing final image at {optimal_speed}...")
                num, den = parse_shutter_speed(optimal_speed)
                img = session.capture(type=0, shutter_num=num, shutter_den=den)
                
                # Get uncorrected linear RAW
                arr_raw = img.to_numpy(half=True)
                img.discard()

                # Correct crosstalk using loaded profile
                GLib.idle_add(self.status_lbl.set_text, "Status: Correcting crosstalk...")
                arr_cc = self.calib.apply(arr_raw)

                # Format to 8-bit preview bytes
                arr_8bit = (arr_cc >> 8).astype(np.uint8)
                raw_bytes = arr_8bit.tobytes()
                h, w, c = arr_cc.shape

                GLib.idle_add(self.on_capture_success, is_target, raw_bytes, w, h, arr_raw, arr_cc)
            except Exception as e:
                GLib.idle_add(self.on_capture_failure, str(e))

        t = threading.Thread(target=thread_func)
        t.daemon = True
        t.start()

    def on_capture_success(self, is_target, raw_bytes, w, h, arr_raw, arr_cc):
        self.spinner.stop()

        glib_bytes = GLib.Bytes.new(raw_bytes)
        pixbuf = GdkPixbuf.Pixbuf.new_from_bytes(
            glib_bytes,
            GdkPixbuf.Colorspace.RGB,
            False,
            8,
            w,
            h,
            w * 3
        )

        if is_target:
            self.arr_raw_target = arr_raw
            self.arr_cc_target = arr_cc
            self.current_pixbuf_target = pixbuf
            self.target_stack.set_visible_child_name("preview")
            self.refresh_preview_image(0)
            self.notebook.set_current_page(0)
            # Reset IT8 mask active status and tab label/table values
            self.it8_mask_active = False
            self.btn_layer_it8.set_label("Layer IT8 Mask")
            self.lbl_target_tab.set_markup("Target (IT8)")
            self.it8_store.clear()
        else:
            self.arr_raw_base = arr_raw
            self.arr_cc_base = arr_cc
            self.current_pixbuf_base = pixbuf
            self.base_stack.set_visible_child_name("preview")
            self.refresh_preview_image(1)
            self.notebook.set_current_page(1)

        self.set_controls_sensitive(self.is_connected)
        self.status_lbl.set_text("Status: Capture successful.")
        self.update_histograms()

    def on_capture_failure(self, err_msg):
        self.spinner.stop()
        self.set_controls_sensitive(self.is_connected)
        self.status_lbl.set_text(f"Status: Capture failed ({err_msg})")

        dialog = Gtk.MessageDialog(
            transient_for=self,
            flags=0,
            message_type=Gtk.MessageType.ERROR,
            buttons=Gtk.ButtonsType.OK,
            text="Capture Error"
        )
        dialog.format_secondary_text(err_msg)
        dialog.run()
        dialog.destroy()

    # =====================================================================
    # LAYOUT AND SIZING
    # =====================================================================
    def refresh_preview_image(self, page):
        if page == 0:
            if not self.current_pixbuf_target:
                return
            alloc = self.target_stack.get_allocation()
            max_w = max(100, alloc.width - 30)
            max_h = max(100, alloc.height - 30)
            w = self.current_pixbuf_target.get_width()
            h = self.current_pixbuf_target.get_height()
            scale = min(max_w / w, max_h / h)
            new_w = max(1, int(w * scale))
            new_h = max(1, int(h * scale))
            self.scaled_pixbuf_target = self.current_pixbuf_target.scale_simple(
                new_w, new_h, GdkPixbuf.InterpType.BILINEAR
            )
            self.image_view_target.queue_draw()
        else:
            if not self.current_pixbuf_base:
                return
            alloc = self.base_stack.get_allocation()
            max_w = max(100, alloc.width - 30)
            max_h = max(100, alloc.height - 30)
            w = self.current_pixbuf_base.get_width()
            h = self.current_pixbuf_base.get_height()
            scale = min(max_w / w, max_h / h)
            new_w = max(1, int(w * scale))
            new_h = max(1, int(h * scale))
            self.scaled_pixbuf_base = self.current_pixbuf_base.scale_simple(
                new_w, new_h, GdkPixbuf.InterpType.BILINEAR
            )
            self.image_view_base.queue_draw()

    def on_window_resized(self, widget, allocation):
        if self.current_pixbuf_target:
            self.refresh_preview_image(0)
        if self.current_pixbuf_base:
            self.refresh_preview_image(1)

    def update_pixbuf_from_arr(self, page):
        if page == 0:
            arr_cc = self.arr_cc_target
            if arr_cc is None:
                self.current_pixbuf_target = None
                return
            h, w, c = arr_cc.shape
            arr_8bit = (arr_cc >> 8).astype(np.uint8)
            glib_bytes = GLib.Bytes.new(arr_8bit.tobytes())
            self.current_pixbuf_target = GdkPixbuf.Pixbuf.new_from_bytes(
                glib_bytes, GdkPixbuf.Colorspace.RGB, False, 8, w, h, w * 3
            )
        else:
            arr_cc = self.arr_cc_base
            if arr_cc is None:
                self.current_pixbuf_base = None
                return
            h, w, c = arr_cc.shape
            arr_8bit = (arr_cc >> 8).astype(np.uint8)
            glib_bytes = GLib.Bytes.new(arr_8bit.tobytes())
            self.current_pixbuf_base = GdkPixbuf.Pixbuf.new_from_bytes(
                glib_bytes, GdkPixbuf.Colorspace.RGB, False, 8, w, h, w * 3
            )

    def rotate_active_tab(self):
        page_num = self.notebook.get_current_page()
        if page_num == 0:
            if self.arr_cc_target is not None:
                self.arr_raw_target = np.rot90(self.arr_raw_target, k=-1)
                self.arr_cc_target = np.rot90(self.arr_cc_target, k=-1)
                self.normalized_selection_target = None
                self.update_pixbuf_from_arr(0)
                self.refresh_preview_image(0)
                self.update_histograms()
                self.update_toolbar_sensitivities()
        else:
            if self.arr_cc_base is not None:
                self.arr_raw_base = np.rot90(self.arr_raw_base, k=-1)
                self.arr_cc_base = np.rot90(self.arr_cc_base, k=-1)
                self.normalized_selection_base = None
                self.update_pixbuf_from_arr(1)
                self.refresh_preview_image(1)
                self.update_histograms()
                self.update_toolbar_sensitivities()

    def hflip_active_tab(self):
        page_num = self.notebook.get_current_page()
        if page_num == 0:
            if self.arr_cc_target is not None:
                self.arr_raw_target = np.fliplr(self.arr_raw_target)
                self.arr_cc_target = np.fliplr(self.arr_cc_target)
                self.normalized_selection_target = None
                self.update_pixbuf_from_arr(0)
                self.refresh_preview_image(0)
                self.update_histograms()
                self.update_toolbar_sensitivities()
        else:
            if self.arr_cc_base is not None:
                self.arr_raw_base = np.fliplr(self.arr_raw_base)
                self.arr_cc_base = np.fliplr(self.arr_cc_base)
                self.normalized_selection_base = None
                self.update_pixbuf_from_arr(1)
                self.refresh_preview_image(1)
                self.update_histograms()
                self.update_toolbar_sensitivities()

    def vflip_active_tab(self):
        page_num = self.notebook.get_current_page()
        if page_num == 0:
            if self.arr_cc_target is not None:
                self.arr_raw_target = np.flipud(self.arr_raw_target)
                self.arr_cc_target = np.flipud(self.arr_cc_target)
                self.normalized_selection_target = None
                self.update_pixbuf_from_arr(0)
                self.refresh_preview_image(0)
                self.update_histograms()
                self.update_toolbar_sensitivities()
        else:
            if self.arr_cc_base is not None:
                self.arr_raw_base = np.flipud(self.arr_raw_base)
                self.arr_cc_base = np.flipud(self.arr_cc_base)
                self.normalized_selection_base = None
                self.update_pixbuf_from_arr(1)
                self.refresh_preview_image(1)
                self.update_histograms()
                self.update_toolbar_sensitivities()

    def crop_active_tab(self):
        page_num = self.notebook.get_current_page()
        if page_num == 0:
            if self.arr_cc_target is not None and self.normalized_selection_target is not None:
                nx1, ny1, nx2, ny2 = self.normalized_selection_target
                h, w, _ = self.arr_cc_target.shape
                x1, x2 = int(nx1 * w), int(nx2 * w)
                y1, y2 = int(ny1 * h), int(ny2 * h)
                if x2 > x1 and y2 > y1:
                    self.arr_raw_target = self.arr_raw_target[y1:y2, x1:x2]
                    self.arr_cc_target = self.arr_cc_target[y1:y2, x1:x2]
                    self.normalized_selection_target = None
                    self.update_pixbuf_from_arr(0)
                    self.refresh_preview_image(0)
                    self.update_histograms()
                    self.update_toolbar_sensitivities()
        else:
            if self.arr_cc_base is not None and self.normalized_selection_base is not None:
                nx1, ny1, nx2, ny2 = self.normalized_selection_base
                h, w, _ = self.arr_cc_base.shape
                x1, x2 = int(nx1 * w), int(nx2 * w)
                y1, y2 = int(ny1 * h), int(ny2 * h)
                if x2 > x1 and y2 > y1:
                    self.arr_raw_base = self.arr_raw_base[y1:y2, x1:x2]
                    self.arr_cc_base = self.arr_cc_base[y1:y2, x1:x2]
                    self.normalized_selection_base = None
                    self.update_pixbuf_from_arr(1)
                    self.refresh_preview_image(1)
                    self.update_histograms()
                    self.update_toolbar_sensitivities()

    def update_toolbar_sensitivities(self):
        # Target tab buttons
        has_target = self.arr_cc_target is not None
        has_target_selection = self.normalized_selection_target is not None
        if hasattr(self, 'btn_rotate_target'):
            self.btn_rotate_target.set_sensitive(has_target)
        if hasattr(self, 'btn_hflip_target'):
            self.btn_hflip_target.set_sensitive(has_target)
        if hasattr(self, 'btn_vflip_target'):
            self.btn_vflip_target.set_sensitive(has_target)
        if hasattr(self, 'btn_crop_target'):
            self.btn_crop_target.set_sensitive(has_target and has_target_selection)

        # Base tab buttons
        has_base = self.arr_cc_base is not None
        has_base_selection = self.normalized_selection_base is not None
        if hasattr(self, 'btn_rotate_base'):
            self.btn_rotate_base.set_sensitive(has_base)
        if hasattr(self, 'btn_hflip_base'):
            self.btn_hflip_base.set_sensitive(has_base)
        if hasattr(self, 'btn_vflip_base'):
            self.btn_vflip_base.set_sensitive(has_base)
        if hasattr(self, 'btn_crop_base'):
            self.btn_crop_base.set_sensitive(has_base and has_base_selection)

        if hasattr(self, 'btn_layer_it8'):
            self.btn_layer_it8.set_sensitive(has_target)
        if hasattr(self, 'btn_read_it8'):
            self.btn_read_it8.set_sensitive(has_target and self.it8_mask_active)

    def get_it8_boxes(self):
        # Base dimensions and values matching ../negicc/read_it8.py layout spacing
        HBASE = 1300.0
        VBASE = 870.0
        VSTEP = 53.0
        HSTEP = 54.0
        box_size = 18.0
        a1_x = 77.0
        a1_y = 79.0
        gs0_x = 23.0
        gs0_y = 800.0

        w_box_base = box_size / HBASE
        h_box_base = box_size / VBASE

        base_boxes = {}
        base_boxes["a1"] = (a1_x / HBASE, a1_y / VBASE)
        
        def add_horizontal_boxes(row, start=2, end=23):
            for j in range(start, end):
                left_x, left_y = base_boxes[row + str(j-1)]
                base_boxes[row + str(j)] = (left_x + HSTEP / HBASE, left_y)

        add_horizontal_boxes('a')
        
        for i in range(1, 12):
            row = chr(ord('a') + i)
            last_row = chr(ord('a') + (i - 1))
            last_x, last_y = base_boxes[last_row + '1']
            base_boxes[row + '1'] = (last_x, last_y + VSTEP / VBASE)
            add_horizontal_boxes(row)

        base_boxes["gs0"] = (gs0_x / HBASE, gs0_y / VBASE)
        add_horizontal_boxes('gs', 1, 24)

        # Scale relative to center (0.5, 0.5) and translate
        scaled_boxes = {}
        for patch, (bx, by) in base_boxes.items():
            cx, cy = 0.5, 0.5
            sx = cx + (bx - cx) * self.it8_scale + self.it8_dx
            sy = cy + (by - cy) * self.it8_scale + self.it8_dy
            sw = w_box_base * self.it8_scale
            sh = h_box_base * self.it8_scale
            scaled_boxes[patch] = (sx, sy, sw, sh)

        return scaled_boxes

    def on_layer_it8_clicked(self, widget):
        self.it8_mask_active = not self.it8_mask_active
        if self.it8_mask_active:
            self.btn_layer_it8.set_label("Remove IT8 Mask")
            self.lbl_target_tab.set_markup("<b>Target (IT8) [Masked]</b>")
            self.status_lbl.set_text("Status: IT8 mask active. Use Arrow keys to move, +/- to scale.")
        else:
            self.btn_layer_it8.set_label("Layer IT8 Mask")
            self.lbl_target_tab.set_markup("Target (IT8)")
            self.status_lbl.set_text("Status: IT8 mask removed.")
            self.it8_store.clear()
        
        self.image_view_target.queue_draw()
        self.update_toolbar_sensitivities()

    def read_it8_values(self):
        if self.arr_cc_target is None:
            return
        
        boxes = self.get_it8_boxes()
        h, w, _ = self.arr_cc_target.shape
        
        self.it8_store.clear()
        
        results = []
        # Print header matching read_it8.py output format
        print("\n=== IT8 Patch Measurements (Crosstalk Corrected & Linear) ===")
        print("patch r g b r_std g_std b_std")
        for patch, (bx, by, bw, bh) in sorted(boxes.items()):
            px1, py1 = int(bx * w), int(by * h)
            px2, py2 = int((bx + bw) * w), int((by + bh) * h)
            
            px1 = max(0, min(px1, w - 1))
            px2 = max(0, min(px2, w))
            py1 = max(0, min(py1, h - 1))
            py2 = max(0, min(py2, h))
            
            patch_img = self.arr_cc_target[py1:py2, px1:px2]
            if patch_img.size > 0:
                # Use average (mean) of each cell as requested
                r = np.mean(patch_img[:, :, 0])
                g = np.mean(patch_img[:, :, 1])
                b = np.mean(patch_img[:, :, 2])
                r_std = np.std(patch_img[:, :, 0])
                g_std = np.std(patch_img[:, :, 1])
                b_std = np.std(patch_img[:, :, 2])
            else:
                r, g, b = 0.0, 0.0, 0.0
                r_std, g_std, b_std = 0.0, 0.0, 0.0
            
            r_int = int(round(r))
            g_int = int(round(g))
            b_int = int(round(b))
            self.it8_store.append([patch, r_int, g_int, b_int, float(r_std), float(g_std), float(b_std)])
            
            val_str = f"{patch} {r_int} {g_int} {b_int} {r_std:.2f} {g_std:.2f} {b_std:.2f}"
            results.append(val_str)
            print(val_str)
        print("=============================================================")
        self.lbl_target_tab.set_markup("<span foreground='#44ff44'><b>Target (IT8) [Read]</b></span>")

        # Show inside a copyable text dialog
        dialog = Gtk.Dialog(title="IT8 Patch Values", parent=self, flags=0)
        dialog.add_button(Gtk.STOCK_CLOSE, Gtk.ResponseType.CLOSE)
        dialog.set_default_size(450, 500)

        box = dialog.get_content_area()
        lbl = Gtk.Label()
        lbl.set_markup("<b>IT8 Patch Values (Crosstalk Corrected & Linear 16-bit):</b>")
        lbl.set_xalign(0.0)
        lbl.set_margin_start(10)
        lbl.set_margin_top(10)
        box.pack_start(lbl, False, False, 5)

        scroll = Gtk.ScrolledWindow()
        scroll.set_policy(Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.AUTOMATIC)
        scroll.set_margin_start(10)
        scroll.set_margin_end(10)
        scroll.set_margin_bottom(10)
        box.pack_start(scroll, True, True, 5)

        text_view = Gtk.TextView()
        text_view.set_editable(False)
        text_view.set_monospace(True)
        buffer = text_view.get_buffer()
        buffer.set_text("patch r g b r_std g_std b_std\n" + "\n".join(results))
        scroll.add(text_view)

        dialog.show_all()
        dialog.run()
        dialog.destroy()

    def on_key_press(self, widget, event):
        if not self.it8_mask_active or self.arr_cc_target is None:
            return False

        page_num = self.notebook.get_current_page()
        if page_num != 0:
            return False

        keyval = event.keyval
        step_translate = 0.002
        step_scale = 0.005

        if keyval == Gdk.KEY_Up:
            self.it8_dy -= step_translate
        elif keyval == Gdk.KEY_Down:
            self.it8_dy += step_translate
        elif keyval == Gdk.KEY_Left:
            self.it8_dx -= step_translate
        elif keyval == Gdk.KEY_Right:
            self.it8_dx += step_translate
        elif keyval in (Gdk.KEY_plus, Gdk.KEY_equal, Gdk.KEY_KP_Add):
            self.it8_scale += step_scale
        elif keyval in (Gdk.KEY_minus, Gdk.KEY_underscore, Gdk.KEY_KP_Subtract):
            self.it8_scale -= step_scale
        else:
            return False

        self.image_view_target.queue_draw()
        return True

    def on_destroy(self, widget):
        if self.camera_session:
            try:
                self.camera_session.close()
            except Exception:
                pass
        Gtk.main_quit()


if __name__ == "__main__":
    win = FilmProfilingAppWindow()
    Gtk.main()
