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
import base64
import tempfile
import json
import film_profiling
from film_profiling import FilmProfile, download_and_parse_reference_file


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


class TargetTabState:
    def __init__(self, index, label_text):
        self.index = index
        self.label_text = label_text
        self.arr_raw = None
        self.img_obj = None
        self.arr_cc = None
        self.filepaths = None
        self.current_pixbuf = None
        self.scaled_pixbuf = None
        self.normalized_selection = None
        self.is_dragging = False
        self.selection_start = None
        self.selection_end = None
        self.img_x_offset = 0
        self.img_y_offset = 0
        self.it8_mask_active = False
        self.it8_scale = 1.0
        self.it8_dx = 0.0
        self.it8_dy = 0.0
        
        # Exposure Info
        self.iso = None
        self.shutter = None
        
        # Widgets
        self.widget_box = None
        self.image_view = None
        self.stack = None
        self.it8_store = None
        self.it8_treeview = None
        self.lbl_tab = None
        self.lbl_exposure_info = None
        
        # Toolbar buttons
        self.btn_rotate = None
        self.btn_hflip = None
        self.btn_vflip = None
        self.btn_crop = None
        self.btn_layer_it8 = None
        self.btn_read_it8 = None


class ProfileProgressDialog(Gtk.Dialog):
    def __init__(self, parent, num_targets):
        super().__init__(title="Generating Film Profile", transient_for=parent, modal=True, destroy_with_parent=True)
        self.set_default_size(400, 150)
        self.set_keep_above(True)
        
        box = self.get_content_area()
        box.set_spacing(12)
        box.set_border_width(12)
        
        self.label = Gtk.Label()
        self.label.set_markup("<b>Starting profile compilation...</b>")
        self.label.set_xalign(0.0)
        box.pack_start(self.label, False, False, 0)
        
        self.pbar = Gtk.ProgressBar()
        box.pack_start(self.pbar, False, False, 0)
        
        self.detail_label = Gtk.Label(label="Initializing...")
        self.detail_label.set_xalign(0.0)
        self.detail_label.get_style_context().add_class("meta-label")
        box.pack_start(self.detail_label, False, False, 0)
        
        self.show_all()
        
    def update_progress(self, text, detail, fraction):
        self.label.set_markup(f"<b>{text}</b>")
        self.detail_label.set_text(detail)
        self.pbar.set_fraction(fraction)


class ProfileReportWindow(Gtk.Window):
    def __init__(self, parent, results_report, film_base_values, arr_raw_base, reference_xyz_path):
        super().__init__(title="Film Profiling Report")
        self.set_transient_for(parent)
        self.set_destroy_with_parent(True)
        self.set_default_size(1100, 750)
        self.set_position(Gtk.WindowPosition.CENTER_ON_PARENT)
        self.reference_xyz_path = reference_xyz_path
        
        # Main layout: Notebook
        self.notebook = Gtk.Notebook()
        self.notebook.set_tab_pos(Gtk.PositionType.LEFT)
        self.add(self.notebook)
        
        # Add a tab for each target
        for target_name, target_data in results_report.items():
            tab_widget = self.create_target_report_tab(target_name, target_data)
            label = Gtk.Label(label=target_name)
            self.notebook.append_page(tab_widget, label)
            
        # Add a tab for the film base
        if film_base_values:
            base_tab = self.create_base_report_tab(film_base_values, arr_raw_base)
            label = Gtk.Label(label="Film Base")
            self.notebook.append_page(base_tab, label)
            
        self.show_all()

    def create_target_report_tab(self, target_name, target_data):
        scrolled = Gtk.ScrolledWindow()
        scrolled.set_policy(Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.AUTOMATIC)
        
        hbox = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=18)
        hbox.set_border_width(12)
        
        # --- Left Column: Matplotlib Figure Canvas (2 subplots) ---
        fig = Figure(figsize=(5, 7), facecolor='#1e1e1e')
        fig.subplots_adjust(hspace=0.4, top=0.95, bottom=0.08, left=0.15, right=0.92)
        
        # Subplot 1: TRC Curves
        ax1 = fig.add_subplot(2, 1, 1)
        ax1.set_facecolor('#121212')
        ax1.spines['bottom'].set_color('#444444')
        ax1.spines['top'].set_color('#444444')
        ax1.spines['left'].set_color('#444444')
        ax1.spines['right'].set_color('#444444')
        ax1.tick_params(colors='#e0e0e0', which='both')
        ax1.yaxis.label.set_color('#e0e0e0')
        ax1.xaxis.label.set_color('#e0e0e0')
        ax1.title.set_color('#e0e0e0')
        
        trc = target_data["trc_curves"]
        x_trc = np.linspace(0.0, 1.0, len(trc[0]))
        ax1.plot(x_trc, trc[0], color='#ff4444', label='Red TRC', linewidth=2)
        ax1.plot(x_trc, trc[1], color='#44ff44', label='Green TRC', linewidth=2)
        ax1.plot(x_trc, trc[2], color='#4444ff', label='Blue TRC', linewidth=2)
        ax1.set_title("Estimated TRC Curves")
        ax1.set_xlabel("Input Intensity")
        ax1.set_ylabel("Output Intensity")
        ax1.legend(facecolor='#1e1e1e', edgecolor='#444444', labelcolor='#e0e0e0')
        ax1.grid(True, color='#2a2a2a')
        
        # Subplot 2: Characteristic Curve (Measured RGB vs Reference Y)
        ax2 = fig.add_subplot(2, 1, 2)
        ax2.set_facecolor('#121212')
        ax2.spines['bottom'].set_color('#444444')
        ax2.spines['top'].set_color('#444444')
        ax2.spines['left'].set_color('#444444')
        ax2.spines['right'].set_color('#444444')
        ax2.tick_params(colors='#e0e0e0', which='both')
        ax2.yaxis.label.set_color('#e0e0e0')
        ax2.xaxis.label.set_color('#e0e0e0')
        ax2.title.set_color('#e0e0e0')
        
        ref_y = []
        measured_r = []
        measured_g = []
        measured_b = []
        
        ref_xyz_path = self.reference_xyz_path
        try:
            with open(ref_xyz_path, 'r') as f:
                ref_json = json.load(f)
            ref_patches = ref_json.get("patches", {})
            
            sc_data = target_data["sc_profile_data"]
            for p_name, p_val in sc_data["patches"].items():
                ref_p = ref_patches.get(p_name.lower()) or ref_patches.get(p_name.upper())
                if ref_p:
                    ref_y.append(ref_p["Y"])
                    measured_r.append(p_val["r"])
                    measured_g.append(p_val["g"])
                    measured_b.append(p_val["b"])
        except Exception as e:
            print(f"[ERROR] Failed to load reference XYZ for report plot: {e}")
            
        if ref_y:
            ref_y = np.array(ref_y)
            log_ref_y = np.log10(np.clip(ref_y, 1e-5, None))
            log_r = np.log10(np.clip(measured_r, 1e-5, None))
            log_g = np.log10(np.clip(measured_g, 1e-5, None))
            log_b = np.log10(np.clip(measured_b, 1e-5, None))
            
            ax2.scatter(log_ref_y, log_r, color='#ff4444', label='Red', alpha=0.6, s=15)
            ax2.scatter(log_ref_y, log_g, color='#44ff44', label='Green', alpha=0.6, s=15)
            ax2.scatter(log_ref_y, log_b, color='#4444ff', label='Blue', alpha=0.6, s=15)
            ax2.set_title("Characteristic Curve (Log-Log)")
            ax2.set_xlabel("Log10(Reference Y)")
            ax2.set_ylabel("Log10(Measured RGB)")
            ax2.legend(facecolor='#1e1e1e', edgecolor='#444444', labelcolor='#e0e0e0')
            ax2.grid(True, color='#2a2a2a')
        else:
            ax2.text(0.5, 0.5, "No Reference Data Available", color='white', ha='center', va='center')
            ax2.set_title("Characteristic Curve")
            
        canvas = FigureCanvas(fig)
        canvas.set_size_request(450, 550)
        hbox.pack_start(canvas, False, False, 0)
        
        # --- Right Column: Text & Conversion Preview ---
        vbox_right = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12)
        hbox.pack_start(vbox_right, True, True, 0)
        
        # 1. Profcheck CIEDE2000 results
        frame_prof = Gtk.Frame(label="IT8 Profile Quality (profcheck)")
        vbox_prof = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        vbox_prof.set_border_width(8)
        frame_prof.add(vbox_prof)
        
        import html
        prof_text = target_data["profcheck_output"] or "No profcheck output."
        lines = [l.strip() for l in prof_text.split('\n') if l.strip()]
        last_line = lines[-1] if lines else "No profcheck output."
        
        lbl_prof = Gtk.Label()
        lbl_prof.set_xalign(0.0)
        lbl_prof.set_yalign(0.5)
        lbl_prof.set_selectable(True)
        lbl_prof.set_markup(f"<span font_family='monospace'>{html.escape(last_line)}</span>")
        vbox_prof.pack_start(lbl_prof, False, False, 4)
        vbox_right.pack_start(frame_prof, False, False, 0)
        
        # 2. Target Converted Preview
        frame_prev = Gtk.Frame(label="Target Converted")
        vbox_prev = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        vbox_prev.set_border_width(8)
        frame_prev.add(vbox_prev)
        
        # Image area
        preview_img_widget = Gtk.Image()
        preview_img_widget.set_size_request(-1, 260)
        align = Gtk.Alignment(xalign=0.5, yalign=0.5, xscale=0, yscale=0)
        align.add(preview_img_widget)
        vbox_prev.pack_start(align, False, False, 0)
        vbox_right.pack_start(frame_prev, False, False, 0)
        
        # 3. Log Details (Expander)
        expander = Gtk.Expander(label="Compilation Logs")
        scroll_logs = Gtk.ScrolledWindow()
        scroll_logs.set_policy(Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.AUTOMATIC)
        scroll_logs.set_min_content_height(250)
        
        tv_logs = Gtk.TextView()
        tv_logs.set_editable(False)
        tv_logs.set_monospace(True)
        tv_logs.get_buffer().set_text("\n".join(target_data["log_messages"]))
        scroll_logs.add(tv_logs)
        expander.add(scroll_logs)
        vbox_right.pack_start(expander, True, True, 0)
        
        # --- Run conversion in background thread once ---
        def conversion_thread():
            try:
                img_obj = target_data.get("img_obj")
                if not img_obj:
                    arr_raw = target_data.get("arr_raw")
                    if arr_raw is not None:
                        arr_8bit = np.clip(arr_raw / 256.0, 0, 255).astype(np.uint8)
                        h, w, c = arr_8bit.shape
                        GLib.idle_add(set_image_from_arr, arr_8bit, w, h)
                    return
                
                shutter_speed_str = target_data.get("shutter") or "1/8s"
                sc_profile = target_data["sc_profile"]
                
                arr_converted = film_profiling.convert_raw_image(
                    img=img_obj,
                    profile=sc_profile,
                    clut_path=None,
                    shutter_str=shutter_speed_str,
                    exposure_comp=1.0,
                    half=True
                )
                
                arr_8bit = np.clip(arr_converted / 256.0, 0, 255).astype(np.uint8)
                h, w, c = arr_8bit.shape
                GLib.idle_add(set_image_from_arr, arr_8bit, w, h)
                
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"[ERROR] Target conversion failed: {e}")
        
        def set_image_from_arr(arr, w, h):
            max_h = 260
            scale = max_h / h
            new_h = max_h
            new_w = int(w * scale)
            
            glib_bytes = GLib.Bytes.new(arr.tobytes())
            pixbuf = GdkPixbuf.Pixbuf.new_from_bytes(
                glib_bytes,
                GdkPixbuf.Colorspace.RGB,
                False,
                8,
                w,
                h,
                w * 3
            )
            scaled_pixbuf = pixbuf.scale_simple(new_w, new_h, GdkPixbuf.InterpType.BILINEAR)
            preview_img_widget.set_from_pixbuf(scaled_pixbuf)
            
        t_conv = threading.Thread(target=conversion_thread)
        t_conv.daemon = True
        t_conv.start()
        
        scrolled.add(hbox)
        return scrolled

    def create_base_report_tab(self, film_base_values, arr_raw_base):
        scrolled = Gtk.ScrolledWindow()
        scrolled.set_policy(Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.AUTOMATIC)
        
        align = Gtk.Alignment(xalign=0.5, yalign=0.1, xscale=0, yscale=0)
        
        vbox_left = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12)
        vbox_left.set_border_width(12)
        vbox_left.set_size_request(450, -1)
        align.add(vbox_left)
        
        lbl_title = Gtk.Label()
        lbl_title.set_markup("<b>Scanned Film Base Statistics</b>")
        lbl_title.set_xalign(0.0)
        vbox_left.pack_start(lbl_title, False, False, 0)
        
        grid = Gtk.Grid()
        grid.set_column_spacing(15)
        grid.set_row_spacing(10)
        grid.set_margin_start(10)
        grid.set_margin_top(10)
        
        def add_row(g, label, val, row):
            lbl_l = Gtk.Label(label=label)
            lbl_l.set_xalign(0.0)
            lbl_l.get_style_context().add_class("meta-label")
            
            lbl_v = Gtk.Label(label=str(val))
            lbl_v.set_xalign(0.0)
            lbl_v.set_selectable(True)
            
            g.attach(lbl_l, 0, row, 1, 1)
            g.attach(lbl_v, 1, row, 1, 1)
            
        add_row(grid, "ISO:", film_base_values.get("iso", 100), 0)
        add_row(grid, "Shutter Speed:", film_base_values.get("shutter", "1/8s"), 1)
        
        # Channels
        r_avg = film_base_values.get("r", {}).get("avg", 0.0)
        r_std = film_base_values.get("r", {}).get("std", 0.0)
        g_avg = film_base_values.get("g", {}).get("avg", 0.0)
        g_std = film_base_values.get("g", {}).get("std", 0.0)
        b_avg = film_base_values.get("b", {}).get("avg", 0.0)
        b_std = film_base_values.get("b", {}).get("std", 0.0)
        
        add_row(grid, "Red Channel Avg:", f"{r_avg:.2f} (std: {r_std:.2f})", 2)
        add_row(grid, "Green Channel Avg:", f"{g_avg:.2f} (std: {g_std:.2f})", 3)
        add_row(grid, "Blue Channel Avg:", f"{b_avg:.2f} (std: {b_std:.2f})", 4)
        
        vbox_left.pack_start(grid, False, False, 0)
        
        scrolled.add(align)
        return scrolled


class FilmProfilingAppWindow(Gtk.Window):
    def __init__(self):
        super().__init__(title="Negative Film Profiling Station")
        self.set_default_size(1280, 800)
        self.connect("destroy", self.on_destroy)

        # Apply custom dark style
        css_provider = Gtk.CssProvider()
        css_provider.load_from_data(b"""
            /* Global theme rules to override light system themes */
            window, dialog {
                background-color: #181818;
                color: #ffffff;
            }
            label {
                color: #ffffff;
            }
            /* Dropdowns, menus, and popups */
            combobox, combobox window, window.popup, menu, menuitem, list, row, popover {
                background-image: none;
                background-color: #1e1e1e;
                color: #ffffff;
            }
            combobox label, window.popup label, menu label, popover label {
                color: #ffffff;
            }
            button {
                background-image: none;
                background-color: #2e2e2e;
                color: #ffffff;
                border: 1px solid #484848;
                border-radius: 6px;
                padding: 6px 14px;
                font-weight: bold;
                text-shadow: none;
            }
            button:hover {
                background-image: none;
                background-color: #3e3e3e;
                border-color: #585858;
            }
            button:active {
                background-image: none;
                background-color: #1e1e1e;
            }
            button:disabled {
                background-image: none;
                color: #666666;
                background-color: #1c1c1c;
                border-color: #2c2c2c;
            }
            entry {
                background-color: #222222;
                color: #ffffff;
                border: 1px solid #444444;
                border-radius: 4px;
                padding: 6px;
            }
            entry:focus {
                border-color: #2ea44f;
            }
            notebook {
                background-color: #181818;
                color: #ffffff;
                border: 1px solid #333333;
            }
            notebook tab {
                background-color: #242424;
                color: #bbbbbb;
                padding: 8px 12px;
                font-weight: bold;
            }
            notebook tab:active {
                background-color: #181818;
                color: #ffffff;
                border-bottom: 2px solid #2ea44f;
            }
            treeview {
                background-color: #1e1e1e;
                color: #ffffff;
            }
            treeview header button {
                background-image: none;
                background-color: #2a2a2a;
                color: #ffffff;
                font-weight: bold;
            }
            textview text {
                background-color: #1e1e1e;
                color: #ffffff;
            }
            scrolledwindow {
                background-color: #181818;
                border: 1px solid #333333;
            }

            /* Custom classes styling */
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
            .capture-btn:disabled {
                background-image: none;
                background-color: #444444;
                color: #888888;
                border-color: #2c2c2c;
            }
            .tool-btn {
                background-image: none;
                background-color: #2e2e2e;
                color: #ffffff;
                border: 1px solid #484848;
                border-radius: 4px;
                padding: 6px 12px;
                font-weight: bold;
            }
            .tool-btn:hover {
                background-image: none;
                background-color: #3e3e3e;
            }
            .tool-btn:disabled {
                background-image: none;
                color: #666666;
                background-color: #1c1c1c;
                border-color: #2c2c2c;
            }
            .sidebar {
                background-color: #181818;
                border-right: 1px solid #333333;
                padding: 15px;
            }
            .right-sidebar {
                background-color: #181818;
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
                color: #cccccc;
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
        self.base_img_obj = None

        # Target tabs and film base state
        self.target_tabs = []
        self.base_values = None
        self.base_iso = None
        self.base_shutter = None

        # Film Base tab state
        self.arr_raw_base = None
        self.arr_cc_base = None
        self.base_filepaths = None
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

        # IT8 Target Reference Panel
        ref_frame = Gtk.Frame(label="IT8 Target Reference")
        ref_vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        ref_vbox.set_border_width(8)
        ref_frame.add(ref_vbox)
        sidebar_box.pack_start(ref_frame, False, False, 5)

        self.zip_entry = Gtk.Entry()
        self.zip_entry.set_text("http://www.colorreference.de/targets/R190808.zip")
        self.zip_entry.set_tooltip_text("Enter reference file (.txt/.it8/.zip) URL or local path")
        ref_vbox.pack_start(self.zip_entry, False, False, 0)

        self.load_ref_btn = Gtk.Button(label="DOWNLOAD REFERENCE IT8")
        self.load_ref_btn.get_style_context().add_class("tool-btn")
        self.load_ref_btn.connect("clicked", self.on_load_ref_clicked)
        ref_vbox.pack_start(self.load_ref_btn, False, False, 5)

        self.ref_status_lbl = Gtk.Label(label="No reference loaded.")
        self.ref_status_lbl.set_xalign(0.0)
        self.ref_status_lbl.get_style_context().add_class("meta-label")
        ref_vbox.pack_start(self.ref_status_lbl, False, False, 5)

        self.reference_xyz_path = None

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
        self.capture_it8_btn = Gtk.Button(label="CAPTURE TARGET")
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
        sidebar_box.pack_start(ae_frame, False, False, 5)
        # Profile Management Frame
        mgmt_frame = Gtk.Frame(label="Profile Management")
        mgmt_vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        mgmt_vbox.set_border_width(8)
        mgmt_frame.add(mgmt_vbox)
        sidebar_box.pack_start(mgmt_frame, False, False, 5)

        self.save_profile_btn = Gtk.Button(label="SAVE PROFILE")
        self.save_profile_btn.get_style_context().add_class("tool-btn")
        self.save_profile_btn.connect("clicked", self.on_save_profile_clicked)
        mgmt_vbox.pack_start(self.save_profile_btn, False, False, 5)

        self.load_profile_btn_side = Gtk.Button(label="LOAD PROFILE")
        self.load_profile_btn_side.get_style_context().add_class("tool-btn")
        self.load_profile_btn_side.connect("clicked", self.on_load_profile_clicked_json)
        mgmt_vbox.pack_start(self.load_profile_btn_side, False, False, 5)

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
        # =====================================================================
        # CENTER NOTEBOOK
        # =====================================================================
        self.notebook = Gtk.Notebook()
        self.notebook.set_scrollable(True)
        main_box.pack_start(self.notebook, True, True, 0)

        # Create Target 1 dynamically
        t1 = TargetTabState(1, "Target 1")
        self.target_tabs.append(t1)
        t1_widget = self.create_target_tab_widget(t1)
        t1.lbl_tab = Gtk.Label(label="Target 1")
        t1.lbl_tab.set_use_markup(True)
        self.notebook.append_page(t1_widget, t1.lbl_tab)

        # Create the "+" tab page
        self.plus_box = Gtk.Box()
        self.lbl_plus_tab = Gtk.Label(label=" + ")
        self.notebook.append_page(self.plus_box, self.lbl_plus_tab)

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

        self.btn_read_base = Gtk.Button(label="Read Film Base Values")
        self.btn_read_base.get_style_context().add_class("tool-btn")
        self.btn_read_base.set_sensitive(False)
        self.btn_read_base.connect("clicked", lambda w: self.read_film_base_values())
        base_tb.pack_start(self.btn_read_base, False, False, 0)

        # Label to display exposure info for Film Base
        self.lbl_exposure_info_base = Gtk.Label()
        self.lbl_exposure_info_base.set_xalign(1.0)
        self.lbl_exposure_info_base.get_style_context().add_class("meta-label")
        base_tb.pack_end(self.lbl_exposure_info_base, True, True, 10)

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

        # Table for Film Base values: Channel (str), Avg (float), Std Dev (float)
        self.base_store = Gtk.ListStore(str, float, float)
        self.base_treeview = Gtk.TreeView(model=self.base_store)
        col1 = Gtk.TreeViewColumn("Channel", Gtk.CellRendererText(), text=0)

        col2 = Gtk.TreeViewColumn("Average (Linear)", Gtk.CellRendererText())
        renderer2 = Gtk.CellRendererText()
        col2.pack_start(renderer2, True)
        col2.set_cell_data_func(renderer2, lambda col, cell, model, iter, data=None: cell.set_property("text", f"{model.get_value(iter, 1):.2f}" if model.get_value(iter, 1) is not None else ""))

        col3 = Gtk.TreeViewColumn("Std Dev", Gtk.CellRendererText())
        renderer3 = Gtk.CellRendererText()
        col3.pack_start(renderer3, True)
        col3.set_cell_data_func(renderer3, lambda col, cell, model, iter, data=None: cell.set_property("text", f"{model.get_value(iter, 2):.2f}" if model.get_value(iter, 2) is not None else ""))
        self.base_treeview.append_column(col1)
        self.base_treeview.append_column(col2)
        self.base_treeview.append_column(col3)

        base_scroll = Gtk.ScrolledWindow()
        base_scroll.set_policy(Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.AUTOMATIC)
        base_scroll.set_min_content_height(140)
        base_scroll.add(self.base_treeview)

        base_frame_widget = Gtk.Frame(label="Read Film Base Values")
        base_frame_widget.add(base_scroll)
        self.base_box.pack_start(base_frame_widget, False, False, 5)

        self.lbl_base_tab = Gtk.Label(label="Film Base")
        self.lbl_base_tab.set_use_markup(True)
        self.notebook.append_page(self.base_box, self.lbl_base_tab)

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
    # MULTI-TAB TARGET MANAGEMENT & STATE REDIRECT PROPERTIES
    # =====================================================================
    def get_active_target_tab(self):
        if not hasattr(self, 'target_tabs') or not self.target_tabs:
            return None
        page_num = self.notebook.get_current_page()
        if page_num == -1:
            return None
        active_widget = self.notebook.get_nth_page(page_num)
        for tab in self.target_tabs:
            if tab.widget_box == active_widget:
                return tab
        return None

    @property
    def arr_raw_target(self):
        t = self.get_active_target_tab()
        return t.arr_raw if t else None
    @arr_raw_target.setter
    def arr_raw_target(self, val):
        t = self.get_active_target_tab()
        if t: t.arr_raw = val

    @property
    def arr_cc_target(self):
        t = self.get_active_target_tab()
        return t.arr_cc if t else None
    @arr_cc_target.setter
    def arr_cc_target(self, val):
        t = self.get_active_target_tab()
        if t: t.arr_cc = val

    @property
    def current_pixbuf_target(self):
        t = self.get_active_target_tab()
        return t.current_pixbuf if t else None
    @current_pixbuf_target.setter
    def current_pixbuf_target(self, val):
        t = self.get_active_target_tab()
        if t: t.current_pixbuf = val

    @property
    def scaled_pixbuf_target(self):
        t = self.get_active_target_tab()
        return t.scaled_pixbuf if t else None
    @scaled_pixbuf_target.setter
    def scaled_pixbuf_target(self, val):
        t = self.get_active_target_tab()
        if t: t.scaled_pixbuf = val

    @property
    def normalized_selection_target(self):
        t = self.get_active_target_tab()
        return t.normalized_selection if t else None
    @normalized_selection_target.setter
    def normalized_selection_target(self, val):
        t = self.get_active_target_tab()
        if t: t.normalized_selection = val

    @property
    def is_dragging_target(self):
        t = self.get_active_target_tab()
        return t.is_dragging if t else False
    @is_dragging_target.setter
    def is_dragging_target(self, val):
        t = self.get_active_target_tab()
        if t: t.is_dragging = val

    @property
    def selection_start_target(self):
        t = self.get_active_target_tab()
        return t.selection_start if t else None
    @selection_start_target.setter
    def selection_start_target(self, val):
        t = self.get_active_target_tab()
        if t: t.selection_start = val

    @property
    def selection_end_target(self):
        t = self.get_active_target_tab()
        return t.selection_end if t else None
    @selection_end_target.setter
    def selection_end_target(self, val):
        t = self.get_active_target_tab()
        if t: t.selection_end = val

    @property
    def img_x_offset_target(self):
        t = self.get_active_target_tab()
        return t.img_x_offset if t else 0
    @img_x_offset_target.setter
    def img_x_offset_target(self, val):
        t = self.get_active_target_tab()
        if t: t.img_x_offset = val

    @property
    def img_y_offset_target(self):
        t = self.get_active_target_tab()
        return t.img_y_offset if t else 0
    @img_y_offset_target.setter
    def img_y_offset_target(self, val):
        t = self.get_active_target_tab()
        if t: t.img_y_offset = val

    @property
    def it8_mask_active(self):
        t = self.get_active_target_tab()
        return t.it8_mask_active if t else False
    @it8_mask_active.setter
    def it8_mask_active(self, val):
        t = self.get_active_target_tab()
        if t: t.it8_mask_active = val

    @property
    def it8_scale(self):
        t = self.get_active_target_tab()
        return t.it8_scale if t else 1.0
    @it8_scale.setter
    def it8_scale(self, val):
        t = self.get_active_target_tab()
        if t: t.it8_scale = val

    @property
    def it8_dx(self):
        t = self.get_active_target_tab()
        return t.it8_dx if t else 0.0
    @it8_dx.setter
    def it8_dx(self, val):
        t = self.get_active_target_tab()
        if t: t.it8_dx = val

    @property
    def it8_dy(self):
        t = self.get_active_target_tab()
        return t.it8_dy if t else 0.0
    @it8_dy.setter
    def it8_dy(self, val):
        t = self.get_active_target_tab()
        if t: t.it8_dy = val

    @property
    def it8_store(self):
        t = self.get_active_target_tab()
        return t.it8_store if t else None

    @property
    def btn_rotate_target(self):
        t = self.get_active_target_tab()
        return t.btn_rotate if t else None

    @property
    def btn_hflip_target(self):
        t = self.get_active_target_tab()
        return t.btn_hflip if t else None

    @property
    def btn_vflip_target(self):
        t = self.get_active_target_tab()
        return t.btn_vflip if t else None

    @property
    def btn_crop_target(self):
        t = self.get_active_target_tab()
        return t.btn_crop if t else None

    @property
    def btn_layer_it8(self):
        t = self.get_active_target_tab()
        return t.btn_layer_it8 if t else None

    @property
    def btn_read_it8(self):
        t = self.get_active_target_tab()
        return t.btn_read_it8 if t else None

    @property
    def lbl_target_tab(self):
        t = self.get_active_target_tab()
        return t.lbl_tab if t else None

    @property
    def target_stack(self):
        t = self.get_active_target_tab()
        return t.stack if t else None

    @property
    def image_view_target(self):
        t = self.get_active_target_tab()
        return t.image_view if t else None

    def create_target_tab_widget(self, tab_state):
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=5)
        box.get_style_context().add_class("preview-container")

        # Target Toolbar
        tb = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        box.pack_start(tb, False, False, 5)

        tab_state.btn_rotate = Gtk.Button(label="Rotate 90°")
        tab_state.btn_rotate.get_style_context().add_class("tool-btn")
        tab_state.btn_rotate.connect("clicked", lambda w: self.rotate_active_tab())
        tb.pack_start(tab_state.btn_rotate, False, False, 0)

        tab_state.btn_hflip = Gtk.Button(label="H-Flip")
        tab_state.btn_hflip.get_style_context().add_class("tool-btn")
        tab_state.btn_hflip.connect("clicked", lambda w: self.hflip_active_tab())
        tb.pack_start(tab_state.btn_hflip, False, False, 0)

        tab_state.btn_vflip = Gtk.Button(label="V-Flip")
        tab_state.btn_vflip.get_style_context().add_class("tool-btn")
        tab_state.btn_vflip.connect("clicked", lambda w: self.vflip_active_tab())
        tb.pack_start(tab_state.btn_vflip, False, False, 0)

        tab_state.btn_crop = Gtk.Button(label="Crop to Selection")
        tab_state.btn_crop.get_style_context().add_class("tool-btn")
        tab_state.btn_crop.connect("clicked", lambda w: self.crop_active_tab())
        tb.pack_start(tab_state.btn_crop, False, False, 0)

        tab_state.btn_layer_it8 = Gtk.Button(label="Layer IT8 Mask")
        tab_state.btn_layer_it8.get_style_context().add_class("tool-btn")
        tab_state.btn_layer_it8.connect("clicked", self.on_layer_it8_clicked)
        tb.pack_start(tab_state.btn_layer_it8, False, False, 0)

        tab_state.btn_read_it8 = Gtk.Button(label="Read Mask Values")
        tab_state.btn_read_it8.get_style_context().add_class("tool-btn")
        tab_state.btn_read_it8.connect("clicked", lambda w: self.read_it8_values())
        tb.pack_start(tab_state.btn_read_it8, False, False, 0)

        # Label to display exposure info (ISO and Shutter)
        tab_state.lbl_exposure_info = Gtk.Label()
        tab_state.lbl_exposure_info.set_xalign(1.0)
        tab_state.lbl_exposure_info.get_style_context().add_class("meta-label")
        tb.pack_end(tab_state.lbl_exposure_info, True, True, 10)

        tab_state.stack = Gtk.Stack()
        tab_state.stack.set_transition_type(Gtk.StackTransitionType.CROSSFADE)
        tab_state.stack.set_transition_duration(150)
        box.pack_start(tab_state.stack, True, True, 0)

        placeholder = Gtk.Label()
        placeholder.set_markup(
            f"<span size='large' foreground='#777777'>No Image Captured for {tab_state.label_text}\n\n"
            "Please load a crosstalk calibration profile,\nthen click CAPTURE TARGET.</span>"
        )
        placeholder.set_justify(Gtk.Justification.CENTER)
        tab_state.stack.add_named(placeholder, "placeholder")

        tab_state.image_view = Gtk.DrawingArea()
        tab_state.image_view.set_can_focus(True)
        tab_state.image_view.add_events(
            Gdk.EventMask.BUTTON_PRESS_MASK |
            Gdk.EventMask.BUTTON_RELEASE_MASK |
            Gdk.EventMask.POINTER_MOTION_MASK |
            Gdk.EventMask.BUTTON_MOTION_MASK
        )
        tab_state.image_view.connect("draw", self.on_draw_image_view_target, tab_state)
        tab_state.image_view.connect("button-press-event", self.on_image_button_press_target, tab_state)
        tab_state.image_view.connect("button-release-event", self.on_image_button_release_target, tab_state)
        tab_state.image_view.connect("motion-notify-event", self.on_image_motion_notify_target, tab_state)
        tab_state.stack.add_named(tab_state.image_view, "preview")
        tab_state.stack.set_visible_child_name("placeholder")

        # Table for values
        tab_state.it8_store = Gtk.ListStore(str, float, float, float, float, float, float)
        tab_state.it8_treeview = Gtk.TreeView(model=tab_state.it8_store)
        cols = [
            ("Patch", 0, False),
            ("R (Linear Avg)", 1, True),
            ("G (Linear Avg)", 2, True),
            ("B (Linear Avg)", 3, True),
            ("R (Std Dev)", 4, True),
            ("G (Std Dev)", 5, True),
            ("B (Std Dev)", 6, True)
        ]
        def _make_float_renderer(idx):
            def _render(col, cell, model, tree_iter, data=None):
                try:
                    if tree_iter is None:
                        cell.set_property("text", "")
                        return
                    val = model.get_value(tree_iter, idx)
                    cell.set_property("text", f"{val:.2f}" if val is not None else "")
                except Exception:
                    cell.set_property("text", "")
            return _render

        for col_title, col_idx, is_float in cols:
            renderer = Gtk.CellRendererText()
            if col_idx > 0:
                renderer.set_property("xalign", 1.0)
            col = Gtk.TreeViewColumn(col_title, renderer)
            if is_float:
                col.set_cell_data_func(renderer, _make_float_renderer(col_idx))
            else:
                col.add_attribute(renderer, "text", col_idx)
            if col_idx > 0:
                col.set_alignment(1.0)
            col.set_sort_column_id(col_idx)
            tab_state.it8_treeview.append_column(col)

        scroll = Gtk.ScrolledWindow()
        scroll.set_policy(Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.AUTOMATIC)
        scroll.set_min_content_height(140)
        scroll.add(tab_state.it8_treeview)

        frame = Gtk.Frame(label="Read IT8 Patch Values")
        frame.add(scroll)
        box.pack_start(frame, False, False, 5)

        tab_state.widget_box = box
        return box

    def prompt_for_film_stock(self):
        dialog = Gtk.Dialog(title="Enter Film Stock", parent=self, flags=0)
        dialog.add_buttons(
            Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL,
            Gtk.STOCK_OK, Gtk.ResponseType.OK
        )
        dialog.set_default_size(300, 100)
        
        box = dialog.get_content_area()
        lbl = Gtk.Label(label="Enter Film Stock name (e.g. Portra400, Gold200):")
        lbl.set_margin_start(10)
        lbl.set_margin_end(10)
        lbl.set_margin_top(10)
        box.pack_start(lbl, False, False, 5)
        
        entry = Gtk.Entry()
        entry.set_margin_start(10)
        entry.set_margin_end(10)
        entry.set_margin_bottom(10)
        box.pack_start(entry, True, True, 5)
        
        dialog.show_all()
        response = dialog.run()
        film_stock = None
        if response == Gtk.ResponseType.OK:
            film_stock = entry.get_text().strip()
        dialog.destroy()
        return film_stock
        
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

    def on_load_ref_clicked(self, widget):
        zip_location = self.zip_entry.get_text().strip()
        if not zip_location:
            self.show_error_dialog("Reference Error", "Please enter a valid URL or local filepath.")
            return

        self.ref_status_lbl.set_text("Downloading/parsing reference...")
        self.load_ref_btn.set_sensitive(False)

        def prompt_zip_callback(ref_filenames):
            event = threading.Event()
            result = {}
            
            def show_dialog():
                try:
                    dialog = Gtk.Dialog(
                        title="Select Reference File",
                        parent=self,
                        flags=0
                    )
                    dialog.add_buttons(
                        Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL,
                        Gtk.STOCK_OK, Gtk.ResponseType.OK
                    )
                    
                    dialog.set_default_size(350, 150)
                    box = dialog.get_content_area()
                    box.set_spacing(10)
                    
                    lbl = Gtk.Label()
                    lbl.set_markup("<b>Multiple reference files found.</b>\nSelect the file to use:")
                    lbl.set_xalign(0.0)
                    box.pack_start(lbl, False, False, 10)
                    
                    combo = Gtk.ComboBoxText()
                    for name in ref_filenames:
                        combo.append(name, name)
                    # Try to select the main one matching the zip name or first
                    zip_basename = os.path.splitext(os.path.basename(zip_location))[0].lower()
                    select_idx = 0
                    for idx, name in enumerate(ref_filenames):
                        if "extras" not in name.lower() and zip_basename in os.path.basename(name).lower():
                            select_idx = idx
                            break
                    combo.set_active(select_idx)
                    box.pack_start(combo, False, False, 10)
                    
                    dialog.show_all()
                    response = dialog.run()
                    if response == Gtk.ResponseType.OK:
                        result['selected'] = combo.get_active_text()
                    else:
                        result['selected'] = None
                    dialog.destroy()
                except Exception as ex:
                    result['error'] = ex
                finally:
                    event.set()
            
            GLib.idle_add(show_dialog)
            event.wait()
            
            if 'error' in result:
                raise result['error']
            selected = result.get('selected')
            if not selected:
                raise ValueError("Reference file selection cancelled by user.")
            return selected

        def run():
            try:
                # Use data/ directory in the workspace
                cache_dir = os.path.join(project_dir, "data")
                patches, loaded_filename, reference_dir = download_and_parse_reference_file(
                    zip_location, cache_dir, prompt_zip_callback=prompt_zip_callback
                )
                
                # Write to f"{ref_base_name}_ref.json" next to reference file
                ref_base_name = os.path.splitext(os.path.basename(loaded_filename))[0]
                out_json_path = os.path.join(reference_dir, f"{ref_base_name}_ref.json")
                ref_data = {
                    "description": "IT8.7/2 Reference XYZ values",
                    "source": zip_location,
                    "patches": patches
                }
                with open(out_json_path, 'w') as f:
                    json.dump(ref_data, f, indent=2)
                
                GLib.idle_add(self.on_load_ref_success, out_json_path, len(patches), loaded_filename)
            except Exception as e:
                GLib.idle_add(self.on_load_ref_failure, str(e))

        t = threading.Thread(target=run)
        t.daemon = True
        t.start()

    def on_load_ref_success(self, path, num_patches, loaded_filename):
        self.load_ref_btn.set_sensitive(True)
        self.reference_xyz_path = path
        self.ref_status_lbl.set_markup(f"<span foreground='#44ff44'>Loaded {num_patches} patches</span> from {os.path.basename(loaded_filename)}")

    def on_load_ref_failure(self, err_msg):
        self.load_ref_btn.set_sensitive(True)
        self.ref_status_lbl.set_text("Failed to load reference.")
        self.show_error_dialog("Download/Parse Error", err_msg)

    def on_save_profile_clicked(self, widget):
        if not self.calib:
            self.status_lbl.set_text("Status: No crosstalk profile loaded.")
            return
        
        if not hasattr(self, 'base_values') or self.base_values is None:
            dialog = Gtk.MessageDialog(
                transient_for=self,
                flags=0,
                message_type=Gtk.MessageType.WARNING,
                buttons=Gtk.ButtonsType.OK,
                text="Missing Film Base"
            )
            dialog.format_secondary_text("Please capture and read the Film Base values first.")
            dialog.run()
            dialog.destroy()
            return

        if not hasattr(self, 'reference_xyz_path') or self.reference_xyz_path is None:
            self.show_error_dialog("Missing Reference", "Please download or load the IT8 reference file first.")
            return

        active_target_tabs = []
        for tab in self.target_tabs:
            if len(tab.it8_store) > 0:
                active_target_tabs.append(tab)

        if not active_target_tabs:
            dialog = Gtk.MessageDialog(
                transient_for=self,
                flags=0,
                message_type=Gtk.MessageType.WARNING,
                buttons=Gtk.ButtonsType.OK,
                text="Missing Target Data"
            )
            dialog.format_secondary_text("Please read mask values for at least one Target tab.")
            dialog.run()
            dialog.destroy()
            return

        film_stock = self.prompt_for_film_stock()
        if not film_stock:
            self.status_lbl.set_text("Status: Save canceled.")
            return

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"profile_{film_stock}_{timestamp}.json"

        file_dialog = Gtk.FileChooserDialog(
            title="Save Film Profile JSON",
            parent=self,
            action=Gtk.FileChooserAction.SAVE
        )
        file_dialog.add_buttons(
            Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL,
            Gtk.STOCK_SAVE, Gtk.ResponseType.OK
        )
        file_dialog.set_current_name(filename)
        
        filter_json = Gtk.FileFilter()
        filter_json.set_name("JSON files")
        filter_json.add_mime_type("application/json")
        filter_json.add_pattern("*.json")
        file_dialog.add_filter(filter_json)

        response = file_dialog.run()
        if response == Gtk.ResponseType.OK:
            filepath = file_dialog.get_filename()
            file_dialog.destroy()
            
            # Show progress dialog
            progress_dialog = ProfileProgressDialog(self, len(active_target_tabs))
            
            def run_builds():
                try:
                    targets_json_list = []
                    results_report = {}
                    
                    for idx, tab in enumerate(active_target_tabs):
                        t_num = idx + 1
                        t_tot = len(active_target_tabs)
                        
                        GLib.idle_add(progress_dialog.update_progress, 
                                      f"Processing Target {tab.label_text} ({t_num}/{t_tot})", 
                                      "Preparing training data...", 
                                      idx / t_tot + 0.1 / t_tot)
                        
                        patches = {}
                        for row in tab.it8_store:
                            patch_name = row[0]
                            patches[patch_name] = {
                                "r": row[1],
                                "g": row[2],
                                "b": row[3],
                                "r_std": row[4],
                                "g_std": row[5],
                                "b_std": row[6]
                            }
                        
                        calib_dict = {
                            "camera_model": self.calib.camera_model,
                            "crosstalk_matrix_raw": self.calib.M.tolist() if self.calib.M is not None else None,
                            "crosstalk_matrix_normalized": self.calib.M_norm.tolist() if self.calib.M_norm is not None else None,
                            "crosstalk_correction_matrix": self.calib.M_corr.tolist() if self.calib.M_corr is not None else None,
                            "captured_data": self.calib.captured_data
                        }
                        
                        # Prepare temporary dict
                        temp_profile_dict = {
                            "camera_name": self.calib.camera_model,
                            "crosstalk_profile": calib_dict,
                            "targets": [{
                                "name": tab.label_text,
                                "iso": tab.iso if tab.iso is not None else 100,
                                "shutter": tab.shutter if tab.shutter is not None else "1/8s",
                                "patches": patches
                            }],
                            "film_base": self.base_values,
                            "normalization_target": film_profiling.DEFAULT_NORMALIZATION_TARGET
                        }
                        
                        temp_profile = FilmProfile(temp_profile_dict)
                        
                        GLib.idle_add(progress_dialog.update_progress, 
                                      f"Processing Target {tab.label_text} ({t_num}/{t_tot})", 
                                      "Compiling ICC profile (colprof)...", 
                                      idx / t_tot + 0.3 / t_tot)
                        
                        output_profiles_dir = os.path.join(project_dir, "profiles")
                        os.makedirs(output_profiles_dir, exist_ok=True)
                        
                        def make_icc_progress_cb(step, detail):
                            GLib.idle_add(progress_dialog.detail_label.set_text, detail[:50])
                            
                        res = film_profiling.build_icc_profile(
                            temp_profile,
                            self.reference_xyz_path,
                            output_profiles_dir,
                            progress_callback=make_icc_progress_cb
                        )
                        
                        clut_path = res['clut_icc_path']
                        
                        GLib.idle_add(progress_dialog.update_progress, 
                                      f"Processing Target {tab.label_text} ({t_num}/{t_tot})", 
                                      "Encoding ICC profile...", 
                                      idx / t_tot + 0.8 / t_tot)
                        
                        with open(clut_path, 'rb') as f_icc:
                            icc_bytes = f_icc.read()
                        icc_b64 = base64.b64encode(icc_bytes).decode('utf-8')
                        
                        target_dict = {
                            "name": tab.label_text,
                            "iso": tab.iso if tab.iso is not None else 100,
                            "shutter": tab.shutter if tab.shutter is not None else "1/8s",
                            "patches": patches,
                            "icc_profile_base64": icc_b64,
                            "profcheck_output": res['profcheck_output'],
                            "log_messages": res['log_messages']
                        }
                        targets_json_list.append(target_dict)
                        
                        results_report[tab.label_text] = {
                            "trc_curves": res['trc_curves'],
                            "profcheck_output": res['profcheck_output'],
                            "log_messages": res['log_messages'],
                            "sc_profile": FilmProfile(temp_profile_dict),
                            "arr_raw": tab.arr_raw,
                            "filepaths": getattr(tab, 'filepaths', None),
                            "img_obj": getattr(tab, 'img_obj', None),
                            "iso": tab.iso,
                            "shutter": tab.shutter,
                            "cc_matrix": temp_profile.crosstalk_matrix,
                            "sc_profile_data": target_dict
                        }
                        
                        # Set self-contained ICC bytes directly in memory (no base64 decode needed)
                        results_report[tab.label_text]["sc_profile"].icc_profile_bytes = icc_bytes
                        
                        GLib.idle_add(progress_dialog.update_progress, 
                                      f"Processing Target {tab.label_text} ({t_num}/{t_tot})", 
                                      "Done!", 
                                      (idx + 1) / t_tot)
                    
                    GLib.idle_add(progress_dialog.update_progress, "Saving File", "Writing profile JSON...", 0.95)
                    
                    profile_data = {
                        "camera_name": self.calib.camera_model,
                        "crosstalk_profile": calib_dict,
                        "targets": targets_json_list,
                        "film_base": self.base_values,
                        "normalization_target": film_profiling.DEFAULT_NORMALIZATION_TARGET
                    }
                    
                    with open(filepath, 'w') as f:
                        json.dump(profile_data, f, indent=4)
                    
                    GLib.idle_add(progress_dialog.destroy)
                    GLib.idle_add(self.status_lbl.set_text, f"Status: Profile saved to {os.path.basename(filepath)}")
                    
                    # Open Report Window
                    GLib.idle_add(self.open_report_window, results_report)
                    
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    GLib.idle_add(progress_dialog.destroy)
                    GLib.idle_add(self.status_lbl.set_text, f"Status: Save failed ({str(e)})")
                    GLib.idle_add(self.show_error_dialog, "Save Profile Error", f"An error occurred: {str(e)}")
            
            t = threading.Thread(target=run_builds)
            t.daemon = True
            t.start()
        else:
            file_dialog.destroy()

    def open_report_window(self, results_report):
        report_win = ProfileReportWindow(self, results_report, self.base_values, getattr(self, 'arr_raw_base', None), self.reference_xyz_path)
        report_win.show_all()

    def on_load_profile_clicked_json(self, widget):
        import json
        file_dialog = Gtk.FileChooserDialog(
            title="Load Film Profile JSON",
            parent=self,
            action=Gtk.FileChooserAction.OPEN
        )
        file_dialog.add_buttons(
            Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL,
            Gtk.STOCK_OPEN, Gtk.ResponseType.OK
        )
        
        filter_json = Gtk.FileFilter()
        filter_json.set_name("JSON files")
        filter_json.add_mime_type("application/json")
        filter_json.add_pattern("*.json")
        file_dialog.add_filter(filter_json)

        response = file_dialog.run()
        if response == Gtk.ResponseType.OK:
            filepath = file_dialog.get_filename()
            try:
                with open(filepath, 'r') as f:
                    profile_data = json.load(f)
                
                calib_dict = profile_data.get("crosstalk_profile")
                if calib_dict:
                    self.calib = crosstalk_calibration.CrosstalkCalibration(
                        camera_model=calib_dict.get("camera_model"),
                        M=calib_dict.get("crosstalk_matrix_raw"),
                        M_norm=calib_dict.get("crosstalk_matrix_normalized"),
                        M_corr=calib_dict.get("crosstalk_correction_matrix"),
                        captured_data=calib_dict.get("captured_data")
                    )
                    self.lbl_profile_status.set_text(
                        f"Profile loaded successfully.\nCamera: {self.calib.camera_model}"
                    )
                
                for tab in list(reversed(self.target_tabs)):
                    pageNum = self.notebook.page_num(tab.widget_box)
                    if pageNum != -1:
                        self.notebook.remove_page(pageNum)
                self.target_tabs = []
                
                targets = profile_data.get("targets", [])
                for i, t_data in enumerate(targets):
                    tab_name = t_data.get("name", f"Target {i+1}")
                    t = TargetTabState(i+1, tab_name)
                    self.target_tabs.append(t)
                    
                    t_widget = self.create_target_tab_widget(t)
                    t.lbl_tab = Gtk.Label(label=tab_name)
                    t.lbl_tab.set_use_markup(True)
                    
                    plus_position = self.notebook.page_num(self.plus_box)
                    self.notebook.insert_page(t_widget, t.lbl_tab, plus_position)
                    
                    t.iso = t_data.get("iso")
                    t.shutter = t_data.get("shutter")
                    if t.iso and t.shutter:
                        t.lbl_exposure_info.set_text(f"Exposure: ISO {t.iso} | {t.shutter}")
                    
                    t.it8_store.clear()
                    patches = t_data.get("patches", {})
                    for patch_name, p_val in sorted(patches.items()):
                        r_avg = p_val.get("r", 0.0)
                        g_avg = p_val.get("g", 0.0)
                        b_avg = p_val.get("b", 0.0)
                        r_std = p_val.get("r_std", 0.0)
                        g_std = p_val.get("g_std", 0.0)
                        b_std = p_val.get("b_std", 0.0)
                        t.it8_store.append([patch_name, float(r_avg), float(g_avg), float(b_avg), float(r_std), float(g_std), float(b_std)])
                    
                    t.lbl_tab.set_markup(f"<span foreground='#44ff44'><b>{tab_name}</b></span>")
                
                film_base = profile_data.get("film_base")
                if film_base:
                    self.base_values = film_base
                    self.base_iso = film_base.get("iso", 100)
                    self.base_shutter = film_base.get("shutter", "1/8s")
                    self.lbl_exposure_info_base.set_text(f"Exposure: ISO {self.base_iso} | {self.base_shutter}")
                    
                    self.base_store.clear()
                    self.base_store.append(["Red", float(film_base.get("r", {}).get("avg", 0.0)), float(film_base.get("r", {}).get("std", 0.0))])
                    self.base_store.append(["Green", float(film_base.get("g", {}).get("avg", 0.0)), float(film_base.get("g", {}).get("std", 0.0))])
                    self.base_store.append(["Blue", float(film_base.get("b", {}).get("avg", 0.0)), float(film_base.get("b", {}).get("std", 0.0))])
                    
                    self.lbl_base_tab.set_markup("<span foreground='#44ff44'><b>Film Base</b></span>")
                
                self.notebook.show_all()
                if self.target_tabs:
                    self.notebook.set_current_page(0)
                
                self.status_lbl.set_text(f"Status: Profile loaded from {os.path.basename(filepath)}")
                self.set_controls_sensitive(self.is_connected)
            except Exception as e:
                self.status_lbl.set_text(f"Status: Load failed ({str(e)})")
        
        file_dialog.destroy()

    def read_film_base_values(self):
        if self.arr_cc_base is None:
            return
        
        arr_raw_crop, arr_cc_crop = self.get_active_crop(self.arr_raw_base, self.arr_cc_base, self.normalized_selection_base)
        
        print(f"[DEBUG] read_film_base_values:")
        print(f"  - Selection coordinates (normalized): {self.normalized_selection_base}")
        print(f"  - Base image shape: {self.arr_cc_base.shape if self.arr_cc_base is not None else 'None'}")
        if self.normalized_selection_base is not None:
            h, w, _ = self.arr_cc_base.shape
            nx1, ny1, nx2, ny2 = self.normalized_selection_base
            print(f"  - Pixel bounds: x = [{int(nx1*w)}, {int(nx2*w)}], y = [{int(ny1*h)}, {int(ny2*h)}]")
        print(f"  - Cropped array shape: {arr_cc_crop.shape}")
        
        r_avg = np.mean(arr_cc_crop[:, :, 0])
        g_avg = np.mean(arr_cc_crop[:, :, 1])
        b_avg = np.mean(arr_cc_crop[:, :, 2])
        print(f"  - Crop averages (crosstalk-corrected): R={r_avg:.2f}, G={g_avg:.2f}, B={b_avg:.2f}")
        
        r_raw_avg = np.mean(arr_raw_crop[:, :, 0])
        g_raw_avg = np.mean(arr_raw_crop[:, :, 1])
        b_raw_avg = np.mean(arr_raw_crop[:, :, 2])
        print(f"  - Crop averages (raw uncorrected): R={r_raw_avg:.2f}, G={g_raw_avg:.2f}, B={b_raw_avg:.2f}")
        
        r_std = np.std(arr_cc_crop[:, :, 0])
        g_std = np.std(arr_cc_crop[:, :, 1])
        b_std = np.std(arr_cc_crop[:, :, 2])
        
        self.base_store.clear()
        self.base_store.append(["Red", float(r_avg), float(r_std)])
        self.base_store.append(["Green", float(g_avg), float(g_std)])
        self.base_store.append(["Blue", float(b_avg), float(b_std)])
        
        self.base_values = {
            "iso": self.base_iso if self.base_iso is not None else 100,
            "shutter": self.base_shutter if self.base_shutter is not None else "1/8s",
            "r": {"avg": float(r_avg), "std": float(r_std)},
            "g": {"avg": float(g_avg), "std": float(g_std)},
            "b": {"avg": float(b_avg), "std": float(b_std)}
        }
        
        self.lbl_base_tab.set_markup("<span foreground='#44ff44'><b>Film Base</b></span>")
        self.status_lbl.set_text("Status: Read film base values.")

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
        
        active_target = self.get_active_target_tab()
        if active_target:
            arr_raw = active_target.arr_raw
            arr_cc = active_target.arr_cc
            selection = active_target.normalized_selection
            scaled_pixbuf = active_target.scaled_pixbuf
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
        if page == self.plus_box:
            # Add a new target tab dynamically
            new_idx = len(self.target_tabs) + 1
            tab_name = f"Target {new_idx}"
            t = TargetTabState(new_idx, tab_name)
            self.target_tabs.append(t)
            
            t_widget = self.create_target_tab_widget(t)
            t.lbl_tab = Gtk.Label(label=tab_name)
            t.lbl_tab.set_use_markup(True)
            
            plus_position = self.notebook.page_num(self.plus_box)
            self.notebook.insert_page(t_widget, t.lbl_tab, plus_position)
            self.notebook.show_all()
            # Defer the switch: set_current_page inside the page-changed handler
            # has no effect because GTK is mid-signal. idle_add runs it after.
            new_page_idx = plus_position  # new tab sits here; + shifted to +1
            GLib.idle_add(self.notebook.set_current_page, new_page_idx)
            return

        self.update_histograms()
        self.update_toolbar_sensitivities()

    # Target event wrappers
    def on_draw_image_view_target(self, widget, cr, tab_state):
        return self.draw_image_preview_for_tab(cr, tab_state)

    def on_image_button_press_target(self, widget, event, tab_state):
        if not tab_state.scaled_pixbuf:
            return False
        if event.button == 1:
            img_w = tab_state.scaled_pixbuf.get_width()
            img_h = tab_state.scaled_pixbuf.get_height()
            x = max(tab_state.img_x_offset, min(event.x, tab_state.img_x_offset + img_w))
            y = max(tab_state.img_y_offset, min(event.y, tab_state.img_y_offset + img_h))
            tab_state.is_dragging = True
            tab_state.selection_start = (x, y)
            tab_state.selection_end = (x, y)
            widget.queue_draw()
            return True
        return False

    def on_image_motion_notify_target(self, widget, event, tab_state):
        if tab_state.is_dragging and tab_state.selection_start:
            img_w = tab_state.scaled_pixbuf.get_width()
            img_h = tab_state.scaled_pixbuf.get_height()
            x = max(tab_state.img_x_offset, min(event.x, tab_state.img_x_offset + img_w))
            y = max(tab_state.img_y_offset, min(event.y, tab_state.img_y_offset + img_h))
            tab_state.selection_end = (x, y)
            widget.queue_draw()
            return True
        return False

    def on_image_button_release_target(self, widget, event, tab_state):
        if event.button == 1 and tab_state.is_dragging:
            tab_state.is_dragging = False
            if tab_state.selection_start and tab_state.selection_end:
                x1, y1 = tab_state.selection_start
                x2, y2 = tab_state.selection_end
                if abs(x2 - x1) > 5 and abs(y2 - y1) > 5:
                    img_w = tab_state.scaled_pixbuf.get_width()
                    img_h = tab_state.scaled_pixbuf.get_height()
                    img_x1 = max(0, min(x1, x2) - tab_state.img_x_offset)
                    img_x2 = min(img_w, max(x1, x2) - tab_state.img_x_offset)
                    img_y1 = max(0, min(y1, y2) - tab_state.img_y_offset)
                    img_y2 = min(img_h, max(y1, y2) - tab_state.img_y_offset)

                    tab_state.normalized_selection = (
                        img_x1 / img_w,
                        img_y1 / img_h,
                        img_x2 / img_w,
                        img_y2 / img_h
                    )
                else:
                    tab_state.normalized_selection = None
                self.update_histograms()
                self.update_toolbar_sensitivities()
            widget.queue_draw()
            return True
        return False

    # Film base event wrappers
    def on_draw_image_view_base(self, widget, cr):
        if not self.scaled_pixbuf_base:
            return False

        alloc = self.image_view_base.get_allocation()
        img_w = self.scaled_pixbuf_base.get_width()
        img_h = self.scaled_pixbuf_base.get_height()

        x_offset = max(0, (alloc.width - img_w) // 2)
        y_offset = max(0, (alloc.height - img_h) // 2)

        self.img_x_offset_base = x_offset
        self.img_y_offset_base = y_offset

        Gdk.cairo_set_source_pixbuf(cr, self.scaled_pixbuf_base, x_offset, y_offset)
        cr.paint()

        if self.is_dragging_base and self.selection_start_base and self.selection_end_base:
            x1, y1 = self.selection_start_base
            x2, y2 = self.selection_end_base
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

        elif self.normalized_selection_base is not None:
            nx1, ny1, nx2, ny2 = self.normalized_selection_base
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

        return True

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

    def draw_image_preview_for_tab(self, cr, tab_state):
        if not tab_state.scaled_pixbuf:
            return False

        alloc = tab_state.image_view.get_allocation()
        img_w = tab_state.scaled_pixbuf.get_width()
        img_h = tab_state.scaled_pixbuf.get_height()

        x_offset = max(0, (alloc.width - img_w) // 2)
        y_offset = max(0, (alloc.height - img_h) // 2)

        tab_state.img_x_offset = x_offset
        tab_state.img_y_offset = y_offset

        Gdk.cairo_set_source_pixbuf(cr, tab_state.scaled_pixbuf, x_offset, y_offset)
        cr.paint()

        # Draw selection border
        if tab_state.is_dragging and tab_state.selection_start and tab_state.selection_end:
            x1, y1 = tab_state.selection_start
            x2, y2 = tab_state.selection_end
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

        elif tab_state.normalized_selection is not None:
            nx1, ny1, nx2, ny2 = tab_state.normalized_selection
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
        if tab_state.it8_mask_active:
            boxes = self.get_it8_boxes_for_tab(tab_state)
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

        active_target = self.get_active_target_tab()
        if is_target and not active_target:
            self.status_lbl.set_text("Status: No active Target tab selected.")
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
                captured_filepaths = img.filepaths

                # Correct crosstalk using loaded profile in 32-bit float space
                GLib.idle_add(self.status_lbl.set_text, "Status: Correcting crosstalk...")
                arr_cc = self.calib.apply(arr_raw.astype(np.float32)).astype(np.float32)

                # Format to 8-bit preview bytes
                arr_8bit = np.clip(arr_cc / 256.0, 0, 255).astype(np.uint8)
                raw_bytes = arr_8bit.tobytes()
                h, w, c = arr_cc.shape

                exposure_info = {
                    "iso": iso,
                    "shutter": optimal_speed
                }

                target_tab_or_base = active_target if is_target else "base"
                GLib.idle_add(self.on_capture_success, target_tab_or_base, raw_bytes, w, h, arr_raw, arr_cc, exposure_info, captured_filepaths, img)
            except Exception as e:
                GLib.idle_add(self.on_capture_failure, str(e))

        t = threading.Thread(target=thread_func)
        t.daemon = True
        t.start()

    def on_capture_success(self, target_tab_or_base, raw_bytes, w, h, arr_raw, arr_cc, exposure_info, filepaths, img_obj):
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

        if isinstance(target_tab_or_base, TargetTabState):
            if getattr(target_tab_or_base, 'img_obj', None) is not None:
                try:
                    target_tab_or_base.img_obj.discard()
                except Exception:
                    pass
            target_tab_or_base.img_obj = img_obj
            target_tab_or_base.arr_raw = arr_raw
            target_tab_or_base.arr_cc = arr_cc
            target_tab_or_base.filepaths = filepaths
            target_tab_or_base.current_pixbuf = pixbuf
            target_tab_or_base.stack.set_visible_child_name("preview")
            
            target_tab_or_base.iso = exposure_info["iso"]
            target_tab_or_base.shutter = exposure_info["shutter"]
            target_tab_or_base.lbl_exposure_info.set_text(f"Exposure: ISO {target_tab_or_base.iso} | {target_tab_or_base.shutter}")
            
            self.refresh_preview_image(target_tab_or_base)
            pageNum = self.notebook.page_num(target_tab_or_base.widget_box)
            self.notebook.set_current_page(pageNum)
            
            # Reset IT8 mask active status and tab label/table values
            target_tab_or_base.it8_mask_active = False
            target_tab_or_base.btn_layer_it8.set_label("Layer IT8 Mask")
            target_tab_or_base.lbl_tab.set_markup(target_tab_or_base.label_text)
            target_tab_or_base.it8_store.clear()
        else:
            if getattr(self, 'base_img_obj', None) is not None:
                try:
                    self.base_img_obj.discard()
                except Exception:
                    pass
            self.base_img_obj = img_obj
            self.arr_raw_base = arr_raw
            self.arr_cc_base = arr_cc
            self.base_filepaths = filepaths
            self.current_pixbuf_base = pixbuf
            self.base_stack.set_visible_child_name("preview")
            
            self.base_iso = exposure_info["iso"]
            self.base_shutter = exposure_info["shutter"]
            self.lbl_exposure_info_base.set_text(f"Exposure: ISO {self.base_iso} | {self.base_shutter}")
            
            self.refresh_preview_image("base")
            pageNum = self.notebook.page_num(self.base_box)
            self.notebook.set_current_page(pageNum)

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
    def refresh_preview_image(self, target_tab_or_base):
        if isinstance(target_tab_or_base, TargetTabState):
            t = target_tab_or_base
            if not t.current_pixbuf:
                return
            alloc = t.stack.get_allocation()
            max_w = max(100, alloc.width - 30)
            max_h = max(100, alloc.height - 30)
            w = t.current_pixbuf.get_width()
            h = t.current_pixbuf.get_height()
            scale = min(max_w / w, max_h / h)
            new_w = max(1, int(w * scale))
            new_h = max(1, int(h * scale))
            t.scaled_pixbuf = t.current_pixbuf.scale_simple(
                new_w, new_h, GdkPixbuf.InterpType.BILINEAR
            )
            t.image_view.queue_draw()
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
        for tab in self.target_tabs:
            if tab.current_pixbuf:
                self.refresh_preview_image(tab)
        if self.current_pixbuf_base:
            self.refresh_preview_image("base")

    def update_pixbuf_from_arr(self, target_tab_or_base):
        if isinstance(target_tab_or_base, TargetTabState):
            t = target_tab_or_base
            arr_cc = t.arr_cc
            if arr_cc is None:
                t.current_pixbuf = None
                return
            h, w, c = arr_cc.shape
            arr_8bit = np.clip(arr_cc / 256.0, 0, 255).astype(np.uint8)
            glib_bytes = GLib.Bytes.new(arr_8bit.tobytes())
            t.current_pixbuf = GdkPixbuf.Pixbuf.new_from_bytes(
                glib_bytes, GdkPixbuf.Colorspace.RGB, False, 8, w, h, w * 3
            )
        else:
            arr_cc = self.arr_cc_base
            if arr_cc is None:
                self.current_pixbuf_base = None
                return
            h, w, c = arr_cc.shape
            arr_8bit = np.clip(arr_cc / 256.0, 0, 255).astype(np.uint8)
            glib_bytes = GLib.Bytes.new(arr_8bit.tobytes())
            self.current_pixbuf_base = GdkPixbuf.Pixbuf.new_from_bytes(
                glib_bytes, GdkPixbuf.Colorspace.RGB, False, 8, w, h, w * 3
            )

    def rotate_active_tab(self):
        active_target = self.get_active_target_tab()
        if active_target:
            if active_target.arr_cc is not None:
                active_target.arr_raw = np.rot90(active_target.arr_raw, k=-1)
                active_target.arr_cc = np.rot90(active_target.arr_cc, k=-1)
                active_target.normalized_selection = None
                self.update_pixbuf_from_arr(active_target)
                self.refresh_preview_image(active_target)
                self.update_histograms()
                self.update_toolbar_sensitivities()
        else:
            if self.arr_cc_base is not None:
                self.arr_raw_base = np.rot90(self.arr_raw_base, k=-1)
                self.arr_cc_base = np.rot90(self.arr_cc_base, k=-1)
                self.normalized_selection_base = None
                self.update_pixbuf_from_arr("base")
                self.refresh_preview_image("base")
                self.update_histograms()
                self.update_toolbar_sensitivities()

    def hflip_active_tab(self):
        active_target = self.get_active_target_tab()
        if active_target:
            if active_target.arr_cc is not None:
                active_target.arr_raw = np.fliplr(active_target.arr_raw)
                active_target.arr_cc = np.fliplr(active_target.arr_cc)
                active_target.normalized_selection = None
                self.update_pixbuf_from_arr(active_target)
                self.refresh_preview_image(active_target)
                self.update_histograms()
                self.update_toolbar_sensitivities()
        else:
            if self.arr_cc_base is not None:
                self.arr_raw_base = np.fliplr(self.arr_raw_base)
                self.arr_cc_base = np.fliplr(self.arr_cc_base)
                self.normalized_selection_base = None
                self.update_pixbuf_from_arr("base")
                self.refresh_preview_image("base")
                self.update_histograms()
                self.update_toolbar_sensitivities()

    def vflip_active_tab(self):
        active_target = self.get_active_target_tab()
        if active_target:
            if active_target.arr_cc is not None:
                active_target.arr_raw = np.flipud(active_target.arr_raw)
                active_target.arr_cc = np.flipud(active_target.arr_cc)
                active_target.normalized_selection = None
                self.update_pixbuf_from_arr(active_target)
                self.refresh_preview_image(active_target)
                self.update_histograms()
                self.update_toolbar_sensitivities()
        else:
            if self.arr_cc_base is not None:
                self.arr_raw_base = np.flipud(self.arr_raw_base)
                self.arr_cc_base = np.flipud(self.arr_cc_base)
                self.normalized_selection_base = None
                self.update_pixbuf_from_arr("base")
                self.refresh_preview_image("base")
                self.update_histograms()
                self.update_toolbar_sensitivities()

    def crop_active_tab(self):
        active_target = self.get_active_target_tab()
        if active_target:
            if active_target.arr_cc is not None and active_target.normalized_selection is not None:
                nx1, ny1, nx2, ny2 = active_target.normalized_selection
                h, w, _ = active_target.arr_cc.shape
                x1, x2 = int(nx1 * w), int(nx2 * w)
                y1, y2 = int(ny1 * h), int(ny2 * h)
                if x2 > x1 and y2 > y1:
                    active_target.arr_raw = active_target.arr_raw[y1:y2, x1:x2]
                    active_target.arr_cc = active_target.arr_cc[y1:y2, x1:x2]
                    active_target.normalized_selection = None
                    self.update_pixbuf_from_arr(active_target)
                    self.refresh_preview_image(active_target)
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
                    self.update_pixbuf_from_arr("base")
                    self.refresh_preview_image("base")
                    self.update_histograms()
                    self.update_toolbar_sensitivities()

    def update_toolbar_sensitivities(self):
        active_target = self.get_active_target_tab()
        has_target = active_target.arr_cc is not None if active_target else False
        has_target_selection = active_target.normalized_selection is not None if active_target else False

        if active_target:
            pass  # Target tab buttons are always enabled

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
        if hasattr(self, 'btn_read_base'):
            self.btn_read_base.set_sensitive(has_base)

    def get_it8_boxes_for_tab(self, tab_state):
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
            sx = cx + (bx - cx) * tab_state.it8_scale + tab_state.it8_dx
            sy = cy + (by - cy) * tab_state.it8_scale + tab_state.it8_dy
            sw = w_box_base * tab_state.it8_scale
            sh = h_box_base * tab_state.it8_scale
            scaled_boxes[patch] = (sx, sy, sw, sh)

        return scaled_boxes

    def get_it8_boxes(self):
        active_target = self.get_active_target_tab()
        if not active_target:
            return {}
        return self.get_it8_boxes_for_tab(active_target)

    def on_layer_it8_clicked(self, widget):
        active_target = self.get_active_target_tab()
        if not active_target:
            return
        
        active_target.it8_mask_active = not active_target.it8_mask_active
        if active_target.it8_mask_active:
            active_target.btn_layer_it8.set_label("Remove IT8 Mask")
            active_target.lbl_tab.set_markup(f"<b>{active_target.label_text} [Masked]</b>")
            self.status_lbl.set_text("Status: IT8 mask active. Use Arrow keys to move, +/- to scale.")
        else:
            active_target.btn_layer_it8.set_label("Layer IT8 Mask")
            active_target.lbl_tab.set_markup(active_target.label_text)
            self.status_lbl.set_text("Status: IT8 mask removed.")
            active_target.it8_store.clear()
        
        active_target.image_view.queue_draw()
        self.update_toolbar_sensitivities()

    def read_it8_values(self):
        active_target = self.get_active_target_tab()
        if not active_target or active_target.arr_cc is None:
            return
        
        boxes = self.get_it8_boxes_for_tab(active_target)
        h, w, _ = active_target.arr_cc.shape
        
        active_target.it8_store.clear()
        
        results = []
        # Print header matching read_it8.py output format
        print(f"\n=== IT8 Patch Measurements for {active_target.label_text} (Crosstalk Corrected & Linear) ===")
        print("patch r g b r_std g_std b_std")
        for patch, (bx, by, bw, bh) in sorted(boxes.items()):
            px1, py1 = int(bx * w), int(by * h)
            px2, py2 = int((bx + bw) * w), int((by + bh) * h)
            
            px1 = max(0, min(px1, w - 1))
            px2 = max(0, min(px2, w))
            py1 = max(0, min(py1, h - 1))
            py2 = max(0, min(py2, h))
            
            patch_img = active_target.arr_cc[py1:py2, px1:px2]
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
            
            active_target.it8_store.append([patch, float(r), float(g), float(b), float(r_std), float(g_std), float(b_std)])
            
            val_str = f"{patch} {r:.2f} {g:.2f} {b:.2f} {r_std:.2f} {g_std:.2f} {b_std:.2f}"
            results.append(val_str)
            print(val_str)
        print("=============================================================")
        active_target.lbl_tab.set_markup(f"<span foreground='#44ff44'><b>{active_target.label_text}</b></span>")

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
        active_target = self.get_active_target_tab()
        if not active_target or not active_target.it8_mask_active or active_target.arr_cc is None:
            return False

        keyval = event.keyval
        step_translate = 0.002
        step_scale = 0.005

        if keyval == Gdk.KEY_Up:
            active_target.it8_dy -= step_translate
        elif keyval == Gdk.KEY_Down:
            active_target.it8_dy += step_translate
        elif keyval == Gdk.KEY_Left:
            active_target.it8_dx -= step_translate
        elif keyval == Gdk.KEY_Right:
            active_target.it8_dx += step_translate
        elif keyval in (Gdk.KEY_plus, Gdk.KEY_equal, Gdk.KEY_KP_Add):
            active_target.it8_scale += step_scale
        elif keyval in (Gdk.KEY_minus, Gdk.KEY_underscore, Gdk.KEY_KP_Subtract):
            active_target.it8_scale -= step_scale
        else:
            return False

        active_target.image_view.queue_draw()
        return True

    def on_destroy(self, widget):
        for tab in self.target_tabs:
            if getattr(tab, 'img_obj', None) is not None:
                try:
                    tab.img_obj.discard()
                except Exception:
                    pass
        if getattr(self, 'base_img_obj', None) is not None:
            try:
                self.base_img_obj.discard()
            except Exception:
                pass
        if self.camera_session:
            try:
                self.camera_session.close()
            except Exception:
                pass
        Gtk.main_quit()


if __name__ == "__main__":
    win = FilmProfilingAppWindow()
    Gtk.main()
