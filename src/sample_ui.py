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
            .capture-btn:disabled {
                background: #555555;
                color: #888888;
            }
            .sidebar {
                background-color: #1e1e1e;
                border-right: 1px solid #333333;
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
        self.cc_checkbox.connect("toggled", self.on_cc_toggled)
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
        # RIGHT PANEL: Preview Canvas
        # =====================================================================
        self.preview_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        self.preview_box.get_style_context().add_class("preview-container")
        main_box.pack_start(self.preview_box, True, True, 0)

        # Initial Placeholder Label
        self.placeholder_label = Gtk.Label()
        self.placeholder_label.set_markup("<span size='large' foreground='#666666'>No Image Captured\n\nConfigure settings and click CAPTURE to display preview.</span>")
        self.placeholder_label.set_justify(Gtk.Justification.CENTER)
        self.preview_box.pack_start(self.placeholder_label, True, True, 0)

        # Image canvas widget (initially hidden)
        self.image_widget = Gtk.Image()
        self.image_widget.set_no_show_all(True)
        self.preview_box.pack_start(self.image_widget, True, True, 0)

        # Results metrics & Histogram drawing box
        self.results_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        self.results_box.set_no_show_all(True)
        self.preview_box.pack_start(self.results_box, False, False, 0)

        # Dynamic Range display label
        self.dr_label = Gtk.Label()
        self.dr_label.set_use_markup(True)
        self.dr_label.set_xalign(0.0)
        self.dr_label.get_style_context().add_class("meta-label")
        self.results_box.pack_start(self.dr_label, False, False, 0)

        # Histogram Drawing Area
        self.histogram_draw = Gtk.DrawingArea()
        self.histogram_draw.set_size_request(-1, 150) # 150px height
        self.histogram_draw.connect("draw", self.on_draw_histogram)
        self.results_box.pack_start(self.histogram_draw, False, False, 0)

        # Initialize histogram normalization variables
        self.hist_r_norm = None
        self.hist_g_norm = None
        self.hist_b_norm = None

        # Keep reference to the full-size decoded pixbuf to resize on window changes
        self.current_pixbuf = None
        self.connect("size-allocate", self.on_window_resized)

        # Crosstalk Correction Profile data
        self.last_captured_image = None
        self.correction_matrix = None

        self.show_all()

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
                    img = negicc_station.capture(type=0, shutter_num=num, shutter_den=den) # Single-shot
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
            if cc_active and self.correction_matrix is not None:
                matrix = [val for row in self.correction_matrix for val in row]

            # Take the final shot
            GLib.idle_add(self.status_label.set_text, f"Status: Capturing final image at {final_shutter_str}...")
            shutter_num, shutter_den = parse_shutter_speed(final_shutter_str)

            t_cap_start = time.time()
            img = negicc_station.capture(type=mode_id, shutter_num=shutter_num, shutter_den=shutter_den)
            self.last_captured_image = img
            t_cap_duration = time.time() - t_cap_start

            # Fetch metadata
            iso = img.iso
            shutter_sec = img.shutter_speed
            paths = img.filepaths

            # Convert to half-size numpy array for fast screen preview
            t_conv_start = time.time()
            arr = img.to_numpy(half=True, crosstalk_matrix=matrix)
            t_conv_duration = time.time() - t_conv_start

            # Calculate dynamic range and histogram from final image
            avg_dr, (dr_r, dr_g, dr_b) = auto_exposure.calculate_dynamic_range(arr)

            bins = 256
            hist_r, _ = np.histogram(arr[:, :, 0], bins=bins, range=(0, 65535))
            hist_g, _ = np.histogram(arr[:, :, 1], bins=bins, range=(0, 65535))
            hist_b, _ = np.histogram(arr[:, :, 2], bins=bins, range=(0, 65535))
            max_val = max(hist_r.max(), hist_g.max(), hist_b.max(), 1)

            hist_r_norm = hist_r / max_val
            hist_g_norm = hist_g / max_val
            hist_b_norm = hist_b / max_val

            height, width, channels = arr.shape
            arr_8bit = (arr >> 8).astype(np.uint8)
            raw_bytes = arr_8bit.tobytes()

            # Schedule UI updates back onto the GTK main thread safely
            GLib.idle_add(
                self.update_ui_success_with_metrics,
                raw_bytes, width, height, iso, shutter_sec, paths,
                t_cap_duration, t_conv_duration,
                dr_r, dr_g, dr_b, avg_dr,
                hist_r_norm, hist_g_norm, hist_b_norm
            )
        except Exception as e:
            GLib.idle_add(self.update_ui_failure, str(e))

    def update_ui_success(self, raw_bytes, width, height, iso, shutter_sec, paths, t_cap_duration, t_conv_duration):
        # Stop spinner and enable UI controls
        self.spinner.stop()
        self.capture_button.set_sensitive(True)
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

        # Swap placeholder label for image widget
        self.placeholder_label.hide()
        self.image_widget.show()

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

    def update_ui_success_with_metrics(self, raw_bytes, width, height, iso, shutter_sec, paths, t_cap_duration, t_conv_duration, dr_r, dr_g, dr_b, avg_dr, hist_r_norm, hist_g_norm, hist_b_norm):
        self.update_ui_success(raw_bytes, width, height, iso, shutter_sec, paths, t_cap_duration, t_conv_duration)

        # Display dynamic range metrics
        self.dr_label.set_markup(
            f"<b>Final Dynamic Range:</b>  R: {dr_r:.1f} | G: {dr_g:.1f} | B: {dr_b:.1f} | <b>Avg: {avg_dr:.1f}</b>"
        )

        # Update histogram data
        self.hist_r_norm = hist_r_norm
        self.hist_g_norm = hist_g_norm
        self.hist_b_norm = hist_b_norm

        # Show results box and redraw histogram
        self.results_box.show_all()
        self.histogram_draw.queue_draw()

    def update_ui_failure(self, error_msg):
        self.spinner.stop()
        self.capture_button.set_sensitive(True)
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

    def on_draw_histogram(self, widget, cr):
        alloc = widget.get_allocation()
        w = alloc.width
        h = alloc.height

        # Draw background
        cr.set_source_rgb(0.08, 0.08, 0.08)
        cr.paint()

        if self.hist_r_norm is None or self.hist_g_norm is None or self.hist_b_norm is None:
            return

        # Draw vertical grid lines
        cr.set_source_rgba(0.2, 0.2, 0.2, 0.5)
        cr.set_line_width(1.0)
        for pct in [0.25, 0.5, 0.75]:
            x = w * pct
            cr.move_to(x, 0)
            cr.line_to(x, h)
            cr.stroke()

        bins = len(self.hist_r_norm)
        def draw_channel(hist_norm, r, g, b):
            # Fill under the curve
            cr.set_source_rgba(r, g, b, 0.15)
            cr.move_to(0, h)
            for i in range(bins):
                x = (i / (bins - 1)) * w
                y = h - (hist_norm[i] * (h - 10))
                cr.line_to(x, y)
            cr.line_to(w, h)
            cr.close_path()
            cr.fill()

            # Stroke outline
            cr.set_source_rgba(r, g, b, 0.8)
            cr.set_line_width(1.5)
            cr.move_to(0, h - (hist_norm[0] * (h - 10)))
            for i in range(1, bins):
                x = (i / (bins - 1)) * w
                y = h - (hist_norm[i] * (h - 10))
                cr.line_to(x, y)
            cr.stroke()

        # Draw channels
        draw_channel(self.hist_r_norm, 0.9, 0.2, 0.2)
        draw_channel(self.hist_g_norm, 0.2, 0.8, 0.2)
        draw_channel(self.hist_b_norm, 0.2, 0.4, 0.9)

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

        scaled_pixbuf = self.current_pixbuf.scale_simple(new_w, new_h, GdkPixbuf.InterpType.BILINEAR)
        self.image_widget.set_from_pixbuf(scaled_pixbuf)

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
                if self.cc_checkbox.get_active() and self.correction_matrix is not None:
                    # Flatten the 3x3 matrix to 9 elements
                    matrix = [val for row in self.correction_matrix for val in row]

                t_conv_start = time.time()
                arr = self.last_captured_image.to_numpy(half=True, crosstalk_matrix=matrix)
                t_conv_duration = time.time() - t_conv_start

                # Recalculate dynamic range and histogram
                avg_dr, (dr_r, dr_g, dr_b) = auto_exposure.calculate_dynamic_range(arr)

                bins = 256
                hist_r, _ = np.histogram(arr[:, :, 0], bins=bins, range=(0, 65535))
                hist_g, _ = np.histogram(arr[:, :, 1], bins=bins, range=(0, 65535))
                hist_b, _ = np.histogram(arr[:, :, 2], bins=bins, range=(0, 65535))
                max_val = max(hist_r.max(), hist_g.max(), hist_b.max(), 1)

                hist_r_norm = hist_r / max_val
                hist_g_norm = hist_g / max_val
                hist_b_norm = hist_b / max_val

                height, width, channels = arr.shape
                arr_8bit = (arr >> 8).astype(np.uint8)
                raw_bytes = arr_8bit.tobytes()

                GLib.idle_add(
                    self.update_ui_success_with_metrics_no_cap,
                    raw_bytes, width, height, t_conv_duration,
                    dr_r, dr_g, dr_b, avg_dr,
                    hist_r_norm, hist_g_norm, hist_b_norm
                )
            except Exception as e:
                GLib.idle_add(self.update_ui_failure, str(e))

        thread = threading.Thread(target=run)
        thread.daemon = True
        thread.start()

    def update_ui_success_with_metrics_no_cap(self, raw_bytes, width, height, t_conv_duration, dr_r, dr_g, dr_b, avg_dr, hist_r_norm, hist_g_norm, hist_b_norm):
        # Stop spinner and enable UI controls
        self.spinner.stop()
        self.capture_button.set_sensitive(True)
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
        self.placeholder_label.hide()
        self.image_widget.show()
        self.refresh_preview_image()

        # Display dynamic range metrics
        self.dr_label.set_markup(
            f"<b>Final Dynamic Range:</b>  R: {dr_r:.1f} | G: {dr_g:.1f} | B: {dr_b:.1f} | <b>Avg: {avg_dr:.1f}</b>"
        )

        # Update histogram data
        self.hist_r_norm = hist_r_norm
        self.hist_g_norm = hist_g_norm
        self.hist_b_norm = hist_b_norm
        self.results_box.show_all()
        self.histogram_draw.queue_draw()

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
                    self.cc_checkbox.set_active(True)

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
        self.capture_button.set_sensitive(True)
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
