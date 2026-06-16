#!/usr/bin/env python3
"""
GTK3-based GUI application for the Negative Film Scanning Station.
Provides a modern dark theme interface to trigger captures, select shutter speeds,
display 16-bit linear C++ converted previews, and show metadata.
"""

import os
import sys
import threading
import numpy as np
import gi

# Require GTK3
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk, Gdk, GdkPixbuf, GLib

import negicc_station

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
        self.connect("destroy", Gtk.main_quit)

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

        # Action: Capture Button & Spinner
        self.capture_button = Gtk.Button(label="CAPTURE IMAGE")
        self.capture_button.get_style_context().add_class("capture-btn")
        self.capture_button.connect("clicked", self.on_capture_clicked)
        sidebar_box.pack_start(self.capture_button, False, False, 10)

        # Spinner & Status Info
        status_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=10)
        self.spinner = Gtk.Spinner()
        status_box.pack_start(self.spinner, False, False, 0)
        self.status_label = Gtk.Label(label="Status: Idle")
        self.status_label.set_xalign(0.0)
        self.status_label.set_yalign(0.5)
        status_box.pack_start(self.status_label, True, True, 0)
        sidebar_box.pack_start(status_box, False, False, 0)

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

        # Keep reference to the full-size decoded pixbuf to resize on window changes
        self.current_pixbuf = None
        self.connect("size-allocate", self.on_window_resized)

        self.show_all()

    def on_capture_clicked(self, widget):
        # Disable controls during capture thread run
        self.capture_button.set_sensitive(False)
        self.mode_combo.set_sensitive(False)
        self.shutter_combo.set_sensitive(False)
        self.spinner.start()
        self.status_label.set_text("Status: Tethering and capturing...")

        # Run capture in background thread to keep GTK UI responsive
        capture_thread = threading.Thread(target=self.background_capture_and_convert)
        capture_thread.daemon = True
        capture_thread.start()

    def background_capture_and_convert(self):
        try:
            # Read UI values
            mode_id = int(self.mode_combo.get_active_id())
            shutter_str = self.shutter_combo.get_active_text()
            shutter_num, shutter_den = parse_shutter_speed(shutter_str)

            # Trigger C++ capture via bindings
            img = negicc_station.capture(type=mode_id, shutter_num=shutter_num, shutter_den=shutter_den)
            
            # Fetch metadata
            iso = img.iso
            shutter_sec = img.shutter_speed
            paths = img.filepaths

            # Convert to half-size numpy array for fast screen preview
            arr = img.to_numpy(half=True)
            height, width, channels = arr.shape

            # Convert uint16 linear array to uint8 RGB for GTK GdkPixbuf
            arr_8bit = (arr >> 8).astype(np.uint8)
            raw_bytes = arr_8bit.tobytes()

            # Discard temporary RAW files immediately as requested
            img.discard()

            # Schedule UI updates back onto the GTK main thread safely
            GLib.idle_add(self.update_ui_success, raw_bytes, width, height, iso, shutter_sec, paths)
        except Exception as e:
            GLib.idle_add(self.update_ui_failure, str(e))

    def update_ui_success(self, raw_bytes, width, height, iso, shutter_sec, paths):
        # Stop spinner and enable UI controls
        self.spinner.stop()
        self.capture_button.set_sensitive(True)
        self.mode_combo.set_sensitive(True)
        self.shutter_combo.set_sensitive(True)
        self.status_label.set_text("Status: Success!")

        # Update metadata panel
        self.iso_label.set_text(f"ISO: {iso}")
        self.shutter_label.set_text(f"Shutter Speed: {shutter_sec:.4f}s")
        self.size_label.set_text(f"Dimensions: {width} x {height} (Half-size)")
        self.files_label.set_text(f"RAW Filepath(s) [Deleted]:\n" + "\n".join(paths))

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

    def update_ui_failure(self, error_msg):
        self.spinner.stop()
        self.capture_button.set_sensitive(True)
        self.mode_combo.set_sensitive(True)
        self.shutter_combo.set_sensitive(True)
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
