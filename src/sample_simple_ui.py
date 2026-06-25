#!/usr/bin/env python3
"""
Simple GTK3-based GUI application for the Negative Film Scanning Station.
Provides a modern dark theme interface to connect to the camera,
select shutter speed, run Auto Exposure, and capture linear previews.
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

# Ensure project src directory is in python path
src_dir = os.path.dirname(os.path.abspath(__file__))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# Preload the Sony CrSDK shared library from the virtual environment if present
project_dir = os.path.dirname(src_dir)
lib_path = os.path.join(project_dir, 'venv/bin/libCr_Core.so')
if os.path.exists(lib_path):
    import ctypes
    try:
        ctypes.CDLL(lib_path)
    except Exception as e:
        print(f"Warning: Failed to preload CrSDK library: {e}")

import negicc_station
import auto_exposure

# Shutter speeds supported
SHUTTER_SPEEDS = auto_exposure.SHUTTER_SPEEDS

def parse_shutter_speed(shutter_str):
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
    # Calculate 256-bin normalized histograms
    bins = 256
    hist_r, _ = np.histogram(arr[:, :, 0], bins=bins, range=(0, 65535))
    hist_g, _ = np.histogram(arr[:, :, 1], bins=bins, range=(0, 65535))
    hist_b, _ = np.histogram(arr[:, :, 2], bins=bins, range=(0, 65535))
    max_val = max(hist_r.max(), hist_g.max(), hist_b.max(), 1)
    
    hist_r_norm = hist_r / max_val
    hist_g_norm = hist_g / max_val
    hist_b_norm = hist_b / max_val

    # Exclude 5% borders for percentiles
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

    return (hist_r_norm, hist_g_norm, hist_b_norm), (p2_r, p2_g, p2_b), (p98_r, p98_g, p98_b), (dr_r, dr_g, dr_b, avg_dr)

class SimpleScanningAppWindow(Gtk.Window):
    def __init__(self):
        super().__init__(title="Sony Simple Scanner")
        self.set_default_size(900, 600)
        self.connect("destroy", self.on_destroy)

        # Force GTK dark theme
        settings = Gtk.Settings.get_default()
        settings.set_property("gtk-application-prefer-dark-theme", True)

        # Custom CSS styling
        css_provider = Gtk.CssProvider()
        css_provider.load_from_data(b"""
            .capture-btn {
                background-image: linear-gradient(to bottom, #2ea44f, #2c974b);
                color: white;
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

        # Main horizontal box split
        main_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=0)
        self.add(main_box)

        # Sidebar Panel
        sidebar_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=15)
        sidebar_box.get_style_context().add_class("sidebar")
        sidebar_box.set_size_request(280, -1)
        main_box.pack_start(sidebar_box, False, False, 0)

        # Header Title
        title_label = Gtk.Label()
        title_label.set_markup("<span size='large' weight='bold'>Simple Scanner</span>")
        title_label.set_xalign(0.0)
        sidebar_box.pack_start(title_label, False, False, 5)

        # Camera Status Row
        camera_status_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        self.camera_status_label = Gtk.Label()
        self.camera_status_label.set_markup("<span><span foreground='#e6a23c'>●</span> Camera: Connecting...</span>")
        self.camera_status_label.set_xalign(0.0)
        camera_status_box.pack_start(self.camera_status_label, True, True, 0)
        
        self.connect_btn = Gtk.Button(label="Connect")
        self.connect_btn.connect("clicked", lambda w: self.connect_camera(manual=True))
        self.connect_btn.set_sensitive(False)
        camera_status_box.pack_start(self.connect_btn, False, False, 0)
        sidebar_box.pack_start(camera_status_box, False, False, 2)

        # Settings Section
        settings_frame = Gtk.Frame(label="Capture Settings")
        settings_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        settings_box.set_border_width(10)
        settings_frame.add(settings_box)
        sidebar_box.pack_start(settings_frame, False, False, 5)

        # Shutter Dropdown
        shutter_label = Gtk.Label(label="Shutter Speed:")
        shutter_label.set_xalign(0.0)
        settings_box.pack_start(shutter_label, False, False, 0)

        self.shutter_combo = Gtk.ComboBoxText()
        for speed in SHUTTER_SPEEDS:
            self.shutter_combo.append(speed, speed)
        self.shutter_combo.set_active(SHUTTER_SPEEDS.index("1/8s"))
        settings_box.pack_start(self.shutter_combo, False, False, 0)

        # Auto Exposure Checkbox
        self.ae_checkbox = Gtk.CheckButton(label="Auto Exposure")
        self.ae_checkbox.connect("toggled", self.on_ae_toggled)
        settings_box.pack_start(self.ae_checkbox, False, False, 5)

        # Action: Capture Button
        self.capture_button = Gtk.Button(label="CAPTURE IMAGE")
        self.capture_button.get_style_context().add_class("capture-btn")
        self.capture_button.set_sensitive(False)
        self.capture_button.connect("clicked", self.on_capture_clicked)
        sidebar_box.pack_start(self.capture_button, False, False, 10)

        # Status & Spinner Info
        status_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=10)
        self.spinner = Gtk.Spinner()
        status_box.pack_start(self.spinner, False, False, 0)
        self.status_label = Gtk.Label(label="Status: Idle")
        self.status_label.set_xalign(0.0)
        status_box.pack_start(self.status_label, True, True, 0)
        sidebar_box.pack_start(status_box, False, False, 0)

        # Right Panel: Preview Pane
        self.preview_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        self.preview_box.get_style_context().add_class("preview-container")
        main_box.pack_start(self.preview_box, True, True, 0)

        # Preview Placeholder
        self.placeholder_label = Gtk.Label()
        self.placeholder_label.set_markup("<span size='large' foreground='#666666'>No Image Captured\n\nConfigure settings and click CAPTURE.</span>")
        self.placeholder_label.set_justify(Gtk.Justification.CENTER)
        self.preview_box.pack_start(self.placeholder_label, True, True, 0)

        # Image widget
        self.image_widget = Gtk.Image()
        self.image_widget.set_no_show_all(True)
        self.preview_box.pack_start(self.image_widget, True, True, 0)

        # Info Display Box
        self.results_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        self.results_box.set_no_show_all(True)
        self.preview_box.pack_start(self.results_box, False, False, 0)

        # Metrics Label
        self.dr_label = Gtk.Label()
        self.dr_label.set_use_markup(True)
        self.dr_label.set_xalign(0.0)
        self.dr_label.get_style_context().add_class("meta-label")
        self.results_box.pack_start(self.dr_label, False, False, 0)

        self.current_pixbuf = None
        self.connect("size-allocate", self.on_window_resized)
        
        self.last_captured_image = None
        
        self.show_all()

        # Camera session and auto-connect
        self.camera_session = None
        self.is_connected = False
        self.is_connecting = False
        self.was_physically_connected = False
        self.connect_camera()
        GLib.timeout_add_seconds(2, self.poll_camera_connection)

    def poll_camera_connection(self):
        currently_physically_connected = negicc_station.is_camera_connected()
        
        if self.is_connected:
            if not currently_physically_connected:
                self.is_connected = False
                self.update_connection_ui(False, "Camera unplugged.")
        elif not self.is_connecting:
            # Only trigger auto-connect on rising edge of physical connection
            if currently_physically_connected and not self.was_physically_connected:
                self.connect_camera()
                
        self.was_physically_connected = currently_physically_connected
        return True

    def connect_camera(self, manual=False):
        if self.is_connecting or self.is_connected:
            return
        self.is_connecting = True
        self.capture_button.set_sensitive(False)
        self.connect_btn.set_sensitive(False)
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
            self.connect_btn.set_sensitive(False)
            self.status_label.set_text("Status: Camera connected, ready.")
        else:
            self.camera_status_label.set_markup("<span foreground='#ff4444'>●</span> <b>Camera: Disconnected</b>")
            self.capture_button.set_sensitive(False)
            self.connect_btn.set_sensitive(True)
            if error_msg:
                self.status_label.set_text(f"Status: Connection failed ({error_msg})")
            else:
                self.status_label.set_text("Status: Camera disconnected.")

    def on_capture_clicked(self, widget):
        shutter_str = self.shutter_combo.get_active_text()
        is_ae = self.ae_checkbox.get_active()

        # Disable controls during capture
        self.capture_button.set_sensitive(False)
        self.shutter_combo.set_sensitive(False)
        self.ae_checkbox.set_sensitive(False)
        self.spinner.start()
        self.status_label.set_text("Status: Capturing...")

        capture_thread = threading.Thread(
            target=self.background_capture_and_convert,
            args=(shutter_str, is_ae)
        )
        capture_thread.daemon = True
        capture_thread.start()

    def background_capture_and_convert(self, start_shutter_str, is_ae):
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
                def ae_local_capture(idx):
                    shutter_s = auto_exposure.SHUTTER_SPEEDS[idx]
                    GLib.idle_add(self.status_label.set_text, f"AE Search: Capturing {shutter_s}...")
                    num, den = parse_shutter_speed(shutter_s)
                    img = self.camera_session.capture(type=0, shutter_num=num, shutter_den=den)
                    arr = img.to_numpy(half=True)
                    img.discard()
                    return arr

                GLib.idle_add(self.status_label.set_text, "AE Search: Finding optimal exposure...")
                opt_shutter, steps = auto_exposure.run_auto_exposure(
                    start_shutter_str=start_shutter_str,
                    capture_func=ae_local_capture,
                    progress_callback=None
                )
                final_shutter_str = opt_shutter
                GLib.idle_add(self.set_shutter_speed_active, opt_shutter)

            GLib.idle_add(self.status_label.set_text, f"Status: Capturing final image at {final_shutter_str}...")
            shutter_num, shutter_den = parse_shutter_speed(final_shutter_str)

            t_start = time.time()
            img = self.camera_session.capture(type=0, shutter_num=shutter_num, shutter_den=shutter_den)
            self.last_captured_image = img
            t_cap_duration = time.time() - t_start

            # Convert to numpy for preview display
            t_conv_start = time.time()
            arr = img.to_numpy(half=True)
            t_conv_duration = time.time() - t_conv_start

            # Calculate dynamic range metrics
            _, p2_vals, p98_vals, dr_metrics = compute_hist_and_percentiles(arr)
            dr_r, dr_g, dr_b, avg_dr = dr_metrics

            height, width, channels = arr.shape
            arr_8bit = (arr >> 8).astype(np.uint8)
            raw_bytes = arr_8bit.tobytes()

            GLib.idle_add(
                self.update_ui_success,
                raw_bytes, width, height, t_cap_duration, t_conv_duration,
                dr_r, dr_g, dr_b, avg_dr, p2_vals, p98_vals
            )
        except Exception as e:
            GLib.idle_add(self.update_ui_failure, str(e))

    def update_ui_success(self, raw_bytes, width, height, t_cap_duration, t_conv_duration, dr_r, dr_g, dr_b, avg_dr, p2_vals, p98_vals):
        self.spinner.stop()
        self.capture_button.set_sensitive(self.is_connected)
        self.shutter_combo.set_sensitive(not self.ae_checkbox.get_active())
        self.ae_checkbox.set_sensitive(True)
        self.status_label.set_text("Status: Success!")

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

        # Display Metrics
        self.dr_label.set_markup(
            f"<b>Dynamic Range (p2-p98):</b>  R: {dr_r:.1f} | G: {dr_g:.1f} | B: {dr_b:.1f} | <b>Avg: {avg_dr:.1f}</b>\n"
            f"  Red [2%-98%]: [{int(p2_vals[0])} - {int(p98_vals[0])}]\n"
            f"  Green [2%-98%]: [{int(p2_vals[1])} - {int(p98_vals[1])}]\n"
            f"  Blue [2%-98%]: [{int(p2_vals[2])} - {int(p98_vals[2])}]\n"
            f"Capture: {t_cap_duration:.2f}s | Conversion: {t_conv_duration:.2f}s"
        )
        self.results_box.show_all()

    def update_ui_failure(self, error_msg):
        self.spinner.stop()
        self.capture_button.set_sensitive(self.is_connected)
        self.shutter_combo.set_sensitive(not self.ae_checkbox.get_active())
        self.ae_checkbox.set_sensitive(True)
        self.status_label.set_text("Status: Failed!")

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

    def set_shutter_speed_active(self, shutter_str):
        if shutter_str in SHUTTER_SPEEDS:
            self.shutter_combo.set_active(SHUTTER_SPEEDS.index(shutter_str))

    def refresh_preview_image(self):
        if not self.current_pixbuf:
            return
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
        if self.current_pixbuf:
            self.refresh_preview_image()

    def on_destroy(self, widget):
        # Start a watchdog thread to force exit in 1.5 seconds if the camera close hangs
        def watchdog():
            import time
            time.sleep(1.5)
            print("[Watchdog] Cleanup timeout reached. Forcing exit.", file=sys.stderr, flush=True)
            os._exit(0)
        threading.Thread(target=watchdog, daemon=True).start()

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
        os._exit(0)

if __name__ == "__main__":
    win = SimpleScanningAppWindow()
    Gtk.main()
