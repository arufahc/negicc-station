#!/usr/bin/env python3
"""
GTK3-based Crosstalk Correction Calibration application for negicc-station.
Guides the user through capturing 3 images (Red, Blue, Green)
to calculate and save the crosstalk correction matrix.
"""

import os
import sys
import threading
import numpy as np
import gi

# Require GTK3
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk, Gdk, GLib

# Ensure project root is in python path
project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_dir)
sys.path.insert(0, os.path.join(project_dir, 'src'))

# Preload the Sony CrSDK shared library from the virtual environment
lib_path = os.path.join(project_dir, 'venv/bin/libCr_Core.so')
if os.path.exists(lib_path):
    import ctypes
    ctypes.CDLL(lib_path)

import negicc_station
import auto_exposure
import crosstalk_calibration

def get_circle_stats(arr):
    H, W, C = arr.shape
    cy, cx = H / 2.0, W / 2.0
    r = min(H, W) / 6.0  # radius: diameter is 1/3 of the short side
    
    y, x = np.ogrid[:H, :W]
    mask = (y - cy)**2 + (x - cx)**2 <= r**2
    
    # Extract pixels in mask
    pixels = arr[mask]
    means = np.mean(pixels, axis=0)
    stds = np.std(pixels, axis=0)
    return means.tolist(), stds.tolist()

class CrosstalkAppWindow(Gtk.Window):
    def __init__(self):
        super().__init__(title="Sony Crosstalk Calibration Tool")
        self.set_default_size(1050, 680)
        self.connect("destroy", self.on_destroy)

        # Force GTK dark theme
        settings = Gtk.Settings.get_default()
        settings.set_property("gtk-application-prefer-dark-theme", True)

        # Apply CSS style provider
        css_provider = Gtk.CssProvider()
        css_provider.load_from_data(b"""
            .sidebar {
                background-color: #1e1e1e;
                border-right: 1px solid #333333;
                padding: 15px;
            }
            .main-content {
                background-color: #121212;
                padding: 20px;
            }
            .btn-red {
                background-image: linear-gradient(to bottom, #d93838, #b52b2b);
                color: white;
                font-weight: bold;
                border: 1px solid rgba(0,0,0,0.2);
                border-radius: 6px;
                padding: 10px;
            }
            .btn-red:hover {
                background-image: linear-gradient(to bottom, #f24e4e, #d93838);
            }
            .btn-red:disabled, .btn-red:insensitive {
                background-image: none;
                background-color: #444444;
                color: #888888;
                border-color: #222222;
            }
            .btn-blue {
                background-image: linear-gradient(to bottom, #2c75d9, #235eb3);
                color: white;
                font-weight: bold;
                border: 1px solid rgba(0,0,0,0.2);
                border-radius: 6px;
                padding: 10px;
            }
            .btn-blue:hover {
                background-image: linear-gradient(to bottom, #408df2, #2c75d9);
            }
            .btn-blue:disabled, .btn-blue:insensitive {
                background-image: none;
                background-color: #444444;
                color: #888888;
                border-color: #222222;
            }
            .btn-green {
                background-image: linear-gradient(to bottom, #2ea44f, #2c974b);
                color: white;
                font-weight: bold;
                border: 1px solid rgba(0,0,0,0.2);
                border-radius: 6px;
                padding: 10px;
            }
            .btn-green:hover {
                background-image: linear-gradient(to bottom, #30bc5a, #2ea44f);
            }
            .btn-green:disabled, .btn-green:insensitive {
                background-image: none;
                background-color: #444444;
                color: #888888;
                border-color: #222222;
            }
            .btn-save {
                background-image: linear-gradient(to bottom, #8a2be2, #6a1b9a);
                color: white;
                font-weight: bold;
                border: 1px solid rgba(0,0,0,0.2);
                border-radius: 6px;
                padding: 12px;
            }
            .btn-save:hover {
                background-image: linear-gradient(to bottom, #a052ff, #8a2be2);
            }
            .btn-save:disabled, .btn-save:insensitive {
                background-image: none;
                background-color: #444444;
                color: #888888;
                border-color: #222222;
            }
            .grid-header {
                font-weight: bold;
                color: #ffffff;
                background-color: #252525;
                padding: 10px;
                border-bottom: 2px solid #3d3d3d;
            }
            .grid-cell {
                padding: 10px;
                border-bottom: 1px solid #2d2d2d;
            }
            .matrix-label {
                font-family: monospace;
                font-size: 13px;
                color: #00ff66;
                background-color: #1a1a1a;
                padding: 12px;
                border: 1px solid #333333;
                border-radius: 6px;
            }
        """)
        Gtk.StyleContext.add_provider_for_screen(
            Gdk.Screen.get_default(),
            css_provider,
            Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION
        )

        # State Variables
        self.start_speed = "1/8s"
        self.camera_model = "Unknown"
        
        self.means_r = None
        self.stds_r = None
        self.speed_r = None
        
        self.means_b = None
        self.stds_b = None
        self.speed_b = None

        self.means_g = None
        self.stds_g = None
        self.speed_g = None

        self.M = None
        self.M_norm = None
        self.correction_matrix = None

        # Base Layout
        main_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=0)
        self.add(main_box)

        # =====================================================================
        # SIDEBAR PANEL
        # =====================================================================
        sidebar_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=15)
        sidebar_box.get_style_context().add_class("sidebar")
        sidebar_box.set_size_request(300, -1)
        main_box.pack_start(sidebar_box, False, False, 0)

        # Title
        title_label = Gtk.Label()
        title_label.set_markup("<span size='large' weight='bold'>Crosstalk Calibration</span>")
        title_label.set_xalign(0.0)
        sidebar_box.pack_start(title_label, False, False, 5)

        # Camera Connection Status Indicator
        self.camera_status_label = Gtk.Label()
        self.camera_status_label.set_markup("<span><span foreground='#e6a23c'>●</span> Camera: Connecting...</span>")
        self.camera_status_label.set_xalign(0.0)
        sidebar_box.pack_start(self.camera_status_label, False, False, 5)

        # Settings Section
        config_frame = Gtk.Frame(label="Calibration Settings")
        config_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        config_box.set_border_width(10)
        config_frame.add(config_box)
        sidebar_box.pack_start(config_frame, False, False, 5)

        shutter_label = Gtk.Label(label="Start Shutter Speed for AE:")
        shutter_label.set_xalign(0.0)
        config_box.pack_start(shutter_label, False, False, 0)

        self.shutter_combo = Gtk.ComboBoxText()
        for speed in auto_exposure.SHUTTER_SPEEDS:
            self.shutter_combo.append(speed, speed)
        self.shutter_combo.set_active(auto_exposure.SHUTTER_SPEEDS.index("1/8s"))
        config_box.pack_start(self.shutter_combo, False, False, 0)

        # Auto-Exposure Progress Listing
        ae_steps_frame = Gtk.Frame(label="Auto-Exposure Progress")
        ae_steps_scroll = Gtk.ScrolledWindow()
        ae_steps_scroll.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        ae_steps_scroll.set_min_content_height(180)
        self.ae_steps_listbox = Gtk.ListBox()
        self.ae_steps_listbox.set_selection_mode(Gtk.SelectionMode.NONE)
        ae_steps_scroll.add(self.ae_steps_listbox)
        ae_steps_frame.add(ae_steps_scroll)
        sidebar_box.pack_start(ae_steps_frame, False, False, 5)

        # Save Button
        self.save_button = Gtk.Button(label="SAVE PROFILE")
        self.save_button.get_style_context().add_class("btn-save")
        self.save_button.set_sensitive(False)
        self.save_button.connect("clicked", self.on_save_clicked)
        sidebar_box.pack_start(self.save_button, False, False, 5)

        # Load Button
        self.load_button = Gtk.Button(label="LOAD PROFILE")
        self.load_button.get_style_context().add_class("btn-save")
        self.load_button.connect("clicked", self.on_load_clicked)
        sidebar_box.pack_start(self.load_button, False, False, 5)

        # Status spinner & label
        status_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=10)
        self.spinner = Gtk.Spinner()
        status_box.pack_start(self.spinner, False, False, 0)
        self.status_label = Gtk.Label(label="Status: Ready")
        self.status_label.set_xalign(0.0)
        self.status_label.set_yalign(0.5)
        status_box.pack_start(self.status_label, True, True, 0)
        sidebar_box.pack_start(status_box, False, False, 0)

        # =====================================================================
        # RIGHT PANEL: Capture Controls & Matrices Display
        # =====================================================================
        main_content_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=20)
        main_content_box.get_style_context().add_class("main-content")
        main_box.pack_start(main_content_box, True, True, 0)

        # Capture Control buttons
        btn_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=10)
        main_content_box.pack_start(btn_box, False, False, 0)

        self.btn_capture_r = Gtk.Button(label="Step 1: Capture Red")
        self.btn_capture_r.get_style_context().add_class("btn-red")
        self.btn_capture_r.connect("clicked", self.on_capture_clicked, "R")
        btn_box.pack_start(self.btn_capture_r, True, True, 0)

        self.btn_capture_b = Gtk.Button(label="Step 2: Capture Blue")
        self.btn_capture_b.get_style_context().add_class("btn-blue")
        self.btn_capture_b.connect("clicked", self.on_capture_clicked, "B")
        btn_box.pack_start(self.btn_capture_b, True, True, 0)

        self.btn_capture_g = Gtk.Button(label="Step 3: Capture Green")
        self.btn_capture_g.get_style_context().add_class("btn-green")
        self.btn_capture_g.connect("clicked", self.on_capture_clicked, "G")
        btn_box.pack_start(self.btn_capture_g, True, True, 0)

        # Stats display grid
        self.grid = Gtk.Grid()
        self.grid.set_column_spacing(1)
        self.grid.set_row_spacing(1)
        main_content_box.pack_start(self.grid, False, False, 0)

        # Grid headers
        headers = ["Channel / State", "Shutter Speed", "Means (R, G, B)", "Std-Dev (R, G, B)"]
        for col_idx, text in enumerate(headers):
            lbl = Gtk.Label()
            lbl.set_markup(f"<b>{text}</b>")
            lbl.get_style_context().add_class("grid-header")
            lbl.set_xalign(0.0)
            self.grid.attach(lbl, col_idx, 0, 1, 1)

        # Value cell allocations
        self.val_r_speed = self.create_grid_cell()
        self.val_r_means = self.create_grid_cell()
        self.val_r_stds = self.create_grid_cell()

        self.val_b_speed = self.create_grid_cell()
        self.val_b_means = self.create_grid_cell()
        self.val_b_stds = self.create_grid_cell()

        self.val_g_speed = self.create_grid_cell()
        self.val_g_means = self.create_grid_cell()
        self.val_g_stds = self.create_grid_cell()

        # Map grid positions
        self.setup_row(1, "Red (Step 1)", self.val_r_speed, self.val_r_means, self.val_r_stds)
        self.setup_row(2, "Blue (Step 2)", self.val_b_speed, self.val_b_means, self.val_b_stds)
        self.setup_row(3, "Green (Step 3)", self.val_g_speed, self.val_g_means, self.val_g_stds)

        # Matrices output boxes
        matrix_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=20)
        main_content_box.pack_start(matrix_box, True, True, 0)

        # Normalized Matrix
        norm_frame = Gtk.Frame(label="Normalized Crosstalk Matrix (M_norm)")
        norm_vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=5)
        norm_vbox.set_border_width(10)
        self.lbl_m_norm = Gtk.Label(label="Matrix will display after Red, Blue, and Green steps are captured.")
        self.lbl_m_norm.get_style_context().add_class("matrix-label")
        self.lbl_m_norm.set_xalign(0.0)
        self.lbl_m_norm.set_selectable(True)
        norm_vbox.pack_start(self.lbl_m_norm, True, True, 0)
        norm_frame.add(norm_vbox)
        matrix_box.pack_start(norm_frame, True, True, 0)

        # Correction Matrix
        corr_frame = Gtk.Frame(label="Crosstalk Correction Matrix (M_norm^-1)")
        corr_vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=5)
        corr_vbox.set_border_width(10)
        self.lbl_m_corr = Gtk.Label(label="Matrix will display after Red, Blue, and Green steps are captured.")
        self.lbl_m_corr.get_style_context().add_class("matrix-label")
        self.lbl_m_corr.set_xalign(0.0)
        self.lbl_m_corr.set_selectable(True)
        corr_vbox.pack_start(self.lbl_m_corr, True, True, 0)
        corr_frame.add(corr_vbox)
        matrix_box.pack_start(corr_frame, True, True, 0)

        self.show_all()

        # Camera session and auto-connect
        self.camera_session = None
        self.is_connected = False
        self.is_connecting = False
        self.connect_camera()
        GLib.timeout_add_seconds(2, self.poll_camera_connection)

    def on_destroy(self, widget):
        if self.camera_session:
            try:
                self.camera_session.close()
            except Exception:
                pass
        Gtk.main_quit()

    def poll_camera_connection(self):
        if self.is_connected:
            # Check if camera was unplugged
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
            # Check if camera was plugged in
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
            self.status_label.set_text("Status: Camera connected, ready.")
        else:
            self.camera_status_label.set_markup("<span foreground='#ff4444'>●</span> <b>Camera: Disconnected</b>")
            self.set_controls_sensitive(False)
            if error_msg:
                self.status_label.set_text(f"Status: Connection failed ({error_msg})")
            else:
                self.status_label.set_text("Status: Camera disconnected.")

    def create_grid_cell(self):
        lbl = Gtk.Label(label="--")
        lbl.get_style_context().add_class("grid-cell")
        lbl.set_xalign(0.0)
        return lbl

    def setup_row(self, row_idx, title, speed_lbl, means_lbl, stds_lbl):
        t_lbl = Gtk.Label(label=title)
        t_lbl.get_style_context().add_class("grid-cell")
        t_lbl.set_xalign(0.0)
        self.grid.attach(t_lbl, 0, row_idx, 1, 1)
        self.grid.attach(speed_lbl, 1, row_idx, 1, 1)
        self.grid.attach(means_lbl, 2, row_idx, 1, 1)
        self.grid.attach(stds_lbl, 3, row_idx, 1, 1)

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

    def on_capture_clicked(self, widget, channel_id):
        self.start_speed = self.shutter_combo.get_active_text()
        self.set_controls_sensitive(False)
        self.clear_ae_steps()
        self.spinner.start()
        self.status_label.set_text(f"Status: Run auto-exposure for {channel_id}...")

        capture_thread = threading.Thread(
            target=self.background_capture_step,
            args=(channel_id,)
        )
        capture_thread.daemon = True
        capture_thread.start()

    def background_capture_step(self, channel_id):
        if not self.is_connected or self.camera_session is None:
            GLib.idle_add(self.on_step_error, channel_id, "Camera is not connected.")
            return

        session = self.camera_session

        def ae_progress_callback(idx, shutter_str, dr_channels, avg_dr):
            dr_r, dr_g, dr_b = dr_channels
            GLib.idle_add(self.add_ae_step_to_listbox, idx, shutter_str, dr_r, dr_g, dr_b, avg_dr)

        def ae_capture_func(idx):
            shutter_str = auto_exposure.SHUTTER_SPEEDS[idx]
            return auto_exposure.capture_exposure_frame(shutter_str, half=True, session=session)

        try:
            opt_speed, _ = auto_exposure.run_auto_exposure(
                start_shutter_str=self.start_speed,
                capture_func=ae_capture_func,
                progress_callback=ae_progress_callback,
                channel=channel_id
            )
            
            GLib.idle_add(self.status_label.set_text, f"Status: AE complete ({opt_speed}). Capturing final image...")

            num, den = auto_exposure.parse_shutter_speed(opt_speed)
            img = session.capture(type=0, shutter_num=num, shutter_den=den)
            arr = img.to_numpy(half=True)
            model = img.camera_model
            img.discard()

            means, stds = get_circle_stats(arr)
            GLib.idle_add(self.on_step_complete, channel_id, opt_speed, model, means, stds)
        except Exception as e:
            GLib.idle_add(self.on_step_error, channel_id, str(e))

    def on_step_complete(self, channel_id, opt_speed, model, means, stds):
        self.spinner.stop()
        self.set_controls_sensitive(True)
        self.camera_model = model

        if channel_id == "R":
            self.means_r = means
            self.stds_r = stds
            self.speed_r = opt_speed
            self.val_r_speed.set_text(opt_speed)
            self.val_r_means.set_text(f"R:{means[0]:.1f} G:{means[1]:.1f} B:{means[2]:.1f}")
            self.val_r_stds.set_text(f"R:{stds[0]:.1f} G:{stds[1]:.1f} B:{stds[2]:.1f}")
        elif channel_id == "B":
            self.means_b = means
            self.stds_b = stds
            self.speed_b = opt_speed
            self.val_b_speed.set_text(opt_speed)
            self.val_b_means.set_text(f"R:{means[0]:.1f} G:{means[1]:.1f} B:{means[2]:.1f}")
            self.val_b_stds.set_text(f"R:{stds[0]:.1f} G:{stds[1]:.1f} B:{stds[2]:.1f}")
        elif channel_id == "G":
            self.means_g = means
            self.stds_g = stds
            self.speed_g = opt_speed
            self.val_g_speed.set_text(opt_speed)
            self.val_g_means.set_text(f"R:{means[0]:.1f} G:{means[1]:.1f} B:{means[2]:.1f}")
            self.val_g_stds.set_text(f"R:{stds[0]:.1f} G:{stds[1]:.1f} B:{stds[2]:.1f}")

        self.status_label.set_text(f"Status: Capture {channel_id} success!")
        self.update_matrix_calculations()

    def on_step_error(self, channel_id, error_msg):
        self.spinner.stop()
        self.set_controls_sensitive(True)
        self.status_label.set_text(f"Status: Error in {channel_id} capture - {error_msg}")

    def set_controls_sensitive(self, sensitive):
        is_active = sensitive and self.is_connected
        self.shutter_combo.set_sensitive(is_active)
        self.btn_capture_r.set_sensitive(is_active)
        self.btn_capture_b.set_sensitive(is_active)
        self.btn_capture_g.set_sensitive(is_active)

    def format_matrix_with_labels(self, matrix):
        return (
            "# Col:  R         G         B\n"
            "[\n"
            f"  [{matrix[0,0]:8.4f}, {matrix[0,1]:8.4f}, {matrix[0,2]:8.4f}],  # Row: R\n"
            f"  [{matrix[1,0]:8.4f}, {matrix[1,1]:8.4f}, {matrix[1,2]:8.4f}],  # Row: G\n"
            f"  [{matrix[2,0]:8.4f}, {matrix[2,1]:8.4f}, {matrix[2,2]:8.4f}]   # Row: B\n"
            "]"
        )

    def update_matrix_calculations(self):
        if self.means_r is not None and self.means_b is not None and self.means_g is not None:
            try:
                self.M, self.M_norm, self.correction_matrix = crosstalk_calibration.compute_calibration_matrices(
                    self.means_r, self.means_g, self.means_b
                )

                self.lbl_m_norm.set_text(self.format_matrix_with_labels(self.M_norm))
                self.lbl_m_corr.set_text(self.format_matrix_with_labels(self.correction_matrix))

                self.save_button.set_sensitive(True)
            except np.linalg.LinAlgError:
                self.status_label.set_text("Status: Error - Matrix is singular and cannot be inverted!")
                self.lbl_m_norm.set_text("Singular matrix!")
                self.lbl_m_corr.set_text("Cannot invert!")
                self.save_button.set_sensitive(False)

    def on_save_clicked(self, widget):
        if self.correction_matrix is None:
            return

        dialog = Gtk.FileChooserDialog(
            title="Save Calibration Profile",
            parent=self,
            action=Gtk.FileChooserAction.SAVE
        )
        dialog.add_buttons(
            Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL,
            Gtk.STOCK_SAVE, Gtk.ResponseType.OK
        )
        dialog.set_do_overwrite_confirmation(True)

        safe_camera_name = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in self.camera_model)
        default_filename = f"{safe_camera_name}_crosstalk_profile.json"
        dialog.set_current_name(default_filename)

        response = dialog.run()
        if response == Gtk.ResponseType.OK:
            filepath = dialog.get_filename()
            try:
                calib = crosstalk_calibration.CrosstalkCalibration(
                    camera_model=self.camera_model,
                    M=self.M,
                    M_norm=self.M_norm,
                    M_corr=self.correction_matrix,
                    captured_data={
                        "Red": {
                            "shutter_speed": self.speed_r,
                            "means": list(self.means_r) if self.means_r is not None else None,
                            "stds": list(self.stds_r) if self.stds_r is not None else None
                        },
                        "Green": {
                            "shutter_speed": self.speed_g,
                            "means": list(self.means_g) if self.means_g is not None else None,
                            "stds": list(self.stds_g) if self.stds_g is not None else None
                        },
                        "Blue": {
                            "shutter_speed": self.speed_b,
                            "means": list(self.means_b) if self.means_b is not None else None,
                            "stds": list(self.stds_b) if self.stds_b is not None else None
                        }
                    }
                )
                calib.save(filepath)
                self.status_label.set_text(f"Status: Profile saved to {os.path.basename(filepath)}")
            except Exception as e:
                self.status_label.set_text(f"Status: Error saving profile: {str(e)}")

        dialog.destroy()

    def on_load_clicked(self, widget):
        dialog = Gtk.FileChooserDialog(
            title="Load Calibration Profile",
            parent=self,
            action=Gtk.FileChooserAction.OPEN
        )
        dialog.add_buttons(
            Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL,
            Gtk.STOCK_OPEN, Gtk.ResponseType.OK
        )
        
        # Filter for JSON files
        filter_json = Gtk.FileFilter()
        filter_json.set_name("JSON files")
        filter_json.add_mime_type("application/json")
        filter_json.add_pattern("*.json")
        dialog.add_filter(filter_json)

        response = dialog.run()
        if response == Gtk.ResponseType.OK:
            filepath = dialog.get_filename()
            try:
                calib = crosstalk_calibration.CrosstalkCalibration.load(filepath)
                
                self.camera_model = calib.camera_model or "Unknown"
                
                captured_data = calib.captured_data or {}
                
                # Load Red
                r_data = captured_data.get("Red", {})
                self.speed_r = r_data.get("shutter_speed", "N/A")
                self.means_r = r_data.get("means")
                self.stds_r = r_data.get("stds")
                
                # Load Green
                g_data = captured_data.get("Green", {})
                self.speed_g = g_data.get("shutter_speed", "N/A")
                self.means_g = g_data.get("means")
                self.stds_g = g_data.get("stds")
                
                # Load Blue
                b_data = captured_data.get("Blue", {})
                self.speed_b = b_data.get("shutter_speed", "N/A")
                self.means_b = b_data.get("means")
                self.stds_b = b_data.get("stds")
                
                # Update UI elements
                if self.means_r:
                    self.val_r_speed.set_text(self.speed_r)
                    self.val_r_means.set_text(f"R:{self.means_r[0]:.1f} G:{self.means_r[1]:.1f} B:{self.means_r[2]:.1f}")
                    self.val_r_stds.set_text(f"R:{self.stds_r[0]:.1f} G:{self.stds_r[1]:.1f} B:{self.stds_r[2]:.1f}")
                else:
                    self.val_r_speed.set_text("--")
                    self.val_r_means.set_text("--")
                    self.val_r_stds.set_text("--")
                    
                if self.means_g:
                    self.val_g_speed.set_text(self.speed_g)
                    self.val_g_means.set_text(f"R:{self.means_g[0]:.1f} G:{self.means_g[1]:.1f} B:{self.means_g[2]:.1f}")
                    self.val_g_stds.set_text(f"R:{self.stds_g[0]:.1f} G:{self.stds_g[1]:.1f} B:{self.stds_g[2]:.1f}")
                else:
                    self.val_g_speed.set_text("--")
                    self.val_g_means.set_text("--")
                    self.val_g_stds.set_text("--")
                    
                if self.means_b:
                    self.val_b_speed.set_text(self.speed_b)
                    self.val_b_means.set_text(f"R:{self.means_b[0]:.1f} G:{self.means_b[1]:.1f} B:{self.means_b[2]:.1f}")
                    self.val_b_stds.set_text(f"R:{self.stds_b[0]:.1f} G:{self.stds_b[1]:.1f} B:{self.stds_b[2]:.1f}")
                else:
                    self.val_b_speed.set_text("--")
                    self.val_b_means.set_text("--")
                    self.val_b_stds.set_text("--")
                
                # Load matrix values
                self.M = calib.M
                self.M_norm = calib.M_norm
                self.correction_matrix = calib.M_corr
                
                # Format and display matrices
                self.lbl_m_norm.set_text(self.format_matrix_with_labels(self.M_norm))
                self.lbl_m_corr.set_text(self.format_matrix_with_labels(self.correction_matrix))
                
                # Enable Save Button since we have a valid profile loaded
                self.save_button.set_sensitive(True)
                self.status_label.set_text(f"Status: Profile loaded from {os.path.basename(filepath)}")
                
            except Exception as e:
                self.status_label.set_text(f"Status: Error loading profile: {str(e)}")
        
        dialog.destroy()

if __name__ == "__main__":
    win = CrosstalkAppWindow()
    Gtk.main()
