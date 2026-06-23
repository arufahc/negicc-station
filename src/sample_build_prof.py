#!/usr/bin/env python3
"""
Film Profile Builder & Scanner GUI (sample_build_prof.py)

Provides a modern dark theme interface to:
1. Enter location of the IT8 reference ZIP, download, cache, and parse XYZ values.
2. Load a film profile JSON file generated during capture.
3. Build the custom ICC profile (compiling make_icc and invoking ArgyllCMS colprof).
4. Capture raw images and apply linear crosstalk and IT8 profile corrections in C++.
5. Display positive display-ready images directly in Python.
"""

import os
import sys
import time
import threading
import urllib.request
import zipfile
import re
import json
import shutil
import tempfile
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
import film_profiling
from film_profiling import FilmProfile, download_and_parse_reference_file

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
    bins = 256
    hist_r, _ = np.histogram(arr[:, :, 0], bins=bins, range=(0, 65535))
    hist_g, _ = np.histogram(arr[:, :, 1], bins=bins, range=(0, 65535))
    hist_b, _ = np.histogram(arr[:, :, 2], bins=bins, range=(0, 65535))
    max_val = max(hist_r.max(), hist_g.max(), hist_b.max(), 1)
    
    hist_r_norm = hist_r / max_val
    hist_g_norm = hist_g / max_val
    hist_b_norm = hist_b / max_val

    H, W, C = arr.shape
    h_border = int(H * 0.05)
    w_border = int(W * 0.05)
    cropped = arr[h_border:H-h_border, w_border:W-w_border, :]

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

class ProfileBuilderAppWindow(Gtk.Window):
    def __init__(self):
        super().__init__(title="Film Profile Builder & Scanning Client")
        self.set_default_size(1200, 800)
        self.connect("destroy", self.on_destroy)

        # Force GTK dark theme
        settings = Gtk.Settings.get_default()
        settings.set_property("gtk-application-prefer-dark-theme", True)

        css_provider = Gtk.CssProvider()
        css_provider.load_from_data(b"""
            .sidebar {
                background-color: #1a1a1a;
                border-right: 1px solid #333333;
                padding: 12px;
            }
            .sidebar-section {
                border-bottom: 1px solid #2d2d2d;
                padding-bottom: 12px;
                margin-bottom: 12px;
            }
            .section-title {
                font-weight: bold;
                font-size: 13px;
                color: #4a90e2;
                margin-bottom: 8px;
            }
            .action-btn {
                background-image: linear-gradient(to bottom, #3a3a3a, #2b2b2b);
                color: #e0e0e0;
                border: 1px solid #444444;
                border-radius: 4px;
                padding: 6px;
            }
            .action-btn:hover {
                background-image: linear-gradient(to bottom, #4a4a4a, #3b3b3b);
            }
            .capture-btn {
                background-image: linear-gradient(to bottom, #2ea44f, #2c974b);
                color: white;
                font-weight: bold;
                border: 1px solid rgba(27,31,35,0.15);
                border-radius: 6px;
                padding: 8px;
            }
            .capture-btn:hover {
                background-image: linear-gradient(to bottom, #30bc5a, #2ea44f);
            }
            .capture-btn:disabled {
                background-image: none;
                background-color: #444444;
                color: #888888;
            }
            .preview-container {
                background-color: #0f0f0f;
                padding: 12px;
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

        main_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=0)
        self.add(main_box)

        # ----------------------------------------------------
        # Sidebar Panel
        # ----------------------------------------------------
        scroll_sidebar = Gtk.ScrolledWindow()
        scroll_sidebar.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        scroll_sidebar.set_size_request(340, -1)
        sidebar_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=0)
        sidebar_box.get_style_context().add_class("sidebar")
        scroll_sidebar.add(sidebar_box)
        main_box.pack_start(scroll_sidebar, False, False, 0)

        # Header Title
        title_lbl = Gtk.Label()
        title_lbl.set_markup("<span size='large' weight='bold'>Film Profile Builder</span>")
        title_lbl.set_xalign(0.0)
        sidebar_box.pack_start(title_lbl, False, False, 10)

        # Camera Status Row
        camera_status_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        self.camera_status_label = Gtk.Label()
        self.camera_status_label.set_markup("<span><span foreground='#e6a23c'>●</span> Camera: Connecting...</span>")
        self.camera_status_label.set_xalign(0.0)
        camera_status_box.pack_start(self.camera_status_label, True, True, 0)
        
        self.connect_btn = Gtk.Button(label="Connect")
        self.connect_btn.get_style_context().add_class("action-btn")
        self.connect_btn.connect("clicked", lambda w: self.connect_camera(manual=True))
        self.connect_btn.set_sensitive(False)
        camera_status_box.pack_start(self.connect_btn, False, False, 0)
        sidebar_box.pack_start(camera_status_box, False, False, 5)

        # SECTION 1: Target Certificate Loader
        sec1_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        sec1_box.get_style_context().add_class("sidebar-section")
        sidebar_box.pack_start(sec1_box, False, False, 0)

        s1_title = Gtk.Label()
        s1_title.set_markup("<span class='section-title'>1. Reference IT8 File</span>")
        s1_title.set_xalign(0.0)
        sec1_box.pack_start(s1_title, False, False, 0)

        self.zip_entry = Gtk.Entry()
        self.zip_entry.set_text("http://www.colorreference.de/targets/R190808.zip")
        self.zip_entry.set_tooltip_text("Enter reference file (.txt/.it8/.zip) URL or local path")
        sec1_box.pack_start(self.zip_entry, False, False, 0)

        self.load_ref_btn = Gtk.Button(label="Download Reference IT8 File")
        self.load_ref_btn.get_style_context().add_class("action-btn")
        self.load_ref_btn.connect("clicked", self.on_load_ref_clicked)
        sec1_box.pack_start(self.load_ref_btn, False, False, 0)

        self.ref_status_lbl = Gtk.Label(label="Reference status: Not loaded")
        self.ref_status_lbl.set_xalign(0.0)
        self.ref_status_lbl.get_style_context().add_class("meta-label")
        sec1_box.pack_start(self.ref_status_lbl, False, False, 0)

        # SECTION 2: Load Film Profile
        sec2_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        sec2_box.get_style_context().add_class("sidebar-section")
        sidebar_box.pack_start(sec2_box, False, False, 0)

        s2_title = Gtk.Label()
        s2_title.set_markup("<span class='section-title'>2. Loaded Film Profile JSON</span>")
        s2_title.set_xalign(0.0)
        sec2_box.pack_start(s2_title, False, False, 0)

        self.load_profile_btn = Gtk.Button(label="Load Profile JSON...")
        self.load_profile_btn.get_style_context().add_class("action-btn")
        self.load_profile_btn.connect("clicked", self.on_load_profile_clicked)
        sec2_box.pack_start(self.load_profile_btn, False, False, 0)

        self.profile_status_lbl = Gtk.Label(label="Profile: None loaded")
        self.profile_status_lbl.set_xalign(0.0)
        self.profile_status_lbl.set_line_wrap(True)
        self.profile_status_lbl.get_style_context().add_class("meta-label")
        sec2_box.pack_start(self.profile_status_lbl, False, False, 0)

        # SECTION 3: ICC Compilation
        sec3_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        sec3_box.get_style_context().add_class("sidebar-section")
        sidebar_box.pack_start(sec3_box, False, False, 0)

        s3_title = Gtk.Label()
        s3_title.set_markup("<span class='section-title'>3. Generate Custom ICC Profile</span>")
        s3_title.set_xalign(0.0)
        sec3_box.pack_start(s3_title, False, False, 0)

        self.build_profile_btn = Gtk.Button(label="Compile Custom ICC Profile")
        self.build_profile_btn.get_style_context().add_class("action-btn")
        self.build_profile_btn.set_sensitive(False)
        self.build_profile_btn.connect("clicked", self.on_build_profile_clicked)
        sec3_box.pack_start(self.build_profile_btn, False, False, 0)

        self.build_status_lbl = Gtk.Label(label="ICC: Not generated")
        self.build_status_lbl.set_xalign(0.0)
        self.build_status_lbl.set_line_wrap(True)
        self.build_status_lbl.get_style_context().add_class("meta-label")
        sec3_box.pack_start(self.build_status_lbl, False, False, 0)

        # SECTION 4: Live Rendering Parameters
        sec4_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        sec4_box.get_style_context().add_class("sidebar-section")
        sidebar_box.pack_start(sec4_box, False, False, 0)

        s4_title = Gtk.Label()
        s4_title.set_markup("<span class='section-title'>4. Post-Correction Adjustments</span>")
        s4_title.set_xalign(0.0)
        sec4_box.pack_start(s4_title, False, False, 0)

        self.apply_it8_checkbox = Gtk.CheckButton(label="Apply IT8 & CC Corrections")
        self.apply_it8_checkbox.set_active(False)
        sec4_box.pack_start(self.apply_it8_checkbox, False, False, 0)

        # Exposure compensation
        exp_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        exp_lbl = Gtk.Label(label="Exposure Comp:")
        exp_lbl.set_xalign(0.0)
        exp_box.pack_start(exp_lbl, True, True, 0)
        self.exposure_comp_spin = Gtk.SpinButton.new_with_range(0.1, 10.0, 0.1)
        self.exposure_comp_spin.set_value(1.0)
        exp_box.pack_start(self.exposure_comp_spin, False, False, 0)
        sec4_box.pack_start(exp_box, False, False, 0)

        # Gamma
        gamma_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        gamma_lbl = Gtk.Label(label="Post Gamma:")
        gamma_lbl.set_xalign(0.0)
        gamma_box.pack_start(gamma_lbl, True, True, 0)
        self.gamma_spin = Gtk.SpinButton.new_with_range(0.5, 3.0, 0.1)
        self.gamma_spin.set_value(1.0)
        gamma_box.pack_start(self.gamma_spin, False, False, 0)
        sec4_box.pack_start(gamma_box, False, False, 0)



        # Capture Configuration Section
        sec5_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        sidebar_box.pack_start(sec5_box, False, False, 10)

        # Shutter Dropdown
        shutter_lbl = Gtk.Label(label="Capture Shutter Speed:")
        shutter_lbl.set_xalign(0.0)
        sec5_box.pack_start(shutter_lbl, False, False, 0)

        self.shutter_combo = Gtk.ComboBoxText()
        for speed in SHUTTER_SPEEDS:
            self.shutter_combo.append(speed, speed)
        self.shutter_combo.set_active(SHUTTER_SPEEDS.index("1/8s"))
        sec5_box.pack_start(self.shutter_combo, False, False, 0)

        self.ae_checkbox = Gtk.CheckButton(label="Auto Exposure Search")
        self.ae_checkbox.connect("toggled", self.on_ae_toggled)
        sec5_box.pack_start(self.ae_checkbox, False, False, 0)

        # CAPTURE Button
        self.capture_button = Gtk.Button(label="TETHERED CAPTURE PREVIEW")
        self.capture_button.get_style_context().add_class("capture-btn")
        self.capture_button.set_sensitive(False)
        self.capture_button.connect("clicked", self.on_capture_clicked)
        sec5_box.pack_start(self.capture_button, False, False, 5)

        # Spinner & status details
        status_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=10)
        self.spinner = Gtk.Spinner()
        status_box.pack_start(self.spinner, False, False, 0)
        self.status_label = Gtk.Label(label="Status: Idle")
        self.status_label.set_xalign(0.0)
        status_box.pack_start(self.status_label, True, True, 0)
        sec5_box.pack_start(status_box, False, False, 0)

        # ----------------------------------------------------
        # Right Panel: Notebook tabbed layout
        # ----------------------------------------------------
        self.notebook = Gtk.Notebook()
        main_box.pack_start(self.notebook, True, True, 0)

        # Tab 1: Preview Box
        self.preview_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        self.preview_box.get_style_context().add_class("preview-container")
        self.notebook.append_page(self.preview_box, Gtk.Label(label="Capture Preview"))

        # Placeholder Label
        self.placeholder_label = Gtk.Label()
        self.placeholder_label.set_markup("<span size='large' foreground='#666666'>No Image Captured\n\nLoad a profile, download reference, compile, and capture.</span>")
        self.placeholder_label.set_justify(Gtk.Justification.CENTER)
        self.preview_box.pack_start(self.placeholder_label, True, True, 0)

        # GTK Image Widget
        self.image_widget = Gtk.Image()
        self.image_widget.set_no_show_all(True)
        self.preview_box.pack_start(self.image_widget, True, True, 0)

        # Info Display Box
        self.results_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        self.results_box.set_no_show_all(True)
        self.preview_box.pack_start(self.results_box, False, False, 0)

        self.dr_label = Gtk.Label()
        self.dr_label.set_use_markup(True)
        self.dr_label.set_xalign(0.0)
        self.dr_label.get_style_context().add_class("meta-label")
        self.results_box.pack_start(self.dr_label, False, False, 0)

        # Tab 2: Build Log
        log_scroll = Gtk.ScrolledWindow()
        log_scroll.set_policy(Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.AUTOMATIC)
        log_scroll.get_style_context().add_class("preview-container")
        
        self.log_view = Gtk.TextView()
        self.log_view.set_editable(False)
        self.log_view.set_cursor_visible(False)
        self.log_view.set_monospace(True)
        self.log_view.set_left_margin(10)
        self.log_view.set_right_margin(10)
        self.log_view.set_top_margin(10)
        self.log_view.set_bottom_margin(10)
        self.log_buffer = self.log_view.get_buffer()
        log_scroll.add(self.log_view)
        
        self.notebook.append_page(log_scroll, Gtk.Label(label="ICC Compilation Build Log"))

        # App variables
        self.current_pixbuf = None
        self.film_profile = None
        self.reference_xyz_path = None
        self.built_clut_icc_path = None
        self.crosstalk_matrix = None
        self.profile_film_base = None

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

    # ----------------------------------------------------
    # Camera Connection Management
    # ----------------------------------------------------
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

    # ----------------------------------------------------
    # Sidebar Section Handlers
    # ----------------------------------------------------
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
        self.ref_status_lbl.set_text(f"Loaded {num_patches} patches from {loaded_filename}")
        self.check_build_button_sensitivity()

    def on_load_ref_failure(self, err_msg):
        self.load_ref_btn.set_sensitive(True)
        self.ref_status_lbl.set_text("Failed to load reference.")
        self.show_error_dialog("Download/Parse Error", err_msg)

    def on_load_profile_clicked(self, widget):
        dialog = Gtk.FileChooserDialog(
            title="Please choose a film profile JSON file",
            parent=self,
            action=Gtk.FileChooserAction.OPEN
        )
        dialog.add_buttons(
            Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL,
            Gtk.STOCK_OPEN, Gtk.ResponseType.ACCEPT
        )
        
        # Add profile filters
        filter_json = Gtk.FileFilter()
        filter_json.set_name("JSON profiles")
        filter_json.add_pattern("*.json")
        dialog.add_filter(filter_json)

        # Default to profiles directory
        profiles_dir = os.path.join(project_dir, "profiles")
        if os.path.exists(profiles_dir):
            dialog.set_current_folder(profiles_dir)

        response = dialog.run()
        if response == Gtk.ResponseType.ACCEPT:
            profile_path = dialog.get_filename()
            dialog.destroy()
            self.load_film_profile_data(profile_path)
        else:
            dialog.destroy()

    def load_film_profile_data(self, profile_path):
        try:
            profile = FilmProfile(profile_path)
            self.film_profile = profile
            
            # Setup crosstalk matrix
            self.crosstalk_matrix = profile.crosstalk_matrix.flatten().tolist()
            
            # Setup profile film base values
            r_avg = int(round(profile.film_base['r_avg']))
            g_avg = int(round(profile.film_base['g_avg']))
            b_avg = int(round(profile.film_base['b_avg']))
            self.profile_film_base = [r_avg, g_avg, b_avg]



            # Display metadata on label
            self.profile_status_lbl.set_text(
                f"Film: {profile.film_name}\n"
                f"Camera: {profile.camera_name}\n"
                f"Base Avg: R={r_avg}, G={g_avg}, B={b_avg}"
            )
            self.check_build_button_sensitivity()
        except Exception as e:
            self.profile_status_lbl.set_text("Failed to parse JSON profile.")
            self.show_error_dialog("Film Profile Error", str(e))

    def check_build_button_sensitivity(self):
        if self.film_profile and self.reference_xyz_path:
            self.build_profile_btn.set_sensitive(True)
        else:
            self.build_profile_btn.set_sensitive(False)

    def append_build_log(self, text):
        end_iter = self.log_buffer.get_end_iter()
        self.log_buffer.insert(end_iter, text + "\n")
        mark = self.log_buffer.get_insert()
        self.log_view.scroll_to_mark(mark, 0.0, True, 0.0, 1.0)

    def on_build_profile_clicked(self, widget):
        if not self.film_profile or not self.reference_xyz_path:
            return

        self.log_buffer.set_text("")
        self.notebook.set_current_page(1)  # Switch to Build Log tab
        self.build_status_lbl.set_text("Compiling custom ICC profile...")
        self.build_profile_btn.set_sensitive(False)
        self.load_profile_btn.set_sensitive(False)
        self.load_ref_btn.set_sensitive(False)

        def run():
            try:
                output_profiles_dir = os.path.join(project_dir, "profiles")
                os.makedirs(output_profiles_dir, exist_ok=True)
                
                def log_cb(step, detail):
                    GLib.idle_add(self.append_build_log, detail)

                # Call library to compile and generate custom nested cLUT profile
                res = film_profiling.build_icc_profile(
                    self.film_profile,
                    self.reference_xyz_path,
                    output_profiles_dir,
                    progress_callback=log_cb
                )
                
                clut_path = res['clut_icc_path']
                profcheck_out = res['profcheck_output']
                
                # Parse average and maximum Delta E from profcheck output
                # Output usually contains "max = X.XX, avg = Y.YY" or similar
                max_de = "N/A"
                avg_de = "N/A"
                for line in profcheck_out.splitlines():
                    m_max = re.search(r'max\s*=\s*([0-9.]+)', line, re.IGNORECASE)
                    if m_max:
                        max_de = m_max.group(1)
                    m_avg = re.search(r'avg\s*=\s*([0-9.]+)', line, re.IGNORECASE)
                    if m_avg:
                        avg_de = m_avg.group(1)

                GLib.idle_add(self.on_build_profile_success, clut_path, avg_de, max_de)
            except Exception as e:
                GLib.idle_add(self.on_build_profile_failure, str(e))

        t = threading.Thread(target=run)
        t.daemon = True
        t.start()

    def on_build_profile_success(self, clut_path, avg_de, max_de):
        self.build_profile_btn.set_sensitive(True)
        self.load_profile_btn.set_sensitive(True)
        self.load_ref_btn.set_sensitive(True)
        self.built_clut_icc_path = clut_path
        self.apply_it8_checkbox.set_active(True)
        
        self.build_status_lbl.set_markup(
            f"<span foreground='#44ff44'>Profile Compiled!</span>\n"
            f"<b>Path:</b> {os.path.basename(clut_path)}\n"
            f"<b>Delta E Errors:</b> Avg: {avg_de} | Max: {max_de}"
        )

    def on_build_profile_failure(self, err_msg):
        self.build_profile_btn.set_sensitive(True)
        self.load_profile_btn.set_sensitive(True)
        self.load_ref_btn.set_sensitive(True)
        self.build_status_lbl.set_text("ICC Compilation Failed.")
        self.show_error_dialog("ICC Compilation Error", err_msg)

    # ----------------------------------------------------
    # Capture & Processing Pipeline
    # ----------------------------------------------------
    def on_ae_toggled(self, button):
        is_active = button.get_active()
        self.shutter_combo.set_sensitive(not is_active)

    def on_capture_clicked(self, widget):
        shutter_str = self.shutter_combo.get_active_text()
        is_ae = self.ae_checkbox.get_active()
        
        # Read all UI settings on the main thread for thread-safety
        should_apply_it8 = self.apply_it8_checkbox.get_active() and (self.built_clut_icc_path is not None)
        exposure_comp = float(self.exposure_comp_spin.get_value())
        gamma_val = float(self.gamma_spin.get_value())

        self.capture_button.set_sensitive(False)
        self.shutter_combo.set_sensitive(False)
        self.ae_checkbox.set_sensitive(False)
        self.spinner.start()
        self.status_label.set_text("Status: Capturing...")

        capture_thread = threading.Thread(
            target=self.background_capture_and_convert,
            args=(shutter_str, is_ae, should_apply_it8, exposure_comp, gamma_val)
        )
        capture_thread.daemon = True
        capture_thread.start()

    def background_capture_and_convert(self, start_shutter_str, is_ae, should_apply_it8, exposure_comp, gamma_val):
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

            # Convert to numpy for preview display (passing IT8 profile path if enabled)
            t_conv_start = time.time()
            
            if should_apply_it8:
                print("[sample_build_prof] Applying C++ IT8 & Crosstalk Correction...")
                arr = img.to_numpy(
                    half=True,
                    crosstalk_matrix=self.crosstalk_matrix,
                    it8_profile_path=self.built_clut_icc_path,
                    output_profile_path="srgb",
                    profile_film_base=None,
                    film_base=None,
                    exposure_comp=exposure_comp,
                    post_correction_gamma=gamma_val
                )
            else:
                print("[sample_build_prof] Fetching raw/uncorrected preview...")
                arr = img.to_numpy(half=True)
                
            t_conv_duration = time.time() - t_conv_start

            # Calculate dynamic range metrics
            _, p2_vals, p98_vals, dr_metrics = compute_hist_and_percentiles(arr)
            dr_r, dr_g, dr_b, avg_dr = dr_metrics

            height, width, channels = arr.shape
            
            # Since the array is uint16, convert to uint8 for displaying in GdkPixbuf
            arr_8bit = (arr >> 8).astype(np.uint8)
            raw_bytes = arr_8bit.tobytes()

            GLib.idle_add(
                self.update_ui_success,
                raw_bytes, width, height, t_cap_duration, t_conv_duration,
                dr_r, dr_g, dr_b, avg_dr, p2_vals, p98_vals, should_apply_it8
            )
        except Exception as e:
            GLib.idle_add(self.update_ui_failure, str(e))

    def update_ui_success(self, raw_bytes, width, height, t_cap_duration, t_conv_duration, dr_r, dr_g, dr_b, avg_dr, p2_vals, p98_vals, applied_it8):
        self.spinner.stop()
        self.capture_button.set_sensitive(self.is_connected)
        self.shutter_combo.set_sensitive(not self.ae_checkbox.get_active())
        self.ae_checkbox.set_sensitive(True)
        self.status_label.set_text("Status: Preview updated successfully.")

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
        mode_str = "IT8 + Crosstalk Mapped (Positive)" if applied_it8 else "Raw Uncorrected (Linear Negative)"
        self.dr_label.set_markup(
            f"<b>Preview Mode:</b> <span color='#4a90e2'>{mode_str}</span>\n"
            f"<b>Dynamic Range (p2-p98):</b>  R: {dr_r:.1f} | G: {dr_g:.1f} | B: {dr_b:.1f} | <b>Avg: {avg_dr:.1f}</b>\n"
            f"  Red [2%-98%]: [{int(p2_vals[0])} - {int(p98_vals[0])}]\n"
            f"  Green [2%-98%]: [{int(p2_vals[1])} - {int(p98_vals[1])}]\n"
            f"  Blue [2%-98%]: [{int(p2_vals[2])} - {int(p98_vals[2])}]\n"
            f"Capture: {t_cap_duration:.2f}s | Processing: {t_conv_duration:.2f}s"
        )
        self.results_box.show_all()

    def update_ui_failure(self, error_msg):
        self.spinner.stop()
        self.capture_button.set_sensitive(self.is_connected)
        self.shutter_combo.set_sensitive(not self.ae_checkbox.get_active())
        self.ae_checkbox.set_sensitive(True)
        self.status_label.set_text("Status: Capture failed.")
        self.show_error_dialog("Capture Error", error_msg)

    def set_shutter_speed_active(self, shutter_str):
        if shutter_str in SHUTTER_SPEEDS:
            self.shutter_combo.set_active(SHUTTER_SPEEDS.index(shutter_str))

    # ----------------------------------------------------
    # UI Utility Helpers
    # ----------------------------------------------------
    def refresh_preview_image(self):
        if not self.current_pixbuf:
            return
        alloc = self.preview_box.get_allocation()
        max_w = max(100, alloc.width - 30)
        max_h = max(100, alloc.height - 120)

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

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Film Profile Builder CLI / GUI")
    parser.add_argument("--profile", type=str, help="Path to a film profile .json file.")
    parser.add_argument("--reference", type=str,
                        help="URL or local path of the IT8 reference file (.zip or .txt/.it8). Required in CLI mode.")
    # Support both boolean flag (--dry-run as switch) and string value for backwards compatibility
    parser.add_argument("--dry-run", nargs='?', const=True, default=False,
                        help="Run in dry-run mode (do not save final ICC files, print errors only). Can optionally specify profile JSON directly.")
    
    args = parser.parse_args()
    
    profile_path = None
    is_dry_run = False
    
    if args.profile:
        profile_path = args.profile
        is_dry_run = bool(args.dry_run)
    elif args.dry_run and args.dry_run is not True:
        profile_path = args.dry_run
        is_dry_run = True
        
    if profile_path:
        ref_location = args.reference
        if not ref_location:
            print("Error: --reference file or URL must be specified in CLI mode.")
            sys.exit(1)
        
        if is_dry_run:
            print(f"=== CLI DRY-RUN BUILD ===")
        else:
            print(f"=== CLI BUILD & GENERATE PROFILE ===")
            
        print(f"Loading Film Profile: {profile_path}")
        print(f"Reference File: {ref_location}")
        
        try:
            # 1. Download/parse reference target
            cache_dir = tempfile.gettempdir()
            patches, loaded_filename, reference_dir = download_and_parse_reference_file(ref_location, cache_dir, prompt_zip_callback=None)
            
            ref_base_name = os.path.splitext(os.path.basename(loaded_filename))[0]
            out_json_path = os.path.join(reference_dir, f"{ref_base_name}_ref.json")
            ref_data = {
                "description": "IT8.7/2 Reference XYZ values",
                "source": ref_location,
                "patches": patches
            }
            with open(out_json_path, 'w') as f:
                json.dump(ref_data, f, indent=2)
            
            print(f"Loaded {len(patches)} reference patches from {loaded_filename}")
            
            # 2. Load film profile
            film_profile = FilmProfile(profile_path)
            print(f"Film Profile Name: {film_profile.film_name}")
            
            # 3. Build ICC Profile
            if is_dry_run:
                # Use a temp directory so we do not save final ICC files to the profiles dir
                tmp_output_dir = tempfile.mkdtemp(prefix="negicc_dry_run_")
                output_profiles_dir = tmp_output_dir
            else:
                output_profiles_dir = os.path.join(project_dir, "profiles")
                os.makedirs(output_profiles_dir, exist_ok=True)
            
            # Define progress callback
            if is_dry_run:
                # Do not log individual building steps to keep stdout clean
                def log_cb(step, detail):
                    pass
            else:
                def log_cb(step, detail):
                    print(f"[{step}] {detail}")
                
            res = film_profiling.build_icc_profile(
                film_profile,
                out_json_path,
                output_profiles_dir,
                progress_callback=log_cb
            )
            
            clut_path = res['clut_icc_path']
            profcheck_out = res['profcheck_output']
            
            if is_dry_run:
                print("\n=== DRY-RUN RESULTS ===")
                shutil.rmtree(tmp_output_dir, ignore_errors=True)
            else:
                print("\n=== BUILD SUCCESSFUL ===")
                print(f"ICC Profile Path: {clut_path}")
            
            # Print the results (errors) only
            for line in profcheck_out.splitlines():
                if "errors" in line or "Profile check complete" in line:
                    print(line.strip())
            
        except Exception as e:
            print(f"\nError during build: {e}")
            sys.exit(1)
            
    else:
        # Start GUI
        win = ProfileBuilderAppWindow()
        Gtk.main()
