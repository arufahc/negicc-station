#!/usr/bin/env python3
import os
import sys
import subprocess
import gi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk, Gdk, GLib

class LauncherWindow(Gtk.Window):
    def __init__(self):
        super().__init__(title="NEGICC Station Launcher")
        self.set_default_size(480, 520)
        self.set_resizable(False)
        self.set_position(Gtk.WindowPosition.CENTER)

        self.is_launching = False
        self.processes = {}

        # Force GTK dark theme for premium aesthetics
        settings = Gtk.Settings.get_default()
        settings.set_property("gtk-application-prefer-dark-theme", True)

        # Custom CSS for custom buttons and layouts
        css_provider = Gtk.CssProvider()
        css_provider.load_from_data(b"""
            .main-window {
                background-color: #121212;
                padding: 25px;
            }
            .header-title {
                font-family: 'Inter', 'Outfit', 'sans-serif';
                font-size: 22px;
                font-weight: 800;
                color: #ffffff;
            }
            .header-subtitle {
                font-family: 'Inter', 'Outfit', 'sans-serif';
                font-size: 12px;
                color: #888888;
                margin-bottom: 20px;
            }
            .card-box {
                background-color: #1a1a1a;
                border: 1px solid #2e2e2e;
                border-radius: 8px;
                padding: 15px 20px;
            }
            .card-title {
                font-family: 'Inter', 'sans-serif';
                font-size: 15px;
                font-weight: 600;
                color: #ffffff;
            }
            .card-desc {
                font-family: 'Inter', 'sans-serif';
                font-size: 11px;
                color: #aaaaaa;
            }
            button {
                transition: background-image 0.1s ease-in-out, background-color 0.1s ease-in-out, box-shadow 0.1s ease-in-out;
            }
            .btn-action {
                font-family: 'Inter', 'sans-serif';
                font-size: 12px;
                font-weight: bold;
                color: white;
                border-radius: 6px;
                padding: 10px 16px;
                border: none;
            }
            .btn-blue {
                background-image: linear-gradient(to bottom, #1e70e0, #1555b3);
                box-shadow: 0 2px 4px rgba(0,0,0,0.3);
            }
            .btn-blue:hover {
                background-image: linear-gradient(to bottom, #3b88f5, #1e70e0);
                box-shadow: 0 4px 8px rgba(0,0,0,0.4);
            }
            .btn-blue:active {
                background-image: linear-gradient(to bottom, #1555b3, #0f3d82);
                box-shadow: inset 0 2px 4px rgba(0,0,0,0.5);
            }
            .btn-purple {
                background-image: linear-gradient(to bottom, #8a2be2, #5f00b3);
                box-shadow: 0 2px 4px rgba(0,0,0,0.3);
            }
            .btn-purple:hover {
                background-image: linear-gradient(to bottom, #a34df2, #8a2be2);
                box-shadow: 0 4px 8px rgba(0,0,0,0.4);
            }
            .btn-purple:active {
                background-image: linear-gradient(to bottom, #5f00b3, #430080);
                box-shadow: inset 0 2px 4px rgba(0,0,0,0.5);
            }
            .btn-green {
                background-image: linear-gradient(to bottom, #2ea44f, #2c974b);
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
        """)
        Gtk.StyleContext.add_provider_for_screen(
            Gdk.Screen.get_default(),
            css_provider,
            Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION
        )

        self.init_ui()

    def init_ui(self):
        # Main layout container
        main_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=0)
        main_box.get_style_context().add_class("main-window")
        self.add(main_box)

        # Header Title
        title_label = Gtk.Label(label="NEGICC Station")
        title_label.get_style_context().add_class("header-title")
        title_label.set_xalign(0.0)
        main_box.pack_start(title_label, False, False, 0)

        # Header Subtitle
        subtitle_label = Gtk.Label(label="Sony A7R4 Film Scanning Control Panel")
        subtitle_label.get_style_context().add_class("header-subtitle")
        subtitle_label.set_xalign(0.0)
        main_box.pack_start(subtitle_label, False, False, 0)

        # Cards container (Vertical Stack)
        cards_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12)
        main_box.pack_start(cards_box, True, True, 0)

        # Script paths resolved relative to this script
        self.script_dir = os.path.dirname(os.path.abspath(__file__))

        # Card 1: Crosstalk Calibration
        card1 = self.create_launcher_card(
            title="Crosstalk Calibration",
            desc="Calibrate RGB pixel sensor crosstalk matrices using raw R, G, B negative exposures.",
            btn_label="Open Calibration",
            btn_class="btn-blue",
            target_script="ui_crosstalk_correction.py"
        )
        cards_box.pack_start(card1, True, True, 0)

        # Card 2: Film Profiling
        card2 = self.create_launcher_card(
            title="Film Profiling",
            desc="Expose IT8 targets to scan patches, perform curve fitting, and compile custom ICC profiles.",
            btn_label="Open Profiling",
            btn_class="btn-purple",
            target_script="ui_film_profiling.py"
        )
        cards_box.pack_start(card2, True, True, 0)

        # Card 3: Capture Station
        card3 = self.create_launcher_card(
            title="Capture & Scan",
            desc="Connect to A7R4 over USB, trigger captures, apply linear inversions, and save 16-bit TIFFs.",
            btn_label="Open Capture",
            btn_class="btn-green",
            target_script="ui_capture.py"
        )
        cards_box.pack_start(card3, True, True, 0)

        self.connect("destroy", Gtk.main_quit)
        self.show_all()

    def create_launcher_card(self, title, desc, btn_label, btn_class, target_script):
        # Card container (Horizontal for side-by-side alignment of text and button)
        card = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=15)
        card.get_style_context().add_class("card-box")

        # Vertical text box for Title and Description
        text_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)

        # Card Title
        lbl_title = Gtk.Label(label=title)
        lbl_title.get_style_context().add_class("card-title")
        lbl_title.set_xalign(0.0)
        text_box.pack_start(lbl_title, False, False, 0)

        # Card Description
        lbl_desc = Gtk.Label(label=desc)
        lbl_desc.get_style_context().add_class("card-desc")
        lbl_desc.set_line_wrap(True)
        lbl_desc.set_xalign(0.0)
        lbl_desc.set_yalign(0.0)
        text_box.pack_start(lbl_desc, True, True, 0)

        card.pack_start(text_box, True, True, 0)

        # Launch Button vertical container to handle alignment
        btn_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        btn = Gtk.Button(label=btn_label)
        btn.get_style_context().add_class("btn-action")
        btn.get_style_context().add_class(btn_class)
        btn.set_valign(Gtk.Align.CENTER)
        btn.connect("clicked", self.on_launch_clicked, target_script)
        btn_box.pack_start(btn, True, False, 0)
        
        card.pack_end(btn_box, False, False, 0)

        return card

    def on_launch_clicked(self, button, target_script):
        if getattr(self, 'is_launching', False):
            return
        self.is_launching = True

        # Check if this script window is already running
        running_proc = self.processes.get(target_script)
        if running_proc is not None:
            if running_proc.poll() is None:
                self.show_info_dialog("Already Running", f"An instance of '{target_script}' is already running.")
                self.is_launching = False
                return

        script_path = os.path.join(self.script_dir, target_script)
        if not os.path.exists(script_path):
            self.show_error_dialog(f"Target script not found:\n{script_path}")
            self.is_launching = False
            return

        try:
            # Launch via subprocess using the same python interpreter (venv)
            proc = subprocess.Popen([sys.executable, script_path])
            self.processes[target_script] = proc
            
            # Throttles double-clicks on launch buttons
            GLib.timeout_add(1000, lambda: setattr(self, 'is_launching', False) or False)
        except Exception as e:
            self.show_error_dialog(f"Failed to launch script:\n{str(e)}")
            self.is_launching = False

    def show_info_dialog(self, title, message):
        dialog = Gtk.MessageDialog(
            transient_for=self,
            flags=0,
            message_type=Gtk.MessageType.INFO,
            buttons=Gtk.ButtonsType.OK,
            text=title
        )
        dialog.format_secondary_text(message)
        dialog.run()
        dialog.destroy()

    def show_error_dialog(self, message):
        dialog = Gtk.MessageDialog(
            transient_for=self,
            flags=0,
            message_type=Gtk.MessageType.ERROR,
            buttons=Gtk.ButtonsType.OK,
            text="Launch Error"
        )
        dialog.format_secondary_text(message)
        dialog.run()
        dialog.destroy()

def main():
    app = LauncherWindow()
    Gtk.main()

if __name__ == '__main__':
    main()
