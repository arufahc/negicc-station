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
        self.set_default_size(700, 360)
        self.set_resizable(False)
        self.set_position(Gtk.WindowPosition.CENTER)

        # Force GTK dark theme for premium aesthetics
        settings = Gtk.Settings.get_default()
        settings.set_property("gtk-application-prefer-dark-theme", True)

        # Custom CSS for custom buttons and layouts
        css_provider = Gtk.CssProvider()
        css_provider.load_from_data(b"""
            .main-window {
                background-color: #121212;
                padding: 30px;
            }
            .header-title {
                font-family: 'Inter', 'Outfit', 'sans-serif';
                font-size: 26px;
                font-weight: 800;
                color: #ffffff;
            }
            .header-subtitle {
                font-family: 'Inter', 'Outfit', 'sans-serif';
                font-size: 13px;
                color: #888888;
                margin-bottom: 25px;
            }
            .card-box {
                background-color: #1a1a1a;
                border: 1px solid #2e2e2e;
                border-radius: 8px;
                padding: 20px;
            }
            .card-title {
                font-family: 'Inter', 'sans-serif';
                font-size: 16px;
                font-weight: 600;
                color: #ffffff;
            }
            .card-desc {
                font-family: 'Inter', 'sans-serif';
                font-size: 12px;
                color: #aaaaaa;
            }
            .btn-action {
                font-family: 'Inter', 'sans-serif';
                font-size: 13px;
                font-weight: bold;
                color: white;
                border-radius: 6px;
                padding: 10px 16px;
                border: none;
            }
            .btn-blue {
                background-image: linear-gradient(to bottom, #1e70e0, #1555b3);
            }
            .btn-blue:hover {
                background-image: linear-gradient(to bottom, #3b88f5, #1e70e0);
            }
            .btn-purple {
                background-image: linear-gradient(to bottom, #8a2be2, #5f00b3);
            }
            .btn-purple:hover {
                background-image: linear-gradient(to bottom, #a34df2, #8a2be2);
            }
            .btn-green {
                background-image: linear-gradient(to bottom, #2ea44f, #2c974b);
            }
            .btn-green:hover {
                background-image: linear-gradient(to bottom, #3bc262, #2ea44f);
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
        title_label = Gtk.Label()
        title_label.set_markup("<span class='header-title'>NEGICC Station</span>")
        title_label.set_xalign(0.0)
        main_box.pack_start(title_label, False, False, 0)

        # Header Subtitle
        subtitle_label = Gtk.Label()
        subtitle_label.set_markup("<span class='header-subtitle'>Sony A7R4 Film Scanning Control Panel</span>")
        subtitle_label.set_xalign(0.0)
        main_box.pack_start(subtitle_label, False, False, 0)

        # Cards container (Horizontal)
        cards_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=15)
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
        # Card container
        card = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12)
        card.get_style_context().add_class("card-box")

        # Card Title
        lbl_title = Gtk.Label()
        lbl_title.set_markup(f"<span class='card-title'>{title}</span>")
        lbl_title.set_xalign(0.0)
        card.pack_start(lbl_title, False, False, 0)

        # Card Description
        lbl_desc = Gtk.Label()
        lbl_desc.set_markup(f"<span class='card-desc'>{desc}</span>")
        lbl_desc.set_line_wrap(True)
        lbl_desc.set_xalign(0.0)
        lbl_desc.set_yalign(0.0)
        card.pack_start(lbl_desc, True, True, 0)

        # Launch Button
        btn = Gtk.Button(label=btn_label)
        btn.get_style_context().add_class("btn-action")
        btn.get_style_context().add_class(btn_class)
        btn.connect("clicked", self.on_launch_clicked, target_script)
        card.pack_end(btn, False, False, 0)

        return card

    def on_launch_clicked(self, button, target_script):
        script_path = os.path.join(self.script_dir, target_script)
        if not os.path.exists(script_path):
            self.show_error_dialog(f"Target script not found:\n{script_path}")
            return

        try:
            # Launch via subprocess using the same python interpreter (venv)
            subprocess.Popen([sys.executable, script_path])
        except Exception as e:
            self.show_error_dialog(f"Failed to launch script:\n{str(e)}")

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
