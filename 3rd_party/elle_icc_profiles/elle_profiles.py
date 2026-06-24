import os
import re

def get_srgb_srgbtrc_bytes():
    """Parses sRGB_elle_V2_srgbtrc.h to extract and return the raw binary ICC profile bytes."""
    dir_path = os.path.dirname(os.path.abspath(__file__))
    h_path = os.path.join(dir_path, "sRGB_elle_V2_srgbtrc.h")
    with open(h_path, 'r') as f:
        content = f.read()
    hex_vals = re.findall(r'0x[0-9a-fA-F]{2}', content)
    return bytes(int(x, 16) for x in hex_vals)

def get_srgb_g10_bytes():
    """Parses sRGB_elle_V2_g10.h to extract and return the raw binary ICC profile bytes."""
    dir_path = os.path.dirname(os.path.abspath(__file__))
    h_path = os.path.join(dir_path, "sRGB_elle_V2_g10.h")
    with open(h_path, 'r') as f:
        content = f.read()
    hex_vals = re.findall(r'0x[0-9a-fA-F]{2}', content)
    return bytes(int(x, 16) for x in hex_vals)
