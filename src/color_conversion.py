#!/usr/bin/env python3
import os
import sys
import ctypes
import numpy as np
from imageio.plugins.tifffile import _tifffile

# =============================================================================
# Little CMS (lcms2) Library Binding & Parsing Layer
# =============================================================================

# Structs matching ARM64 Linux layout exactly
class cmsInterpParams(ctypes.Structure):
    _fields_ = [
        ("ContextID", ctypes.c_void_p),
        ("dwFlags", ctypes.c_uint32),
        ("nInputs", ctypes.c_uint32),
        ("nOutputs", ctypes.c_uint32),
        ("nSamples", ctypes.c_uint32 * 15),
        ("Domain", ctypes.c_uint32 * 15),
        ("opta", ctypes.c_uint32 * 15),
        ("Table", ctypes.c_void_p),
        ("Interpolation", ctypes.c_void_p),  # cmsInterpFunction union (8-byte pointer)
    ]

class _cmsStageToneCurvesData(ctypes.Structure):
    _fields_ = [
        ("nCurves", ctypes.c_uint32),
        ("TheCurves", ctypes.POINTER(ctypes.c_void_p))
    ]

class _cmsStageCLutData(ctypes.Structure):
    _fields_ = [
        ("Tab", ctypes.c_void_p),
        ("Params", ctypes.POINTER(cmsInterpParams)),
        ("nEntries", ctypes.c_uint32),
        ("HasFloatValues", ctypes.c_uint32),
    ]

class _cmsStageMatrixData(ctypes.Structure):
    _fields_ = [
        ("Double", ctypes.POINTER(ctypes.c_double)),
        ("Offset", ctypes.POINTER(ctypes.c_double)),
    ]

class LittleCMS:
    def __init__(self):
        if sys.platform.startswith("win"):
            lib_name = "lcms2.dll"
        elif sys.platform.startswith("darwin"):
            lib_name = "liblcms2.dylib"
        else:
            lib_name = "liblcms2.so"

        try:
            self.lib = ctypes.CDLL(lib_name)
        except OSError as err:
            raise ImportError(
                f"Could not load native lcms2 library: {err}. "
                "Ensure that Little CMS 2 is installed in the system library path."
            )
        self._setup_bindings()

    def _setup_bindings(self):
        self.lib.cmsOpenProfileFromMem.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        self.lib.cmsOpenProfileFromMem.restype = ctypes.c_void_p

        self.lib.cmsCloseProfile.argtypes = [ctypes.c_void_p]
        self.lib.cmsCloseProfile.restype = ctypes.c_bool

        self.lib.cmsReadTag.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        self.lib.cmsReadTag.restype = ctypes.c_void_p

        self.lib.cmsPipelineGetPtrToFirstStage.argtypes = [ctypes.c_void_p]
        self.lib.cmsPipelineGetPtrToFirstStage.restype = ctypes.c_void_p

        self.lib.cmsStageNext.argtypes = [ctypes.c_void_p]
        self.lib.cmsStageNext.restype = ctypes.c_void_p

        self.lib.cmsStageType.argtypes = [ctypes.c_void_p]
        self.lib.cmsStageType.restype = ctypes.c_uint32

        self.lib.cmsStageData.argtypes = [ctypes.c_void_p]
        self.lib.cmsStageData.restype = ctypes.c_void_p

        self.lib.cmsGetToneCurveEstimatedTableEntries.argtypes = [ctypes.c_void_p]
        self.lib.cmsGetToneCurveEstimatedTableEntries.restype = ctypes.c_uint32

        self.lib.cmsGetToneCurveEstimatedTable.argtypes = [ctypes.c_void_p]
        self.lib.cmsGetToneCurveEstimatedTable.restype = ctypes.POINTER(ctypes.c_uint16)

        self.lib.cmsCreateXYZProfile.argtypes = []
        self.lib.cmsCreateXYZProfile.restype = ctypes.c_void_p

        self.lib.cmsCreateTransform.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32]
        self.lib.cmsCreateTransform.restype = ctypes.c_void_p

        self.lib.cmsDoTransform.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32]
        self.lib.cmsDoTransform.restype = None

# =============================================================================
# Helper Utilities
# =============================================================================

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

def make_gamma_curve(gamma):
    pwr = 1.0 / gamma
    i_vals = np.arange(65535, dtype=np.float64)
    r_vals = i_vals / 65535.0
    curve = np.zeros(65536, dtype=np.uint16)
    curve[:65535] = np.clip(65536.0 * (r_vals ** pwr), 0, 65535).astype(np.uint16)
    curve[65535] = 65535
    return curve

def adjust_correction_matrix(cc_matrix, exposure_comp, profile_film_base=None, film_base=None):
    if len(cc_matrix) == 0:
        return []
    
    r_coef = np.array(cc_matrix[0:3], dtype=np.float32)
    g_coef = np.array(cc_matrix[3:6], dtype=np.float32)
    b_coef = np.array(cc_matrix[6:9], dtype=np.float32)
    
    p_fb = np.array(profile_film_base if profile_film_base is not None else [1, 1, 1], dtype=np.float32)
    c_fb = np.array(film_base if film_base is not None else [1, 1, 1], dtype=np.float32)
    
    cc_average_r = np.dot(r_coef, c_fb)
    cc_average_g = np.dot(g_coef, c_fb)
    cc_average_b = np.dot(b_coef, c_fb)
    
    cc_profile_r = np.dot(r_coef, p_fb)
    cc_profile_g = np.dot(g_coef, p_fb)
    cc_profile_b = np.dot(b_coef, p_fb)
    
    g_scale = 1.0
    b_scale = 1.0
    
    if cc_average_g > 0 and cc_profile_r > 0:
        g_scale = (cc_average_r / cc_average_g) * (cc_profile_g / cc_profile_r)
    if cc_average_b > 0 and cc_profile_r > 0:
        b_scale = (cc_average_r / cc_average_b) * (cc_profile_b / cc_profile_r)
        
    r_coef *= exposure_comp
    g_coef *= (g_scale * exposure_comp)
    b_coef *= (b_scale * exposure_comp)
    
    return [
        r_coef[0], r_coef[1], r_coef[2],
        g_coef[0], g_coef[1], g_coef[2],
        b_coef[0], b_coef[1], b_coef[2]
    ]

# =============================================================================
# Core Pipeline Conversion Function
# =============================================================================

# Python pipeline design notes:
# - For colorspace in ("srgb", "srgb-g10"):
#   - Manually parses the film profile stages (TRCs and cLUT) using Little CMS via ctypes.
#   - Applies them manually, projects XYZ to sRGB via a hardcoded Bradford matrix (D50->D65) and XYZ-to-sRGB matrix.
#   - Applies a predefined sRGB gamma curve.
#   - Saves the output TIFF file and embeds/attaches the sRGB profile bytes (elle profile) as Tag 34675.
#     The profile file itself is NOT used for pixel conversion (only as metadata).
# - For custom colorspaces (non-sRGB paths):
#   - Uses a real Little CMS transform (cmsCreateTransform/cmsDoTransform) from XYZ to the custom output colorspace profile.
#   - In this case, the output profile is actually used for pixel conversion.
def convert_raw_to_tiff(img, profile, output_path, colorspace="srgb", clut_path=None, shutter_str=None, exposure_comp=1.0, half=True, film_base_rgb=None, film_base_img=None):
    """
    Decodes and converts RAW image entirely in Python using Little CMS ctypes metadata extraction
    and NumPy-vectorized transformations. Saves output as 16-bit linear/sRGB TIFF with embedded ICC profile.
    """
    # 1. Resolve input Film ICC Profile bytes
    icc_bytes = None
    if clut_path is None:
        if getattr(profile, 'icc_profile_bytes', None):
            icc_bytes = profile.icc_profile_bytes
        else:
            raise ValueError("clut_path is None and profile has no self-contained ICC profile bytes.")
    else:
        with open(clut_path, 'rb') as f:
            icc_bytes = f.read()

    # 2. Resolve output Working Space Profile bytes
    src_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(src_dir)
    
    if colorspace in ("srgb", "srgb-g10"):
        elle_dir = os.path.join(project_dir, "3rd_party/elle_icc_profiles")
        if elle_dir not in sys.path:
            sys.path.insert(0, elle_dir)
        import elle_profiles
        if colorspace == "srgb":
            out_icc_bytes = elle_profiles.get_srgb_srgbtrc_bytes()
        else:
            out_icc_bytes = elle_profiles.get_srgb_g10_bytes()
    else:
        # Custom profile path
        if os.path.exists(colorspace):
            with open(colorspace, 'rb') as f:
                out_icc_bytes = f.read()
        else:
            raise FileNotFoundError(f"Output colorspace profile not found: {colorspace}")

    # 3. Determine film base RGB values
    if film_base_rgb is not None:
        fb_r, fb_g, fb_b = film_base_rgb
    else:
        fb_r = profile.film_base['r_avg']
        fb_g = profile.film_base['g_avg']
        fb_b = profile.film_base['b_avg']

    # 4. Compute exposure ratio
    if shutter_str is not None:
        scan_num, scan_den = parse_shutter_speed(shutter_str)
        t_scan = scan_num / scan_den
    else:
        t_scan = img.shutter_speed

    # Shutter speed and ISO of film base
    if film_base_img is not None:
        t_base = film_base_img.shutter_speed
        iso_base = film_base_img.iso
    else:
        print("[Warning] color_conversion: film_base_img is None. Falling back to profile film_base_shutter and film_base_iso.", file=sys.stdout)
        sys.stdout.flush()
        base_num, base_den = parse_shutter_speed(profile.film_base_shutter)
        t_base = base_num / base_den
        iso_base = profile.film_base_iso

    iso_scan = img.iso

    # Exposure: t * ISO
    exposure_profile = t_base * (iso_base / 100.0)
    exposure_scan = t_scan * (iso_scan / 100.0)
    exposure_ratio = exposure_profile / exposure_scan if exposure_scan > 0 else 1.0

    # Scale factors to map film base at current exposure to normalization_target
    target_val = profile.normalization_target
    scale_r = (target_val / fb_r) * exposure_ratio if fb_r > 0 else 1.0
    scale_g = (target_val / fb_g) * exposure_ratio if fb_g > 0 else 1.0
    scale_b = (target_val / fb_b) * exposure_ratio if fb_b > 0 else 1.0

    # Merge normalization scale factors into crosstalk matrix on the fly
    raw_crosstalk = np.array(profile.crosstalk_matrix)
    scales = np.array([scale_r, scale_g, scale_b])
    merged_matrix = raw_crosstalk * scales[:, np.newaxis]
    flat_merged_matrix = merged_matrix.flatten().tolist()

    # 5. Load uncorrected linear RAW image to NumPy (half or full size)
    arr_raw = img.to_numpy(half=half, crosstalk_matrix=None)
    
    # Convert to float32 normalized range [0.0, 1.0] immediately
    img_float = arr_raw.astype(np.float32) / 65535.0
    
    # 6. Apply adjusted crosstalk matrix in Python (entirely in float32, clipped to [0.0, 1.0])
    adjusted_cc = adjust_correction_matrix(flat_merged_matrix, exposure_comp, None, None)
    
    if len(adjusted_cc) > 0:
        r_c = img_float[..., 0] * adjusted_cc[0] + img_float[..., 1] * adjusted_cc[1] + img_float[..., 2] * adjusted_cc[2]
        g_c = img_float[..., 0] * adjusted_cc[3] + img_float[..., 1] * adjusted_cc[4] + img_float[..., 2] * adjusted_cc[5]
        b_c = img_float[..., 0] * adjusted_cc[6] + img_float[..., 1] * adjusted_cc[7] + img_float[..., 2] * adjusted_cc[8]
        img_float = np.stack([r_c, g_c, b_c], axis=-1)
        np.clip(img_float, 0.0, 1.0, out=img_float)

    # 7. Post-correction gamma removed (no-op)

    # 8. Parse Film ICC Profile Stages using Little CMS via ctypes
    lcms = LittleCMS()
    h_profile = lcms.lib.cmsOpenProfileFromMem(icc_bytes, len(icc_bytes))
    if not h_profile:
        raise ValueError("Failed to open Film ICC profile from memory.")

    try:
        # Read AtoB0 multi-stage lookup table tag (Tag ID: 0x41324230)
        h_pipeline = lcms.lib.cmsReadTag(h_profile, 0x41324230)
        if not h_pipeline:
            raise ValueError("The film profile does not contain a valid A2B0 lookup table tag.")

        stage_ptr = lcms.lib.cmsPipelineGetPtrToFirstStage(h_pipeline)
        
        while stage_ptr:
            stage_type = lcms.lib.cmsStageType(stage_ptr)
            
            # Curves stage: cmsSigCurveSetStage (0x63767374) or cmsSigCurvesStage (0x63757276)
            if stage_type == 0x63767374 or stage_type == 0x63757276:
                curve_data = ctypes.cast(lcms.lib.cmsStageData(stage_ptr), ctypes.POINTER(_cmsStageToneCurvesData))
                num_curves = curve_data.contents.nCurves
                curve_pointers = [curve_data.contents.TheCurves[i] for i in range(num_curves)]
                
                # Apply 1D interpolation channel-wise
                for c in range(min(3, num_curves)):
                    curve_ptr = curve_pointers[c]
                    entries = lcms.lib.cmsGetToneCurveEstimatedTableEntries(curve_ptr)
                    table_ptr = lcms.lib.cmsGetToneCurveEstimatedTable(curve_ptr)
                    raw_table = np.ctypeslib.as_array(table_ptr, shape=(entries,))
                    normalized_trc = raw_table.astype(np.float32) / 65535.0
                    
                    x_ref = np.linspace(0.0, 1.0, entries, dtype=np.float32)
                    img_float[..., c] = np.interp(img_float[..., c], x_ref, normalized_trc)
            
            # Matrix stage: cmsSigMatrixStage (0x6d617478)
            elif stage_type == 0x6d617478:
                matrix_data = ctypes.cast(lcms.lib.cmsStageData(stage_ptr), ctypes.POINTER(_cmsStageMatrixData))
                # Double is a pointer to 9 floats, Offset is a pointer to 3 floats
                mat_vals = np.ctypeslib.as_array(matrix_data.contents.Double, shape=(9,)).astype(np.float32)
                offset_vals = np.ctypeslib.as_array(matrix_data.contents.Offset, shape=(3,)).astype(np.float32)
                
                mat_3x3 = mat_vals.reshape(3, 3)
                img_float = np.tensordot(img_float, mat_3x3, axes=(-1, 1)) + offset_vals
            
            # CLUT stage: cmsSigCLutStage (0x636c7574)
            elif stage_type == 0x636c7574:
                clut_data = ctypes.cast(lcms.lib.cmsStageData(stage_ptr), ctypes.POINTER(_cmsStageCLutData))
                params = clut_data.contents.Params.contents
                n_inputs = params.nInputs
                n_outputs = params.nOutputs
                nSamples = [params.nSamples[i] for i in range(n_inputs)]
                Domain = [params.Domain[i] for i in range(n_inputs)]
                
                total_elements = n_outputs
                for ns in nSamples:
                    total_elements *= ns
                
                reshape_dims = tuple(nSamples) + (n_outputs,)
                if clut_data.contents.HasFloatValues:
                    raw_lut_ptr = ctypes.cast(clut_data.contents.Tab, ctypes.POINTER(ctypes.c_float))
                    flat_lut = np.ctypeslib.as_array(raw_lut_ptr, shape=(total_elements,))
                    clut_grid = flat_lut.astype(np.float32).reshape(reshape_dims)
                else:
                    raw_lut_ptr = ctypes.cast(clut_data.contents.Tab, ctypes.POINTER(ctypes.c_uint16))
                    flat_lut = np.ctypeslib.as_array(raw_lut_ptr, shape=(total_elements,))
                    clut_grid = (flat_lut.astype(np.float32) / 65535.0).reshape(reshape_dims)
                
                # Perform vectorized tetrahedral interpolation
                scaled = img_float * np.array(Domain, dtype=np.float32)
                floor_idx = np.floor(scaled).astype(np.int32)
                ceil_idx = np.clip(floor_idx + 1, 0, np.array(Domain))
                
                delta = scaled - floor_idx
                dr = delta[..., 0]
                dg = delta[..., 1]
                db = delta[..., 2]
                
                dr_e = dr[..., np.newaxis]
                dg_e = dg[..., np.newaxis]
                db_e = db[..., np.newaxis]
                
                rf, gf, bf = floor_idx[..., 0], floor_idx[..., 1], floor_idx[..., 2]
                rc, gc, bc = ceil_idx[..., 0], ceil_idx[..., 1], ceil_idx[..., 2]
                
                v000 = clut_grid[rf, gf, bf]
                v100 = clut_grid[rc, gf, bf]
                v010 = clut_grid[rf, gc, bf]
                v110 = clut_grid[rc, gc, bf]
                v001 = clut_grid[rf, gf, bc]
                v101 = clut_grid[rc, gf, bc]
                v011 = clut_grid[rf, gc, bc]
                v111 = clut_grid[rc, gc, bc]
                
                res = np.zeros(img_float.shape[:-1] + (n_outputs,), dtype=np.float32)
                
                # Masks for the 6 tetrahedra
                m1 = (dr >= dg) & (dg >= db)
                m2 = (dr >= db) & (db > dg)
                m3 = (db > dr) & (dr >= dg)
                m4 = (dg > dr) & (dr >= db)
                m5 = (dg >= db) & (db > dr)
                m6 = (db > dg) & (dg > dr)
                
                if np.any(m1):
                    res[m1] = (v000[m1] * (1.0 - dr_e[m1]) +
                               v100[m1] * (dr_e[m1] - dg_e[m1]) +
                               v110[m1] * (dg_e[m1] - db_e[m1]) +
                               v111[m1] * db_e[m1])
                if np.any(m2):
                    res[m2] = (v000[m2] * (1.0 - dr_e[m2]) +
                               v100[m2] * (dr_e[m2] - db_e[m2]) +
                               v101[m2] * (db_e[m2] - dg_e[m2]) +
                               v111[m2] * dg_e[m2])
                if np.any(m3):
                    res[m3] = (v000[m3] * (1.0 - db_e[m3]) +
                               v001[m3] * (db_e[m3] - dr_e[m3]) +
                               v101[m3] * (dr_e[m3] - dg_e[m3]) +
                               v111[m3] * dg_e[m3])
                if np.any(m4):
                    res[m4] = (v000[m4] * (1.0 - dg_e[m4]) +
                               v010[m4] * (dg_e[m4] - dr_e[m4]) +
                               v110[m4] * (dr_e[m4] - db_e[m4]) +
                               v111[m4] * db_e[m4])
                if np.any(m5):
                    res[m5] = (v000[m5] * (1.0 - dg_e[m5]) +
                               v010[m5] * (dg_e[m5] - db_e[m5]) +
                               v011[m5] * (db_e[m5] - dr_e[m5]) +
                               v111[m5] * dr_e[m5])
                if np.any(m6):
                    res[m6] = (v000[m6] * (1.0 - db_e[m6]) +
                               v001[m6] * (db_e[m6] - dg_e[m6]) +
                               v011[m6] * (dg_e[m6] - dr_e[m6]) +
                               v111[m6] * dr_e[m6])
                
                img_float = res
                
            stage_ptr = lcms.lib.cmsStageNext(stage_ptr)
            
        img_float = img_float * (65535.0 / 32768.0)

    finally:
        lcms.lib.cmsCloseProfile(h_profile)

    # 9. Transform D50 PCS XYZ to Output Color Space
    if colorspace in ("srgb", "srgb-g10"):
        # Bradford Chromatic Adaptation Matrix (D50 -> D65)
        m_adapt = np.array([
            [ 0.9555766, -0.0230393,  0.0631636],
            [-0.0282895,  1.0099416,  0.0210077],
            [ 0.0122982, -0.0204830,  1.3299098]
        ], dtype=np.float32)
        
        xyz_d65 = np.tensordot(img_float, m_adapt, axes=(-1, 1))
        
        # XYZ to Linear sRGB Matrix Projection
        m_xyz_to_srgb = np.array([
            [ 3.2406255, -1.5372080, -0.4986286],
            [-0.9689307,  1.8757561,  0.0415175],
            [ 0.0557101, -0.2040211,  1.0569959]
        ], dtype=np.float32)
        
        linear_rgb = np.tensordot(xyz_d65, m_xyz_to_srgb, axes=(-1, 1))
        clamped_rgb = np.clip(linear_rgb, 0.0, 1.0)
        
        if colorspace == "srgb":
            # Apply Standard Piece-wise sRGB EOTF Mapping
            srgb_mapped = np.where(
                clamped_rgb <= 0.0031308,
                clamped_rgb * 12.92,
                (clamped_rgb ** (1.0 / 2.4)) * 1.055 - 0.055
            )
        else:
            # Linear sRGB (srgb-g10)
            srgb_mapped = clamped_rgb
            
        srgb_uint16 = (srgb_mapped * 65535.0).round().astype(np.uint16)
    else:
        # Custom colorspace profile using Little CMS transform (as a robust fallback)
        h_xyz_profile = lcms.lib.cmsCreateXYZProfile()
        h_out_profile = lcms.lib.cmsOpenProfileFromMem(out_icc_bytes, len(out_icc_bytes))
        
        # TYPE_XYZ_FLT: 4784156, TYPE_RGB_16: 262170
        transform = lcms.lib.cmsCreateTransform(h_xyz_profile, 4784156, h_out_profile, 262170, 0, 0)
        if not transform:
            lcms.lib.cmsCloseProfile(h_out_profile)
            lcms.lib.cmsCloseProfile(h_xyz_profile)
            raise ValueError("Failed to create Little CMS transform to custom output colorspace.")
            
        try:
            h, w, c = img_float.shape
            srgb_uint16 = np.zeros_like(img_float, dtype=np.uint16)
            
            # Pass pointers directly to cmsDoTransform
            in_ptr = img_float.ctypes.data_as(ctypes.c_void_p)
            out_ptr = srgb_uint16.ctypes.data_as(ctypes.c_void_p)
            lcms.lib.cmsDoTransform(transform, in_ptr, out_ptr, w * h)
        finally:
            lcms.lib.cmsCloseProfile(h_out_profile)
            lcms.lib.cmsCloseProfile(h_xyz_profile)

    # 10. Save array to TIFF with the output profile embedded as Tag 34675
    _tifffile.imsave(
        output_path,
        srgb_uint16,
        extratags=[(34675, 'B', len(out_icc_bytes), out_icc_bytes, True)]
    )
    
    return True
