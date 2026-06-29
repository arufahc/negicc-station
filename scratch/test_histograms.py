import time
import numpy as np
import cv2
import json
import negicc_station
import film_profiling

raw_path = "/home/alpha/Pictures/Portra400_raw_0004.ARW"
print(f"Loading {raw_path}")
img = negicc_station.CapturedImage(0, 0.125, 100, [raw_path])
img_wrapper = img

with open("profiles/profile_Portra400_20260624_012038.json", "r") as f:
    prof_data = json.load(f)
    
prof_data['targets'] = [prof_data['targets'][5]] # Target 6
tgt = prof_data['targets'][0]
if 'icc_profile_base64' in tgt:
    prof_data['icc_profile_base64'] = tgt['icc_profile_base64']
profile = film_profiling.FilmProfile(prof_data)

film_base_rgb = np.array([8262.8, 4359.6, 6698.1])

w = 4752
h = 3168
cW = int(0.2 * w)
cH = int(0.2 * h)
x_start = (w - cW) // 2
y_start = (h - cH) // 2

# Global Gains to search
gains = np.arange(0.1, 3.1, 0.1).astype(np.float32)

print("\n--- Python + CUDA Pipeline ---")
t0 = time.time()
best_gain = 1.0
best_L_diff = float('inf')

for gain in gains:
    srgb_16 = film_profiling.convert_raw_to_numpy(
        img=img_wrapper, profile=profile,
        shutter_str=None, exposure_comp=gain, g_gain=1.0, b_gain=1.0, half=True,
        film_base_rgb=film_base_rgb, film_base_img=None, pipeline="cuda", to_uint8=False
    )
    srgb_float = srgb_16.astype(np.float32) / 65535.0
    lab = cv2.cvtColor(srgb_float, cv2.COLOR_RGB2Lab)
    crop = lab[y_start:y_start+cH, x_start:x_start+cW, :]
    L = crop[:, :, 0]
    mean_L = np.mean(L)
    print(f"Gain {gain:.1f}: old L={mean_L:.2f}")
    
    if abs(mean_L - 50.0) < best_L_diff:
        best_L_diff = abs(mean_L - 50.0)
        best_gain = gain

t1 = time.time()
print(f"Old Global Gain search took {t1 - t0:.2f}s, best gain: {best_gain:.2f}")

g_gains = []
b_gains = []
for g in np.arange(0.8, 1.21, 0.05):
    for b in np.arange(0.8, 1.21, 0.05):
        g_gains.append(g)
        b_gains.append(b)

g_gains = np.array(g_gains, dtype=np.float32)
b_gains = np.array(b_gains, dtype=np.float32)

best_g = 1.0
best_b = 1.0
min_cast = float('inf')

t2 = time.time()
for g, b in zip(g_gains, b_gains):
    srgb_16 = film_profiling.convert_raw_to_numpy(
        img=img_wrapper, profile=profile,
        shutter_str=None, exposure_comp=best_gain, g_gain=g, b_gain=b, half=True,
        film_base_rgb=film_base_rgb, film_base_img=None, pipeline="cuda", to_uint8=False
    )
    srgb_float = srgb_16.astype(np.float32) / 65535.0
    lab = cv2.cvtColor(srgb_float, cv2.COLOR_RGB2Lab)
    crop = lab[y_start:y_start+cH, x_start:x_start+cW, :]
    a = crop[:, :, 1]
    b_chan = crop[:, :, 2]
    cast = np.mean(np.square(a) + np.square(b_chan))
    
    if cast < min_cast:
        min_cast = cast
        best_g = g
        best_b = b
t3 = time.time()
print(f"Old GB Gain search took {t3 - t2:.2f}s, best g: {best_g:.2f}, best b: {best_b:.2f}")

print("\n--- Specialized CUDA Histograms Pipeline ---")

c_gains = np.ones_like(gains, dtype=np.float32)

target_val = getattr(profile, 'normalization_target', 55000.0)
exposure_ratio = 1.0 # assume t_base and t_scan are the same for this test
scale_r = (target_val / film_base_rgb[0]) * exposure_ratio if film_base_rgb[0] > 0 else 1.0
scale_g = (target_val / film_base_rgb[1]) * exposure_ratio if film_base_rgb[1] > 0 else 1.0
scale_b = (target_val / film_base_rgb[2]) * exposure_ratio if film_base_rgb[2] > 0 else 1.0
scales = np.array([scale_r, scale_g, scale_b])
merged_matrix = np.array(profile.crosstalk_matrix) * scales[:, np.newaxis]
flat_cc = merged_matrix.flatten().astype(float).tolist()

t4 = time.time()
hists_global = img.search_gains_histogram(
    half=True, crop_w=cW, crop_h=cH,
    crosstalk_matrix=flat_cc,
    it8_profile_path="",
    profile_film_base=[int(x) for x in profile.get_film_base_rgb()],
    film_base=film_base_rgb.astype(int).tolist(),
    global_gains=gains.tolist(),
    g_gains=c_gains.tolist(),
    b_gains=c_gains.tolist(),
    it8_profile_bytes=profile.icc_profile_bytes
)
# hists_global is (30, 3, 65536)
best_gain_hist = 1.0
best_L_diff_hist = float('inf')
bins_L = np.arange(65536) / 655.35
total_pixels = cW * cH
for i, gain in enumerate(gains):
    L_hist = hists_global[i, 0, :]
    mean_L_hist = np.sum(bins_L * L_hist) / total_pixels
    print(f"Gain {gain:.1f}: new L={mean_L_hist:.2f}")
    if abs(mean_L_hist - 50.0) < best_L_diff_hist:
        best_L_diff_hist = abs(mean_L_hist - 50.0)
        best_gain_hist = gain
        
t5 = time.time()
print(f"New Global Gain search took {t5 - t4:.2f}s, best gain: {best_gain_hist:.2f}")

# GB search
gg_global = np.full_like(g_gains, best_gain_hist, dtype=np.float32)
t6 = time.time()
hists_gb = img.search_gains_histogram(
    half=True, crop_w=cW, crop_h=cH,
    crosstalk_matrix=flat_cc,
    it8_profile_path="",
    profile_film_base=[int(x) for x in profile.get_film_base_rgb()],
    film_base=film_base_rgb.astype(int).tolist(),
    global_gains=gg_global.tolist(),
    g_gains=g_gains.tolist(),
    b_gains=b_gains.tolist(),
    it8_profile_bytes=profile.icc_profile_bytes
)
best_g_hist = 1.0
best_b_hist = 1.0
min_cast_hist = float('inf')

bins_a = np.arange(65536) / 256.0 - 128.0
bins_b = np.arange(65536) / 256.0 - 128.0
bins_a2 = np.square(bins_a)
bins_b2 = np.square(bins_b)

for i, (g, b) in enumerate(zip(g_gains, b_gains)):
    a_hist = hists_gb[i, 1, :]
    b_hist = hists_gb[i, 2, :]
    cast = (np.sum(bins_a2 * a_hist) + np.sum(bins_b2 * b_hist)) / total_pixels
    if cast < min_cast_hist:
        min_cast_hist = cast
        best_g_hist = g
        best_b_hist = b
t7 = time.time()
print(f"New GB Gain search took {t7 - t6:.2f}s, best g: {best_g_hist:.2f}, best b: {best_b_hist:.2f}")

print("\n--- Summary ---")
print(f"Total Old Time: {t1 - t0 + t3 - t2:.2f}s")
print(f"Total New Time: {t5 - t4 + t7 - t6:.2f}s")
print(f"Speedup: {(t1 - t0 + t3 - t2) / (t5 - t4 + t7 - t6):.2f}x")
print(f"Old result: Gain={best_gain:.2f}, G={best_g:.2f}, B={best_b:.2f}")
print(f"New result: Gain={best_gain_hist:.2f}, G={best_g_hist:.2f}, B={best_b_hist:.2f}")
