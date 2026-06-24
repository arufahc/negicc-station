import sys
import os
import numpy as np
import json

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))
from film_profiling import FilmProfile
from target_selection import find_best_target_index

def test_selection():
    # Mock profile
    mock_json = {
        "film_base": {
            "g": {"avg": 10000.0}
        },
        "targets": [
            {
                "name": "Target 1 (Dark)",
                "patches": {
                    f"gs{i}": {"g": 10000.0 * (1.0 - (i/23.0)*0.8)} for i in range(24) # Transmittance 1.0 to 0.2
                }
            },
            {
                "name": "Target 2 (Mid)",
                "patches": {
                    f"gs{i}": {"g": 10000.0 * (0.8 - (i/23.0)*0.6)} for i in range(24) # Transmittance 0.8 to 0.2
                }
            },
            {
                "name": "Target 3 (Bright)",
                "patches": {
                    f"gs{i}": {"g": 10000.0 * (0.4 - (i/23.0)*0.3)} for i in range(24) # Transmittance 0.4 to 0.1
                }
            }
        ]
    }
    
    profile = FilmProfile(mock_json)
    
    # Mock raw image (100x100) with dynamic range mimicking "Mid" target
    raw_img = np.zeros((100, 100, 3), dtype=np.float32)
    # Give it a 98% percentile of ~0.75 transmittance, 2% of ~0.25
    raw_img[..., 1] = np.random.uniform(2500, 7500, size=(100, 100)) 
    
    film_base_rgb = (10000.0, 10000.0, 10000.0)
    
    best_idx, dist = find_best_target_index(profile, raw_img, film_base_rgb)
    print(f"Selected target index: {best_idx}, Distance to mid-grey: {dist}")
    
    # For this mock, Target 2 should be the best fit
    assert best_idx == 1, f"Expected 1, got {best_idx}"
    print("Test passed!")

def test_crosstalk_only_profile():
    profile_path = os.path.join(os.path.dirname(__file__), '../profiles/ILCE-7RM4_crosstalk_profile.json')
    profile = FilmProfile(profile_path)
    
    assert profile.target_name == 'None'
    assert profile.crosstalk_matrix is not None
    assert profile.crosstalk_matrix.shape == (3, 3)
    assert not np.allclose(profile.crosstalk_matrix, np.eye(3))
    print("Crosstalk-only profile test passed!")

if __name__ == "__main__":
    test_selection()
    test_crosstalk_only_profile()
