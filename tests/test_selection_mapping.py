import unittest
import numpy as np
import sys
import os

# Add src to python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))
from ui_capture import (
    map_raw_to_transformed_coords,
    map_transformed_to_raw_coords,
    map_raw_rect_to_transformed,
    map_transformed_rect_to_raw,
    apply_transforms_numpy
)

class TestSelectionMapping(unittest.TestCase):
    def test_coordinate_mapping_roundtrip(self):
        w_raw, h_raw = 300, 200
        grid = np.arange(h_raw * w_raw).reshape((h_raw, w_raw))

        options_hflip = [False, True]
        options_vflip = [False, True]
        options_rot = [0, 90, 180, 270]

        for hflip in options_hflip:
            for vflip in options_vflip:
                for rot in options_rot:
                    grid_trans = apply_transforms_numpy(grid, hflip, vflip, rot)
                    h_trans, w_trans = grid_trans.shape
                    
                    # Test points
                    for y_raw in [0, 50, 100, h_raw - 1]:
                        for x_raw in [0, 75, 150, w_raw - 1]:
                            # Forward map
                            x_t, y_t = map_raw_to_transformed_coords(x_raw, y_raw, w_raw, h_raw, hflip, vflip, rot)
                            
                            # Value equivalence check
                            self.assertEqual(grid[y_raw, x_raw], grid_trans[y_t, x_t])
                            
                            # Inverse map
                            x_r, y_r = map_transformed_to_raw_coords(x_t, y_t, w_trans, h_trans, hflip, vflip, rot)
                            self.assertEqual((x_r, y_r), (x_raw, y_raw))

    def test_rect_mapping_roundtrip(self):
        w_raw, h_raw = 300, 200
        rect_raw = (50, 40, 150, 120)
        
        options_hflip = [False, True]
        options_vflip = [False, True]
        options_rot = [0, 90, 180, 270]

        for hflip in options_hflip:
            for vflip in options_vflip:
                for rot in options_rot:
                    grid_trans = apply_transforms_numpy(np.zeros((h_raw, w_raw)), hflip, vflip, rot)
                    h_trans, w_trans = grid_trans.shape
                    
                    # Map to transformed space
                    rect_trans = map_raw_rect_to_transformed(rect_raw, w_raw, h_raw, hflip, vflip, rot)
                    self.assertIsNotNone(rect_trans)
                    tx1, ty1, tx2, ty2 = rect_trans
                    
                    # Ensure dimensions are bounded
                    self.assertTrue(0 <= tx1 <= tx2 <= w_trans)
                    self.assertTrue(0 <= ty1 <= ty2 <= h_trans)
                    
                    # Map back to raw
                    rect_back = map_transformed_rect_to_raw(rect_trans, w_trans, h_trans, hflip, vflip, rot)
                    self.assertEqual(rect_back, rect_raw)

if __name__ == '__main__':
    unittest.main()
