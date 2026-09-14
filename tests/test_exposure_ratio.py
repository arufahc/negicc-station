import os
import sys
import unittest
from types import SimpleNamespace

# Add src to python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))
from film_profiling import (
    FilmProfile,
    parse_shutter_speed,
    shutter_to_seconds,
    compute_exposure_ratio,
)


class TestExposureRatio(unittest.TestCase):
    def test_parse_shutter_speed_strings(self):
        self.assertEqual(parse_shutter_speed("1/8s"), (1, 8))
        self.assertEqual(parse_shutter_speed("1/125s"), (1, 125))
        self.assertEqual(parse_shutter_speed("1s"), (1, 1))
        self.assertEqual(parse_shutter_speed("30s"), (30, 1))
        self.assertEqual(parse_shutter_speed("0.5s"), (5, 10))
        self.assertEqual(parse_shutter_speed("2.5s"), (25, 10))

    def test_parse_shutter_speed_numbers(self):
        self.assertEqual(parse_shutter_speed(1), (1, 1))
        self.assertEqual(parse_shutter_speed(1.0), (1, 1))
        self.assertEqual(parse_shutter_speed(0.125), (1250, 10000))

    def test_shutter_to_seconds(self):
        self.assertAlmostEqual(shutter_to_seconds("1/8s"), 0.125)
        self.assertAlmostEqual(shutter_to_seconds("1/125s"), 0.008)
        self.assertAlmostEqual(shutter_to_seconds("0.5s"), 0.5)
        self.assertAlmostEqual(shutter_to_seconds(0.125), 0.125)
        self.assertAlmostEqual(shutter_to_seconds(1), 1.0)
        self.assertAlmostEqual(shutter_to_seconds(None), 1.0)

    def test_compute_exposure_ratio_explicit(self):
        # Base: 1/8s @ ISO 100 -> E_base = 0.125
        # Scan: 1/125s @ ISO 100 -> E_scan = 0.008
        # Ratio = 0.125 / 0.008 = 15.625
        ratio = compute_exposure_ratio(
            t_base="1/8s", iso_base=100, t_scan="1/125s", iso_scan=100
        )
        self.assertAlmostEqual(ratio, 15.625)

        # Same shutter, ISO 200 scan -> E_scan = 0.125 * 2 = 0.25 -> Ratio = 0.5
        ratio_iso = compute_exposure_ratio(
            t_base=0.125, iso_base=100, t_scan=0.125, iso_scan=200
        )
        self.assertAlmostEqual(ratio_iso, 0.5)

        # Zero scan exposure -> fallback to 1.0
        ratio_zero = compute_exposure_ratio(t_scan=0.0, iso_scan=0)
        self.assertEqual(ratio_zero, 1.0)

    def test_compute_exposure_ratio_return_details(self):
        ratio, t_base, iso_base, t_scan, iso_scan = compute_exposure_ratio(
            t_base="1/8s", iso_base=100, t_scan="1/125s", iso_scan=100, return_details=True
        )
        self.assertAlmostEqual(ratio, 15.625)
        self.assertAlmostEqual(t_base, 0.125)
        self.assertAlmostEqual(iso_base, 100.0)
        self.assertAlmostEqual(t_scan, 0.008)
        self.assertAlmostEqual(iso_scan, 100.0)

    def test_compute_exposure_ratio_with_profile(self):
        profile_path = os.path.join(
            os.path.dirname(__file__), "../profiles/profile_Portra400_20260623_170610.json"
        )
        profile = FilmProfile(profile_path)
        # Profile film base shutter is 1/8s, ISO 100
        self.assertEqual(profile.film_base_shutter, "1/8s")
        self.assertEqual(profile.film_base_iso, 100)

        ratio = compute_exposure_ratio(profile=profile, t_scan="1/125s", iso_scan=100)
        self.assertAlmostEqual(ratio, 15.625)

        # Via profile method
        method_ratio = profile.get_exposure_ratio(t_scan="1/125s", iso_scan=100)
        self.assertAlmostEqual(method_ratio, 15.625)

    def test_compute_exposure_ratio_with_images(self):
        img_scan = SimpleNamespace(shutter_speed=0.008, iso=100)
        img_base = SimpleNamespace(shutter_speed=0.125, iso=100)

        # Using img scan and img base
        ratio = compute_exposure_ratio(img=img_scan, film_base_img=img_base)
        self.assertAlmostEqual(ratio, 15.625)

        # Overriding scan shutter string
        ratio_override = compute_exposure_ratio(
            img=img_scan, film_base_img=img_base, shutter_str="1/8s"
        )
        self.assertAlmostEqual(ratio_override, 1.0)


if __name__ == "__main__":
    unittest.main()
