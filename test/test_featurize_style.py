import pytest

import pandas as pd
import numpy as np
from src.featurize_style import featurize_style_features


def test_featurize_style_features():
    """Testing happy path for featurize_style_features()."""

    images_path = 'test/test_images/'
    image_extensions = ['.jpg']
    colors = {
        'red': [0, 0, 255],
        'orange': [0, 127, 255],
        'yellow': [0, 255, 255],
        'green': [0, 255, 0],
        'cyan': [255, 255, 0],
        'blue': [255, 0, 0],
        'purple': [255, 0, 143]
    }

    true_norms = np.array([6.43144108e+01, 3.43282188e+04, 1.21471875e+03, 9.90562500e+02,
                           3.30500000e+02, 2.96700938e+04, 3.45287500e+03, 2.28243750e+03,
                           1.45859375e+03, 2.71602656e+04, 2.17903125e+03, 2.11631250e+03,
                           5.40839062e+03, 1.23896166e+01, 3.09990332e+01, 5.33470277e+01,
                           1.14817414e+00, 1.89049300e+06, 3.60995000e+05, 9.23960000e+04,
                           4.19900000e+03, 8.29000000e+02, 4.00700000e+03, 6.37700000e+03])

    true_style_features = pd.DataFrame({'contrast': {'test.jpg': 1.0},
                                        'B_shadow': {'test.jpg': 1.0},
                                        'B_dark': {'test.jpg': 1.0},
                                        'B_light': {'test.jpg': 1.0},
                                        'B_highlight': {'test.jpg': 1.0},
                                        'G_shadow': {'test.jpg': 1.0},
                                        'G_dark': {'test.jpg': 1.0},
                                        'G_light': {'test.jpg': 1.0},
                                        'G_highlight': {'test.jpg': 1.0},
                                        'R_shadow': {'test.jpg': 1.0},
                                        'R_dark': {'test.jpg': 1.0},
                                        'R_light': {'test.jpg': 1.0},
                                        'R_highlight': {'test.jpg': 1.0},
                                        'B_average': {'test.jpg': 1.0},
                                        'G_average': {'test.jpg': 1.0},
                                        'R_average': {'test.jpg': 1.0},
                                        'sharpness': {'test.jpg': 1.0},
                                        'red_average': {'test.jpg': 1.0},
                                        'orange_average': {'test.jpg': 1.0},
                                        'yellow_average': {'test.jpg': 1.0},
                                        'green_average': {'test.jpg': 1.0},
                                        'cyan_average': {'test.jpg': 1.0},
                                        'blue_average': {'test.jpg': 1.0},
                                        'purple_average': {'test.jpg': 1.0}})

    style_features, norms = featurize_style_features(images_path, image_extensions, colors)
    pd.testing.assert_frame_equal(true_style_features, style_features)
    np.testing.assert_allclose(true_norms, norms, rtol=1e-04, atol=10)


def test_featurize_style_features_unhappy():
    """Testing happy path for featurize_style_features()."""

    images_path = 123
    image_extensions = ['.jpg']
    colors = {
        'red': [0, 0, 255],
        'orange': [0, 127, 255],
        'yellow': [0, 255, 255],
        'green': [0, 255, 0],
        'cyan': [255, 255, 0],
        'blue': [255, 0, 0],
        'purple': [255, 0, 143]
    }

    with pytest.raises(TypeError):
        featurize_style_features(images_path, image_extensions, colors)
