import os
import numpy as np
import pandas as pd
import logging

import cv2
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import normalize

logger = logging.getLogger(__name__)


def _get_contrast(img):
    """
    Given an image calculate the overall contrast level
    Used grayscale version picture's standard deviation to reflect contrast.
    (Reference: https://stackoverflow.com/questions/58821130/how-to-calculate-the-contrast-of-an-image)

    Args:
        img(`obj`:`np.array`): pixel data of a picture

    Returns:
        contrast(float): contrast value
    """
    # gray scale
    img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # use std as contrast
    contrast = img_grey.std()
    return contrast


def _get_brightness_distribution(img):
    """
    Given an image calculate the pixel distribution in shadow, dark, light, highlight (for RGB channels).
    This is essentially the channel histograms splitting to 4 exposure tiers and take mean.
    (Reference: https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_histograms/py_histogram_begins/py_histogram_begins.html)

    Args:
        img(`obj`:`np.array`): pixel data of a picture

    Returns:
        exposures(dict): {color:(shadow, dark, light, highlight)}
    """
    exposures = {}
    color = ('b', 'g', 'r')
    for i, col in enumerate(color):
        # calculate average pixel counts for each light area
        count = cv2.calcHist([img], [i], None, [256], [0, 256])
        unit = len(count) // 4
        shadow = count[0:unit].mean()
        dark = count[unit:2 * unit].mean()
        light = count[2 * unit:3 * unit].mean()
        highlight = count[3 * unit:].mean()
        exposures[col] = (shadow, dark, light, highlight)
    return exposures


def _get_RGB_average(img):
    """
    Given an image calculate average B,G,R for the whole picture.
    This is basically averaging all B,G,R.
    (Reference: https://andi-siess.de/rgb-to-color-temperature/)

    Args:
        img(`obj`:`np.array`): pixel data of a picture

    Returns:
        tuple: (B,G,R), tuple (average B,G,R for whole picture)
    """
    avg_color_per_row = np.average(img, axis=0)
    avg_color = np.average(avg_color_per_row, axis=0)
    return tuple(avg_color)


def _get_sharpness(img):
    """
    Given an image calculate average sharpness for the whole picture.
    The sharpness can be estimated by the average gradient magnitude.
    (Reference:
    https://stackoverflow.com/questions/6646371/detect-which-image-is-sharper
    https://www.mathworks.com/matlabcentral/fileexchange/32397-sharpness-estimation-from-image-gradients)

    Args:
        img(`obj`:`np.array`): pixel data of a picture

    Returns:
        sharpness(float): sharpness value
    """
    img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    array = np.asarray(img_grey, dtype=np.int32)

    gy, gx = np.gradient(array)  # get gradient
    gnorm = np.sqrt(gx ** 2 + gy ** 2)
    sharpness = np.average(gnorm)
    return sharpness


def _get_color_weight_for_major_color(img, colors):
    """
    Calculate weight of the seven rainbow colors.

    Args:
        img(`obj`:`np.array`): pixel data of a picture

    Returns:
        color_weight(dict): weight for each color
    """
    # Parse color centroid
    color_names = [e for e in colors]
    color_centroid = np.asarray([colors[e] for e in colors])

    # Calculate pixel distance to major color
    pixels = img.reshape(-1, img.shape[-1])
    dist = euclidean_distances(pixels, color_centroid)

    # Get color label
    labels = pd.Series([color_names[e] for e in np.argmin(dist, axis=1)])

    # Calculate
    color_weight = {}
    for color in color_names:
        if sum(labels == color) == 0:
            color_weight[color] = 0
        else:
            color_weight[color] = sum(labels == color)
    return color_weight


def get_style_feature(image_path, colors):
    """
    Get style feature vector for single image.

    Args:
        image_path(str): path to the target image
        colors(dict): centroid for rainbow colors

    Returns:
        features(np.array): style feature vector
    """
    # Read image
    img_array = cv2.imread(image_path)

    # Get features
    f1 = _get_contrast(img_array)
    f2 = _get_brightness_distribution(img_array)
    f3 = _get_RGB_average(img_array)
    f4 = _get_sharpness(img_array)
    f5 = _get_color_weight_for_major_color(img_array, colors)
    features = [f1] + [e for key in f2 for e in f2[key]] + [e for e in f3] + [f4] + [f5[e] for e in f5]
    return features


def _get_style_feature_matrix(images, colors):
    """
    Get style feature matrix for all the images.

    Args:
        images(list): list of images' path
        colors(dict): centroid for rainbow colors

    Returns:
        features(`obj`:`np.array`): style features
    """
    features = []
    for i, image_path in enumerate(images):
        if i % 50 == 0:
            logger.info("analyzing image %d / %d." % (i, len(images)))
        feat = get_style_feature(image_path, colors)
        features.append(feat)

    features = np.array(features)
    return features


def featurize_style_features(images_path, image_extensions, colors):
    """
    Featurize the raw pictures.

    Args:
        images_path(str): path to all the raw images
        image_extensions(list): acceptable file extensions
        colors(dict): centroid for rainbow colors

    Returns:
        features(`obj`:`pd.DataFrame`): style features for the images
    """
    # Get image paths
    images = [os.path.join(dp, f) for dp, dn, filenames in os.walk(images_path) for f in filenames if
              os.path.splitext(f)[1].lower() in image_extensions]
    image_names = [f for dp, dn, filenames in os.walk(images_path) for f in filenames if
                   os.path.splitext(f)[1].lower() in image_extensions]

    # Get raw features
    style_features = _get_style_feature_matrix(images, colors)
    style_features = style_features.reshape(len(images), -1)  # flatten, needed if using convolutional layer
    logger.info("Raw Style features extracted successfully.")

    # normalize
    style_features, norms = normalize(style_features, axis=0, norm='l1', return_norm=True)
    logger.info("Raw features extracted successfully.")

    # Return dataframe
    style_features = pd.DataFrame(style_features, index=image_names,
                                  columns=['contrast', 'B_shadow', 'B_dark', 'B_light', 'B_highlight',
                                           'G_shadow', 'G_dark', 'G_light', 'G_highlight', 'R_shadow', 'R_dark',
                                           'R_light', 'R_highlight', 'B_average', 'G_average', 'R_average', 'sharpness',
                                           'red_average', 'orange_average', 'yellow_average', 'green_average',
                                           'cyan_average', 'blue_average', 'purple_average'])
    return style_features, norms


