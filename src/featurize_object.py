import os
import logging

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.models import Model

from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

logger = logging.getLogger(__name__)


def _load_image(path, target_size=None):
    """
    Helper to load image.

    Args:
        path(str): path to the target image
        target_size: target size to load the image so it can fit in the pretrained model

    Returns:
        img(`obj`,`PIL.ImageFile`), x(obj`,`np.array`): image object and pixel array
    """
    img = image.load_img(path, target_size=target_size)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return img, x


def get_object_feature_extractor(extractor_config, extractor_layer):
    """
    Get feature extractor from pretrained VGG16 model.

    Args:
        extractor_config(dict): configuration for VGG6 model
        extractor_layer(str): layer to use as feature extractor

    Returns:
        feat_extractor(keras.engine.functional.Functional): feature extractor
    """
    model = tf.keras.applications.VGG16(**extractor_config)
    feat_extractor = Model(inputs=model.input, outputs=model.get_layer(extractor_layer).output)
    return feat_extractor


def get_object_feature(image_path, feat_extractor):
    """
    Helper to get feature vector for one image.

    Args:
        image_path(str): path to the target image
        feat_extractor(keras.engine.functional.Functional): feature extractor

    Returns:
        feat(`obj`:`pd.DataFrame`): features for the image
    """
    img, img_array = _load_image(image_path, feat_extractor.input_shape[1:3]);
    feat = feat_extractor.predict(img_array)[0]
    return feat


def _get_object_feature_matrix(images, feat_extractor):
    """
    Get raw feature matrix
    Args:
        images(list): path to all the raw images
        feat_extractor(keras.engine.functional.Functional): feature extractor

    Returns:
        features(`obj`:`pd.DataFrame`): raw features for the images
    """
    features = []
    for i, image_path in enumerate(images):
        if i % 50 == 0:
            logger.info("Analyzing image %d / %d." % (i, len(images)))
        feat = get_object_feature(image_path, feat_extractor)
        features.append(feat)
    logger.info("Image analyzing finished.")
    features = np.array(features)

    return features


def _get_pca30_model(features):
    """
    Get PCA model from feature matrix.

    Args:
        features(`obj`:`np.array`): raw features from feature extractor

    Returns:
        pca(`obj`:`sklearn.decomposition._pca.PCA`)
    """
    if features.shape[0] < 30:
        logger.error("Need at least 30 images to run the featurize pipeline.")
    pca = PCA(n_components=30)
    pca.fit(features)
    return pca


def get_pca30_features(features, pca):
    """
    Get PCA features.

    Args:
        features(`obj`:`np.array`): raw features from feature extractor
        pca(`obj`:`sklearn.decomposition._pca.PCA`): PCA decomposition model

    Returns:
        pca_features(`obj`:`np.array`): PCA features
    """
    pca_features = pca.transform(features)
    return pca_features


def featurize_object_features(images_path, image_extensions, extractor_config, extractor_layer):
    """
    Featurize the raw pictures.

    Args:
        images_path(str): path to all the raw images
        image_extensions(list): acceptable file extensions
        extractor_config(dict): configuration for extractor
        extractor_layer(str): name of layer used for extractor

    Returns:
        features(`obj`:`pd.DataFrame`): features for the images
    """
    # Get image paths
    images = [os.path.join(dp, f) for dp, dn, filenames in os.walk(images_path) for f in filenames if
              os.path.splitext(f)[1].lower() in image_extensions]
    image_names = [f for dp, dn, filenames in os.walk(images_path) for f in filenames if
                   os.path.splitext(f)[1].lower() in image_extensions]

    # Get feature extractor
    feat_extractor = get_object_feature_extractor(extractor_config, extractor_layer)
    logger.info("Feature extractor loaded successfully.")

    # Get raw features
    raw_features = _get_object_feature_matrix(images, feat_extractor)
    raw_features = raw_features.reshape(len(images), -1)  # flatten, need if using convolutional layer
    logger.info("Raw Object features extracted successfully.")

    # Normalize
    raw_features, norms = normalize(raw_features, axis=0, norm='l1', return_norm=True)

    # Get  PCA features
    pca = _get_pca30_model(raw_features)
    features = get_pca30_features(raw_features, pca)
    logger.info("PCA features extracted successfully.")

    # return PCA features
    features = pd.DataFrame(features, index=image_names,
                            columns=['pc1', 'pc2', 'pc3', 'pc4', 'pc5', 'pc6', 'pc7', 'pc8', 'pc9',
                                     'pc10', 'pc11', 'pc12', 'pc13', 'pc14', 'pc15', 'pc16', 'pc17', 'pc18', 'pc19',
                                     'pc20',
                                     'pc21', 'pc22', 'pc23', 'pc24', 'pc25', 'pc26', 'pc27', 'pc28', 'pc29', 'pc30'])
    return pca, features, norms

