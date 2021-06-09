import logging

from scipy.spatial.distance import cosine
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from tensorflow.keras.preprocessing import image

logger = logging.getLogger()


def _calculate_silhouette_score(features, min_k, max_k, sil_metric, affinity, linkage):
    """
    Helper function to calculate silhouette score chart to find out the optimal cluster number.

    Args:
        features(`obj`:`pd.DataFrame`): features dataframe
        min_k(int): minimal cluster number to try
        max_k(int): maximal cluster number to try
        sil_metric(str): metric used for calculating silhouette score
        affinity(str): affinity for hierarchical clustering
        linkage(str): linkage for hierarchical clustering

    Returns:
        sil(list): silhouette score list
    """
    sil = []

    for k in range(min_k, max_k + 1):
        cluster = AgglomerativeClustering(n_clusters=k, affinity=affinity, linkage=linkage)
        labels = cluster.fit_predict(features)
        sil.append(silhouette_score(features, labels, metric=sil_metric))

    return sil


def _get_feature_df(input_path_style, input_path_object, feature_selection, style_weight=1):
    """
    Helper function to get feature matrix for training.

    Args:
        input_path_style(str): path for style feature data
        input_path_object(str): path for object feature data
        feature_selection(str): how the pipeline should combine the two data source
        style_weight(float): weight of style features

    Returns:
        features(`obj`:`pd.DataFrame`): features dataframe
    """

    # Load feature data frame
    if feature_selection == 'style':
        features = pd.read_csv(input_path_style, index_col=0)
    elif feature_selection == 'object':
        features = pd.read_csv(input_path_object, index_col=0)
    elif feature_selection == 'combined':
        f1 = pd.read_csv(input_path_style, index_col=0)
        f2 = pd.read_csv(input_path_object, index_col=0)
        features = pd.concat([f1 * style_weight, f2], axis=1)
    else:
        logger.error("Invalid feature selection, available options: ['style', 'object', 'combined']")
    return features


def _get_concatenated_images(images, thumb_height):
    """
    Helper function to get concatenated images for each cluster,

    Args:
        images(list): images' paths
        thumb_height(int): image height

    Returns:
        concat_image(`obj`:`np.array`): concatenated images
    """
    thumbs = []
    for i in images:
        img = image.load_img(i)
        img = img.resize((int(img.width * thumb_height / img.height), thumb_height))
        thumbs.append(img)
    concat_image = np.concatenate([np.asarray(t) for t in thumbs], axis=1)
    return concat_image


def tune_model(input_path_style, input_path_object, plot_save_path,
               max_k, min_k, affinity, linkage, feature_selection,
               sil_metric, style_weight):
    """
    Function to run the clustering model given different configurations. Produces silhouette
    score chart for cluster number picking.

    Args:
        input_path_style(str): path for style feature data
        input_path_object(str): path for object feature data
        plot_save_path(str): path to save the output plot
        min_k(int): minimal cluster number to try
        max_k(int): maximal cluster number to try
        affinity(str): affinity for hierarchical clustering
        linkage(str): linkage for hierarchical clustering
        feature_selection(str): how the pipeline should combine the two data source
        sil_metric(str): metric used for calculating silhouette score
        style_weight(float): weight of style features

    Returns:
        None
    """

    # Get silhouette scores
    features = _get_feature_df(input_path_style, input_path_object, feature_selection, style_weight)
    logger.info('Features loaded.')
    x = _calculate_silhouette_score(features, min_k, max_k, sil_metric, affinity, linkage)
    logger.info('Silhouette score calculated.')

    # Plot
    plt_name = 'Silhouette:min=' + str(min_k) + '_' + 'max=' + str(max_k) + '_' \
               + 'sil_metric=' + sil_metric + '_' \
               + 'affinity=' + affinity + '_' \
               + 'linkage=' + linkage \
               + 'feature=' + feature_selection + '.png'
    plt.plot(x)
    plt.xlim(min_k, max_k)
    plt.title('silhouette_score')
    plt.savefig(plot_save_path + plt_name)
    logger.info('Performance plot saved.')


def run_model(input_path_style, input_path_object, plot_save_path, image_path_prefix,
              k_optimal, affinity, linkage, feature_selection,
              sil_metric, style_weight, thumb_height, figsize):
    """
    Function to run the model given configurations and produce clustered images.

    Args:
        input_path_style(str): path for style feature data
        input_path_object(str): path for object feature data
        plot_save_path(str): path to save the output plot
        image_path_prefix(str): file path to the image folder, used to add in front of the image names to read the actual images
        k_optimal(int): optimal number of clusters
        affinity(str): affinity for hierarchical clustering
        linkage(str): linkage for hierarchical clustering
        feature_selection(str): how the pipeline should combine the two data source
        sil_metric(str): metric used for calculating silhouette score
        style_weight(float): weight of style features
        thumb_height(int): image height
        figsize(tuple): figsize for the output plot

    Returns:
        None
    """

    # Init figure
    fig, axes = plt.subplots(ncols=1, nrows=k_optimal, figsize=(figsize[0], figsize[1]))

    # Get pics for each cluster
    features = _get_feature_df(input_path_style, input_path_object, feature_selection, style_weight)
    cluster = AgglomerativeClustering(n_clusters=k_optimal, affinity=affinity, linkage=linkage)
    labels = cluster.fit_predict(features)
    for i, k in enumerate(range(k_optimal)):
        cluster = features.index[labels == k]
        cluster = [image_path_prefix + e for e in cluster]
        cluster_image = _get_concatenated_images(cluster, thumb_height)
        axes[i].imshow(cluster_image)

    # Save plot
    plt_name = 'Result:min=' + 'k_optimal' + str(k_optimal) + '_' \
               + 'sil_metric=' + sil_metric + '_' \
               + 'affinity=' + affinity + '_' \
               + 'linkage=' + linkage \
               + 'feature=' + feature_selection + '.png'

    plt.savefig(plot_save_path + plt_name)
    logger.info('Result plot saved.')


def get_clusters_df(input_path_style, input_path_object,
                    k_optimal, affinity, linkage,
                    feature_selection, style_weight):
    """
    Function to generate cluster dataframe which contains the centroid information for the clusters.

    Args:
        input_path_style(str): path for style feature data
        input_path_object(str): path for object feature data
        k_optimal(int): optimal number of clusters
        affinity(str): affinity for hierarchical clustering
        linkage(str): linkage for hierarchical clustering
        feature_selection(str): how the pipeline should combine the two data source
        style_weight(float): weight of style features

    Returns:
        cluster_df(`obj`:`pd.DataFrame`): cluster centroid dataframe

    """
    # Get pics for each cluster
    features = _get_feature_df(input_path_style, input_path_object, feature_selection, style_weight)
    cluster = AgglomerativeClustering(n_clusters=k_optimal, affinity=affinity, linkage=linkage)
    labels = cluster.fit_predict(features)

    # Create cluster df
    cluster_df = pd.DataFrame(index=list(set(labels)), columns=list(features.columns) + ['images_names'])
    for k in list(set(labels)):
        images_names = ','.join(features.index[labels == k])
        cluster_df.loc[k, list(features.columns)] = features.loc[features.index[labels == k], :].mean()
        cluster_df.loc[k, 'images_names'] = images_names

    return cluster_df


def make_recommendation(features, new_feature_vector, rec_num):
    """
    Function to make recommendations given the referencing feature matrix and the new feature vector.

    Args:
        features(`obj`:`pd.DataFrame`): features dataframe
        new_feature_vector(`obj`:`np.array`): new image's feature vector
        rec_num(int): number of recommendations to make

    Returns:
        recommended_pics(list): list of recommended picture names
    """
    distances = [cosine(new_feature_vector, features.loc[pic, :]) for pic in features.index]
    idx_closest = sorted(range(len(distances)), key=lambda k: distances[k])[0:rec_num]
    recommended_pics = list(features.index[idx_closest])

    return recommended_pics


def make_clusters(clusters, new_feature_vector, rec_num):
    """

    Args:
        clusters(`obj`:`pd.DataFrame`): cluster centroid dataframe
        new_feature_vector(`obj`:`np.array`): new image's feature vector
        rec_num(int): number of recommendations to make

    Returns:
        recommended_pics(list): list of recommended picture names' clusters
    """
    distances = [cosine(new_feature_vector, clusters.loc[pic, :]) for pic in clusters.index]
    idx_closest = sorted(range(len(distances)), key=lambda k: distances[k])[0:rec_num]
    recommended_clusters = list(clusters.index[idx_closest])
    recommended_clusters = list(map(lambda e: e.split(','), recommended_clusters))
    return recommended_clusters
