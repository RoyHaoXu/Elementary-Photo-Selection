import logging.config
import os
import pickle
import random

import numpy as np
import pandas as pd
from flask import Flask, flash
from flask import render_template, request, redirect, send_from_directory
from werkzeug.utils import secure_filename

from src.model import make_recommendation, make_clusters
from src.rds_db import PhotoManager, PhotoStyleFeatures, PhotoObjectFeatures, PhotoClusters
from src.featurize_object import get_object_feature, get_pca30_features, get_object_feature_extractor
from src.featurize_style import get_style_feature

# Initialize the Flask application and config
app = Flask(__name__, template_folder="app/templates", static_folder="app/static")
app.config.from_pyfile('config/flaskconfig.py')
logging.config.fileConfig(app.config["LOGGING_CONFIG"])
logger = logging.getLogger(app.config["APP_NAME"])
logger.debug('Web app log')

# Initialize the database session
PHOTO_MANAGER = PhotoManager(app=app)

# Initialize model and model serving data
query = PHOTO_MANAGER.session.query(PhotoObjectFeatures)
OBJECT_FEATURES = pd.read_sql(query.statement, PHOTO_MANAGER.session.bind)
OBJECT_FEATURES = OBJECT_FEATURES.drop(['id'], axis=1)
OBJECT_FEATURES = OBJECT_FEATURES.set_index(['name'])

query = PHOTO_MANAGER.session.query(PhotoStyleFeatures)
STYLE_FEATURES = pd.read_sql(query.statement, PHOTO_MANAGER.session.bind)
STYLE_FEATURES = STYLE_FEATURES.drop(['id'], axis=1)
STYLE_FEATURES = STYLE_FEATURES.set_index(['name'])

query = PHOTO_MANAGER.session.query(PhotoClusters)
CLUSTER_FEATURES = pd.read_sql(query.statement, PHOTO_MANAGER.session.bind)
CLUSTER_FEATURES = CLUSTER_FEATURES.drop(['id'], axis=1)
CLUSTER_FEATURES = CLUSTER_FEATURES.set_index(['images_names'])

PCA = pickle.load(open(app.config['PCA_PATH'], 'rb'))

OBJECT_NORMS = pickle.load(open(app.config['OBJECT_NORMS_PATH'], 'rb'))
STYLE_NORMS = pickle.load(open(app.config['STYLE_NORMS_PATH'], 'rb'))

# Initialize VGG feature extractor
FEAT_EXTRACTOR = get_object_feature_extractor(app.config["EXTRACTOR_CONFIG"], app.config["EXTRACTOR_LAYER"])


def _allowed_file(filename):
    """Check if file format is correct."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def _query_features(feature_selection, style_weight=1):
    if feature_selection == 'Style':
        features = STYLE_FEATURES
    elif feature_selection == 'Object':
        features = OBJECT_FEATURES
    elif feature_selection == 'Combined':
        features = pd.concat([STYLE_FEATURES * style_weight, OBJECT_FEATURES], axis=1)
    return features


def _process_image_features(upload_filepath, feature_selection, style_weight):
    if feature_selection == 'Style':
        image_features = get_style_feature(upload_filepath, app.config['COLORS'])
        image_features = image_features / STYLE_NORMS
        logger.info("Style features extracted.")
        return image_features

    elif feature_selection == 'Object':
        image_features = get_object_feature(upload_filepath, FEAT_EXTRACTOR).reshape(1, -1)
        image_features = image_features / OBJECT_NORMS
        logger.info("Object features extracted.")
        image_features = get_pca30_features(image_features, PCA)
        logger.info("PCA features extracted.")
        return image_features

    elif feature_selection == 'Combined':
        image_object_features = get_object_feature(upload_filepath, FEAT_EXTRACTOR).reshape(1, -1)
        image_object_features = image_object_features / OBJECT_NORMS
        logger.info("Object features extracted.")
        image_object_features = get_pca30_features(image_object_features, PCA)
        logger.info("PCA features extracted.")
        image_style_features = get_style_feature(upload_filepath, app.config['COLORS'])
        image_style_features = image_style_features / STYLE_NORMS
        logger.info("Style features extracted.")
        image_style_features = np.array(image_style_features).reshape(1, -1) * style_weight
        image_features = np.concatenate((image_style_features, image_object_features), axis=1)
        logger.info("Combined features extracted.")
        return image_features


@app.route('/')
def index():
    all_images = os.listdir(app.config["PHOTO_PATH"])
    samples = list(map(lambda e: app.config["PHOTO_LINK_PREFIX"] + e, random.sample(all_images, 10)))
    return render_template('index.html', samples=samples)


@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),
                               'favicon.ico', mimetype='image/vnd.microsoft.icon')


@app.route('/image_upload')
def image_upload():
    return render_template('image_upload.html')


@app.route('/result', methods=['GET', 'POST'])
def result():
    if request.method == 'GET':
        return render_template('image_upload.html')

    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)

    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)

    if file and _allowed_file(file.filename):
        # Save uploaded file
        filename = secure_filename(file.filename)
        upload_filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(upload_filepath)

        # Parse name of style from dropdown menu
        option = request.form.get('rec_option', 'Recommendation')
        feature_selection = request.form.get('feature_selection', 'Object')
        try:
            style_weight = float(request.form.get('style_weight', '1'))
        except:
            style_weight = 1

        if option == 'Recommendation':
            # Query image features
            features = _query_features(feature_selection, style_weight)

            # Process input image
            image_features = _process_image_features(upload_filepath, feature_selection, style_weight)

            # Make Recommendations
            recommendations = make_recommendation(features, image_features, app.config['REC_NUM_PIC'])
            recommendations = [app.config['PHOTO_LINK_PREFIX'] + e for e in recommendations]
            upload_file_location = app.config['UPLOAD_LINK_PREFIX'] + filename
            return render_template('result_recommendation.html', upload_file_location=upload_file_location,
                                   recommendations=recommendations)

        elif option == 'Album Generation':
            # Query cluster features
            cluster_features = CLUSTER_FEATURES

            # Process input image
            image_features = _process_image_features(upload_filepath, app.config['CLUSTER_FEATURE_SELECTION'],
                                                     app.config['CLUSTER_STYLE_WEIGHT'])

            # Make Recommendations
            clusters = make_clusters(cluster_features, image_features, app.config['REC_NUM_ALBUM'])
            clusters = [[app.config['PHOTO_LINK_PREFIX'] + e for e in cluster] for cluster in clusters]

            upload_file_location = app.config['UPLOAD_LINK_PREFIX'] + filename
            return render_template('result_album.html', upload_file_location=upload_file_location,
                                   clusters=clusters)


    else:
        flash('Allowed image types are: {}'.format(', '.join(app.config['ALLOWED_EXTENSIONS'])))
        return redirect(request.url)


if __name__ == '__main__':
    app.run(debug=app.config["DEBUG"], port=app.config["PORT"], host=app.config["HOST"])
