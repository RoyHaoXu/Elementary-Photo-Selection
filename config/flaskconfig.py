import os

# App settings
DEBUG = True
LOGGING_CONFIG = "config/logging/local.conf"
PORT = 5000
APP_NAME = "image_recommendation"
SQLALCHEMY_TRACK_MODIFICATIONS = True
HOST = "0.0.0.0"
SQLALCHEMY_ECHO = False  # If true, SQL for queries made will be printed
MAX_ROWS_SHOW = 100

# DB Connection string
DB_HOST = os.environ.get('MYSQL_HOST')
DB_PORT = os.environ.get('MYSQL_PORT')
DB_USER = os.environ.get('MYSQL_USER')
DB_PW = os.environ.get('MYSQL_PASSWORD')
DATABASE = os.environ.get('DATABASE_NAME')
DB_DIALECT = 'mysql+pymysql'
SQLALCHEMY_DATABASE_URI = os.environ.get('SQLALCHEMY_DATABASE_URI')
if SQLALCHEMY_DATABASE_URI is not None:
    pass
elif DB_HOST is None:
    SQLALCHEMY_DATABASE_URI = 'sqlite:///data/photos.db'
else:
    SQLALCHEMY_DATABASE_URI = '{dialect}://{user}:{pw}@{host}:{port}/{db}'.format(dialect=DB_DIALECT, user=DB_USER,
                                                                                  pw=DB_PW, host=DB_HOST, port=DB_PORT,
                                                                                  db=DATABASE)

# Photos Serving

# image paths
UPLOAD_FOLDER = 'app/static/uploads/'  # save user upload image
PHOTO_PATH = 'app/static/raw_images/'  # path to app serving images
PHOTO_LINK_PREFIX = 'static/raw_images/'  # path prefix for app serving images, used to display image in html
UPLOAD_LINK_PREFIX = 'static/uploads/'  # path prefix for uploaded images, used to display image in html

# model paths
PCA_PATH = 'models/pca.pkl'  # PCA model path
OBJECT_NORMS_PATH = 'models/object_norms.pkl'  # path to object feature normalization norms
STYLE_NORMS_PATH = 'models/style_norms.pkl'  # path to style feature normalization norms

# model pipeline config
CLUSTER_FEATURE_SELECTION = 'Combined'  # feature method used by the pre-trained clusters
CLUSTER_STYLE_WEIGHT = 2  # style weight used by the pre-trained cluster model
EXTRACTOR_CONFIG = {'weights': "imagenet", 'include_top': True}  # configuration for VGG feature extrctor
EXTRACTOR_LAYER = "fc2"  # layer used in VGG16 as feature extractor
REC_NUM_PIC = 5  # number of recommendations to make
REC_NUM_ALBUM = 3  # number of cluster recommendations to make
# rainbow color BGR
COLORS = {
    'red' : [0, 0, 255],
    'orange' : [0, 127, 255],
    'yellow' : [0, 255, 255],
    'green' : [0, 255, 0],
    'cyan' : [255, 255, 0],
    'blue' : [255, 0, 0],
    'purple' : [255, 0, 143]
}

# others
ALLOWED_EXTENSIONS = ['png', 'jpg', 'jpeg']  # acceptable file types for uploading

