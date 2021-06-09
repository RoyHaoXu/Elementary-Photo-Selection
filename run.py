import argparse
import logging.config
import yaml
import pickle

from src.s3_bucket import upload_to_s3, download_from_s3
from src.rds_db import PhotoManager, create_db
from src.featurize_object import featurize_object_features
from src.featurize_style import featurize_style_features
from src.model import tune_model, run_model, get_clusters_df
from config.flaskconfig import SQLALCHEMY_DATABASE_URI

logging.config.fileConfig('config/logging/local.conf')
logger = logging.getLogger()

if __name__ == '__main__':

    # Parse args and run corresponding pipeline
    parser = argparse.ArgumentParser(description="")

    parser.add_argument('--config', default='config/model_config.yaml', help='Path to pipeline configuration file')
    parser.add_argument('step', help='Which step to run',
                        choices=['upload', 'download', 'create_db', 'featurize_object', 'get_cluster_df',
                                 'featurize_style', 'inject_data', 'tune_model', 'model_result'])
    parser.add_argument('--s3_path', help='Path to store raw data on S3')
    parser.add_argument('--local_path', help='Local path of raw data')
    parser.add_argument("--engine_string", default=SQLALCHEMY_DATABASE_URI,
                        help="SQLAlchemy connection URI for database")
    parser.add_argument('--input_folder', help='Path of input raw images path')
    parser.add_argument('--input_style', help='Path of input csv data for style')
    parser.add_argument('--input_object', help='Path of input csv data for object')
    parser.add_argument('--input_cluster', help='Path of input csv data for object')
    parser.add_argument('--output', help='Path to save output')
    parser.add_argument('--plot_output', help='Path to save output')
    parser.add_argument('--model_dump_path', help='Path to save models')
    parser.add_argument('--norms_dump_path', help='Path to save norms')

    args = parser.parse_args()

    # Load configuration file for parameters and tmo path
    with open(args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    logger.info("Configuration file loaded from %s" % args.config)

    if args.step == 'upload':
        upload_to_s3(args.s3_path, args.local_path)

    elif args.step == 'download':
        download_from_s3(args.s3_path, args.local_path)

    elif args.step == 'create_db':
        create_db(args.engine_string)

    elif args.step == 'featurize_object':
        pca, output, norms = featurize_object_features(args.input_folder, **config['featurize_object']['featurize_object_features'])
        pickle.dump(pca, open(args.model_dump_path, "wb"))
        logger.info(f'PCA model saved to {args.model_dump_path}')
        pickle.dump(norms, open(args.norms_dump_path, "wb"))
        logger.info(f'Norms model saved to {args.norms_dump_path}')

    elif args.step == 'featurize_style':
        output, norms = featurize_style_features(args.input_folder, **config['featurize_style']['featurize_style_features'])
        pickle.dump(norms, open(args.norms_dump_path, "wb"))
        logger.info(f'Norms model saved to {args.norms_dump_path}')

    elif args.step == 'get_cluster_df':
        output = get_clusters_df(args.input_style, args.input_object, **config['model']['get_cluster_df'])

    elif args.step == 'inject_data':
        photo_manager = PhotoManager(engine_string=args.engine_string)
        photo_manager.add_offline_df(args.input_object, table_name='object_features',
                                     **config['rds_db']['PhotoManager']['add_offline_df'])
        photo_manager.add_offline_df(args.input_style, table_name='style_features',
                                     **config['rds_db']['PhotoManager']['add_offline_df'])
        photo_manager.add_offline_df(args.input_cluster, table_name='photo_clusters',
                                     **config['rds_db']['PhotoManager']['add_offline_df'])
        photo_manager.close()

    elif args.step == 'tune_model':
        tune_model(args.input_style, args.input_object, args.plot_output, **config['model']['tune_model'])

    elif args.step == 'model_result':
        run_model(args.input_style, args.input_object, args.plot_output, args.input_folder, **config['model']['run_model'])

    if args.output is not None and output is not None:
        output.to_csv(args.output, index=True)
        logger.info("Output saved to %s" % args.output)
