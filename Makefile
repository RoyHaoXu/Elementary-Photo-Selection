.PHONY: image, upload_to_s3, download_from_s3, featurize_object, featurize_style, featurize, tune_model, run_model, get_cluster, create_db_rds, create_db_engine_string, inject_data_rds, inject_data_engine_string, pipeline_rds, pipeline_engine_string, remove_outdated_app_serving_images, move_raw_images_to_static, run_app_rds, run_app_engine_string, stop_containers, remove_containers, remove_images, clean_docker, tests
S3_PATH="s3://2021-msia423-xu-hao/raw/"  # path of images on S3 bucket
LOCAL_IMAGE_PATH="./data/raw_images/"  # local path of images for offline model pipeline
LOCAL_APP_IMAGE_PATH="./app/static/raw_images/"  # local path of images for app image serving
OBJECT_FEATURE_DATA_PATH="data/model_data/featurized_data_object.csv"  # path to save featurized_data_object.csv
STYLE_FEATURE_DATA_PATH="data/model_data/featurized_data_style.csv"  # path to save featurized_data_style.csv
CLUSTER_FEATURE_DATA_PATH="data/model_data/clusters.csv"  # path to save clusters.csv
PCA_MODEL_PATH="models/pca.pkl"  # path to save pca model
OBJECT_NORM_PATH="models/object_norms.pkl"  # path to save object_norms.pkl
STYLE_NORM_PATH="models/style_norms.pkl"  # path to save style_norms.pkl
PLOTS_PATH="plots/"  # path to save plots
LOCAL_IMAGE_PATH_GITKEEP="./data/raw_images/.gitkeep"  # gitkeep for local path of images for offline model pipeline

image:
	docker build -t photo_hxq9433 .

upload_to_s3:
	docker run \
		-e AWS_ACCESS_KEY_ID \
		-e AWS_SECRET_ACCESS_KEY \
		photo_hxq9433 run.py upload \
		--s3_path=${S3_PATH} \
		--local_path=${LOCAL_IMAGE_PATH}

download_from_s3:
	docker run --mount type=bind,source="`pwd`",target=/app/ \
		-e AWS_ACCESS_KEY_ID \
		-e AWS_SECRET_ACCESS_KEY \
		photo_hxq9433 run.py download \
		--s3_path=${S3_PATH} \
		--local_path=${LOCAL_IMAGE_PATH}

featurize_object:
	docker run --mount type=bind,source="`pwd`",target=/app/ \
		photo_hxq9433 run.py featurize_object \
		--input_folder=${LOCAL_IMAGE_PATH} \
		--output=${OBJECT_FEATURE_DATA_PATH}\
		--model_dump_path=${PCA_MODEL_PATH} \
		--norms_dump_path=${OBJECT_NORM_PATH}

featurize_style:
	docker run --mount type=bind,source="`pwd`",target=/app/ \
		photo_hxq9433 run.py featurize_style \
		--input_folder=${LOCAL_IMAGE_PATH} \
		--output=${STYLE_FEATURE_DATA_PATH}  \
		--norms_dump_path=${STYLE_NORM_PATH}

featurize: featurize_object featurize_style

tune_model:
	docker run --mount type=bind,source="`pwd`",target=/app/ \
		photo_hxq9433 run.py tune_model \
		--input_object=${OBJECT_FEATURE_DATA_PATH} \
		--input_style=${STYLE_FEATURE_DATA_PATH} \
		--plot_output=${PLOTS_PATH}

run_model:
	docker run --mount type=bind,source="`pwd`",target=/app/ \
		photo_hxq9433 run.py model_result \
		--input_object=${OBJECT_FEATURE_DATA_PATH} \
		--input_style=${STYLE_FEATURE_DATA_PATH} \
		--plot_output=${PLOTS_PATH} \
		--input_folder=${LOCAL_IMAGE_PATH}

get_cluster:
	docker run --mount type=bind,source="`pwd`",target=/app/ \
		photo_hxq9433 run.py get_cluster_df \
		--input_object=${OBJECT_FEATURE_DATA_PATH} \
		--input_style=${STYLE_FEATURE_DATA_PATH} \
		--output=${CLUSTER_FEATURE_DATA_PATH}

create_db_rds:
	docker run --mount type=bind,source="`pwd`",target=/app/ \
		-e MYSQL_USER \
		-e MYSQL_PASSWORD \
		-e MYSQL_PORT \
		-e DATABASE_NAME \
		-e MYSQL_HOST \
		photo_hxq9433 run.py create_db

create_db_engine_string:
	docker run --mount type=bind,source="`pwd`",target=/app/ \
		-e SQLALCHEMY_DATABASE_URI \
		photo_hxq9433 run.py create_db

inject_data_rds:
	docker run --mount type=bind,source="`pwd`",target=/app/ \
		-e MYSQL_USER \
		-e MYSQL_PASSWORD \
		-e MYSQL_PORT \
		-e DATABASE_NAME \
	    -e MYSQL_HOST \
		photo_hxq9433 run.py inject_data \
		--input_object=${OBJECT_FEATURE_DATA_PATH} \
		--input_style=${STYLE_FEATURE_DATA_PATH} \
		--input_cluster=${CLUSTER_FEATURE_DATA_PATH}

inject_data_engine_string:
	docker run --mount type=bind,source="`pwd`",target=/app/ \
		-e SQLALCHEMY_DATABASE_URI \
		photo_hxq9433 run.py inject_data \
		--input_object=${OBJECT_FEATURE_DATA_PATH} \
		--input_style=${STYLE_FEATURE_DATA_PATH} \
		--input_cluster=${CLUSTER_FEATURE_DATA_PATH}

pipeline_rds: download_from_s3 featurize get_cluster create_db_rds inject_data_rds
pipeline_engine_string: download_from_s3 featurize get_cluster create_db_engine_string inject_data_engine_string

remove_outdated_app_serving_images:
	rm -r ${LOCAL_APP_IMAGE_PATH}

move_raw_images_to_static:
	mv ${LOCAL_IMAGE_PATH} ${LOCAL_APP_IMAGE_PATH}
	mkdir ${LOCAL_IMAGE_PATH}
	touch ${LOCAL_IMAGE_PATH_GITKEEP}

run_app_rds:
	docker run --mount type=bind,source="`pwd`",target=/app/ \
		-e MYSQL_USER \
		-e MYSQL_PASSWORD \
		-e MYSQL_PORT \
		-e DATABASE_NAME \
	    -e MYSQL_HOST \
	    -p 5000:5000 photo_hxq9433 app.py

run_app_engine_string:
	docker run --mount type=bind,source="`pwd`",target=/app/ \
		-e SQLALCHEMY_DATABASE_URI \
	    -p 5000:5000 photo_hxq9433 app.py

stop_containers:
	docker kill $(docker ps -q)

remove_containers:
	docker rm $$(docker ps -aq)

remove_images:
	docker rmi $$(docker images -q)

clean_docker: remove_containers remove_images

tests:
	docker run photo_hxq9433 -m pytest test/*

