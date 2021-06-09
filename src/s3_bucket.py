import os
import re
import logging
import boto3
import botocore.exceptions

logger = logging.getLogger(__name__)


def _parse_s3(s3path):
    """Parse s3 path. Source: https://github.com/MSIA/2021-msia423/blob/main/aws-s3/s3.py

    Args:
        s3path (str): full s3 path

    Returns:
        str,str: s3bucket name, s3path to store the data
    """

    regex = r"s3://([\w._-]+)/([\w./_-]+)"

    m = re.match(regex, s3path)
    s3bucket = m.group(1)
    s3path = m.group(2)

    return s3bucket, s3path


def upload_to_s3(s3_path, local_path):
    """Upload raw data to S3 bucket.

    Args:
        s3_path (str): target path for uploading raw data on s3 bucket
        local_path (str): local path to the raw data directory

    Returns:
        None

    """
    # Connect to s3 using aws access key
    try:
        s3 = boto3.client('s3',
                          aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
                          aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"))
        logger.info("AWS S3 Connected.")
    except botocore.exceptions.PartialCredentialsError:
        logger.error("AWS Credentials Invalid.")

    # Upload all raw pictures under the local path to s3
    bucket_name, s3_store_path = _parse_s3(s3_path)
    if len(list(os.walk(local_path))) > 0:
        for root, dirs, files in os.walk(local_path):
            for file in files:
                s3.upload_file(os.path.join(root, file), bucket_name, os.path.join(s3_store_path, file))
                logger.info("{} Uploaded.".format(file))  # log progress

    # If a single file path submitted, upload the single file
    else:
        filename = local_path.split('/')[-1]
        s3.upload_file(local_path, bucket_name, os.path.join(s3_store_path, filename))
        logger.info("{} Uploaded.".format(filename))  # log progress

    logger.info("All Image Uploaded to S3.")


def _download_s3_folder(s3, bucket_name, s3_store_path, local_dir):
    """
    Download the contents of a folder directory
    Args:
        s3(`obj`:`s3.ServiceResource`):
        bucket_name(str): the name of the s3 bucket
        s3_store_path(str): the folder path in the s3 bucket
        local_dir(str): a relative or absolute directory path in the local file system
    """
    bucket = s3.Bucket(bucket_name)
    for obj in bucket.objects.filter(Prefix=s3_store_path):
        target = os.path.join(local_dir, os.path.relpath(obj.key, s3_store_path))
        if not os.path.exists(os.path.dirname(target)):
            os.makedirs(os.path.dirname(target))
        if obj.key[-1] == '/':
            continue
        bucket.download_file(obj.key, target)
        logger.info("{} Downloaded.".format(obj.key))  # log progress


def download_from_s3(s3_path, local_path):
    """Download raw data from S3 bucket.

    Args:
        s3_path (str): target path for uploading raw data on s3 bucket
        local_path (str): local path to the raw data directory

    Returns:
        None

    """
    # Connect to s3 using aws access key
    try:
        s3 = boto3.resource('s3',
                            aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
                            aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"))
        logger.info("AWS S3 Connected.")
    except botocore.exceptions.PartialCredentialsError:
        logger.error("AWS Credentials Invalid.")

    bucket_name, s3_store_path = _parse_s3(s3_path)
    _download_s3_folder(s3, bucket_name, s3_store_path, local_path)
    logger.info("All Image Downloaded from S3.")


