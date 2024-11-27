import logging
import os
from venv import logger

from minio import Minio
from minio.error import S3Error


class MinIOClient:
    def __init__(self, url, access_key, secret_key, secure):
        self.client = Minio(
            endpoint=url,
            access_key=access_key,
            secret_key=secret_key,
            secure=True  # if HTTPS is used, set this to True
        )
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def upload_file(self, bucket_name, object_name, file_path):
        try:
            # Check if bucket exists; if not, create it
            if not self.client.bucket_exists(bucket_name):
                self.client.make_bucket(bucket_name)
                self.logger.info("Bucket created successfully.")

            # Remove leading slash from object_name
            if object_name.startswith("/"):
                object_name = object_name[1:]

            logger.info(f"Object Name in for uploading {object_name}")
            logger.info(f"File path in for uploading {file_path}")
            # Upload the file
            self.client.fput_object(
                bucket_name,
                object_name,
                file_path
            )
            self.logger.info("File uploaded successfully.")

        except S3Error as exc:
            self.logger.error("S3 error occurred: %s", exc)
        except Exception as exc:
            self.logger.error("Unexpected error occurred: %s", exc)

    def download_file(self, bucket_name, object_name, file_path):
        try:
            # Check if the object exists before downloading
            if not self.stat_object(bucket_name, object_name):
                raise FileNotFoundError(f"Object {object_name} does not exist in bucket {bucket_name}")

            # Ensure the download directory exists
            download_dir = os.path.dirname(file_path)
            os.makedirs(download_dir, exist_ok=True)

            # Remove leading slash from object_name
            if object_name.startswith("/"):
                object_name = object_name[1:]
            logger.info(f" {bucket_name, object_name, file_path}")
            # Download the file
            self.client.fget_object(
                bucket_name,
                object_name,
                file_path
            )
            self.logger.info("File downloaded successfully.")
            return file_path  # Return the file path

        except S3Error as exc:
            self.logger.error("S3 error occurred: %s", exc)
        except Exception as exc:
            self.logger.error("Unexpected error occurred: %s", exc)
            raise  # Re-raise the exception to signal failure

    def stat_object(self, bucket_name, object_name):
        try:
            # Check if the object exists
            self.client.stat_object(bucket_name, object_name)
            self.logger.info("Object exists: %s/%s", bucket_name, object_name)
            return True
        except S3Error as exc:
            if exc.code == 'NoSuchKey':
                self.logger.info("Object does not exist: %s/%s", bucket_name, object_name)
                return False
            else:
                self.logger.error("Error checking object existence: %s", exc)
                raise

    # List Objects in a Bucket
    def list_objects(self, bucket_name, prefix=''):
        try:
            # List objects in the bucket with an optional prefix
            objects = self.client.list_objects(bucket_name, prefix=prefix)
            return [obj.object_name for obj in objects]
        except S3Error as exc:
            self.logger.error("Error listing objects: %s", exc)
            raise
