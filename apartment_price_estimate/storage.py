"""
Basic MInIO client to get and save feedback in a fixed file: group7/feedback.csv
"""

from minio import Minio
from minio.error import S3Error

from dotenv import load_dotenv
import os

# Loading environment variables for MinIO: ACCESS_KEY and SECRET_KEY
load_dotenv()


def create_client() -> object:
    """
    Create client for MinIO server.

    :return: client object
    """
    client = Minio(
        "api.storage.sws.informatik.uni-leipzig.de",
        access_key=os.environ.get('ACCESS_KEY'),
        secret_key=os.environ.get('SECRET_KEY')
    )
    return client


def upload_to_minio(client):
    """
    Uploads feedback.csv (has to be in working directory) to minio group7
    s3 bucket.
    """
    try:
        client.fput_object("group7", "feedback.csv", "./feedback.csv")
        print("'feedback.csv' was successfully uploaded to bucket 'group7'.")
    except S3Error as e:
        print(e.message, e.args)


def get_feedback_from_minio(client):
    """
    Connects to group7 minio bucket and retrieves feedback.csv as bytestream.
    Bytestream gets saved as feedback.csv in working directory.
    """
    try:
        # Get data of an object.
        response = client.get_object("group7", "feedback.csv")
        print("'feedback.csv' was successfully received from bucket 'group7'.")
        # Split the long string into a list of lines
        lines = response.data.decode('utf-8').splitlines()
        # Open CSV file for writing
        with open("feedback.csv", "w") as csv_file:
            for line in lines:
                if not line.isspace():  # necessary to get rid of empty lines in CSV
                    csv_file.write(line + "\n")
    except S3Error as e:
        print(e.message, e.args)
    finally:
        response.close()
        response.release_conn()
