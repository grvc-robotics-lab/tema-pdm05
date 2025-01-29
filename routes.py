import threading
import requests
import time
import math
import os
import numpy as np
import rasterio
import logging
import config
import geopandas as gpd
import queue
import json
import fiona

from flask import Flask, request, jsonify, Response
from flask_socketio import SocketIO
from osgeo import gdal, osr, ogr
from rasterio.features import geometry_mask, rasterize
from rasterio.transform import from_origin, from_bounds, rowcol, xy
from Kalman_filter_estimating_Objects import KalmanFilter
from georeferencing_module import main
from minio_client import MinIOClient
from logging_config import logger
from pyproj import CRS, Transformer
from rasterio.windows import Window
from datetime import datetime
from shapely.geometry import Polygon, box, mapping, shape
from shapely.ops import transform
from concurrent.futures import ThreadPoolExecutor
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.mask import mask
from threading import Thread, Lock
from datetime import datetime, timezone
from urllib.parse import urlparse

print(config.CALLBACK_URL)
logger.info(config.CALLBACK_URL)

# Initialize Flask app and SocketIO
app = Flask(__name__)
socketio = SocketIO(app)

DOWNLOAD_DIR = 'downloads/'
if not os.path.exists(DOWNLOAD_DIR):
    os.makedirs(DOWNLOAD_DIR)

# Initialize MinIO client
minio_client = MinIOClient(
    url=config.MINIO_ENDPOINT,
    access_key=config.MINIO_ACCESS_KEY,
    secret_key=config.MINIO_SECRET_KEY,
    secure=True
)

ND_entity_ID = None
obj_entity_ID = None
polygon_coordinates = None

global_cache = {"processing": False,
                "natural_disaster": "No disaster info",
                "roi": "No disaster info",
                "expiration": "No info",
                "ignitionPoints": "No info",
                }
global_cache_lock = threading.Lock()

alert_event = threading.Event()  # Event triggered by Alert notifications
other_entity_event = threading.Event()  # Event triggered by other notifications
notification_queue = queue.Queue()  # Thread-safe queue for non-Alert notifications
entities_initialized = False  # Tracks whether initialize_entities has run
is_processing = False
processing_lock = Lock()

ogm_flag = False
ogm_counter = 0


###################################################################
@app.route(f'/{config.BASE_PATH}')
def index():
    return jsonify({"status": "App is running", "message": "Occupancy Grid Estimation System"})


@app.route(f'/{config.BASE_PATH}/logs')
def logs():
    html_content = f'''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Logs</title>
        <script>
            function fetchLogs() {{
                fetch('/{config.BASE_PATH}/log_data')  // Use the correct route
                    .then(response => response.text())
                    .then(data => {{
                        document.getElementById('log-container').innerText = data;
                    }});
            }}
            setInterval(fetchLogs, 2000); // Refresh logs every 2 seconds
        </script>
    </head>
    <body>
        <h1>Application Logs</h1>
        <pre id="log-container">Loading logs...</pre>
    </body>
    </html>
    '''
    return Response(html_content, content_type='text/html')


@app.route(f'/{config.BASE_PATH}/log_data')
def logs_data():
    try:
        with open('app_routes.log', 'r') as log_file:
            return log_file.read(), 200
    except FileNotFoundError:
        return "Log file not found.", 404


@app.route(f'/{config.BASE_PATH}/{config.API_ENDPOINT}', methods=['POST'])
def notify():
    try:
        notification_data = request.get_json()
        if not isinstance(notification_data, dict):
            logger.error("Invalid notification data: Expected a JSON object.")
            return jsonify({"error": "Invalid notification data format"}), 400

        if not isinstance(notification_data.get("data"), list):
            logger.error("Invalid notification data: 'data' must be a list.")
            return jsonify({"error": "Invalid notification data format"}), 400

        for notification in notification_data["data"]:
            if not isinstance(notification, dict):
                logger.error(f"Invalid notification: Expected a dictionary, got {type(notification)}.")
                return jsonify({"error": f"Invalid notification format: {notification}"}), 400

        executor = ThreadPoolExecutor(max_workers=10)
        futures = []

        for notification in notification_data["data"]:
            try:
                logger.info(f"Processing notification: {notification['type']}")
                print(f"Processing notification: {notification['type']}")

                # process_notification(notification) # without concurrent processing
                futures.append(executor.submit(process_notification, notification))
            except Exception as e:
                logger.error(f"Error processing notification {notification}: {e}")

        # Wait for all submitted tasks to complete
        for future in futures:
            try:
                future.result()  # Ensure exceptions in threads are raised
            except Exception as e:
                logger.error(f"Error in processing notification: {e}")

        with global_cache_lock:
            # Ensure that initialize_processing() is not already running
            if global_cache['natural_disaster'] != 'No disaster info':
                if not global_cache['processing']:
                    global_cache['processing'] = True
                    try:
                        logger.info("Starting initialize_processing() in a new thread")
                        print("Starting initialize_processing() in a new thread")

                        processing_thread = threading.Thread(target=initialize_processing)
                        processing_thread.start()
                    except Exception as e:
                        logger.error(f"Error in initialize_processing: {e}")
                        global_cache['processing'] = False  # Reset on failure
                else:
                    logger.info("initialize_processing() is already running, skipping.")
                    print("initialize_processing() is already running, skipping.")

            else:
                logger.info("No relevant natural disaster data, skipping processing.")
                print("No relevant natural disaster data, skipping processing.")

        return jsonify({"status": "Notifications processed successfully"}), 200

    except Exception as e:
        logger.error(f"Unexpected error in notify function: {e}")
        return jsonify({"error": "Internal server error"}), 500


###################################################################

def handle_person_vehicle_detection(notification, parameters):
    auth_filename = notification.get('parameters', {}).get('value', {}).get('FileName', None)
    detection = notification.get("detection", {}).get('value', {})

    if not auth_filename:
        logger.warning("No filename found for PersonVehicleDetection.")
        return

    metadata_file_base = auth_filename.split('.')[0]
    detection_file = f"{metadata_file_base}_obj.json"
    metadata_file = f"{metadata_file_base}_objects_metadata.json"

    # Getting disaster info from the global cache
    disaster_info = global_cache.get('natural_disaster', 'No disaster info')
    if disaster_info == 'No disaster info':
        logger.warning("No disaster information found in the cache.")
        return

    # Constructing file paths
    download_path = os.path.join("downloads", "drone_imgs", disaster_info)
    os.makedirs(download_path, exist_ok=True)  # Ensure directory exists

    json_file_path_metadata = os.path.join(download_path, metadata_file)
    json_file_path_detection = os.path.join(download_path, detection_file)
    # Writing metadata and detection to JSON files
    try:
        with open(json_file_path_metadata, 'w', encoding='utf-8') as json_file:
            json.dump(parameters, json_file, indent=4)

        with open(json_file_path_detection, 'w', encoding='utf-8') as json_file:
            json.dump(detection, json_file, indent=4)

        logger.info(f"Files written successfully: {json_file_path_metadata}, {json_file_path_detection}")

        # Call main function for geo-referencing
        try:
            disaster_info = global_cache.get('natural_disaster', 'No disaster info')
            main(disaster_info, "bbox")
            logger.info("Geo-referencing is done correctly for person and vehicles.")
        except Exception as e:
            logger.error(f"Issue in geo-referencing due to {e}")

    except Exception as e:
        logger.error(f"Error writing detection files: {e}")
        return


def handle_segmentation(notification):
    bucket = notification.get("bucket", {}).get('value')
    auth_filename = notification.get('segmentation', {}).get('value')

    if not auth_filename or not isinstance(auth_filename, dict):
        logger.warning("Invalid or missing 'segmentation' value in notification.")
        return

    mask_id = auth_filename.get('mask_id')

    if not mask_id:
        logger.warning("No mask_id found in auth_filename.")
        return

    file_path = f"downloads/drone_imgs/{global_cache.get('natural_disaster', 'No disaster info')}/{mask_id}"
    logger.info(f"path of download {file_path}")
    print(f"path of download {file_path}")

    try:
        # Synchronous file download simulation
        downloaded_file_path = minio_client.download_file(bucket, mask_id, file_path)
        if downloaded_file_path:
            logger.info(f"File downloaded successfully to {downloaded_file_path}")
            print(f"File downloaded successfully to {downloaded_file_path}")

            # Call main function for geo-referencing
            try:
                disaster_info = global_cache.get('natural_disaster', 'No disaster info')
                main(disaster_info, "segmented")
                logger.info("Geo-referencing is done correctly for segmented images.")
            except Exception as e:
                logger.error(f"Issue in geo-referencing due to {e}")

        else:
            logger.error("File download failed.")

    except Exception as e:
        logger.error(f"Error downloading file for mask_id {mask_id}: {e}")
        return

    metadata_file_base = mask_id.split('.')[0]
    metadata_file = f"{metadata_file_base}_metadata.json"
    json_file_path = f"downloads/drone_imgs/{global_cache.get('natural_disaster', 'No disaster info')}/{metadata_file}"

    parameters = notification.get("parameters", {}).get('value', {})
    try:
        with open(json_file_path, 'w') as json_file:
            json_file.write(json.dumps(parameters, indent=4))
    except Exception as e:
        logger.error(f"Error writing segmentation metadata: {e}")


def process_alert(notification, entity_id):
    #############################################################
    location = notification.get("location", {}).get("value", {})
    if isinstance(location, dict) and "coordinates" in location:
        try:
            global_cache['roi'] = location['coordinates']
            logger.info(f"ROI set for entity ID {entity_id}: {global_cache['roi']}")
            print(f"ROI set for entity ID {entity_id}: {global_cache['roi']}")

        except Exception as e:
            logger.error(f"Error accessing 'coordinates' for entity ID {entity_id}: {e}")
            global_cache['roi'] = "No disaster info"
    else:
        global_cache['roi'] = "No disaster info"
        logger.warning(f"Unexpected 'location' format for entity ID {entity_id}: {location}")
    #############################################################
    event_ = notification.get("event", {}).get("value", None)
    if event_:
        try:
            global_cache['natural_disaster'] = event_
            logger.info(f"Natural disaster set for entity ID {entity_id}: {global_cache['natural_disaster']}")
            print(f"Natural disaster set for entity ID {entity_id}: {global_cache['natural_disaster']}")

        except Exception as e:
            logger.error(f"Error accessing 'event' value for entity ID {entity_id}: {e}")
            global_cache['natural_disaster'] = "Unknown event"
    else:
        global_cache['natural_disaster'] = "Unknown event"
        logger.warning(f"Unexpected 'event' format for entity ID {entity_id}: {event_}")
    #############################################################
    expiration = notification.get("expires", {}).get("value")
    if expiration:
        try:
            global_cache['expiration'] = expiration
            logger.info(f"expiration for entity ID {entity_id}: {global_cache['expiration']}")
        except Exception as e:
            logger.error(f"Error accessing 'expires' value for entity ID {entity_id}: {e}")
            global_cache['expiration'] = "No info"
    #############################################################
    ignitionPoints = notification.get("ignitionPoints", {}).get("value", {})
    if isinstance(ignitionPoints, dict) and "coordinates" in ignitionPoints:
        try:
            global_cache['ignitionPoints'] = ignitionPoints['coordinates']
            logger.info(f"ignitionPoints set for entity ID {entity_id}: {global_cache['ignitionPoints']}")
            print(f"ignitionPoints set for entity ID {entity_id}: {global_cache['ignitionPoints']}")

        except Exception as e:
            logger.error(f"Error accessing 'ignitionPoints' for entity ID {entity_id}: {e}")
            global_cache['ignitionPoints'] = "No info"
    else:
        global_cache['ignitionPoints'] = "No info"
        logger.warning(f"Unexpected 'ignitionPoints' format for entity ID {entity_id}: {ignitionPoints}")
    #############################################################


def extract_bucket_and_filename(url):
    """Extract bucket name and file name from the URL."""
    parsed_url = urlparse(url)
    path_parts = parsed_url.path.strip("/").split("/")

    # Extract bucket name and file name based on the new requirement
    bucket = "/".join(path_parts[:2])  # First two parts form the bucket name
    filename = "/".join(path_parts[2:])  # Remaining part is the file name

    return bucket, filename


# def handle_file_download(notification):
#     """
#         Handle downloading a file based on notification data.
#         Args:
#             notification (dict): Notification containing file and bucket information.
#         """
#     downloaded_file_path = None
#     entity_type = notification.get('type')
#     minio_url = notification.get('minio_url', {}).get('value')
#     ##############################################################################################
#     # Handle StandardArrivalTime
#     if entity_type == 'StandardArrivalTime':
#         if minio_url:
#             logger.info(f"minio url {minio_url}")
#             logger.info(f"Handling download via URL for entity type: {entity_type}")
#             bucket_name, file_name = extract_bucket_and_filename(minio_url)
#             logger.info(f"bucket_name {bucket_name} ---- filename {file_name}")
#             try:
#                 downloaded_file_path = download_file(entity_type, file_name, bucket_name)
#                 if downloaded_file_path:
#                     logger.info(f"File downloaded successfully to {downloaded_file_path}")
#                     print(f"File downloaded successfully to {downloaded_file_path}")
#                 else:
#                     logger.error("File download failed.")
#             except Exception as e:
#                 logger.error(f"Error in downloading file: {e}", exc_info=True)
#             # try:
#             #     downloaded_file_path = download_file_from_url(entity_type, minio_url)
#             #     if downloaded_file_path:
#             #         logger.info(f"File downloaded successfully to {downloaded_file_path}")
#             #         print(f"File downloaded successfully to {downloaded_file_path}")
#             #     else:
#             #         logger.error("File download via URL failed.")
#             # except Exception as e:
#             #     logger.error(f"Error downloading file via URL: {e}", exc_info=True)
#             return
#     ##############################################################################################
#     # Handle cases with data.href for EOBurntArea or similar
#     if entity_type == "EOBurntArea":
#         data_href = notification.get('data', {}).get('value', {}).get('href')
#         if data_href:
#             logger.info(f"data_href url {data_href}")
#             logger.info(f"Handling download via URL for entity type: {entity_type}")
#             bucket_name, file_name = extract_bucket_and_filename(data_href)
#             logger.info(f"bucket_name {bucket_name} ---- filename {file_name}")
#             try:
#                 downloaded_file_path = download_file(entity_type, file_name, bucket_name)
#                 if downloaded_file_path:
#                     logger.info(f"File downloaded successfully to {downloaded_file_path}")
#                     print(f"File downloaded successfully to {downloaded_file_path}")
#                 else:
#                     logger.error("File download failed.")
#             except Exception as e:
#                 logger.error(f"Error in downloading file: {e}", exc_info=True)
#             # try:
#             #     downloaded_file_path = download_file_from_url(entity_type, minio_url)
#             #     if downloaded_file_path:
#             #         logger.info(f"File downloaded successfully to {downloaded_file_path}")
#             #         print(f"File downloaded successfully to {downloaded_file_path}")
#             #     else:
#             #         logger.error("File download via URL failed.")
#             # except Exception as e:
#             #     logger.error(f"Error downloading file via URL: {e}", exc_info=True)
#             return
#     ##########################################################################################################
#     filename_ = notification.get('filename')
#     bucket = notification.get("bucket")
#     # Handle cases with filename and bucket
#     if not (isinstance(filename_, dict) and 'value' in filename_ and
#             isinstance(bucket, dict) and 'value' in bucket):
#         logger.warning("Invalid or missing 'filename' or 'bucket' in notification.")
#         return
#
#     if (isinstance(filename_, dict) and 'value' in filename_ and filename_['value'] and
#             isinstance(bucket, dict) and 'value' in bucket and bucket['value']):
#         try:
#             downloaded_file_path = download_file(entity_type, filename_, bucket)
#             if downloaded_file_path:
#                 logger.info(f"File downloaded successfully to {downloaded_file_path}")
#                 print(f"File downloaded successfully to {downloaded_file_path}")
#             else:
#                 logger.error("File download failed.")
#         except Exception as e:
#             logger.error(f"Error in downloading file: {e}", exc_info=True)
#     ##########################################################################################################
#     if not downloaded_file_path:
#         logger.warning(
#             "No valid download source found in notification. Ensure filename, bucket, data.href, or minio_url are "
#             "correctly provided.")
#         return
def handle_file_download(notification):
    """
    Handle downloading a file based on notification data.
    Args:
        notification (dict): Notification containing file and bucket information.
    """
    downloaded_file_path = None
    entity_type = notification.get('type')
    minio_url = notification.get('minio_url', {}).get('value')

    def log_and_download(entity_type, url, download_type):
        """Helper function to extract bucket/filename and perform download."""
        try:
            bucket_name, file_name = extract_bucket_and_filename(url)
            logger.info(f"Extracted bucket: {bucket_name}, filename: {file_name}")
            return download_file(entity_type, file_name, bucket_name)
        except Exception as e:
            logger.error(f"Failed to handle {download_type} download: {e}", exc_info=True)
            return None

    # Handle StandardArrivalTime entity type
    if entity_type == 'StandardArrivalTime' and minio_url:
        logger.info(f"Handling StandardArrivalTime download with URL: {minio_url}")
        downloaded_file_path = log_and_download(entity_type, minio_url, "StandardArrivalTime")

    # Handle EOBurntArea entity type
    elif entity_type == "EOBurntArea":
        data_href = notification.get('data', {}).get('value', {}).get('href')
        if data_href:
            logger.info(f"Handling EOBurntArea download with data_href: {data_href}")
            downloaded_file_path = log_and_download(entity_type, data_href, "EOBurntArea")

    # Handle cases with filename and bucket
    else:
        filename_ = notification.get('filename')
        bucket = notification.get("bucket")

        if not (isinstance(filename_, dict) and 'value' in filename_ and
                isinstance(bucket, dict) and 'value' in bucket):
            logger.warning("Invalid or missing 'filename' or 'bucket' in notification.")
        elif filename_['value'] and bucket['value']:
            try:
                downloaded_file_path = download_file(entity_type, filename_['value'], bucket['value'])
                if downloaded_file_path:
                    logger.info(f"File downloaded successfully to {downloaded_file_path}")
                    print(f"File downloaded successfully to {downloaded_file_path}")
                else:
                    logger.error("File download failed.")
            except Exception as e:
                logger.error(f"Error in downloading file: {e}", exc_info=True)

    # Final fallback if no valid source was handled
    if not downloaded_file_path:
        logger.warning(
            "No valid download source found in notification. Ensure filename, bucket, data.href, or minio_url are "
            "correctly provided.")


def download_file_from_url(entity_type, url):
    """
    Download a file directly from a URL.
    Args:
        entity_type (str): The type of entity triggering the download.
        url (str): The URL of the file to download.
    Returns:
        str: Path to the downloaded file, or None if the download fails.
    """
    # Define the base directory structure
    base_dirs = {
        "HotspotResult": "downloads/SocialMedia",
        "SinglePostResult": "downloads/SocialMedia",
        "UAVTrajectory": "downloads/drone_planning",
        "StandardArrivalTime": "downloads/FireSim",
        "FloodCalculationResults": "downloads/FloodSim",
        "EOBurntArea": "downloads/satellite_imgs"
    }

    # Get the directory for the entity type
    download_dir = base_dirs.get(entity_type, "downloads/Other")
    os.makedirs(download_dir, exist_ok=True)  # Ensure the directory exists

    # Extract the file name from the URL
    file_name = url.split('/')[-1]
    file_path = os.path.join(download_dir, file_name)

    try:
        # Download the file
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an error for HTTP errors

        with open(file_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        logger.info(f"File downloaded via URL to: {file_path}")
        return file_path
    except Exception as e:
        logger.error(f"Error downloading file from URL {url}: {e}", exc_info=True)
        return None


def download_file(entity_type, filename_, bucket):
    file_path = ""

    if entity_type in ["HotspotResult", "SinglePostResult"]:
        file_path = f"downloads/SocialMedia/{filename_['value']}"
    elif entity_type == "UAVTrajectory":
        file_path = f"downloads/drone_planning/{filename_['value']}"
    elif entity_type == "StandardArrivalTime":
        file_path = f"downloads/FireSim/{filename_['value']}"
    elif entity_type == "FloodCalculationResults":
        file_path = f"downloads/FloodSim/{filename_['value']}"
    elif entity_type == "EOBurntArea":
        file_path = f"downloads/satellite_imgs/{filename_['value']}"

    try:
        if entity_type == "StandardArrivalTime" or entity_type == "EOBurntArea":
            logger.info(f"Downloading file '{filename_['value']}' from bucket '{bucket['value']}' to '{file_path}'.")
            return minio_client.download_file(bucket, filename_, file_path)
        else:
            logger.info(f"Downloading file '{filename_['value']}' from bucket '{bucket['value']}' to '{file_path}'.")
            return minio_client.download_file(bucket['value'], filename_['value'], file_path)
    except Exception as e:
        logger.error(f"Error downloading file {filename_['value']} from bucket {bucket['value']}: {e}")
        return None


def process_notification(notification):
    entity_id = notification.get("id")
    entity_type = notification.get("type")

    if not entity_id or not entity_type:
        logger.error(f"Invalid notification received: {notification}")
        return

    logger.info(f"Processing notification for entity ID: {entity_id}, Entity type: {entity_type}")
    print(f"Processing notification for entity ID: {entity_id}, Entity type: {entity_type}")

    try:
        # Process different entity types accordingly
        if entity_type == "Alert":
            try:
                logger.info("Triggering Alert entity processing")
                print("Triggering Alert entity processing")

                process_alert(notification, entity_id)
                alert_event.set()
            except Exception as e:
                logger.error(f"Error processing Alert entity ID {entity_id}: {e}")

        elif entity_type == "PersonVehicleDetection":
            try:
                parameters = notification.get("parameters", {}).get('value', {})
                if not parameters:
                    raise ValueError(f"Missing parameters in notification: {notification}")
                handle_person_vehicle_detection(notification, parameters)
            except Exception as e:
                logger.error(f"Error handling PersonVehicleDetection for entity ID {entity_id}: {e}")
        elif entity_type in ["FireSegmentation", "BurntSegmentation", "FloodSegmentation"]:
            try:
                handle_segmentation(notification)
            except Exception as e:
                logger.error(f"Error handling segmentation for entity ID {entity_id}: {e}")
        else:
            try:
                handle_file_download(notification)
            except Exception as e:
                logger.error(f"Error handling file download for entity ID {entity_id}: {e}")

        # Notify initialize_processing for non-Alert entities
        if entity_type != "Alert":
            try:
                logger.info("Setting event for non-Alert entity")
                print("Setting event for non-Alert entity")

                other_entity_event.set()
            except Exception as e:
                logger.error(f"Error setting other_entity_event for entity ID {entity_id}: {e}")
    except Exception as e:
        logger.error(f"Error processing notification {entity_id} of type {entity_type}: {e}")


def initialize_processing():
    global entities_initialized, ogm_flag, polygon_coordinates, ogm_counter, polygon_coordinates
    # polygon_coordinates = convert_to_polygon()
    while True:
        ################################################################################################################
        expiration = global_cache.get('expiration')
        if expiration == "No info":
            time.sleep(1)
            continue
        try:
            expiration_time = datetime.strptime(expiration, "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=timezone.utc)
            current_time = datetime.now(timezone.utc)
            if current_time >= expiration_time:
                logger.info(f"Expiration time reached: {expiration}. Exiting the loop.")
                break
        except Exception as e:
            logger.error(f"Error parsing expiration time: {e}")
            break
        ################################################################################################################
        # Wait for alert_event or other_entity_event to be set
        if not (alert_event.is_set() or other_entity_event.is_set()):
            time.sleep(0.1)
            continue
        try:
            # Handle Alert-specific instantiation
            if alert_event.is_set():
                ogm_flag = False
                logger.info("Received new alert notification. Resetting the information fusion.")
                ###########################################################
                # Delete existing files in the estimated_OGM directory
                ogm_dir = "estimated_OGM"
                if os.path.isdir(ogm_dir):
                    for file in os.listdir(ogm_dir):
                        file_path = os.path.join(ogm_dir, file)
                        if os.path.isfile(file_path):
                            os.remove(file_path)
                            logger.info(f"Deleted file: {file_path}")
                ###########################################################
                disaster_type = global_cache.get('natural_disaster', 'No disaster info')
                drone_imgs_fire = f"downloads/drone_imgs/{disaster_type}"
                if os.path.isdir(drone_imgs_fire):
                    for file in os.listdir(drone_imgs_fire):
                        if file.endswith('.tif'):
                            continue
                        else:
                            file_path = os.path.join(drone_imgs_fire, file)
                            if os.path.isfile(file_path):
                                os.remove(file_path)
                                logger.info(f"Deleted file: {file_path}")
                ###########################################################
                georeferenced_imgs = "georeferenced_drone_images"
                if os.path.isdir(georeferenced_imgs):
                    for file in os.listdir(georeferenced_imgs):
                        file_path = os.path.join(georeferenced_imgs, file)
                        if os.path.isfile(file_path):
                            os.remove(file_path)
                            logger.info(f"Deleted file: {file_path}")
                ###########################################################
                initialize_entities()
                entities_initialized = True
                ###########################################################
                if not ogm_flag:
                    polygon_coordinates = convert_to_polygon()
                    disaster_type = global_cache.get('natural_disaster', 'No disaster info')
                    ogm_path_ND = f"estimated_OGM/occupancy_grid_map_{disaster_type}.tif"
                    print(f"polygon_coordinates --> {polygon_coordinates}")
                    get_roi(polygon_coordinates, ogm_path_ND, resolution=10)
                    ##############################################################
                    result = mask_geotiff_with_polygon_exact(ogm_path_ND, global_cache.get('roi'), ogm_path_ND)
                    if result:
                        print(f"Masked GeoTIFF saved to {result}")
                    else:
                        print("Error occurred while masking the GeoTIFF.")
                    ##############################################################
                    ogm_path_obj = f"estimated_OGM/occupancy_grid_map_{disaster_type}_Objects.tif"
                    get_roi(polygon_coordinates, ogm_path_obj, resolution=10)
                    ##############################################################
                    result = mask_geotiff_with_polygon_exact(ogm_path_obj, global_cache.get('roi'), ogm_path_obj)
                    if result:
                        print(f"Masked GeoTIFF saved to {result}")
                    else:
                        print("Error occurred while masking the GeoTIFF.")
                    ##############################################################
                    ogm_metadata = get_geo_dict(f"estimated_OGM/occupancy_grid_map_{disaster_type}_Objects.tiff")
                    logger.info(f"OGM counter --> {ogm_counter}")
                    ogm_counter += 1
                    ogm_flag = True
                ###########################################################
                alert_event.clear()  # Reset Alert event for future triggers

            if other_entity_event.is_set() and entities_initialized:
                try:
                    estimate_ND_status()
                except Exception as e:
                    print(f"No OGM for ND due to {e}")
                    logger.info(f"No OGM for ND due to {e}")
                try:
                    estimate_Objects_status()
                except Exception as e:
                    print(f"No OGM for objects due to {e}")
                    logger.info(f"No OGM for objects due to {e}")
                other_entity_event.clear()  # Reset other_entity_event for future triggers
        except Exception as e:
            logger.error(f"Error during initialize_processing: {e}")


def subscribe_to_entities():
    subscription_url = f"{config.BROKER_URL}/ngsi-ld/v1/subscriptions"

    headers = {
        "Content-Type": "application/ld+json",
        "Accept": "application/ld+json"
    }
    subscription_payload_template = {
        "type": "Subscription",
        "notification": {
            "endpoint": {
                "uri": config.CALLBACK_URL,
                "accept": "application/json"
            }
        },
        '@context': ['https://uri.etsi.org/ngsi-ld/v1/ngsi-ld-core-context.jsonld']
    }

    entity_types_with_ids = {
        "Alert": "urn:ngsi-ld:tema:subscription:USE:PDM05:011",
        # "Alert": "urn:ngsi-ld:Subscription:1a68a5ec-d718-11ef-8fd9-0a580a821b57",
        "FloodCalculationResults": "urn:ngsi-ld:tema:subscription:USE:PDM05:002",
        "StandardArrivalTime": "urn:ngsi-ld:tema:subscription:USE:PDM05:003",
        "FireSegmentation": "urn:ngsi-ld:tema:subscription:USE:PDM05:004",
        "BurntSegmentation": "urn:ngsi-ld:tema:subscription:USE:PDM05:005",
        "FloodSegmentation": "urn:ngsi-ld:tema:subscription:USE:PDM05:006",
        "PersonVehicleDetection": "urn:ngsi-ld:tema:subscription:USE:PDM05:007",
        "HotspotResult": "urn:ngsi-ld:tema:subscription:USE:PDM05:008",
        "SinglePostResult": "urn:ngsi-ld:tema:subscription:USE:PDM05:009",
        "EOBurntArea": "urn:ngsi-ld:tema:subscription:USE:PDM05:010"
    }

    try:
        response = requests.get(subscription_url, headers=headers)
        if response.status_code != 200:
            print(f"Error fetching subscriptions. Status: {response.status_code}, Body: {response.text}")
            logger.error(f"Error fetching subscriptions. Status: {response.status_code}, Body: {response.text}")
            return

        existing_subscriptions = response.json()
        existing_subscription_ids = {sub.get("id") for sub in existing_subscriptions}

        for entity_type, subscription_id in entity_types_with_ids.items():
            if subscription_id in existing_subscription_ids:
                print(f"Subscription for {entity_type} with ID {subscription_id} already exists.")
                logger.info(f"Subscription for {entity_type} with ID {subscription_id} already exists.")
                continue

            subscription_payload = subscription_payload_template.copy()
            subscription_payload["id"] = subscription_id
            subscription_payload["entities"] = [{"type": entity_type}]

            create_subscription(subscription_url, subscription_payload, headers)

    except Exception as e:
        logger.exception("An error occurred during subscription management.")
        print(f"An error occurred: {e}")


def create_subscription(subscription_url, payload, headers):
    try:
        response = requests.post(subscription_url, json=payload, headers=headers)
        if response.status_code in (200, 201):
            print(f"Subscription {payload['id']} created successfully.")
            logger.info(f"Subscription {payload['id']} created successfully.")
        else:
            print(
                f"Failed to create subscription {payload['id']}. Status: {response.status_code}, Body: {response.text}")
            logger.error(
                f"Failed to create subscription {payload['id']}. Status: {response.status_code}, Body: {response.text}")
    except Exception as e:
        logger.exception("An error occurred while creating the subscription.")
        print(f"An error occurred while creating the subscription: {e}")


def estimate_ND_status():
    global polygon_coordinates
    # polygon_coordinates = convert_to_polygon()
    disaster_type = global_cache.get('natural_disaster', 'No disaster info')
    ogm_path = f"estimated_OGM/occupancy_grid_map_{disaster_type}.tif"
    roi_data = global_cache.get('roi')
    logger.info(f"ROI data -> {roi_data}")
    ogm_data, ogm_gt_, ogm_proj = load_image(ogm_path, 0)

    # Update OGM with drone data
    try:
        observe_data_drone, observe_gt_drone, observe_proj_drone = load_image(
            "georeferenced_drone_images", 1)
        #############################
        # Debug the type and shape of observe_data_drone
        if observe_data_drone is not None:
            logger.debug(f"Type of observe_data_drone: {type(observe_data_drone)}")
            if isinstance(observe_data_drone, np.ndarray):
                logger.debug(f"Shape of observe_data_drone: {observe_data_drone.shape}")
                logger.debug(f"Shape of OGM_Data: {ogm_data.shape}")
            else:
                logger.debug("observe_data_drone is not a numpy array.")
        else:
            logger.debug("observe_data_drone is None.")
        #############################
        logger.info(f"ogm_gt_ {ogm_gt_}")
        logger.info(f"observe_gt_drone {observe_gt_drone}")
        observe_data_drone = observe_data_drone / 255
        ogm_data = update_occupancy_grid(ogm_data, ogm_gt_, observe_data_drone, observe_gt_drone)
        logger.info(f"OGM done successfully {ogm_data.shape}")
    except FileNotFoundError:
        logger.warning("Drone images directory not found.")
    except Exception as e:
        logger.info(f"Drone measurements are not available: {e}")

    # Update OGM with satellite data
    try:
        observ_sat_data_, observ_sat_gt_, observ_sat_proj = extract_roi_from_satellite_files(
            "downloads/satellite_imgs", roi_data)
        ogm_data = update_occupancy_grid(ogm_data, ogm_gt_, observ_sat_data_, observ_sat_gt_)
        logger.info(f"OGM has successfully generated and fused sate file")
    except FileNotFoundError:
        logger.warning("Satellite ROI directory not found.")
    except Exception as e:
        logger.info(f"Sat measurements are not available {e}")

    # Update OGM with hotspot data from the SocialMedia directory
    try:
        social_media_dir = "downloads/SocialMedia"  # Path to the SocialMedia directory
        hotspot_files = [os.path.join(social_media_dir, f) for f in os.listdir(social_media_dir)
                         if f.startswith("hotspot_results") and f.endswith(".geojson")]
        # Sort files by timestamp (assumes the timestamp is part of the filename)
        hotspot_files.sort()
        for hotspot_file in hotspot_files:
            try:
                # Load the GeoJSON file
                with open(hotspot_file, 'r') as f:
                    hotspot_data = json.load(f)

                for feature in hotspot_data["features"]:
                    try:
                        properties = feature.get("properties", {})
                        geometry = feature.get("geometry", {})
                        if geometry.get("type") == "Polygon":
                            # Extract polygon coordinates and calculate grid cells it covers
                            polygon = Polygon(geometry.get("coordinates")[0])
                            if not polygon.is_valid:
                                logger.warning(f"Invalid polygon in file: {hotspot_file}")
                                continue
                            # Iterate through grid cells covered by the polygon
                            for x, y in polygon_to_pixels(polygon, ogm_gt_, ogm_data.shape):
                                if 0 <= x < ogm_data.shape[1] and 0 <= y < ogm_data.shape[0]:
                                    # Extract hotspot data properties
                                    hotspot_count = properties.get("count", 0)
                                    hotspot_ratio = properties.get("ratio", 0)

                                    # Validate properties before using them
                                    if not isinstance(hotspot_count, (int, float)):
                                        hotspot_count = 0
                                        logger.warning(f"Invalid 'count' value in {hotspot_file}, defaulting to 0.")
                                    if not isinstance(hotspot_ratio, (int, float)):
                                        hotspot_ratio = 0
                                        logger.warning(f"Invalid 'ratio' value in {hotspot_file}, defaulting to 0.")

                                    # Calculate the hotspot value to be fused
                                    hotspot_value = max(hotspot_count, hotspot_ratio)

                                    # Fuse hotspot value into the OGM
                                    ogm_data[y, x] = fuse_fire_probability(
                                        existing_prob=ogm_data[y, x],
                                        hotspot_value=hotspot_value,
                                        weight=0.5  # Adjust weight as needed for confidence in hotspot data
                                    )
                    except Exception as feature_error:
                        logger.error(f"Error processing feature in {hotspot_file}: {feature_error}")

                # Delete the file after successful processing
                os.remove(hotspot_file)
                logger.info(f"Successfully processed and deleted: {hotspot_file}")
            except Exception as e:
                logger.error(f"Error processing {hotspot_file}: {e}")

    except FileNotFoundError:
        logger.warning("SocialMedia directory not found.")
    except Exception as e:
        logger.error(f"Error integrating hotspot data: {e}")

    # Save updated OGM as GeoTIFF
    try:
        output_tiff_path = f"occupancy_grid_map_{disaster_type}.tif"
        metadata = save_geotiff("estimated_OGM", output_tiff_path, ogm_data, ogm_gt_, crs_epsg=4326)
        logger.info(f"{ND_entity_ID}")
        logger.info(f"Metadata {metadata}")
        process_and_upload_ogm(ND_entity_ID, os.path.join("estimated_OGM", output_tiff_path),
                               config.BUCKET_NAME, metadata)
    except Exception as e:
        logger.error(f"Error saving or uploading OGM: {e}")
    ###################################################################################################################


def polygon_to_pixels(polygon, geo_transform, grid_shape):
    """
    Map a polygon to the pixel coordinates of the grid.

    Args:
        polygon (shapely.geometry.Polygon): The polygon to map.
        geo_transform (tuple): GeoTransform of the grid.
        grid_shape (tuple): Shape of the grid (height, width).

    Yields:
        tuple: Pixel coordinates (x, y) of grid cells covered by the polygon.
    """
    bounds = polygon.bounds
    x_min, y_min = coordinates_to_pixel(geo_transform, bounds[0], bounds[1])
    x_max, y_max = coordinates_to_pixel(geo_transform, bounds[2], bounds[3])

    x_min, x_max = max(0, x_min), min(grid_shape[1] - 1, x_max)
    y_min, y_max = max(0, y_min), min(grid_shape[0] - 1, y_max)
    # Iterate through the bounding box
    for y in range(y_min, y_max + 1):
        for x in range(x_min, x_max + 1):
            # Create a polygon for the grid cell
            cell_polygon = box(*pixel_to_coordinates(geo_transform, x, y),
                               *pixel_to_coordinates(geo_transform, x + 1, y + 1))
            # Check intersection with the input polygon
            if polygon.intersects(cell_polygon):
                yield x, y


# fuse_fire_probability: Fuses the existing fire probability with the hotspot value.
def fuse_fire_probability(existing_prob, hotspot_value, weight=0.5):
    """
    Fuse hotspot data into the existing probability using weighted Bayesian updating.

    Args:
        existing_prob (float): Existing probability in the OGM cell.
        hotspot_value (float): Hotspot value (e.g., count or ratio from social media analysis).
        weight (float): Weight factor for the hotspot data (range: 0 to 1).

    Returns:
        float: Updated probability, clamped to [0, 1].
    """
    epsilon = 1e-9  # Small value to avoid division by zero

    # Ensure probabilities are within [0, 1]
    existing_prob = np.clip(existing_prob, 0, 1)
    hotspot_value = np.clip(hotspot_value, 0, 1)

    # Adjust weights to reflect the confidence in the data
    if weight < 0 or weight > 1:
        raise ValueError("Weight must be between 0 and 1.")

    # Bayesian fusion formula:
    # P(x|z) = [P(z|x) * P(x)] / P(z)
    # Where:
    # - P(z|x) = hotspot_value (likelihood of data given event x)
    # - P(x) = existing_prob (prior probability)
    # - P(z) = marginal likelihood, calculated to normalize the result
    P_z_given_x = hotspot_value
    P_x = existing_prob
    P_not_x = 1 - P_x
    P_z_given_not_x = 1 - P_z_given_x

    # Marginal likelihood (normalizing constant)
    P_z = (P_z_given_x * P_x) + (P_z_given_not_x * P_not_x)
    if P_z < epsilon:
        P_z = epsilon  # Avoid division by zero

    # Posterior probability
    posterior_prob = (P_z_given_x * P_x) / P_z

    # Weighted combination with prior
    updated_prob = (1 - weight) * existing_prob + weight * posterior_prob

    # Clamp the result to ensure valid probability
    return np.clip(updated_prob, 0, 1)


def estimate_Objects_status():
    """
    Estimate objects' status and process the occupancy grid map using data
    from drones and social media posts.
    """
    global polygon_coordinates
    # polygon_coordinates = convert_to_polygon()
    disaster_type = global_cache.get('natural_disaster', 'No disaster info')
    ogm_path = f"estimated_OGM/occupancy_grid_map_{disaster_type}_Objects.tif"

    if disaster_type != 'No disaster info':
        ogm_metadata = get_geo_dict(f"estimated_OGM/occupancy_grid_map_{disaster_type}_Objects.tiff")
        if not ogm_metadata:
            logger.error("Failed to load or generate OGM metadata. Aborting process.")
            return

        #######################################################################################
        # Integrate Drone data
        #######################################################################################
        try:
            # Retrieve observation metadata
            observ_metadata = get_geotiff_metadata_rasterio("georeferenced_drone_images")
            print("Observation Metadata Type:", type(observ_metadata))
            logger.info("Observation Metadata Type:", type(observ_metadata))
            print("First Entry of Observation Metadata:", observ_metadata[0] if observ_metadata else "No Data")

            if not observ_metadata:
                logger.info("Drone metadata is empty or not available.")
            else:
                # Update OGM Metadata
                ogm_metadata = get_observations_in_FOV(ogm_metadata, observ_metadata)
                print("Updated OGM Metadata Type:", type(ogm_metadata))
                print("First Entry of Updated OGM Metadata:", ogm_metadata[0] if ogm_metadata else "No Data")

                # Initialize the Kalman filter
                state_dim = 2  # For position (x, y, z)
                measurement_dim = 2  # For measurements (x, y, z)
                assert state_dim == 2 and measurement_dim == 2, "Kalman filter dimensions mismatch"
                kf = KalmanFilter(state_dim, measurement_dim)

                # Loop through each observation
                for entry in ogm_metadata:
                    try:
                        # Ensure the entry has a valid label
                        label = entry.get("label", -1)
                        if label == -1:
                            # logger.warning(f"Skipping entry with invalid label: {entry}")
                            continue

                        # Extract the position data
                        key = next((k for k in entry if isinstance(entry[k], list)), None)
                        if key:
                            position = entry[key]  # Position is [lon, lat, elev]
                            print(f"Position Key: {key}, Position: {position}")

                            # Validate and prepare the position
                            if len(position) == 3:  # Ensure it has an [x, y, z] format
                                z = np.array(position).reshape((measurement_dim, 1))  # Reshape for the Kalman filter
                                kf.predict()
                                kf.update(z)

                                # Update the entry with the new state (updated position)
                                updated_position = kf.x.flatten().tolist()  # Convert updated state to list
                                entry[key] = updated_position  # Update position in the entry
                                print(f"Updated Position: {updated_position}")
                            else:
                                logger.warning(f"Invalid position format in entry: {position}")
                        else:
                            logger.warning(f"No valid position key found in entry: {entry}")
                    except Exception as e:
                        logger.error(f"Error updating Kalman filter for entry {entry}: {e}")
        except Exception as e:
            logger.info(f"Object measurements are not available. Exception: {e}")

        #######################################################################################
        # Save updated metadata to JSON
        #######################################################################################
        try:
            metadata_path = f"estimated_OGM/occupancy_grid_map_{disaster_type}_Objects.json"
            # Validate ogm_metadata before saving
            if not isinstance(ogm_metadata, list):
                raise ValueError("ogm_metadata is not a list.")

            with open(metadata_path, 'w') as f:
                json.dump(ogm_metadata, f, indent=4)
            logger.info(f"OGM metadata successfully saved to {metadata_path}")

        except ValueError as e:
            logger.error(f"Validation error for OGM metadata: {e}")
        except Exception as e:
            logger.error(f"Failed to save OGM metadata: {e}")

        geojson_path = f"estimated_OGM/occupancy_grid_map_{disaster_type}_Objects_filtered.json"

        # Generate and Save GeoJSON
        try:
            geojson = {"type": "FeatureCollection", "features": []}

            # Iterate through OGM metadata to create GeoJSON features
            for entry in ogm_metadata:
                # Validate each entry
                geo_coords = entry.get("geo_coords", [])
                if not (isinstance(geo_coords, list) and len(geo_coords) == 3):
                    # logger.warning(f"Skipping invalid geo_coords in entry: {entry}")
                    continue

                score = entry.get("score", 0.0)
                label = entry.get("label", -1)

                if label == -1:
                    # logger.info(f"Skipping entry with label -1: {entry}")
                    continue

                # Construct the GeoJSON feature
                feature = {
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [geo_coords[0], geo_coords[1], geo_coords[2]]  # Use lon, lat, elev
                    },
                    "properties": {
                        "score": score,
                        "label": label
                    }
                }
                geojson["features"].append(feature)

            # Save the GeoJSON
            with open(geojson_path, 'w') as f:
                json.dump(geojson, f, indent=4)
            logger.info(f"Filtered GeoJSON successfully saved to {geojson_path}")

        except ValueError as e:
            logger.error(f"Validation error while creating GeoJSON: {e}")
        except PermissionError as e:
            logger.error(f"Permission denied when saving filtered GeoJSON to {geojson_path}: {e}")
        except FileNotFoundError as e:
            logger.error(f"Directory not found for saving filtered GeoJSON to {geojson_path}: {e}")
        except IOError as e:
            logger.error(f"I/O error occurred while saving filtered GeoJSON to {geojson_path}: {e}")
        except Exception as e:
            logger.error(f"Unexpected error while saving filtered GeoJSON: {e}")

        #######################################################################################
        # Convert Geo Json to Geo Tiff
        #######################################################################################
        try:
            geotiff_path = f"estimated_OGM/occupancy_grid_map_{disaster_type}_Objects_filtered.tif"
            output_tiff_path = f"occupancy_grid_map_{disaster_type}_Objects_filtered.tif"

            logger.info(f"Converting GeoJSON to GeoTIFF at {geotiff_path}")

            # Validate the GeoJSON file
            if not os.path.exists(geojson_path):
                raise FileNotFoundError(f"GeoJSON file not found: {geojson_path}")

            with open(geojson_path, 'r') as f:
                geojson_data = json.load(f)

            if not geojson_data.get("features", []):
                raise ValueError("GeoJSON file contains no features. Cannot generate raster.")

            # Calculate raster size and validate pixel sizes
            bbox = get_geojson_bbox(geojson_path)  # Function to compute bounding box
            logger.info(f"GeoJSON bounding box: {bbox}")

            if not bbox or any(k not in bbox for k in ["min_lon", "max_lon", "min_lat", "max_lat"]):
                raise ValueError(f"Invalid or incomplete bounding box: {bbox}")

            # Calculate pixel size in degrees based on the desired resolution (10 meters per pixel)
            meters_per_degree = 111320  # Approximate value at the equator
            pixel_size_lat = 10 / meters_per_degree  # Latitude pixel size in degrees
            pixel_size_lon = 10 / (meters_per_degree * np.cos(np.radians((bbox["min_lat"] + bbox["max_lat"]) / 2)))
            logger.info(f"Pixel size (degrees): Lat = {pixel_size_lat}, Lon = {pixel_size_lon}")

            # Calculate raster dimensions
            raster_width = int((bbox["max_lon"] - bbox["min_lon"]) / pixel_size_lon)
            raster_height = int((bbox["max_lat"] - bbox["min_lat"]) / pixel_size_lat)

            logger.info(f"Calculated raster size: {raster_width}x{raster_height}")
            if raster_width <= 0 or raster_height <= 0:
                raise ValueError("Raster dimensions must be greater than zero. Check pixel sizes or bounding box.")

            # Convert GeoJSON to GeoTIFF
            logger.info(f"Converting GeoJSON to GeoTIFF at {geotiff_path}")
            geojson_to_multi_band_geotiff(
                geojson_path,
                geotiff_path,
                pixel_size_lat=pixel_size_lat,
                pixel_size_lon=pixel_size_lon
            )

            # Load the GeoTIFF data
            logger.info(f"Loading GeoTIFF data from {geotiff_path}")
            ogm_data, ogm_gt_, ogm_proj = load_image(geotiff_path, 0)

            # Save the GeoTIFF to the output path
            logger.info(f"Saving GeoTIFF to {output_tiff_path}")
            metadata = save_geotiff("estimated_OGM", output_tiff_path, ogm_data, ogm_gt_, crs_epsg=4326)

            # Upload the GeoTIFF to the cloud bucket
            logger.info(f"Uploading GeoTIFF to cloud bucket: {config.BUCKET_NAME}")
            process_and_upload_ogm(obj_entity_ID, output_tiff_path, config.BUCKET_NAME, metadata)

            logger.info(f"OGM successfully processed and uploaded: {geotiff_path}")

        except FileNotFoundError as e:
            logger.error(f"File not found during GeoTIFF conversion: {e}")
        except PermissionError as e:
            logger.error(f"Permission error during GeoTIFF conversion: {e}")
        except ValueError as e:
            logger.error(f"Value error during GeoTIFF conversion: {e}")
        except Exception as e:
            logger.error(f"Unexpected error during GeoTIFF conversion: {e}")

        #######################################################################################
        # Integrate SinglePostResult data with Kalman filter
        #######################################################################################
        try:
            # Load OGM data
            ogm_data, ogm_gt_, ogm_proj = load_image(ogm_path, 0)
            social_media_dir = "downloads/SocialMedia"  # Directory containing SinglePostResult files

            # Gather SinglePostResult files
            single_post_files = [os.path.join(social_media_dir, f) for f in os.listdir(social_media_dir)
                                 if f.startswith("single_posts_results") and f.endswith(".geojson")]

            if not single_post_files:
                logger.info("No SinglePostResult files found. Skipping this step.")
                return

            # Process files in chronological order
            single_post_files.sort()  # Process files in chronological order
            logger.info(f"Found {len(single_post_files)} SinglePostResult files.")

            # Initialize the Kalman filter
            state_dim = 3  # For position (x, y, z)
            measurement_dim = 3  # For measurements (x, y, z)
            kf = KalmanFilter(state_dim, measurement_dim)

            # Function to check if coordinates are within OGM bounds and FOV
            def is_within_fov_ogm(gt, lon, lat):
                try:
                    x, y = coordinates_to_pixel(gt, lon, lat)
                    return 0 <= x < ogm_data.shape[1] and 0 <= y < ogm_data.shape[0]
                except Exception as e:
                    logger.warning(f"FOV check failed for coordinates ({lon}, {lat}): {e}")
                    return False

            # Process each SinglePostResult file
            for single_post_file in single_post_files:
                try:
                    with open(single_post_file, 'r') as f:
                        single_post_data = json.load(f)
                    logger.info(f"Processing SinglePostResult file: {single_post_file}")

                    for feature in single_post_data.get("features", []):
                        properties = feature.get("properties", {})
                        geometry = feature.get("geometry", {})

                        if geometry.get("type") == "Point":
                            coordinates = geometry.get("coordinates", [])
                            if len(coordinates) == 2:  # Ensure valid geographic coordinates
                                lon, lat = coordinates
                                # logger.debug(f"Mapping post coordinates: {lon}, {lat}")

                                # Check if the observation is within the FOV of the OGM
                                if not is_within_fov_ogm(ogm_gt_, lon, lat):
                                    # logger.info(f"Coordinates ({lon}, {lat}) are outside the FOV. Skipping.")
                                    continue

                                # Construct the observation (x, y, z)
                                z = np.array([lon, lat, 0]).reshape((3, 1))  # Assuming no altitude

                                # Kalman filter prediction and update
                                try:
                                    kf.predict()
                                    kf.update(z)

                                    # Get the refined position
                                    refined_position = kf.x.flatten().tolist()

                                    # Map the refined position to the closest metadata entry
                                    try:
                                        # Find the closest metadata entry
                                        closest_entry = None
                                        min_distance = float('inf')

                                        for entry in ogm_metadata:
                                            geo_coords = entry.get("geo_coords")
                                            if not geo_coords or len(geo_coords) != 3:
                                                continue

                                            distance = np.sqrt(
                                                (geo_coords[0] - refined_position[0]) ** 2 +
                                                (geo_coords[1] - refined_position[1]) ** 2
                                            )

                                            if distance < min_distance:
                                                min_distance = distance
                                                closest_entry = entry

                                        # Update the closest entry with the refined position
                                        if closest_entry and min_distance < 0.0005:  # Define tolerance
                                            closest_entry["coordinates"] = refined_position
                                            logger.info(
                                                f"Updated metadata entry with refined position: {refined_position}")
                                        else:
                                            logger.warning(
                                                f"No close metadata entry found for refined position: {refined_position}")

                                    except Exception as e:
                                        logger.warning(
                                            f"Error mapping refined coordinates ({refined_position}) to metadata: {e}")
                                except Exception as e:
                                    logger.warning(f"Kalman filter update failed for coordinates ({lon}, {lat}): {e}")

                    # Delete the file after successful processing
                    os.remove(single_post_file)
                    logger.info(f"Successfully processed and deleted: {single_post_file}")

                except Exception as e:
                    logger.error(f"Error processing {single_post_file}: {e}")
        except Exception as e:
            logger.error(f"Error integrating SinglePostResult data with Kalman filter: {e}")

        #######################################################################################
        # Save updated metadata to JSON
        #######################################################################################
        try:
            metadata_path = f"estimated_OGM/occupancy_grid_map_{disaster_type}_Objects.json"
            # Validate ogm_metadata before saving
            if not isinstance(ogm_metadata, list):
                raise ValueError("ogm_metadata is not a list.")

            with open(metadata_path, 'w') as f:
                json.dump(ogm_metadata, f, indent=4)
            logger.info(f"OGM metadata successfully saved to {metadata_path}")

        except ValueError as e:
            logger.error(f"Validation error for OGM metadata: {e}")
        except Exception as e:
            logger.error(f"Failed to save OGM metadata: {e}")

        geojson_path = f"estimated_OGM/occupancy_grid_map_{disaster_type}_Objects_filtered.json"

        # Generate and Save GeoJSON
        try:
            geojson = {"type": "FeatureCollection", "features": []}

            # Iterate through OGM metadata to create GeoJSON features
            for entry in ogm_metadata:
                # Validate each entry
                geo_coords = entry.get("geo_coords", [])
                if not (isinstance(geo_coords, list) and len(geo_coords) == 3):
                    # logger.warning(f"Skipping invalid geo_coords in entry: {entry}")
                    continue

                score = entry.get("score", 0.0)
                label = entry.get("label", -1)

                if label == -1:
                    # logger.info(f"Skipping entry with label -1: {entry}")
                    continue

                # Construct the GeoJSON feature
                feature = {
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [geo_coords[0], geo_coords[1], geo_coords[2]]  # Use lon, lat, elev
                    },
                    "properties": {
                        "score": score,
                        "label": label
                    }
                }
                geojson["features"].append(feature)

            # Save the GeoJSON
            with open(geojson_path, 'w') as f:
                json.dump(geojson, f, indent=4)
            logger.info(f"Filtered GeoJSON successfully saved to {geojson_path}")

        except ValueError as e:
            logger.error(f"Validation error while creating GeoJSON: {e}")
        except PermissionError as e:
            logger.error(f"Permission denied when saving filtered GeoJSON to {geojson_path}: {e}")
        except FileNotFoundError as e:
            logger.error(f"Directory not found for saving filtered GeoJSON to {geojson_path}: {e}")
        except IOError as e:
            logger.error(f"I/O error occurred while saving filtered GeoJSON to {geojson_path}: {e}")
        except Exception as e:
            logger.error(f"Unexpected error while saving filtered GeoJSON: {e}")

        #######################################################################################
        # Convert Geo Json to Geo Tiff
        #######################################################################################
        try:
            geotiff_path = f"estimated_OGM/occupancy_grid_map_{disaster_type}_Objects_filtered.tif"
            output_tiff_path = f"occupancy_grid_map_{disaster_type}_Objects_filtered.tif"

            logger.info(f"Converting GeoJSON to GeoTIFF at {geotiff_path}")

            # Validate the GeoJSON file
            if not os.path.exists(geojson_path):
                raise FileNotFoundError(f"GeoJSON file not found: {geojson_path}")

            with open(geojson_path, 'r') as f:
                geojson_data = json.load(f)

            if not geojson_data.get("features", []):
                raise ValueError("GeoJSON file contains no features. Cannot generate raster.")

            # Calculate raster size and validate pixel sizes
            bbox = get_geojson_bbox(geojson_path)  # Function to compute bounding box
            logger.info(f"GeoJSON bounding box: {bbox}")

            if not bbox or any(k not in bbox for k in ["min_lon", "max_lon", "min_lat", "max_lat"]):
                raise ValueError(f"Invalid or incomplete bounding box: {bbox}")

            # Calculate pixel size in degrees based on the desired resolution (10 meters per pixel)
            meters_per_degree = 111320  # Approximate value at the equator
            pixel_size_lat = 10 / meters_per_degree  # Latitude pixel size in degrees
            pixel_size_lon = 10 / (meters_per_degree * np.cos(np.radians((bbox["min_lat"] + bbox["max_lat"]) / 2)))
            logger.info(f"Pixel size (degrees): Lat = {pixel_size_lat}, Lon = {pixel_size_lon}")

            # Calculate raster dimensions
            raster_width = int((bbox["max_lon"] - bbox["min_lon"]) / pixel_size_lon)
            raster_height = int((bbox["max_lat"] - bbox["min_lat"]) / pixel_size_lat)

            logger.info(f"Calculated raster size: {raster_width}x{raster_height}")
            if raster_width <= 0 or raster_height <= 0:
                raise ValueError("Raster dimensions must be greater than zero. Check pixel sizes or bounding box.")

            # Convert GeoJSON to GeoTIFF
            logger.info(f"Converting GeoJSON to GeoTIFF at {geotiff_path}")
            geojson_to_multi_band_geotiff(
                geojson_path,
                geotiff_path,
                pixel_size_lat=pixel_size_lat,
                pixel_size_lon=pixel_size_lon
            )

            # Load the GeoTIFF data
            logger.info(f"Loading GeoTIFF data from {geotiff_path}")
            ogm_data, ogm_gt_, ogm_proj = load_image(geotiff_path, 0)

            # Save the GeoTIFF to the output path
            logger.info(f"Saving GeoTIFF to {output_tiff_path}")
            metadata = save_geotiff("estimated_OGM", output_tiff_path, ogm_data, ogm_gt_, crs_epsg=4326)

            # Upload the GeoTIFF to the cloud bucket
            logger.info(f"Uploading GeoTIFF to cloud bucket: {config.BUCKET_NAME}")
            process_and_upload_ogm(obj_entity_ID, output_tiff_path, config.BUCKET_NAME, metadata)

            logger.info(f"OGM successfully processed and uploaded: {geotiff_path}")

        except FileNotFoundError as e:
            logger.error(f"File not found during GeoTIFF conversion: {e}")
        except PermissionError as e:
            logger.error(f"Permission error during GeoTIFF conversion: {e}")
        except ValueError as e:
            logger.error(f"Value error during GeoTIFF conversion: {e}")
        except Exception as e:
            logger.error(f"Unexpected error during GeoTIFF conversion: {e}")


def get_geojson_bbox(geojson_path):
    """
    Calculate the bounding box of a GeoJSON file.

    Parameters:
        geojson_path (str): Path to the GeoJSON file.

    Returns:
        dict: Bounding box with min/max longitude and latitude.

    Raises:
        ValueError: If no valid coordinates are found in the GeoJSON.
    """
    with open(geojson_path, 'r') as f:
        geojson_data = json.load(f)

    coordinates = []

    for feature in geojson_data.get("features", []):
        geometry = feature.get("geometry", {})
        geom_type = geometry.get("type")

        if geom_type == "Point":
            # Append single coordinate
            coordinates.append(geometry["coordinates"])
        elif geom_type == "Polygon":
            # Extract and flatten polygon coordinates
            for polygon in geometry.get("coordinates", []):
                coordinates.extend(polygon)
        elif geom_type == "MultiPolygon":
            # Extract and flatten multi-polygon coordinates
            for multi_polygon in geometry.get("coordinates", []):
                for polygon in multi_polygon:
                    coordinates.extend(polygon)

    if not coordinates:
        raise ValueError("No valid coordinates found in GeoJSON.")

    # Separate longitudes and latitudes
    try:
        lons, lats = zip(*coordinates)
    except ValueError:
        raise ValueError("Malformed coordinate structure in GeoJSON.")

    return {
        "min_lon": min(lons),
        "max_lon": max(lons),
        "min_lat": min(lats),
        "max_lat": max(lats)
    }


def load_raster(filename):
    """Load a raster file and return the dataset, data array, and transform."""
    try:
        with rasterio.open(filename) as src:
            data = src.read(1)
            transform_ = src.transform
            crs = src.crs
        return data, transform_, crs
    except Exception as e:
        logger.info(f"Raster was not loaded due to {e}")


def create_empty_grid(polygon_coords_, pixel_width_, pixel_height_):
    """Create an empty occupancy grid based on polygon coordinates."""
    minx_, miny_ = np.min(polygon_coords_, axis=0)
    maxx_, maxy_ = np.max(polygon_coords_, axis=0)

    # Define dimensions based on bounding box
    ogm_width = 1414
    ogm_height = 1414

    occupancy_grid_data = np.zeros((ogm_height, ogm_width), dtype=np.float32)
    occupancy_grid_data = np.full_like(occupancy_grid_data, 0.01)  # Start with neutral prior

    transform_ = from_origin(minx_, maxy_, pixel_width_, pixel_height_)

    return occupancy_grid_data, transform_


def get_intersected_data(existing_data, existing_transform, new_polygon_coords):
    """Transfer intersected data from an existing ROI map to a new map."""
    new_polygon = Polygon(new_polygon_coords)
    mask = geometry_mask([new_polygon], transform=existing_transform, invert=True, out_shape=existing_data.shape)
    intersected_data = np.where(mask, existing_data, 0)
    return intersected_data


def reorder_polygon(polygon1, polygon2):
    """Reorder polygon2 to match polygon1's vertex order."""

    # Rotate polygon2 to start from the same point as polygon1
    start_point = polygon1[0]

    if start_point in polygon2:
        # Find the index of the starting point in polygon2
        start_idx = polygon2.index(start_point)

        # Reorder polygon2 to start from the same point
        reordered_polygon2 = polygon2[start_idx:] + polygon2[:start_idx]

        # Check if polygon2 is in reverse order and needs to be reversed
        if reordered_polygon2[1] != polygon1[1]:
            reordered_polygon2.reverse()

        return reordered_polygon2
    else:
        return None


def generate_metadata(occupancy_grid_data, transform):
    """
    Generate metadata for each pixel in the occupancy grid.

    Args:
        occupancy_grid_data (np.ndarray): The occupancy grid data array.
        transform (Affine): The transform object used to convert pixel indices to geographical coordinates.

    Returns:
        dict: A dictionary mapping pixel coordinates (as strings) to their geographical coordinates and pixel values.
    """
    metadata = {}
    height, width = occupancy_grid_data.shape

    for row in range(height):
        for col in range(width):
            # Get pixel value and convert to a standard float
            pixel_value = float(occupancy_grid_data[row, col])  # or use occupancy_grid_data[row, col].item()

            # Calculate geographical coordinates
            x, y = transform * (col, row)  # Transform from pixel (col, row) to (x, y)

            # Use a string representation for the pixel coordinates
            pixel_key = f"{col},{row}"

            # Store in the dictionary
            metadata[pixel_key] = {
                'geo_coordinates': (x, y),
                'pixel_value': pixel_value
            }

    return metadata


def get_roi(new_polygon_coords, existing_map_path, resolution, crs_epsg=4326):
    """
    Create a GeoTIFF based on polygon coordinates with a specified resolution.

    Args:
        new_polygon_coords (list): Polygon coordinates.
        existing_map_path (str): Path for saving the GeoTIFF.
        resolution (float): Resolution (in meters or degrees per pixel).
        crs_epsg (int): EPSG code for CRS.

    Returns:
        data_set: GDAL dataset, or None if an error occurs.
    """
    try:
        if resolution <= 0:
            raise ValueError("Resolution must be a positive value.")

        polygon = Polygon(new_polygon_coords)
        if not polygon.is_valid:
            polygon = polygon.buffer(0)
            if not polygon.is_valid:
                raise ValueError("Polygon is invalid.")

        minx, miny, maxx, maxy = polygon.bounds

        logger.debug(f"Polygon bounds: minx={minx}, maxx={maxx}, miny={miny}, maxy={maxy}")

        if crs_epsg == 4326:
            width = max(1, int((maxx - minx) / (resolution / 111320)))  # Convert meters to degrees
            height = max(1, int((maxy - miny) / (resolution / 111320)))
        else:
            width = max(1, int((maxx - minx) / resolution))
            height = max(1, int((maxy - miny) / resolution))

        transform_ = from_bounds(minx, miny, maxx, maxy, width, height)

        occupancy_grid_data = np.ones((height, width), dtype=np.uint8) * 0.5
        with rasterio.open(
                existing_map_path,
                'w',
                driver='GTiff',
                height=height,
                width=width,
                count=1,
                dtype=occupancy_grid_data.dtype,
                crs=CRS.from_epsg(crs_epsg),
                transform=transform_,
        ) as dst:
            dst.write(occupancy_grid_data, 1)
        return gdal.Open(existing_map_path, gdal.GA_ReadOnly)

    except Exception as e:
        logger.error(f"Error in get_roi: {e}")
        return None


def mask_geotiff_with_polygon_exact(geotiff_path, polygon_coords, output_path):
    """
    Mask a GeoTIFF file with a given polygon. Anything outside the polygon is set to zero.
    The polygon's exact shape is preserved.

    Args:
        geotiff_path (str): Path to the input GeoTIFF.
        polygon_coords (list): List of polygon coordinates (e.g., [[(x1, y1), (x2, y2), ...]]).
        output_path (str): Path to save the masked GeoTIFF.

    Returns:
        str: Path to the masked GeoTIFF, or None if an error occurs.
    """
    try:
        # Open the GeoTIFF
        with rasterio.open(geotiff_path) as src:
            # Convert polygon coordinates to GeoJSON-like format
            polygon_geojson = {
                "type": "Polygon",
                "coordinates": polygon_coords
            }

            # Mask the data using the polygon
            masked_data, masked_transform = mask(
                src,
                [shape(polygon_geojson)],
                crop=False,  # Do not crop; retain original raster extent
                filled=True,
                invert=False  # Mask everything outside the polygon
            )

            # Replace masked areas with zero
            masked_data = np.where(masked_data == src.nodata, 0, masked_data)

            # Update metadata for the output file
            out_meta = src.meta.copy()
            out_meta.update({
                "driver": "GTiff",
                "height": masked_data.shape[1],
                "width": masked_data.shape[2],
                "transform": masked_transform,
                "nodata": 0  # Set nodata value explicitly
            })

            # Write the masked data to the new GeoTIFF
            with rasterio.open(output_path, "w", **out_meta) as dst:
                dst.write(masked_data)

        return output_path

    except Exception as e:
        print(f"Error masking GeoTIFF: {e}")
        return None


def convert_to_polygon(return_bounding_box=True):
    """
    Convert an ROI (Region of Interest) from the cache into polygon coordinates.

    Args:
        return_bounding_box (bool): If True, return the bounding box coordinates.
                                    If False, return the original polygon.

    Returns:
        list: Polygon coordinates (bounding box or original polygon).
        None: If an error occurs.

    Expected ROI Format:
        - global_cache['roi'] should be a list containing a single polygon.
        - Example:
          [
              [[x1, y1], [x2, y2], [x3, y3], [x4, y4], [x1, y1]]
          ]
    """
    try:
        # Retrieve ROI from the cache
        roi = global_cache.get('roi')
        if not roi or not isinstance(roi, list) or not isinstance(roi[0], list):
            raise ValueError("ROI is not available or improperly set. Must be a non-empty list of polygons.")

        # Validate the structure of the first polygon in the ROI
        roi_polygon = roi[0]
        if not all(isinstance(coord, list) and len(coord) == 2 for coord in roi_polygon):
            raise ValueError("Invalid ROI structure. Each coordinate must be a [x, y] pair.")

        # Ensure the polygon is closed
        if roi_polygon[0] != roi_polygon[-1]:
            raise ValueError("ROI polygon is not closed. The first and last coordinates must match.")

        logger.info(f"Retrieved ROI polygon: {roi_polygon}")

        if not return_bounding_box:
            return roi_polygon  # Return the original polygon if requested

        # Separate x and y coordinates (longitudes and latitudes)
        x_coords = [coord[0] for coord in roi_polygon]
        y_coords = [coord[1] for coord in roi_polygon]

        # Calculate minx, maxx, miny, maxy
        minx, maxx = min(x_coords), max(x_coords)
        miny, maxy = min(y_coords), max(y_coords)

        logger.debug(f"Computed bounds: minx={minx}, maxx={maxx}, miny={miny}, maxy={maxy}")

        # Create the polygon coordinates
        polygon_coords_ = [
            [minx, maxy],  # Top-left
            [maxx, maxy],  # Top-right
            [maxx, miny],  # Bottom-right
            [minx, miny],  # Bottom-left
            [minx, maxy]  # Closing the polygon
        ]

        logger.info(f"Generated polygon coordinates: {polygon_coords_}")
        return polygon_coords_

    except ValueError as ve:
        logger.error(f"Validation error in convert_to_polygon: {ve}")
    except Exception as e:
        logger.exception(f"Unexpected error in convert_to_polygon: {e}")

    # Return None in case of failure
    return None


def initialize_entities():
    """Initialize entities based on a natural disaster type and objects."""
    global ND_entity_ID, obj_entity_ID

    # Log the natural disaster info from the cache
    natural_disaster = global_cache.get('natural_disaster', 'No disaster info')
    logger.info(f"Natural disaster info from global_cache: {natural_disaster}")

    # Determine ND_entity_ID based on a natural disaster type
    if natural_disaster == 'Fire':
        ND_entity_ID = config.ENTITY_Maps4Fire_ID
        logger.info(f"Set ND_entity_ID for Fire: {ND_entity_ID}")
    elif natural_disaster == 'Flood':
        ND_entity_ID = config.ENTITY_Maps4Flood_ID
        logger.info(f"Set ND_entity_ID for Flood: {ND_entity_ID}")
    else:
        logger.warning(f"Unrecognized natural disaster type: {natural_disaster}")
        ND_entity_ID = None

    # Set the object entity ID
    obj_entity_ID = config.ENTITY_Maps4Object_ID

    # List of entities to process
    entity_ids = [ND_entity_ID, obj_entity_ID] if ND_entity_ID else [obj_entity_ID]

    for entity_id in entity_ids:
        if not check_entity_exists(entity_id):
            logger.info(f"Entity {entity_id} does not exist, attempting to create it.")
            try:
                parts = entity_id.split(":")
                entity_type = parts[-2]
                response = create_entity(entity_id, entity_type)

                if isinstance(response, dict):
                    status = response.get("status")
                    if status == "Entity already exists":
                        logger.info(f"Entity {entity_id} already exists, no action needed.")
                    elif status == "Error":
                        logger.error(f"Failed to create entity {entity_id}: {response}")
                    else:
                        logger.info(f"Entity {entity_id} created successfully.")
                else:
                    logger.error(f"Unexpected response type for entity creation: {response}")
            except Exception as e:
                logger.exception(f"Exception occurred while creating entity {entity_id}: {e}")
        else:
            logger.info(f"Entity {entity_id} already exists, skipping creation.")


########################################################################################################################
# Check Entity Existence
########################################################################################################################
def check_entity_exists(entity_id_):
    url_ = f'{config.BROKER_URL}/ngsi-ld/v1/entities/{entity_id_}'
    headers = {'Accept': 'application/ld+json'}

    try:
        response_ = requests.get(url_, headers=headers)
        if response_.status_code == 200:
            logger.info(f'Entity {entity_id_} exists.')
            return True
        elif response_.status_code == 404:
            logger.info(f'Entity {entity_id_} does not exist.')
            return False
        else:
            logger.error(f'Error checking entity existence: {response_.status_code} - {response_.text}')
            return None
    except requests.exceptions.RequestException as e_:
        logger.error(f'Error checking entity existence: {e_}')
        return None


########################################################################################################################
# Create Entity
########################################################################################################################
def create_entity(entity_ID, entity_type_):
    natural_disaster = global_cache.get('natural_disaster', 'No disaster info')
    logger.info("Attempting to create entity...")  # Log for debugging
    url = f'{config.BROKER_URL}/ngsi-ld/v1/entities/'

    headers = {
        'Content-Type': 'application/ld+json',
        'Accept': 'application/ld+json'
    }

    # Check if entity exists
    if check_entity_exists(entity_ID):
        logger.info(f"Entity {entity_ID} already exists.")
        return {"status": "Entity already exists"}, 409

    data = {
        "id": entity_ID,
        "type": entity_type_,
        "creator": "USE",
        "description": {
            "type": "Property",
            "value": "Occupancy grid maps for ND estimation"
        },
        "creationDate": {
            "type": "Property",
            "value": time.strftime('%Y-%m-%dT%H:%M:%SZ', time.localtime())
        },
        "bandName": 'Occupancy',
        "software": {
            "type": "Property",
            "value": "GDAL Python"
        },
        "geotransform":
            {
                "type": "Property",
                "value":
                    {
                        "XMin": {"type": "Property", "value": 0.0},
                        "XRes": {"type": "Property", "value": 0.0},
                        "YMax": {"type": "Property", "value": 0.0},
                        "YRes": {"type": "Property", "value": 0.0}
                    }
            },
        "spatialReference": {
            "type": "Property",
            "value": "EPSG:4326"
        },
        "location": {
            "type": "Polygon",
            "coordinates": [
                [
                    [-0.165825, 51.495065],
                    [-0.165825, 51.513123],
                    [-0.111065, 51.513123],
                    [-0.111065, 51.495065],
                    [-0.165825, 51.495065]
                ]
            ]
        },
        "minio_url": {
            "type": "Property",
            "value": f'https://{config.MINIO_ENDPOINT}/{config.BUCKET_NAME}/occupancy_grid_map{natural_disaster}.tif'
        },
        "filename": {
            "type": "Property",
            "value": f"occupancy_grid_map_{natural_disaster}.tif"
        },
        "bucket": {
            "type": "Property",
            "value": "naples"
        },
        "@context": [
            "https://uri.etsi.org/ngsi-ld/v1/ngsi-ld-core-context.jsonld"
        ]
    }

    try:
        response_ = requests.post(url, json=data, headers=headers)
        response_.raise_for_status()  # Raise exception for HTTP errors

        if response_.status_code == 201:
            logger.info(f"Entity {entity_ID} created successfully.")
            return response_
        elif response_.status_code == 409:
            logger.info(f"Entity {entity_ID} already exists.")
            return {"status": "Entity already exists"}, 409
    except requests.exceptions.RequestException as e:
        logger.error(f"Error creating entity {entity_ID}: {e}")
        return {"status": "Connection Error"}, 500

    logger.warning(f"Failed to create entity: {response_.status_code} - {response_.text}")
    return {"status": "Error", "message": response_.text}, response_.status_code


########################################################################################################################
# Update Entity
########################################################################################################################
def update_entity(entity_id_, payload):
    response = None
    url_ = f'{config.BROKER_URL}/ngsi-ld/v1/entities/{entity_id_}/attrs'
    headers = {'Content-Type': 'application/ld+json'}
    data_to_send = {
        "description": payload["description"],
        "creationDate": payload["creationDate"],
        "geotransform": {
            "type": "Property",
            "value": {
                "XMin": {"type": "Property", "value": payload["XMin"]},
                "XRes": {"type": "Property", "value": payload["XRes"]},
                "YMax": {"type": "Property", "value": payload["YMax"]},
                "YRes": {"type": "Property", "value": payload["YRes"]}
            }
        },
        "minio_url": payload["minio_url"],
        "filename": payload["file_name"],
        "bucket": payload["bucket"],
        "location": {
            "type": "Polygon",
            "coordinates": payload["coordinates"]
        }
    }
    # Ensure @context is included in the payload
    payload_with_context = {
        "@context": "https://uri.etsi.org/ngsi-ld/v1/ngsi-ld-core-context.jsonld",
        **data_to_send
    }

    # Validate coordinates
    if not isinstance(payload["coordinates"], list) or len(payload["coordinates"]) == 0:
        logger.error(f"Invalid coordinates in payload for entity {entity_id_}")
        return None

    try:
        response = requests.post(url_, json=payload_with_context, headers=headers)
        response.raise_for_status()  # Will raise HTTPError for bad responses (4th and 5th)

        if response.status_code in [200, 204]:
            logger.info(f"Entity {entity_id_} updated successfully.")
        else:
            logger.warning(
                f"Unexpected response status code {response.status_code} when updating entity {entity_id_}. Response "
                f"text: {response.text}")

        return response
    except requests.exceptions.HTTPError as e:
        logger.error(f"HTTP error occurred while updating entity {entity_id_}: {e}. Response text: {response.text}")
    except requests.exceptions.RequestException as e:
        logger.error(f"Error occurred while updating entity {entity_id_}: {e}")

    return None


def distance_components(lat1, lon1, lat2, lon2):
    # Differences in coordinates
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    # Latitude component (in meters)
    lat_distance_ = dlat * 111320

    # Longitude component (in meters)
    lon_distance_ = dlon * 111320 * math.cos(((lat1 + lat2) * np.pi / 180) / 2)

    return lat_distance_, lon_distance_


def lonlat_to_pixel(GTransform, lon, lat):
    x = (lon - GTransform[0]) / GTransform[1]
    y = (lat - GTransform[3]) / GTransform[5]
    return int(x), int(y)


def upscale_geotiff(input_file_, output_file_, scale_factor_x_, scale_factor_y_):
    # Set GDAL_DATA environment variable
    # Open input raster
    input_ds = gdal.Open(input_file_)
    if input_ds is None:
        print(f"Error: Could not open input file: {input_file_}")
        logger.info(f"Error: Could not open input file: {input_file_}")
        return
    # Get input raster information
    input_proj = input_ds.GetProjection()
    input_geotrans = input_ds.GetGeoTransform()
    input_band = input_ds.GetRasterBand(1)
    # Calculate output raster dimensions
    cols = int(input_ds.RasterXSize * scale_factor_x_)
    rows = int(input_ds.RasterYSize * scale_factor_y_)
    # Create output raster
    driver = gdal.GetDriverByName('GTiff')
    output_ds = driver.Create(output_file_, cols, rows, 1, input_band.DataType)
    if output_ds is None:
        print(f"Error: Could not create output file: {output_file_}")
        logger.info(f"Error: Could not create output file: {output_file_}")
        return
    output_ds.SetProjection(input_proj)
    # Calculate new geo-transform
    new_geotrans = (input_geotrans[0], input_geotrans[1] / scale_factor_x_, input_geotrans[2],
                    input_geotrans[3], input_geotrans[4], input_geotrans[5] / scale_factor_y_)
    output_ds.SetGeoTransform(new_geotrans)
    # Perform resampling
    gdal.ReprojectImage(input_ds, output_ds, None, None, gdal.GRA_Bilinear)


def get_sate_roi(PolygonCoords):
    if global_cache.get('natural_disaster', 'No disaster info') == "Fire":
        try:
            polygon = Polygon(PolygonCoords)

            # Load the vector data
            gpkg_path = ('Satellite_DLR/S2A_MSIL2A_20210814T102031_N0301_R065_T32TMK_20210814T132326.SAFE_d_f'
                         '.gpkg')
            gdf = gpd.read_file(gpkg_path, layer='S2A_MSIL2A_20210814T102031_N0301_R065_T32TMK_20210814T132326')

            # Ensure both the data and AOI polygon are in EPSG:4326
            gdf_4326 = gdf.to_crs(epsg=4326)
            polygon_gdf = gpd.GeoDataFrame([1], geometry=[polygon], crs="EPSG:4326")

            # Perform cropping operation
            cropped_gdf = gpd.clip(gdf_4326, polygon_gdf)

            if cropped_gdf.empty:
                print("Cropping resulted in an empty GeoDataFrame. Check CRS alignment and polygon.")
                logger.info("Cropping resulted in an empty GeoDataFrame. Check CRS alignment and polygon.")
                return
            else:
                print(f"Cropped GeoDataFrame has {len(cropped_gdf)} geometries.")
                logger.info(f"Cropped GeoDataFrame has {len(cropped_gdf)} geometries.")

            # Generate timestamp for the output file name
            now = datetime.now()
            timestamp = now.strftime("%Y%m%d%H%M%S")
            output_tiff_path = f"segmented_sat_roi_imgs/cropped_sat_data_{global_cache.get('natural_disaster', 'No disaster info')}_{timestamp}.tif"

            # Save the cropped area as a GeoTIFF file
            save_cropped_as_geotiff(cropped_gdf, output_tiff_path)
        except Exception as e:
            print(f"No Sat File due to {e}")
            logger.info(f"No Sat File due to {e}")
    else:
        try:
            for sat_folders in sorted(os.listdir("Satellite_DLR"), reverse=False):
                sat_title_splits = sat_folders.split('_')
                geo_tiff = os.path.join("Satellite_DLR", sat_folders, f"{'_'.join(sat_title_splits)}_WATER.tif")

                #########################################################################
                # Rasterio
                #########################################################################
                raster_layer = rasterio.open(geo_tiff)
                ######################################################################
                # GDAL
                ######################################################################
                raster_gdal = gdal.Open(geo_tiff)
                ###################################
                # x,y coordinates
                ###################################
                x_pixels = raster_gdal.RasterXSize
                y_pixels = raster_gdal.RasterYSize
                ###################################
                gt = raster_gdal.GetGeoTransform()

                x_t_0 = gt[0]
                y_t_0 = gt[3]

                x_w = gt[1]
                y_w = gt[5]

                x_t_1 = x_t_0 + x_pixels * x_w
                y_t_1 = y_t_0 + y_pixels * y_w

                # Calculate resolution (pixels per meter) in x and y directions
                # resolution_x = 1 / x_w
                # resolution_y = 1 / y_w
                ############################################################
                # Crop satellite image to ROI (to the information fusion map)
                ############################################################

                # Calculate latitude and longitude components
                lat_distance, lon_distance = distance_components(y_t_0, x_t_0, y_t_1, x_t_1)

                # displacement_lng = abs(x_t_1 - x_t_0)
                # displacement_lat = abs(y_t_1 - y_t_0)

                ######################################################################
                # Calculating the lat and lng distance in meters for the satellite
                ######################################################################
                # lat_distance = lon_or_lat_to_meter(y_t_0, displacement_lat, 1)
                # lon_distance = lon_or_lat_to_meter(y_t_0, displacement_lng, 0)

                ######################################################################
                lat_res_meters_per_pixel = lat_distance / raster_layer.height
                lng_res_meters_per_pixel = lon_distance / raster_layer.width
                ######################################################################
                # define the dimensions of the occupancy grid map in meters
                ######################################################################
                cam_lat, cam_lng = 50.51681305555555, 6.985609833333333
                w = 707.1067  # meter
                h = 707.1067  # meter
                map_width, map_height = abs(w / lng_res_meters_per_pixel), abs(h / lat_res_meters_per_pixel)

                ######################################################################
                # Defining a map from the Camera GPS location,
                ######################################################################
                lat_max = cam_lat + map_height * x_w  # Upper-left x coordinate
                # lng_max = cam_lng - map_width * y_w # Upper-left y coordinate
                # lat_min = cam_lat - map_height * x_w # Lower-right x coordinate
                lng_min = cam_lng + map_width * y_w  # Lower-right y coordinate

                # Get the geo_transform
                ulx, uly = lonlat_to_pixel(gt, lng_min, lat_max)
                #################################################################
                # Window(col_off, row_off, width, height)
                #################################################################
                window = Window(ulx, uly, width=map_width * 2, height=map_height * 2)

                # window = Window.from_slices((ulx, lrx), (lry, uly))
                data = raster_layer.read(window=window, masked=True)
                transform = raster_layer.window_transform(window)

                # Create a new cropped raster to write to

                profile = raster_layer.profile
                profile.update({
                    'height': map_height * 2,
                    'width': map_width * 2,
                    'transform': transform})
                # Generate timestamp for the output file name
                now = datetime.now()
                timestamp = now.strftime("%Y%m%d%H%M%S")
                path_roi = f"segmented_sat_roi_imgs/cropped_sat_data_{global_cache.get('natural_disaster', 'No disaster info')}_{timestamp}"
                with rasterio.open(path_roi + '%s_%s.tif' % (sat_title_splits[4], sat_title_splits[5]), 'w',
                                   **profile) as dst:
                    # Read the data from the window and write it to the output raster
                    dst.write(data)  # raster.read(window=window)

                #################################################################################
                # Unifying Resolutions
                #################################################################################
                input_file = path_roi + 'cropped_sat_roi_%s_%s.tif' % (
                    sat_title_splits[4], sat_title_splits[5])  # Path to the
                output_file = path_roi + 'cropped_sat_roi_%s_%s.tif' % (
                    sat_title_splits[4], sat_title_splits[5])  # Path to the
                scale_factor_x = abs(lng_res_meters_per_pixel)  # Upscale factor (lng_res_meters_per_pixel)
                scale_factor_y = abs(lat_res_meters_per_pixel)  # Upscale factor (lat_res_meters_per_pixel)
                upscale_geotiff(input_file, output_file, scale_factor_x, scale_factor_y)

                #################################################################################
                # raster_layer = None
                # folder_to_remove = os.path.join(dir_sat, sat_folders)
                # try:
                #     shutil.rmtree(folder_to_remove)
                #     print("**********************************************************")
                #     print("Sat folder removed")
                # except Exception as e:
                #     print("**********************************************************")
                #     print(f"Sat error in removing {e}")
                # break
        except Exception as e:
            print(f"No Sat file due to {e}")


def save_cropped_as_geotiff(cropped_gdf, output_path):
    # Get the bounding box of the cropped GeoDataFrame
    minx_, miny_, maxx_, maxy_ = cropped_gdf.total_bounds

    pixel_width_ = (maxx_ - minx_) / 1414
    pixel_height_ = (maxy_ - miny_) / 1414

    # Define dimensions based on bounding box
    ogm_width = 1414
    ogm_height = 1414

    # Define the resolution (in degrees per pixel)
    # You can adjust this to fit your AOI size/resolution requirements
    # resolution = 0.0001  # Adjust resolution for higher or lower pixel density
    #
    # # Calculate the number of rows and columns
    # width = int((maxx - minx) / resolution)
    # height = int((maxy - miny) / resolution)

    # Define the transform (affine transformation for the GeoTIFF)
    # transform = from_origin(minx, maxy, resolution, resolution)
    transform = from_origin(minx_, maxy_, pixel_width_, pixel_height_)
    # Rasterize the cropped GeoDataFrame geometries
    shapes = [(geom, 1) for geom in cropped_gdf.geometry]  # Assign '1' to all geometries

    # Create an empty raster with the dimensions based on the AOI bounding box
    raster_data = rasterize(
        shapes=shapes,
        out_shape=(ogm_height, ogm_width),
        transform=transform,
        fill=0,  # Background value
        dtype='float32'
    )

    # Write the rasterized output to a GeoTIFF file
    with rasterio.open(
            output_path,
            'w',
            driver='GTiff',
            height=ogm_height,
            width=ogm_width,
            count=1,  # Single-band image
            dtype='float32',
            crs=CRS.from_epsg(4326),  # Use EPSG:4326 (WGS84)
            transform=transform
    ) as dst:
        dst.write(raster_data, 1)


########################################################################################################################
def load_existing_geo_dict(tiff_path):
    """
    Loads an existing geo dictionary from a JSON file.

    Parameters:
        tiff_path (str): Path to the GeoTIFF file (without `.json` extension).

    Returns:
        dict: The loaded geo dictionary if successful.
        None: If the file does not exist, is empty, or cannot be decoded.
    """
    geo_dict_path = os.path.splitext(tiff_path)[0] + ".json"

    if not os.path.exists(geo_dict_path):
        print(f"Geo dictionary file not found: {geo_dict_path}")
        return None

    try:
        with open(geo_dict_path, 'r') as f:
            content = f.read().strip()  # Read and strip whitespace

            if not content:  # Check if the file is empty
                print("The JSON file is empty.")
                return None

            try:
                return json.loads(content)  # Parse JSON content
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")
                return None
    except IOError as e:
        print(f"Error reading the file {geo_dict_path}: {e}")
        return None


# Function to generate a new geo dict from a GeoTIFF file
def generate_geo_dict_from_tiff(tiff_path):
    """
    Generates a geographic dictionary from a GeoTIFF file.

    Parameters:
        tiff_path (str): Path to the GeoTIFF file.

    Returns:
        list: A list of dictionaries with pixel coordinates, geographic coordinates (including elevation), and metadata.
        None: If an error occurs.
    """
    if not os.path.exists(tiff_path):
        logger.error(f"GeoTIFF file not found: {tiff_path}")
        return None

    try:
        geo_dict_list = []

        # Open the GeoTIFF file
        logger.debug(f"Opening GeoTIFF file: {tiff_path}")
        with rasterio.open(tiff_path) as dataset:
            logger.debug(f"CRS: {dataset.crs}")
            logger.debug(f"Width: {dataset.width}, Height: {dataset.height}, Bands: {dataset.count}")

            if dataset.count < 1:
                raise ValueError(f"GeoTIFF file {tiff_path} contains no bands.")

            # Handle CRS
            if str(dataset.crs).startswith("LOCAL_CS"):
                logger.warning("CRS is non-standard (LOCAL_CS). Assuming EPSG:3857 for transformations.")
                dataset_crs = "EPSG:3857"
            else:
                dataset_crs = dataset.crs

            # Read elevation data
            elevation_data = dataset.read(1)
            logger.debug(f"Elevation data shape: {elevation_data.shape}")

            # Handle NoData values (if any)
            nodata_value = dataset.nodata
            if nodata_value is not None:
                mask = elevation_data != nodata_value
            else:
                mask = np.ones_like(elevation_data, dtype=bool)

            # Initialize a transformer for coordinate conversion
            transformer = Transformer.from_crs(dataset_crs, "EPSG:4326", always_xy=True)

            # Get pixel indices
            rows, cols = np.where(mask)
            xs, ys = dataset.xy(rows, cols)
            lons, lats = transformer.transform(xs, ys)

            # Create the geo dictionary
            for i in range(len(rows)):
                pixel_coords = (int(rows[i]), int(cols[i]))
                elevation = float(elevation_data[rows[i], cols[i]])
                geo_dict = {
                    "pixel_coords": pixel_coords,
                    "geo_coords": [float(lons[i]), float(lats[i]), elevation],  # Include elevation in geo_coords
                    "score": 0.0,
                    "label": -1
                }
                geo_dict_list.append(geo_dict)

        logger.info(f"Generated geo dictionary with {len(geo_dict_list)} entries.")
        return geo_dict_list

    except rasterio.errors.RasterioIOError as e:
        logger.error(f"Rasterio I/O error while reading {tiff_path}: {e}")
    except ValueError as e:
        logger.error(f"Value error in {tiff_path}: {e}")
    except Exception as e:
        logger.error(f"Unexpected error in {tiff_path}: {e}")

    return None


# To either load or generate the geo dict
def get_geo_dict(tiff_path):
    """
    Retrieves or generates a geo dictionary from a GeoTIFF file.

    Parameters:
        tiff_path (str): Path to the GeoTIFF file (without `.tif` extension).

    Returns:
        dict: The geo dictionary, either loaded from an existing file or generated anew.
    """
    try:
        # Safely handle paths and extensions
        base_path, _ = os.path.splitext(tiff_path)
        tiff_file = f"{base_path}.tif"
        json_file = f"{base_path}.json"

        # Attempt to load the existing geo dictionary
        logger.info(f"Checking for existing geo dict: {json_file}")
        existing_geo_dict = load_existing_geo_dict(json_file)

        if existing_geo_dict:
            logger.info("Using existing geo dict.")
            return existing_geo_dict

        # Generate a new geo dictionary
        logger.info(f"Generating new geo dict from {tiff_file}")
        print(f"Generating new geo dict from {tiff_file}")
        if not os.path.exists(tiff_file):
            logger.error(f"GeoTIFF file not found: {tiff_file}")
            print(f"GeoTIFF file not found: {tiff_file}")
            return None

        new_geo_dict = generate_geo_dict_from_tiff(tiff_file)
        if not new_geo_dict:
            logger.error(f"Failed to generate geo dict for {tiff_file}.")
            print(f"Failed to generate geo dict for {tiff_file}.")
            return None

        # Save the generated geo dictionary to JSON
        try:
            with open(json_file, 'w') as f:
                json.dump(new_geo_dict, f, indent=4)
                logger.info(f"Geo dict saved to {json_file}")
                print(f"Geo dict saved to {json_file}")
        except IOError as e:
            logger.error(f"Failed to save geo dict to {json_file}: {e}")
            print(f"Failed to save geo dict to {json_file}: {e}")
            return None

        return new_geo_dict

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
    except json.JSONDecodeError as e:
        logger.error(f"JSON decoding error: {e}")
    except IOError as e:
        logger.error(f"I/O error while accessing files for {tiff_path}: {e}")
    except Exception as e:
        logger.error(f"Unexpected error in get_geo_dict for {tiff_path}: {e}")

    return None


########################################################################################################################
def load_image(image_path, mode):
    """
    Load the image and return its pixel values, geo-transform, and CRS.
    Supports multiple modes for different input types.
    """

    """
    Load the image and return its pixel values, geo-transform, and CRS
    """
    data, geo_transform, spatial_ref = None, None, None
    print(f"Loading image from {image_path} with mode {mode}")

    def process_observation(observation_file):
        nonlocal data, geo_transform, spatial_ref
        logger.info(f"Processing observation: {observation_file}")
        dataset__ = gdal.Open(observation_file, gdal.GA_ReadOnly)
        if dataset__ is None:
            raise FileNotFoundError(f"Could not open file {observation_file}")
        band__ = dataset__.GetRasterBand(1)
        data = band__.ReadAsArray()
        geo_transform = dataset__.GetGeoTransform()
        spatial_ref = dataset__.GetProjection()
        return

    if mode == 0:  # Load previous OGM
        # Open the GeoTIFF file
        dataset_ = gdal.Open(image_path, gdal.GA_ReadOnly)
        if dataset_ is None:
            logger.info(f"Could not open file {image_path}")
            raise FileNotFoundError(f"Could not open file {image_path}")

        # Retrieve the first band
        band_ = dataset_.GetRasterBand(1)
        if band_ is None:
            logger.info(f"Could not access the first band of {image_path}")
            raise ValueError(f"Could not access the first band of {image_path}")

        # Read data as an array
        data = band_.ReadAsArray()
        if data is None:
            logger.info(f"Failed to read data from the first band of {image_path}")
            raise ValueError(f"Failed to read data from the first band of {image_path}")

        # Get GeoTransform
        geo_transform = dataset_.GetGeoTransform()
        if geo_transform is None:
            logger.info(f"GeoTransform not found for file {image_path}")
            logger.warning(f"GeoTransform not found for file {image_path}")

        # Get spatial reference
        spatial_ref = dataset_.GetProjection()
        if not spatial_ref:
            logger.warning(f"Spatial reference system not defined for file {image_path}")

    elif mode == 1:  # Load the Observation of geo-referenced segmented drone image TFA-06
        for observation in sorted(os.listdir(image_path), reverse=False):
            if observation.endswith("_Segment.tif"):
                logger.info(f"processing segmented drone images {observation}")
                full_path = os.path.join(image_path, observation)
                process_observation(os.path.join(image_path, observation))
                ##############################################################

                ##############################################################
                # Clean up the old temporary file if it exists
                if os.path.exists(full_path):
                    os.remove(full_path)
                break

    elif mode == 2:  # Load the Observation of geo-referenced object drone image TFA-05
        for observation in sorted(os.listdir(image_path), reverse=False):
            if observation.endswith("_Objects.tif"):
                process_observation(os.path.join(image_path, observation))
                # Clean up the old temporary file if it exists
                full_path = os.path.join(image_path, observation)
                if os.path.exists(full_path):
                    os.remove(full_path)
                break

    elif mode == 3:  # Load the Observation of ROI satellite images TFA-08/09
        roi_polygon = Polygon(global_cache.get('roi', 'No disaster info')[0])
        for observation in sorted(os.listdir(image_path), reverse=False):
            if observation.endswith(".jp2"):
                observation_path = os.path.join(image_path, observation)

                # Step 1: Inspect CRS of the satellite image
                with rasterio.open(observation_path) as src:
                    image_crs = src.crs
                    logger.info(f"Image CRS: {image_crs}")

                # Step 2: Reproject ROI polygon to match the image CRS (if necessary)
                if str(image_crs) != "EPSG:4326":
                    transformer = Transformer.from_crs("EPSG:4326", str(image_crs), always_xy=True)
                    roi_polygon = transform(transformer.transform, roi_polygon)
                    logger.info(f"Reprojected ROI Polygon: {roi_polygon}")

                # Step 3: Crop the satellite image to the ROI
                with rasterio.open(observation_path) as src:
                    out_image, out_transform = mask(src, [roi_polygon], crop=True)
                    out_meta = src.meta.copy()

                    # Update metadata for cropped image
                    out_meta.update({
                        "driver": "GTiff",
                        "height": out_image.shape[1],
                        "width": out_image.shape[2],
                        "transform": out_transform
                    })

                    # Save the cropped image temporarily
                    cropped_image_path = "cropped_sate_roi_image.tif"
                    with rasterio.open(cropped_image_path, "w", **out_meta) as dest:
                        dest.write(out_image)
                    logger.info(f"Cropped image saved to {cropped_image_path}")

                # Step 4: Resample the cropped image to 1 meter resolution
                with rasterio.open(cropped_image_path) as src:
                    # Calculate scale factors for resampling
                    current_resolution = src.res  # Current resolution (x_res, y_res)
                    logger.info(f"Current resolution: {current_resolution}")

                    scale_factor_x = current_resolution[0] / 1  # Desired resolution: 1m/pixel
                    scale_factor_y = current_resolution[1] / 1

                    # New dimensions for upscaled image
                    new_width = int(src.width * scale_factor_x)
                    new_height = int(src.height * scale_factor_y)

                    # Update metadata for the upscaled image
                    upscale_meta = src.meta.copy()
                    upscale_meta.update({
                        "driver": "GTiff",
                        "height": new_height,
                        "width": new_width,
                        "transform": src.transform * rasterio.Affine.scale(1 / scale_factor_x, 1 / scale_factor_y)
                    })

                    # Resample the image to the new resolution
                    upscaled_image = src.read(
                        out_indexes=1,
                        out_shape=(new_height, new_width),
                        resampling=Resampling.bilinear
                    )

                    # Save the upscaled image
                    upscaled_image_path = "cropped_sate_roi_image_1m_resolution.tif"
                    with rasterio.open(upscaled_image_path, "w", **upscale_meta) as dest:
                        dest.write(upscaled_image, 1)
                    logger.info(f"Upscaled cropped image saved to {upscaled_image_path}")

                process_observation(os.path.join(image_path, upscaled_image_path))
                full_path = os.path.join(image_path, observation)
                full_path_ = os.path.join(image_path, upscaled_image_path)
                if os.path.exists(full_path):
                    os.remove(full_path)
                if os.path.exists(full_path_):
                    os.remove(full_path_)
                break

    elif mode == 4:  # Load the Observation of predictive models
        for observation in sorted(os.listdir(image_path), reverse=False):
            if observation.endswith((".tif", ".kmz")):
                process_observation(os.path.join(image_path, observation))
                full_path = os.path.join(image_path, observation)
                if os.path.exists(full_path):
                    os.remove(full_path)
                break

    return data, geo_transform, spatial_ref


def geojson_to_multi_band_geotiff(geojson_file, geotiff_file, pixel_size_lat, pixel_size_lon):
    # Open the GeoJSON file as a datasource
    source_ds = ogr.Open(geojson_file)
    if source_ds is None:
        raise RuntimeError(f"Failed to open GeoJSON file: {geojson_file}")

    source_layer = source_ds.GetLayer()

    # Get the spatial reference and extent (bounding box) of the layer
    spatial_ref = source_layer.GetSpatialRef()

    x_min, x_max, y_min, y_max = source_layer.GetExtent()
    print(f"x_min, x_max, y_min, y_max {x_min, x_max, y_min, y_max}")

    # Calculate raster size in pixels (resolution) for latitude and longitude
    x_res = int((x_max - x_min) / pixel_size_lon)
    y_res = int((y_max - y_min) / pixel_size_lat)
    # Ensure valid raster size
    if x_res <= 0 or y_res <= 0:
        raise ValueError("Raster size (x_res, y_res) must be greater than 0. Please check pixel sizes.")
    print(f"x_res: {x_res}, y_res: {y_res}")

    print(f"GeoTIFF file path: {geotiff_file}")
    # Create the target raster dataset (GeoTIFF) with two bands
    target_ds = gdal.GetDriverByName('GTiff').Create(geotiff_file, x_res, y_res, 2, gdal.GDT_Float32)
    if target_ds is None:
        raise RuntimeError(f"Failed to create GeoTIFF file: {geotiff_file}")

    # Define the geotransform: [top-left x, pixel width (lon), 0, top-left y, 0, -pixel height (lat)]
    geotransform = (x_min, pixel_size_lon, 0, y_max, 0, -pixel_size_lat)
    target_ds.SetGeoTransform(geotransform)

    # Set the projection (spatial reference) of the raster to match the source layer
    target_ds.SetProjection(spatial_ref.ExportToWkt())

    # Validate attributes in the GeoJSON file
    layer_defn = source_layer.GetLayerDefn()
    attribute_names = [layer_defn.GetFieldDefn(i).GetName() for i in range(layer_defn.GetFieldCount())]
    if "label" not in attribute_names or "score" not in attribute_names:
        raise ValueError("GeoJSON file must contain 'label' and 'score' attributes.")

    # Rasterize the 'label' attribute into Band 1
    print(f"Rasterizing 'label' attribute to Band 1")
    gdal.RasterizeLayer(target_ds, [1], source_layer, options=["ATTRIBUTE=label"])

    # Rasterize the 'score' attribute into Band 2
    print(f"Rasterizing 'score' attribute to Band 2")
    gdal.RasterizeLayer(target_ds, [2], source_layer, options=["ATTRIBUTE=score"])

    print(f"Rasterization complete. Output saved as: {geotiff_file}")
    target_ds.FlushCache()  # Ensure all changes are written to the file
    target_ds = None  # Close the dataset to free resources


def get_geotiff_metadata_rasterio(geotiff_path):
    """
    Extract and transform metadata from GeoTIFF files with GDAL.

    Parameters:
        geotiff_path (str): Path to a directory containing GeoTIFF files or a single GeoTIFF file.

    Returns:
        list: Transformed metadata from the GeoTIFF file(s).
        None: If an error occurs or no valid files are found.
    """
    if os.path.isdir(geotiff_path):
        file_paths = [os.path.join(geotiff_path, f) for f in os.listdir(geotiff_path) if f.endswith("_Objects.tif")]
    else:
        file_paths = [geotiff_path] if geotiff_path.endswith("_Objects.tif") else []

    if not file_paths:
        print("No valid '_Objects.tif' files found.")
        return None

    for file_path in sorted(file_paths):
        dataset = gdal.Open(file_path)
        if dataset is None:
            print(f"Unable to open {file_path}")
            continue

        # Extract general metadata
        metadata = dataset.GetMetadata()
        if 'georef_data' not in metadata:
            print(f"'georef_data' field missing in metadata for {file_path}")
            continue

        try:
            data = json.loads(metadata['georef_data'])
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON in 'georef_data' for {file_path}: {e}")
            continue

            # Transform the data
        transformed_data = []
        for index_, item in enumerate(data):
            key = list(item.keys())[0]  # Extract the first key (e.g., "1,114")
            coordinates = item[key]  # Extract geographic coordinates [lon, lat, elevation]
            score = item.get('score', 0.0)  # Extract score, default to 0.0 if missing
            label = item.get('label', -1)  # Extract label, default to -1 if missing

            # Parse pixel coordinates from the key
            try:
                row, col = map(int, key.split(","))
                pixel_coords = (row, col)  # Convert to tuple
            except ValueError:
                logger.warning(f"Invalid pixel coordinate format in key: {key}")
                continue

            # Build the geo_dict for this entry
            geo_dict = {
                "pixel_coords": pixel_coords,
                "geo_coords": [float(coordinates[0]), float(coordinates[1]), float(coordinates[2])],
                # Lon, Lat, Elevation
                "score": score,
                "label": label
            }

            transformed_data.append(geo_dict)

        if transformed_data:
            return transformed_data  # Return the transformed data for the first valid file

    logger.info("No metadata extracted from valid files.")
    return None


# def save_geotiff(output_path_maps_, FileName, data, GTransform, projection):
#     """
#         Save data as a GeoTIFF file
#     """
#     data = np.array(data)
#     driver = gdal.GetDriverByName('GTiff')
#     if not driver:
#         logging.error('GTiff driver is not available.')
#         return
#     height, width = data.shape
#
#     dataset_ogm = driver.Create(os.path.join(output_path_maps_, FileName), width, height, 1, gdal.GDT_Float32)
#     dataset_ogm.SetGeoTransform(GTransform)
#     dataset_ogm.SetProjection(projection)
#
#     # Define the CRS using EPSG code (EPSG:4326 for WGS 84)
#     srs = osr.SpatialReference()
#     srs.ImportFromEPSG(4326)
#     dataset_ogm.SetProjection(srs.ExportToWkt())
#     band_ = dataset_ogm.GetRasterBand(1)
#     band_.WriteArray(data)
#     band_.SetDescription('Estimated OGM')  # Band description
#
#     metadata = {
#         'title': 'OGM',
#         'author': 'GRVC lab, University of Seville',
#         'description': f"Estimated OGM for {global_cache.get('natural_disaster', 'No disaster info')}.",
#         'creationDate': datetime.now().isoformat(),  # Current date and time
#         "XMin": GTransform[0],
#         "XRes": GTransform[1],
#         "YMax": GTransform[3],
#         "YRes": GTransform[5],
#         "spatialReference": 'EPSG:4326 for WGS 84',
#         "file_name": FileName,
#         "bucket": config.BUCKET_NAME
#     }
#     dataset_ogm.SetMetadata(metadata)
#     dataset_ogm.FlushCache()  # Write to disk
#     return metadata
def save_geotiff(output_path_maps_, FileName, data, GTransform, crs_epsg=4326):
    """
    Save data as a GeoTIFF file.

    Args:
        output_path_maps_ (str): Directory to save the GeoTIFF file.
        FileName (str): Name of the GeoTIFF file.
        data (ndarray): 2D array of the data to save.
        GTransform (list): GeoTransform values [XMin, XRes, 0, YMax, 0, YRes].
        crs_epsg (int): EPSG code for the CRS.

    Returns:
        dict: Metadata of the saved GeoTIFF.
    """
    try:
        # Ensure data is a numpy array
        data = np.array(data)
        driver = gdal.GetDriverByName('GTiff')
        if not driver:
            logging.error('GTiff driver is not available.')
            return None

        # Validate GeoTransform
        if len(GTransform) != 6:
            raise ValueError("GTransform must contain exactly six elements.")

        logging.debug(f"Input GTransform: {GTransform}")
        logging.debug(f"Input CRS EPSG: {crs_epsg}")

        # Create the dataset
        height, width = data.shape
        output_file_path = os.path.join(output_path_maps_, FileName)
        dataset_ogm = driver.Create(output_file_path, width, height, 1, gdal.GDT_Float32)
        if not dataset_ogm:
            logging.error("Failed to create GeoTIFF dataset.")
            return None

        dataset_ogm.SetGeoTransform(GTransform)

        # Set CRS
        srs = osr.SpatialReference()
        if crs_epsg:
            srs.ImportFromEPSG(crs_epsg)
            dataset_ogm.SetProjection(srs.ExportToWkt())
        else:
            raise ValueError("CRS EPSG code is not defined.")

        logging.debug(f"CRS set to: EPSG:{crs_epsg}")

        # Write data to the raster band
        band_ = dataset_ogm.GetRasterBand(1)
        band_.WriteArray(data)
        band_.SetDescription('Estimated OGM')

        natural_disaster = global_cache.get('natural_disaster', 'No disaster info')
        # Define metadata
        metadata = {
            'title': 'OGM',
            'author': 'GRVC lab, University of Seville',
            'description': f"Estimated OGM for Fire.",
            'creationDate': datetime.now().isoformat(),  # Current date and time
            "XMin": GTransform[0],
            "XRes": GTransform[1],
            "YMax": GTransform[3],
            "YRes": GTransform[5],
            "spatialReference": f"EPSG:{crs_epsg}",
            "file_name": FileName,
            "coordinates": global_cache.get('roi', 'No disaster info'),
            "bucket": "naples",
            "minio_url": f'https://{config.MINIO_ENDPOINT}/{config.BUCKET_NAME}/occupancy_grid_map{natural_disaster}.tif'

        }
        dataset_ogm.SetMetadata(metadata)

        logging.debug(f"GeoTIFF metadata: {metadata}")

        # Flush cache and close dataset
        dataset_ogm.FlushCache()
        dataset_ogm = None

        logging.info(f"GeoTIFF saved successfully at {output_file_path}")
        return metadata

    except Exception as e:
        logging.error(f"Error saving GeoTIFF: {e}")
        return None


def pixel_to_coordinates(geo_transform, row, col):
    """
    Convert pixel coordinates to geographic coordinates (latitude, longitude).

    Args:
        geo_transform (tuple): GeoTransform of the GeoTIFF image, typically in the form:
            (origin_x, pixel_width, skew_x, origin_y, skew_y, pixel_height).
        row (float): Pixel row (y-coordinate in image).
        col (float): Pixel column (x-coordinate in image).

    Returns:
        tuple: Geographic coordinates (latitude, longitude).

    Raises:
        ValueError: If GeoTransform does not have exactly six elements.
        TypeError: If row or col are not numbers.
    """
    # Validate GeoTransform
    if len(geo_transform) != 6:
        raise ValueError("GeoTransform must be a tuple of six elements: "
                         "(origin_x, pixel_width, skew_x, origin_y, skew_y, pixel_height).")

    if not isinstance(row, (int, float)) or not isinstance(col, (int, float)):
        raise TypeError("Row and column must be integers or floats.")

    # Extract GeoTransform parameters
    origin_x = geo_transform[0]
    pixel_width = geo_transform[1]
    skew_x = geo_transform[2]
    origin_y = geo_transform[3]
    skew_y = geo_transform[4]
    pixel_height = geo_transform[5]

    # Check for valid pixel dimensions
    if pixel_width == 0 or pixel_height == 0:
        raise ValueError("Pixel width and height must be non-zero.")

    # Calculate coordinates (handles skew/rotation)
    x = origin_x + col * pixel_width + row * skew_x
    y = origin_y + col * skew_y + row * pixel_height
    return y, x


def coordinates_to_pixel(geo_transform, x_coord, y_coord):
    """
    Convert geographic coordinates (latitude, longitude) to pixel coordinates (row, column).

    Args:
        geo_transform (tuple): GeoTransform of the GeoTIFF image, typically in the form:
            (origin_x, pixel_width, skew_x, origin_y, skew_y, pixel_height).
        x_coord (float): Longitude (geographic x-coordinate).
        y_coord (float): Latitude (geographic y-coordinate).

    Returns:
        tuple: Pixel coordinates (row, column).

    Raises:
        ValueError: If GeoTransform is not invertible.
    """
    # Validate GeoTransform
    if len(geo_transform) != 6:
        raise ValueError("GeoTransform must be a tuple of six elements.")

    # Extract GeoTransform parameters
    origin_x = geo_transform[0]
    pixel_width = geo_transform[1]
    skew_x = geo_transform[2]
    origin_y = geo_transform[3]
    skew_y = geo_transform[4]
    pixel_height = geo_transform[5]

    # Compute the transformation matrix and its inverse
    transform_matrix = np.array([
        [pixel_width, skew_x],
        [skew_y, pixel_height]
    ])

    try:
        inverse_transform = np.linalg.inv(transform_matrix)
    except np.linalg.LinAlgError:
        raise ValueError("GeoTransform is not invertible. Ensure pixel width/height and skew values are correct.")

    # Compute the pixel coordinates
    geo_diff = np.array([x_coord - origin_x, y_coord - origin_y])
    pixel_coords = np.dot(inverse_transform, geo_diff)

    # Convert to integer pixel indices
    col, row = pixel_coords
    return int(round(row)), int(round(col))


# This one is the function

# def update_occupancy_grid(OGMData, OGM_gt_, observation_data_, observation_gt_):
#     """
#     Update the occupancy grid map using satellite, drone, and geo social-media measurements.
#
#     Parameters:
#         OGMData (np.ndarray): Occupancy grid data.
#         OGM_gt_ (dict): Geotransform for the occupancy grid.
#         observation_data_ (np.ndarray): Observation data (e.g., drone or satellite image).
#         observation_gt_ (dict): Geotransform for the observation.
#
#     Returns:
#         np.ndarray: Updated occupancy grid map.
#     """
#
#     epsilon = 1e-9  # Small value to avoid log(0) or division by zero
#     # Get grid dimensions
#
#     grid_height, grid_width = OGMData.shape
#     obs_height, obs_width = observation_data_.shape
#
#     for y in range(grid_height):
#         for x in range(grid_width):
#             try:
#                 # Get the center pixel coordinates in geospatial terms
#                 y_geo, x_geo = pixel_to_coordinates(OGM_gt_, x , y ) # + 0.5
#
#                 # Convert geospatial coordinates back to pixel coordinates
#                 ogm_pixel = coordinates_to_pixel(OGM_gt_, x_geo, y_geo)
#                 observation_pixel = coordinates_to_pixel(observation_gt_, x_geo, y_geo)
#
#                 # Bounds checking
#                 if not (0 <= ogm_pixel[0] < grid_width and 0 <= ogm_pixel[1] < grid_height):
#                     continue
#                 if not (0 <= observation_pixel[0] < observation_data_.shape[1] and
#                         0 <= observation_pixel[1] < observation_data_.shape[0]):
#                     continue
#
#                 # Retrieve prior probability from OGM
#                 ogm_y_axis, ogm_x_axis = int(ogm_pixel[1]), int(ogm_pixel[0])
#                 prior_prob = OGMData[ogm_y_axis, ogm_x_axis]
#
#                 # Measurement (likelihood) at the observation pixel
#                 obs_y_axis, obs_x_axis = int(observation_pixel[1]), int(observation_pixel[0])
#                 likelihood = observation_data_[obs_y_axis, obs_x_axis]
#
#                 if likelihood > 0:
#                     logger.info(f"likelihood-> {likelihood}")
#
#                 # Compute posterior probability if valid observation
#                 if likelihood > 0:
#                     P_z_given_x = likelihood
#                     P_z_given_not_x = 1 - P_z_given_x
#                     P_not_x = 1 - prior_prob
#                     P_z = P_z_given_x * prior_prob + P_z_given_not_x * P_not_x
#
#                     if P_z > 0:
#                         posterior_prob = (P_z_given_x * prior_prob) / P_z
#                         log_odds_prev = math.log((prior_prob + epsilon) / (1 - prior_prob + epsilon))
#                         log_odds_obs = math.log((posterior_prob + epsilon) / (1 - posterior_prob + epsilon))
#                         log_odds_updated = log_odds_prev + log_odds_obs
#
#                         # Clamp log-odds and convert back to probability
#                         log_odds_clamped = np.clip(log_odds_updated, -10, 10)
#                         OGMData[ogm_y_axis, ogm_x_axis] = 1 - (1 / (1 + np.exp(log_odds_clamped)))
#                 else:
#                     # No valid measurement: retain prior
#                     log_odds_prev = math.log((prior_prob + epsilon) / (1 - prior_prob + epsilon))
#                     OGMData[ogm_y_axis, ogm_x_axis] = 1 - (1 / (1 + np.exp(log_odds_prev)))
#
#                 # Clamp probabilities to [0, 1]
#                 OGMData[ogm_y_axis, ogm_x_axis] = np.clip(OGMData[ogm_y_axis, ogm_x_axis], 0, 1)
#
#             except Exception as e:
#                 # Log error for this cell and continue
#                 logger.debug(f"Error processing cell ({grid_height}, {grid_width}): {e}")
#     return OGMData

def update_occupancy_grid(OGMData, OGM_gt_, observation_data_, observation_gt_):
    """
    Update the occupancy grid map using satellite, drone, and geo-social media measurements.

    Parameters:
        OGMData (np.ndarray): Occupancy grid data.
        OGM_gt_ (dict): Geotransform for the occupancy grid.
        observation_data_ (np.ndarray): Observation data (e.g., drone or satellite image).
        observation_gt_ (dict): Geotransform for the observation.

    Returns:
        np.ndarray: Updated occupancy grid map.
    """
    import numpy as np
    import math
    import logging

    epsilon = 1e-9  # Small value to avoid log(0) or division by zero
    grid_height, grid_width = OGMData.shape

    for y in range(grid_height):
        for x in range(grid_width):
            # Convert grid coordinates to geospatial coordinates
            y_geo, x_geo = pixel_to_coordinates(OGM_gt_, x, y)

            # Map geospatial coordinates back to pixel coordinates
            ogm_pixel = coordinates_to_pixel(OGM_gt_, x_geo, y_geo)
            observation_pixel = coordinates_to_pixel(observation_gt_, x_geo, y_geo)

            # Bounds checking
            if not (0 <= ogm_pixel[0] < grid_width and 0 <= ogm_pixel[1] < grid_height):
                continue
            if not (0 <= observation_pixel[0] < observation_data_.shape[1] and
                    0 <= observation_pixel[1] < observation_data_.shape[0]):
                continue

            ogm_y, ogm_x = int(ogm_pixel[1]), int(ogm_pixel[0])
            prior_prob = OGMData[ogm_y, ogm_x]

            obs_y, obs_x = int(observation_pixel[1]), int(observation_pixel[0])
            likelihood = observation_data_[obs_y, obs_x]
            likelihood = 0.5 + 0.5 * (likelihood - 0.1)

            if 0 < likelihood <= 1:  # Valid probability range
                # Compute log-odds for prior and observation
                log_odds_prior = math.log((prior_prob + epsilon) / (1 - prior_prob + epsilon))
                log_odds_obs = math.log((likelihood + epsilon) / (1 - likelihood + epsilon))

                # decay_factor = 0.95
                log_odds_updated = log_odds_prior + log_odds_obs

                # Update log-odds
                # log_odds_updated = log_odds_prior + log_odds_obs

                # Clamp log-odds to avoid extreme probabilities
                log_odds_clamped = np.clip(log_odds_updated, -5, 5)

                # Convert log-odds back to probability
                updated_prob = 1 - (1 / (1 + np.exp(log_odds_clamped)))

                # Update the OGM cell
                OGMData[ogm_y, ogm_x] = updated_prob

            # Clamp final probabilities to [0, 1] to prevent numerical errors
            OGMData[ogm_y, ogm_x] = np.clip(OGMData[ogm_y, ogm_x], 0, 1)

    return OGMData


# def update_occupancy_grid(OGMData, OGM_gt_, observation_data_, observation_gt_):
#     """
#     Update the occupancy grid map using satellite, drone, and geo social-media measurements,
#     accounting for differences in spatial resolution.
#
#     Parameters:
#         OGMData (np.ndarray): Occupancy grid data.
#         OGM_gt_ (Affine): Geotransform for the occupancy grid.
#         observation_data_ (np.ndarray): Observation data (e.g., drone or satellite image).
#         observation_gt_ (Affine): Geotransform for the observation.
#
#     Returns:
#         np.ndarray: Updated occupancy grid map.
#     """
#
#     epsilon = 1e-9  # Small value to avoid log(0) or division by zero
#     grid_height, grid_width = OGMData.shape
#
#     # Dynamically resample observation data to match OGM resolution and extent
#     resampling_factor_x = OGM_gt_.a / observation_gt_.a
#     resampling_factor_y = OGM_gt_.e / observation_gt_.e
#     resampled_observation = rasterio.warp.reproject(
#         source=observation_data_,
#         destination=np.zeros_like(OGMData, dtype=observation_data_.dtype),
#         src_transform=observation_gt_,
#         src_crs="EPSG:4326",  # Assumed CRS; adapt if different
#         dst_transform=OGM_gt_,
#         dst_crs="EPSG:4326",  # Assumed CRS; adapt if different
#         resampling=Resampling.average
#     )[1]
#
#     # Precompute all geospatial coordinates for OGM
#     x_coords, y_coords = np.meshgrid(np.arange(grid_width), np.arange(grid_height))
#     y_geo, x_geo = rasterio.transform.xy(OGM_gt_, y_coords, x_coords, offset="center")
#     x_geo, y_geo = np.array(x_geo).flatten(), np.array(y_geo).flatten()
#
#     # Map geospatial coordinates to observation pixel space
#     obs_cols, obs_rows = ~OGM_gt_ * (x_geo, y_geo)
#     obs_cols, obs_rows = obs_cols.astype(int), obs_rows.astype(int)
#
#     # Mask for valid observation pixels
#     valid_mask = (
#         (0 <= obs_cols) & (obs_cols < resampled_observation.shape[1]) &
#         (0 <= obs_rows) & (obs_rows < resampled_observation.shape[0])
#     )
#
#     # Process valid pixels only
#     valid_indices = np.where(valid_mask)[0]
#     for idx in valid_indices:
#         y, x = divmod(idx, grid_width)
#         obs_col, obs_row = obs_cols[idx], obs_rows[idx]
#
#         # Retrieve prior probability from OGM
#         prior_prob = OGMData[y, x]
#
#         # Measurement (likelihood) at the observation pixel
#         likelihood = resampled_observation[obs_row, obs_col] / 255.0
#
#         if likelihood > 0:
#             # Compute posterior probability
#             P_z_given_x = likelihood
#             P_z_given_not_x = 1 - P_z_given_x
#             P_not_x = 1 - prior_prob
#             P_z = P_z_given_x * prior_prob + P_z_given_not_x * P_not_x
#
#             if P_z > epsilon:  # Prevent division by zero
#                 posterior_prob = (P_z_given_x * prior_prob) / P_z
#                 log_odds_prev = math.log((prior_prob + epsilon) / (1 - prior_prob + epsilon))
#                 log_odds_obs = math.log((posterior_prob + epsilon) / (1 - posterior_prob + epsilon))
#                 log_odds_updated = log_odds_prev + log_odds_obs
#
#                 # Clamp log-odds and convert back to probability
#                 log_odds_clamped = np.clip(log_odds_updated, -10, 10)
#                 OGMData[y, x] = 1 - (1 / (1 + np.exp(log_odds_clamped)))
#
#         # Clamp probabilities to [0, 1]
#         OGMData[y, x] = np.clip(OGMData[y, x], 0, 1)
#
#     return OGMData


def is_within_fov(ogm_coords, obs_coords, tolerance=0.001):
    """
    Checks if the observation is within the FOV of the OGM based on geographic coordinates.

    Parameters:
        ogm_coords (list): [lon, lat, elev] of the OGM entry.
        obs_coords (list): [lon, lat, elev] of the observation entry.
        tolerance (float): Allowable distance difference to consider within FOV.

    Returns:
        bool: True if within FOV, False otherwise.
    """
    if len(ogm_coords) != 3 or len(obs_coords) != 3:
        return False  # Malformed coordinates

    distance = np.sqrt(
        (ogm_coords[0] - obs_coords[0]) ** 2 +
        (ogm_coords[1] - obs_coords[1]) ** 2
    )
    return distance <= tolerance


def get_observations_in_FOV(georef_ogm, georef_observation):
    """
    Update the occupancy grid using georeferenced dictionaries.

    Parameters:
        georef_ogm (list): List of dictionaries representing the occupancy grid map (OGM).
        georef_observation (list): List of dictionaries representing observations.

    Returns:
        list: Updated occupancy grid with modified "score" and "label" where applicable.
    """
    updated_ogm = []

    for ogm in georef_ogm:
        updated_entry = ogm.copy()  # Copy existing entry

        # Extract pixel key and values
        pixel_key = next((key for key in ogm if key not in ["score", "label"]), None)
        if not pixel_key or pixel_key not in ogm:
            logger.warning(f"Malformed OGM entry: {ogm}")
            continue

        ogm_values = ogm[pixel_key]  # The [lon, lat, elev] values
        prior_probability = ogm.get("score", 0)  # Default to 0 if "score" is missing
        prior_label = ogm.get("label", -1)  # Default to -1 if "label" is missing

        updated_score = prior_probability
        updated_label = prior_label
        observation_found = False

        for obs in georef_observation:
            obs_pixel_key = next((key for key in obs if key not in ["score", "label"]), None)
            if not obs_pixel_key or obs_pixel_key not in obs:
                logger.warning(f"Malformed observation entry: {obs}")
                continue

            obs_values = obs[obs_pixel_key]
            obs_score = obs.get("score", 0)
            obs_label = obs.get("label", -1)

            if is_within_fov(ogm_values, obs_values):
                observation_found = True
                if obs_score > updated_score or (obs_score == updated_score and obs_label > updated_label):
                    updated_score = obs_score
                    updated_label = obs_label

        if observation_found:
            updated_entry["score"] = np.clip(updated_score, 0, 1)
            updated_entry["label"] = updated_label
        else:
            updated_entry["score"] = prior_probability
            updated_entry["label"] = prior_label

        updated_ogm.append(updated_entry)

    return updated_ogm


def predict_ogm():
    return load_image("downloads/", 4)


def process_and_upload_ogm(entity_id, file_path_, bucket_name, metadata):
    """
        Process the occupancy grid, update the entity and upload the OGM to MinIO.
    """

    logger.info(f"inside process_and_upload_ogm entity_id ---> {entity_id}")
    print(f"inside process_and_upload_ogm entity_id ---> {entity_id}")

    object_name = file_path_.split("/")[-1]
    logger.info(f"Object name {object_name}")
    print(f"file path {file_path_}")
    try:
        minio_client.upload_file(bucket_name, object_name, file_path_)
        # minio_client.download_file(bucket_name, object_name,
        #                            f"/home/abdalraheem/Documents/GitHub/Information_Fusion_PDM_tech_05/{object_name}")
        logger.info(f"File '{file_path_}' uploaded to bucket '{bucket_name}' successfully.")
        print(f"File '{file_path_}' uploaded to bucket '{bucket_name}' successfully.")
    except Exception as e:
        logger.error(f"Error uploading file: {e}")
    #############################################
    # List Objects in a Bucket
    #############################################
    # try:
    #     objects = minio_client.list_objects(bucket_name)
    #     print(f"Objects in bucket '{bucket_name}': {objects}")
    # except Exception as e:
    #     print(f"Error listing objects in bucket: {e}")

    ############################################
    try:
        response = update_entity(entity_id, metadata)
        if response:
            logger.info(f"Payload published successfully for entity: {entity_id}")
            print(f"Payload published successfully for entity: {entity_id}")
        else:
            logger.error(f"Failed to publish payload for entity: {entity_id}")
    except Exception as e:
        logger.error(f"Error updating NGSI-LD entity: {e}")


def extract_roi_from_satellite_files(image_path, roi_coords):
    """
    Extract the ROI area from .jp2 or .gpkg satellite files and return data, GeoTransform, and projection.
    Upscales the raster ROI to 1 pixel per meter resolution.

    Args:
        image_path (str): Directory path containing satellite images.
        roi_coords (list): Coordinates of the ROI polygon in EPSG:4326.

    Returns:
        tuple: Three lists:
            - observ_sat_data_ (list): Extracted data (numpy array for .jp2, GeoDataFrame for .gpkg).
            - observ_sat_gt_ (list): GeoTransform for .jp2 files (None for .gpkg).
            - observ_sat_proj (list): Projection (CRS string) for each file.
    """
    # Validate ROI coordinates
    if not isinstance(roi_coords, list) or not roi_coords or not isinstance(roi_coords[0], list):
        logger.error("ROI coordinates must be a nested list of [lon, lat] pairs.")
        raise ValueError("ROI coordinates must be a nested list of [lon, lat] pairs.")

    try:
        roi_polygon = Polygon(roi_coords[0])  # Create Polygon from the outer ring
    except Exception as e:
        logger.error(f"Invalid ROI coordinates: {e}")
        raise ValueError(f"Invalid ROI coordinates: {e}")

    # Validate image path
    if not os.path.isdir(image_path):
        logger.warning(f"Image path does not exist or is not a directory: {image_path}")
        raise FileNotFoundError(f"Image path does not exist or is not a directory: {image_path}")

    # Initialize results lists
    observ_sat_data_ = None
    observ_sat_gt_ = None
    observ_sat_proj = None

    # Search for satellite files
    satellite_files = [f for f in os.listdir(image_path) if f.endswith((".jp2", ".gpkg"))]
    if not satellite_files:
        print("No satellite files (.jp2 or .gpkg) found in the directory.")
        return observ_sat_data_, observ_sat_gt_, observ_sat_proj

    for sat_file in satellite_files:
        logger.info(f"Processing satellite file {sat_file}")
        sat_file_path = os.path.join(image_path, sat_file)
        try:
            if sat_file.endswith(".jp2"):
                with rasterio.open(sat_file_path) as src:
                    image_crs = src.crs

                    # Assign CRS if missing
                    if image_crs is None:
                        image_crs = CRS.from_epsg(32632)

                    transformer = Transformer.from_crs("EPSG:4326", str(image_crs), always_xy=True)
                    roi_polygon_projected = Polygon([
                        transformer.transform(*coord) for coord in roi_polygon.exterior.coords
                    ])

                    raster_bounds_polygon = box(*src.bounds)
                    if not raster_bounds_polygon.intersects(roi_polygon_projected):
                        logger.warning("ROI does not intersect raster bounds. Skipping.")
                        continue

                    try:
                        geojson_roi = [mapping(roi_polygon_projected)]
                        out_image, out_transform = mask(src, geojson_roi, crop=True)

                        if src.nodata is not None:
                            out_image[out_image == src.nodata] = np.nan

                        non_nan_pixels = np.count_nonzero(~np.isnan(out_image))

                        if non_nan_pixels == 0:
                            logger.warning("Cropped satellite area contains only NoData values. Skipping.")
                            continue

                        # Target CRS
                        dst_crs = CRS.from_epsg(4326)
                        dst_transform, dst_width, dst_height = calculate_default_transform(
                            image_crs, dst_crs, out_image.shape[2], out_image.shape[1], *src.bounds
                        )
                        reprojected_image = np.empty((out_image.shape[0], dst_height, dst_width), dtype=out_image.dtype)

                        try:
                            # Explicitly pass the manually assigned CRS (image_crs) to src_crs
                            reproject(
                                source=out_image,
                                destination=reprojected_image,
                                src_transform=out_transform,
                                src_crs=image_crs,  # Use the manually assigned CRS
                                dst_transform=dst_transform,
                                dst_crs=dst_crs,
                                resampling=Resampling.bilinear,
                            )
                        except Exception as e:
                            logger.warning(f"Reprojection failed: {e}")
                            continue

                        logger.info(f"Reprojected image shape: {reprojected_image.shape}")
                        logger.info(f"Reprojected transform: {dst_transform}")
                        logger.info(f"Reprojected CRS: {dst_crs.to_string()}")

                        # Append results
                        observ_sat_data_ = (reprojected_image)
                        observ_sat_gt_ = (dst_transform)
                        observ_sat_proj = (dst_crs.to_string())

                        # Attempt to delete processed file
                        try:
                            os.remove(sat_file_path)
                            print(f"Deleted processed file: {sat_file}")
                        except Exception as e:
                            print(f"Error deleting file {sat_file}: {e}")

                        return observ_sat_data_, observ_sat_gt_, observ_sat_proj
                    except Exception as e:
                        logger.error(f"Error while processing {sat_file}: {e}", exc_info=True)
                        continue

            elif sat_file.endswith(".gpkg"):
                layers = fiona.listlayers(sat_file_path)
                layer_name = layers[0]
                gdf = gpd.read_file(sat_file_path, layer=layer_name)
                aoi_coords = roi_coords
                aoi_gdf = gpd.GeoDataFrame([{"geometry": Polygon(aoi_coords[0])}], crs="EPSG:4326")
                aoi_gdf = aoi_gdf.to_crs(gdf.crs)

                # Step 4: Perform the intersection
                # Validate geometries
                gdf["geometry"] = gdf["geometry"].buffer(0)
                aoi_gdf["geometry"] = aoi_gdf["geometry"].buffer(0)

                # Perform intersection
                intersected_area = gpd.overlay(gdf, aoi_gdf, how="intersection")

                # Step 5: Check results and reproject to EPSG:4326
                if intersected_area.empty:
                    logger.warning("No intersection found between AOI and satellite image.")
                else:
                    intersected_area = intersected_area.to_crs(epsg=4326)
                    # Step 6: Extract data, geotransform, and projection
                    if intersected_area.crs is None or intersected_area.crs != "EPSG:4326":
                        intersected_area = intersected_area.to_crs("EPSG:4326")
                    ##############################################################
                    # Get the bounds of the intersected area
                    minx, miny, maxx, maxy = intersected_area.total_bounds
                    # Calculate the center latitude of the intersected area
                    center_latitude = (miny + maxy) / 2
                    # Constants
                    meters_per_degree_at_equator = 111319  # Approx. meters per degree at the equator
                    # Adjust meters per degree for the center latitude
                    meters_per_degree = meters_per_degree_at_equator * math.cos(math.radians(center_latitude))
                    # Calculate degrees per pixel for 15 meters
                    resolution = 15 / meters_per_degree
                    # Define raster parameters
                    width = int((maxx - minx) / resolution)
                    height = int((maxy - miny) / resolution)
                    ##############################################################
                    if width <= 0 or height <= 0:
                        raise ValueError("Raster width and height must be greater than 0.")
                    # Compute the GeoTransform
                    gt_transform = from_bounds(minx, miny, maxx, maxy, width, height)
                    # Rasterize the geometry
                    geometry = [mapping(geom) for geom in intersected_area.geometry]
                    rasterized = rasterize(
                        [(geom, 1) for geom in geometry],  # Assign a value of 1 to intersected areas
                        out_shape=(height, width),
                        transform=gt_transform,
                        fill=0,
                        dtype="uint8"
                    )
                    # Projection
                    crs = "EPSG:4326"
                    # Output results
                    observ_sat_data_ = rasterized
                    observ_sat_gt_ = gt_transform
                    observ_sat_proj = crs

                # Attempt to delete processed file
                try:
                    os.remove(sat_file_path)
                    print(f"Deleted processed file: {sat_file}")
                except Exception as e:
                    print(f"Error deleting file {sat_file}: {e}")

                return observ_sat_data_, observ_sat_gt_, observ_sat_proj

        except Exception as e:
            print(f"Error processing {sat_file}: {e}")


#####################################################################
# Subscriptions: listing and deletion
#####################################################################
def delete_all_subscriptions():
    subscription_url = f"{config.BROKER_URL}/ngsi-ld/v1/subscriptions"
    headers = {"Accept": "application/ld+json"}

    response = requests.get(subscription_url, headers=headers)
    if response.status_code == 200:
        subscriptions = response.json()
        print(f"Found {len(subscriptions)} subscriptions to delete.")

        for subscription in subscriptions:
            subscription_id = subscription.get("id")
            if subscription_id:
                delete_url = f"{config.BROKER_URL}/ngsi-ld/v1/subscriptions/{subscription_id}"
                delete_response = requests.delete(delete_url)
                if delete_response.status_code == 204:
                    print(f"Subscription {subscription_id} deleted successfully.")
                else:
                    print(f"Failed to delete subscription {subscription_id}.")
                    print(f"Error: {delete_response.text}")
    else:
        print(f"Failed to retrieve subscriptions. Status code: {response.status_code}")
        print(f"Error: {response.text}")


def list_subscriptions():
    subscription_url = f"{config.BROKER_URL}/ngsi-ld/v1/subscriptions"
    headers = {
        "Accept": "application/ld+json"
    }
    response = requests.get(subscription_url, headers=headers)
    if response.status_code == 200:
        subscriptions = response.json()
        print("Active subscriptions:", subscriptions)
        return subscriptions
    else:
        print(f"Failed to retrieve subscriptions. Status code: {response.status_code}")
        print(f"Error: {response.text}")
        return []


def log_active_subscriptions():
    subscription_url = f"{config.BROKER_URL}/ngsi-ld/v1/subscriptions"
    headers = {"Accept": "application/ld+json"}

    response = requests.get(subscription_url, headers=headers)
    if response.status_code == 200:
        subscriptions = response.json()
        for sub in subscriptions:
            print(f"Subscription ID: {sub.get('id')}")
            print(f"Entities: {sub.get('entities')}")
            print(f"Notification Endpoint: {sub['notification']['endpoint']['uri']}")
    else:
        print(f"Failed to retrieve subscriptions. Status code: {response.status_code}")
        print(f"Error: {response.text}")