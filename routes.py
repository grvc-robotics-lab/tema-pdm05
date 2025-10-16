# Standard libraries
import copy
import logging
import tempfile
import threading
from operator import concat
from threading import Lock
import time
import os
import queue
import json
import math
# import shutil
import uuid
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

# Third‑party libraries
import cv2
import numpy as np
import requests
import geopandas as gpd
import rasterio
from rasterio.crs import CRS
from rasterio.transform import Affine
from rasterio.features import rasterize, geometry_mask
from rasterio.transform import from_bounds, rowcol
from rasterio.mask import mask
from rasterio.warp import calculate_default_transform, reproject, Resampling
from requests.auth import HTTPBasicAuth
from scipy.optimize import linear_sum_assignment
from flask import Flask, request, jsonify, Response, current_app, abort
from flask_socketio import SocketIO
from osgeo import gdal, osr, ogr
from pyproj import CRS, Transformer
from shapely.geometry import Polygon, box, shape, mapping
from shapely.ops import transform as shapely_transform

# Local modules
import config
from Kalman_filter_estimating_Objects import KalmanFilter
from georeferencing_module import main
from minio_client import MinIOClient
from logging_config import logger
import pyproj
from affine import Affine
from typing import Union
from rasterio.crs import CRS as RioCRS

# --------------------------------------------------------
# Initialize Flask app and SocketIO
app = Flask(__name__)
socketio = SocketIO(app)
# --------------------------------------------------------
DOWNLOAD_DIR = 'downloads/'
if not os.path.exists(DOWNLOAD_DIR):
    os.makedirs(DOWNLOAD_DIR)
# --------------------------------------------------------
# Initialize MinIO client
minio_client = MinIOClient(
    url=config.MINIO_ENDPOINT,
    access_key=config.MINIO_ACCESS_KEY,
    secret_key=config.MINIO_SECRET_KEY,
    secure=True
)
# --------------------------------------------------------
ND_entity_ID = None
obj_entity_ID = None
polygon_coordinates = None
# --------------------------------------------------------
global_cache = {"processing": False,
                "natural_disaster": "No disaster info",
                "roi": [],
                "expiration": "No info",
                "ignitionPoints": "No info",
                "alert_timestamp": "No info",
                "bm_id": "No info",
                "kf_states": {},
                "alert_received": False,
                "waiting_for_non_alert": False
                }
global_cache_lock = threading.Lock()
# --------------------------------------------------------
alert_event = threading.Event()  # Event triggered by Alert notifications
other_entity_event = threading.Event()  # Event triggered by other notifications
notification_queue = queue.Queue()  # Thread-safe queue for non-Alert notifications
entities_initialized = False  # Tracks whether initialize_entities has run
is_processing = False
processing_lock = Lock()
ogm_flag = False
ogm_counter = 0
# --------------------------------------------------------
# initialize at module level
UPLOAD_INTERVAL = 5 * 60.0
_last_upload = time.monotonic() - UPLOAD_INTERVAL
_upload_lock = threading.Lock()
# --------------------------------------------------------
MAX_WORKERS = min(8, (os.cpu_count() or 4) * 2)

# executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)
# --------------------------------------------------------
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


"""
This is the notify function used in the Dmalian Trail ROLL BACK TO THIS ONE IN CASE OF FAILURE
"""
"""@app.route(f'/{config.BASE_PATH}/{config.API_ENDPOINT}', methods=['POST'])
def notify():
    try:
        notification_data = request.get_json()
        if not isinstance(notification_data, dict):
            logger.error("Invalid notification data: Expected a JSON object.", exc_info=True)
            return jsonify({"error": "Invalid notification data format"}), 400

        data_list = notification_data.get("data")
        if not isinstance(data_list, list):
            logger.error("Invalid notification data: 'data' must be a list.", exc_info=True)
            return jsonify({"error": "Invalid notification data format"}), 400

        for n in data_list:
            if not isinstance(n, dict):
                logger.error(f"Invalid notification: Expected a dictionary, got {type(n)}.", exc_info=True)
                return jsonify({"error": f"Invalid notification format: {n}"}), 400

        # --- Split by type so we can process Alerts first ---
        alerts = [n for n in data_list if n.get("type") == "Alert"]
        others = [n for n in data_list if n.get("type") != "Alert"]

        # --- 1) Process ALERTS FIRST (even if disaster_type not set yet) ---
        alerts_processed = 0
        try:
            if alerts:
                with ThreadPoolExecutor(max_workers=8) as executor:
                    futures = []
                    for notification in alerts:
                        if "type" not in notification:
                            logger.error(f"Missing 'type' in notification: {notification}", exc_info=True)
                            continue
                        # logger.info(f"Processing notification: {notification['type']}")
                        futures.append(executor.submit(process_notification, notification))
                    for f in futures:
                        try:
                            f.result()
                        except Exception as e:
                            logger.error(f"Error in processing Alert notification: {e}", exc_info=True)
                    alerts_processed = len(futures)
        except Exception as e:
            logger.warning(f'Issue in alert entity due to {e}')


        # --- Re-read disaster type after processing Alerts ---
        with global_cache_lock:
            disaster_type = global_cache.get('natural_disaster', 'No disaster info')
        have_alert_gate = (alerts_processed > 0) or (disaster_type in ["Fire", "Flood"])

        # --- 2) ONLY NOW process non-Alert notifications (downloads, etc.) ---
        others_processed = 0
        try:
            if have_alert_gate and disaster_type in ["Fire", "Flood"]:
                with ThreadPoolExecutor(max_workers=8) as executor:
                    futures = []
                    for notification in others:
                        if "type" not in notification:
                            logger.error(f"Missing 'type' in notification: {notification}", exc_info=True)
                            continue
                        futures.append(executor.submit(process_notification, notification))
                    for f in futures:
                        try:
                            f.result()
                        except Exception as e:
                            logger.error(f"Error in processing notification: {e}", exc_info=True)
                    others_processed = len(futures)
            else:
                logger.info(f"Skipping non-Alert notifications "
                            f"(have_alert_gate={have_alert_gate}, disaster_type={disaster_type}).")
        except Exception as e:
            logger.warning(f"Issues in non-Alert entities due to {e}")

        # --- 3) Initialization stays in notify(), but only AFTER an Alert gate ---
        started_init = False
        with global_cache_lock:
            # refresh disaster_type once more in case non-Alerts changed state (unlikely, but safe)
            disaster_type = global_cache.get('natural_disaster', 'No disaster info')
            if have_alert_gate and disaster_type in ["Fire", "Flood"]:
                if not global_cache.get('processing'):
                    global_cache['processing'] = True
                    try:
                        logger.info("Starting initialize_processing() in a new thread (post-Alert gate passed).")
                        processing_thread = threading.Thread(target=initialize_processing, daemon=True)
                        processing_thread.start()
                        started_init = True
                    except Exception as e:
                        logger.error(f"Error in initialize_processing: {e}", exc_info=True)
                        global_cache['processing'] = False
                else:
                    logger.info("initialize_processing() is already running, skipping.")
            else:
                logger.info(f"Skipping initialization (have_alert_gate={have_alert_gate}, "
                            f"disaster_type={disaster_type}).")

        return jsonify({
            "status": "Notifications processed",
            "alerts_processed": alerts_processed,
            "others_processed": others_processed,
            "started_initialization": started_init,
            "disaster_type": disaster_type
        }), 200

    except Exception as e:
        logger.error(f"Unexpected error in notify function: {e}", exc_info=True)
        return jsonify({"error": "Internal server error"}), 500
"""


@app.route(f'/{config.BASE_PATH}/{config.API_ENDPOINT}', methods=['POST'])
def notify():
    try:
        notification_data = request.get_json()
        if not isinstance(notification_data, dict):
            logger.error("Invalid notification data: Expected a JSON object.", exc_info=True)
            return jsonify({"error": "Invalid notification data format"}), 400

        data_list = notification_data.get("data")
        if not isinstance(data_list, list):
            logger.error("Invalid notification data: 'data' must be a list.", exc_info=True)
            return jsonify({"error": "Invalid notification data format"}), 400

        for n in data_list:
            if not isinstance(n, dict):
                logger.error(f"Invalid notification: Expected a dictionary, got {type(n)}.", exc_info=True)
                return jsonify({"error": f"Invalid notification format: {n}"}), 400

        # --- PHASE 1: Separate and prioritize Alert entities ---
        alerts = [n for n in data_list if n.get("type") == "Alert"]
        others = [n for n in data_list if n.get("type") != "Alert"]

        # Track if we have any alerts in this request
        has_alerts_in_this_request = bool(alerts)

        # --- CRITICAL: If we have Alerts, stop current processing and reset ---
        if has_alerts_in_this_request:
            logger.info(f"Alert received - stopping current processing and resetting system")

            # Stop current processing thread
            with global_cache_lock:
                global_cache['processing'] = False
                # Set a flag that alert was received
                global_cache['alert_received'] = True
                # CRITICAL: Set a flag that we're waiting for new non-alert entities
                global_cache['waiting_for_non_alert'] = True

            # Clear events to ensure clean state
            alert_event.clear()
            other_entity_event.clear()

            # Small delay to ensure processing thread stops
            time.sleep(0.5)

        # --- PHASE 2: Process ALL Alert entities FIRST ---
        alerts_processed = 0
        if has_alerts_in_this_request:
            logger.info(f"Processing {len(alerts)} Alert(s) with HIGHEST priority")

            # Process Alerts SYNCHRONOUSLY
            for alert_notification in alerts:
                try:
                    if "type" not in alert_notification:
                        logger.error(f"Missing 'type' in Alert notification: {alert_notification}", exc_info=True)
                        continue

                    logger.info(f"Processing HIGH-PRIORITY Alert: {alert_notification.get('id', 'Unknown ID')}")
                    process_notification(alert_notification)
                    alerts_processed += 1

                except Exception as e:
                    logger.error(f"Error processing Alert notification: {e}", exc_info=True)

            logger.info(f"Completed processing {alerts_processed} Alert(s)")

        # --- PHASE 3: Process non-Alert entities if appropriate ---
        others_processed = 0
        started_init = False

        # Check if we should process non-Alert entities
        with global_cache_lock:
            disaster_type = global_cache.get('natural_disaster', 'No disaster info')
            # Check if alert was received (either in this request or previously)
            alert_was_received = global_cache.get('alert_received', False)
            # Check if we're waiting for new non-alert entities after an alert
            waiting_for_non_alert = global_cache.get('waiting_for_non_alert', False)

        # MODIFIED: Should process others if we have alerts OR disaster type requires it
        # But if we're waiting for new non-alert entities, only process if we actually have non-alert entities
        should_process_others = (alert_was_received or disaster_type in ["Fire", "Flood"])

        if should_process_others and others:
            logger.info(f"Processing {len(others)} non-Alert entities")

            try:
                with ThreadPoolExecutor(max_workers=8) as executor:
                    futures = []
                    for notification in others:
                        if "type" not in notification:
                            logger.error(f"Missing 'type' in notification: {notification}", exc_info=True)
                            continue
                        futures.append(executor.submit(process_notification, notification))

                    for future in as_completed(futures):
                        try:
                            future.result()
                            others_processed += 1
                        except Exception as e:
                            logger.error(f"Error in processing non-Alert notification: {e}", exc_info=True)

            except Exception as e:
                logger.warning(f"Issues in non-Alert entities processing: {e}")

        # --- PHASE 4: CRITICAL FIX - Only start processing when we have non-alert entities AFTER an alert ---
        with global_cache_lock:
            disaster_type = global_cache.get('natural_disaster', 'No disaster info')
            alert_was_received = global_cache.get('alert_received', False)
            current_processing = global_cache.get('processing', False)
            waiting_for_non_alert = global_cache.get('waiting_for_non_alert', False)

            # NEW LOGIC: Only start processing when:
            # 1. We have non-alert entities in this request AND
            # 2. Alert was received (current or previous) AND
            # 3. Valid disaster type AND
            # 4. Processing is not already running AND
            # 5. We're either NOT waiting for non-alert OR we have non-alert entities now
            should_start_processing = (
                    bool(others) and  # Only start if we have non-alert entities to process
                    alert_was_received and
                    disaster_type in ["Fire", "Flood"] and
                    not current_processing and
                    (not waiting_for_non_alert or bool(others))
            # Only start if we're not waiting OR we have non-alert now
            )

            if should_start_processing:
                global_cache['processing'] = True
                # CRITICAL: Clear the waiting flag since we're starting processing
                global_cache['waiting_for_non_alert'] = False
                try:
                    logger.info(
                        "Starting initialize_processing() because we have non-alert entities to process after alert")
                    processing_thread = threading.Thread(target=initialize_processing, daemon=True)
                    processing_thread.start()
                    started_init = True
                    logger.info("New processing thread started successfully")
                except Exception as e:
                    logger.error(f"Error starting initialize_processing: {e}", exc_info=True)
                    global_cache['processing'] = False
            elif current_processing:
                logger.info("Processing already running with existing thread")
            else:
                if waiting_for_non_alert:
                    logger.info(
                        f"Waiting for non-alert entities after alert. Current request has non-alert entities: {bool(others)}")
                else:
                    logger.info(
                        f"No valid conditions to start processing: has_non_alert_entities={bool(others)}, alert_received={alert_was_received}, disaster_type={disaster_type}")

        return jsonify({
            "status": "Notifications processed with Alert prioritization",
            "alerts_processed": alerts_processed,
            "others_processed": others_processed,
            "started_initialization": started_init,
            "disaster_type": disaster_type,
            "alert_received": alert_was_received,
            "waiting_for_non_alert": waiting_for_non_alert,
            "priority_handling": "Alerts trigger system reset and wait for new non-alert entities"
        }), 200

    except Exception as e:
        logger.error(f"Unexpected error in notify function: {e}", exc_info=True)
        return jsonify({"error": "Internal server error"}), 500


# @app.route(f'/{config.BASE_PATH}/{config.API_ENDPOINT}', methods=['POST'])
# def notify():
#     try:
#         notification_data = request.get_json()
#         if not isinstance(notification_data, dict):
#             logger.error("Invalid notification data: Expected a JSON object.", exc_info=True)
#             return jsonify({"error": "Invalid notification data format"}), 400
#
#         data_list = notification_data.get("data")
#         if not isinstance(data_list, list):
#             logger.error("Invalid notification data: 'data' must be a list.", exc_info=True)
#             return jsonify({"error": "Invalid notification data format"}), 400
#
#         for n in data_list:
#             if not isinstance(n, dict):
#                 logger.error(f"Invalid notification: Expected a dictionary, got {type(n)}.", exc_info=True)
#                 return jsonify({"error": f"Invalid notification format: {n}"}), 400
#
#         # --- PHASE 1: Separate and prioritize Alert entities ---
#         alerts = [n for n in data_list if n.get("type") == "Alert"]
#         others = [n for n in data_list if n.get("type") != "Alert"]
#
#         # --- CRITICAL: If we have Alerts, stop current processing and reset ---
#         if alerts:
#             logger.info(f"Alert received - stopping current processing and resetting system")
#
#             # Stop current processing thread
#             with global_cache_lock:
#                 global_cache['processing'] = False
#
#             # Clear events to ensure clean state
#             alert_event.clear()
#             other_entity_event.clear()
#
#             # Small delay to ensure processing thread stops
#             time.sleep(0.5)
#
#         # --- PHASE 2: Process ALL Alert entities FIRST ---
#         alerts_processed = 0
#         if alerts:
#             logger.info(f"Processing {len(alerts)} Alert(s) with HIGHEST priority")
#
#             # Process Alerts SYNCHRONOUSLY
#             for alert_notification in alerts:
#                 try:
#                     if "type" not in alert_notification:
#                         logger.error(f"Missing 'type' in Alert notification: {alert_notification}", exc_info=True)
#                         continue
#
#                     logger.info(f"Processing HIGH-PRIORITY Alert: {alert_notification.get('id', 'Unknown ID')}")
#                     process_notification(alert_notification)
#                     alerts_processed += 1
#
#                 except Exception as e:
#                     logger.error(f"Error processing Alert notification: {e}", exc_info=True)
#
#             logger.info(f"Completed processing {alerts_processed} Alert(s)")
#
#         # --- PHASE 3: Process non-Alert entities if appropriate ---
#         others_processed = 0
#         started_init = False
#
#         logger.info(f"alerts_processed  {alerts_processed}")
#         # Check if we should process non-Alert entities
#         with global_cache_lock:
#             disaster_type = global_cache.get('natural_disaster', 'No disaster info')
#
#         should_process_others = (alerts_processed > 0 or disaster_type in ["Fire", "Flood"])
#
#         if should_process_others and others:
#             logger.info(f"Processing {len(others)} non-Alert entities")
#
#             try:
#                 with ThreadPoolExecutor(max_workers=8) as executor:
#                     futures = []
#                     for notification in others:
#                         if "type" not in notification:
#                             logger.error(f"Missing 'type' in notification: {notification}", exc_info=True)
#                             continue
#                         futures.append(executor.submit(process_notification, notification))
#
#                     for future in as_completed(futures):
#                         try:
#                             future.result()
#                             others_processed += 1
#                         except Exception as e:
#                             logger.error(f"Error in processing non-Alert notification: {e}", exc_info=True)
#
#             except Exception as e:
#                 logger.warning(f"Issues in non-Alert entities processing: {e}")
#
#         # --- PHASE 4: ALWAYS restart processing when we have Alerts ---
#         with global_cache_lock:
#             disaster_type = global_cache.get('natural_disaster', 'No disaster info')
#
#             # CRITICAL FIX: Always restart processing when we have new Alerts
#             if alerts_processed > 0 and disaster_type in ["Fire", "Flood"]:
#                 global_cache['processing'] = True
#                 try:
#                     logger.info("Starting/Restarting initialize_processing() due to new Alert")
#                     processing_thread = threading.Thread(target=initialize_processing, daemon=True)
#                     processing_thread.start()
#                     started_init = True
#                     logger.info("New processing thread started successfully")
#                 except Exception as e:
#                     logger.error(f"Error starting initialize_processing: {e}", exc_info=True)
#                     global_cache['processing'] = False
#             elif global_cache.get('processing'):
#                 logger.info("Processing already running with existing thread")
#             else:
#                 logger.info(
#                     f"No valid conditions to start processing: alerts_processed={alerts_processed}, disaster_type={disaster_type}")
#
#         return jsonify({
#             "status": "Notifications processed with Alert prioritization",
#             "alerts_processed": alerts_processed,
#             "others_processed": others_processed,
#             "started_initialization": started_init,
#             "disaster_type": disaster_type,
#             "priority_handling": "Alerts trigger system reset"
#         }), 200
#
#     except Exception as e:
#         logger.error(f"Unexpected error in notify function: {e}", exc_info=True)
#         return jsonify({"error": "Internal server error"}), 500
###################################################################

def handle_person_vehicle_detection(notification, parameters):
    auth_filename = notification.get('parameters', {}).get('value', {}).get('FileName', None)
    detection = notification.get("detection", {}).get('value', {})

    if not auth_filename:
        logger.warning("No filename found for PersonVehicleDetection.")
        return

    # Ensure filename contains a dot (.)
    if "." not in auth_filename:
        logger.error("Invalid filename format received: %s", auth_filename)
        return

    # Getting disaster info from the global cache
    disaster_info = global_cache.get('natural_disaster', 'No disaster info')
    if disaster_info == 'No disaster info':
        logger.warning("No disaster information found in the cache.")
        return

    ####################################################################
    scaling_factor = float(config.SCALING_FACTOR)
    for bbox in detection.get("boxes", []):
        current_bbox = bbox.get("bbox", [])
        try:
            upscaled_bbox = [int(float(coord) * scaling_factor) for coord in current_bbox]
        except (ValueError, TypeError) as e:
            raise ValueError("One or more bbox coordinates could not be converted to a number.") from e
        bbox["bbox"] = upscaled_bbox
        # logger.info(f'current bbox {current_bbox}')
        # logger.info(f'scaled bbox {upscaled_bbox}')
    ####################################################################
    # TO ADAPT TO AUTH NEW UPLOAD STRUCTURE
    auth_filename = auth_filename.split("/")[-1]
    ####################################################################
    metadata_file_base = auth_filename.split('.')[0]

    detection_file = f"{disaster_info}_{metadata_file_base}_obj.json"
    # logger.info(f"detection_file of PersonVehicle   {detection_file}")

    metadata_file = f"{disaster_info}_{metadata_file_base}_objects_metadata.json"
    # logger.info(f"metadata_file of PersonVehicle   {metadata_file}")

    download_path = None
    # Constructing file paths
    if 'Flood' in metadata_file:
        download_path = os.path.join("downloads", "drone_imgs", 'Flood')
    elif 'Fire' in metadata_file:
        download_path = os.path.join("downloads", "drone_imgs", 'Fire')
    # logger.info(f"download_path   {download_path}")
    os.makedirs(download_path, exist_ok=True)  # Ensure directory exists

    json_file_path_metadata = os.path.join(download_path, metadata_file)
    json_file_path_detection = os.path.join(download_path, detection_file)
    # Writing metadata and detection to JSON files
    with open(json_file_path_metadata, 'w', encoding='utf-8') as json_file:
        json.dump(parameters, json_file, indent=4)

    with open(json_file_path_detection, 'w', encoding='utf-8') as json_file:
        json.dump(detection, json_file, indent=4)

    # logger.info(f"Files written successfully: {json_file_path_metadata}, {json_file_path_detection}")
    try:
        main(disaster_info, "bbox", ground_resolution=1.00)
        # logger.info("Geo-referencing is done correctly for person and vehicles.")
    except Exception as e:
        logger.error(f"Issue in geo-referencing due to {e}")

        return


def handle_segmentation(notification):
    bucket = notification.get("bucket", {}).get('value')
    auth_filename = notification.get('segmentation', {}).get('value')
    disaster = global_cache.get('natural_disaster', 'No disaster info')

    if not auth_filename or not isinstance(auth_filename, dict):
        logger.warning("Invalid or missing 'segmentation' value in notification.")
        return

    mask_id = auth_filename.get('mask_id')
    # logger.info(f"complete path {mask_id}")
    mask_id_temp = mask_id.split('/')[-1]
    file_path = None
    if not mask_id_temp:
        logger.warning("No mask_id found in auth_filename.")
        return

    if 'Flood' in mask_id_temp:
        file_path = f"downloads/drone_imgs/Flood/{mask_id_temp}"
    elif 'Burnt' in mask_id_temp or 'Fire' in mask_id_temp:
        file_path = f"downloads/drone_imgs/Fire/{mask_id_temp}"
    else:
        logger.warning(f"Could not determine file_path for file: {mask_id_temp}")
        return

    os.makedirs(file_path, exist_ok=True)
    metadata_file_base = mask_id_temp.split('.')[0]
    metadata_file = f"{metadata_file_base}_metadata.json"
    json_file_path = None
    if 'Flood' in metadata_file_base:
        json_file_path = f"downloads/drone_imgs/Flood/{metadata_file}"
    elif 'Fire' in metadata_file_base or 'Burnt' in metadata_file_base:
        json_file_path = f"downloads/drone_imgs/Fire/{metadata_file}"

    parameters = notification.get("parameters", {}).get('value', {})
    try:
        with open(json_file_path, 'w') as json_file:
            json_file.write(json.dumps(parameters, indent=4))
    except Exception as e:
        logger.error(f"Error writing segmentation metadata: {e}")

    # try:
    # Synchronous file download simulation
    try:
        downloaded_file_path = minio_client.download_file(bucket, mask_id, file_path)
        if downloaded_file_path:
            # logger.info(f"File downloaded successfully to {downloaded_file_path}")
            try:
                main(disaster, "segmented", ground_resolution=1.00)
            except Exception as e:
                logger.error(f"Error in Geo referencing due to {e}", exc_info=True)
        else:
            logger.error("File download failed.")

    except Exception as e:
        logger.error(f"Error downloading file for mask_id {mask_id}: {e}")
        return


def process_alert(notification, entity_id):
    location = notification.get("area", {}).get("value", {})
    if isinstance(location, dict) and "coordinates" in location:
        try:
            global_cache['roi'] = location['coordinates']
            # logger.info(f"ROI set for entity ID {entity_id}: {global_cache['roi']}")
            # print(f"ROI set for entity ID {entity_id}: {global_cache['roi']}")

        except Exception as e:
            logger.error(f"Error accessing 'coordinates' for entity ID {entity_id}: {e}")
            global_cache['roi'] = [None]
    else:
        global_cache['roi'] = [None]
        logger.warning(f"Unexpected 'location' format for entity ID {entity_id}: {location}")
    #############################################################
    event_ = notification.get("event", {}).get("value", None)
    if event_:
        try:
            global_cache['natural_disaster'] = event_
            # logger.info(f"Natural disaster set for entity ID {entity_id}: {global_cache['natural_disaster']}")
            # print(f"Natural disaster set for entity ID {entity_id}: {global_cache['natural_disaster']}")

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
            # logger.info(f"expiration for entity ID {entity_id}: {global_cache['expiration']}")
        except Exception as e:
            logger.error(f"Error accessing 'expires' value for entity ID {entity_id}: {e}")
            global_cache['expiration'] = "No info"
    #############################################################
    ignitionPoints = notification.get("ignitionPoints", {}).get("value", {})
    if isinstance(ignitionPoints, dict) and "coordinates" in ignitionPoints:
        try:
            global_cache['ignitionPoints'] = ignitionPoints['coordinates']
            # logger.info(f"ignitionPoints set for entity ID {entity_id}: {global_cache['ignitionPoints']}")
            # print(f"ignitionPoints set for entity ID {entity_id}: {global_cache['ignitionPoints']}")

        except Exception as e:
            logger.error(f"Error accessing 'ignitionPoints' for entity ID {entity_id}: {e}")
            global_cache['ignitionPoints'] = "No info"
    else:
        global_cache['ignitionPoints'] = "No info"
        logger.warning(f"Unexpected 'ignitionPoints' format for entity ID {entity_id}: {ignitionPoints}")
    #############################################################
    alert_timestamp = notification.get("sent", {}).get("value")
    if alert_timestamp:
        try:
            global_cache['alert_timestamp'] = alert_timestamp
            # logger.info(f"alert_timestamp for entity ID {entity_id}: {global_cache['alert_timestamp']}")
        except Exception as e:
            logger.error(f"Error accessing 'expires' value for entity ID {entity_id}: {e}")
            global_cache['alert_timestamp'] = "No info"
    ############################################################
    bm_id = notification.get("bm_id", {}).get("value", None)
    if bm_id:
        try:
            global_cache['bm_id'] = bm_id
            # logger.info(f"bm_id: {global_cache['bm_id']}")
            # print(f"bm_id: {global_cache['bm_id']}")

        except Exception as e:
            logger.error(f"Error accessing bm_id value for entity ID {entity_id}: {e}")
            global_cache['bm_id'] = "Unknown event"
    else:
        global_cache['bm_id'] = "Unknown event"
        logger.warning(f"Unexpected bm_id format for entity ID {entity_id}: {event_}")
    #############################################################


def fetch_burnt_area(notification, polygon):
    """
    Fetch burnt-area features from the notification's URL,
    rasterize 'confidence' at 10 m resolution (normalized [0,1]),
    mask to the input polygon, and save a single-band GeoTIFF.
    """
    url = notification.get('data').get('value').get('href')
    if not url:
        logger.info("No valid URL found in the notification.")
        return

    # Try to download via MinIO first, fall back to HTTP if it fails
    file_content = None
    minio_success = False

    # Try MinIO download first
    try:
        bucket_name = url.split('/')[-3]  # Extract bucket name from URL
        file_path = f"{url.split('/')[-2]}/{url.split('/')[-1]}"  # Extract file path

        # Download via MinIO to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.gpkg') as tmp_file:
            temp_path = tmp_file.name

        downloaded_file_path = minio_client.download_file(bucket_name, file_path, temp_path)

        if downloaded_file_path:
            logger.info(f"File downloaded successfully via MinIO to {downloaded_file_path}")
            with open(temp_path, 'rb') as f:
                file_content = f.read()
            minio_success = True
            os.unlink(temp_path)  # Clean up temp file
        else:
            logger.warning("MinIO download returned None, falling back to HTTP")

    except Exception as e:
        logger.warning(f"MinIO download failed: {e}, falling back to HTTP")

    # If MinIO failed, try HTTP download
    if not minio_success:
        try:
            logger.info(f"Downloading via HTTP from {url}...")
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            file_content = resp.content
            logger.info("HTTP download completed successfully")

        except requests.exceptions.RequestException as e:
            logger.error(f"Both MinIO and HTTP downloads failed. MinIO error, HTTP error: {e}")
            return

    # Parse the downloaded content
    try:
        # For GeoPackage files, we need to read them differently
        if url.endswith('.gpkg'):
            # Use the temporary file approach for GeoPackage
            with tempfile.NamedTemporaryFile(delete=False, suffix='.gpkg') as tmp_file:
                tmp_file.write(file_content)
                tmp_path = tmp_file.name

            # Read GeoPackage using geopandas
            gdf = gpd.read_file(tmp_path)
            os.unlink(tmp_path)  # Clean up temp file
        else:
            # Assume it's GeoJSON
            import json
            features_data = json.loads(file_content.decode('utf-8'))
            features = features_data.get("features", [])

            if not features:
                logger.info("No burnt-area features returned.")
                return

            # Load into GeoDataFrame (EPSG:4326)
            gdf = gpd.GeoDataFrame.from_features(features, crs="EPSG:4326")

    except Exception as e:
        logger.error(f"Failed to parse downloaded file: {e}")
        return

    if len(gdf) == 0:
        logger.info("No burnt-area features in the data.")
        return

    # 4. Pick UTM zone from polygon centroid
    # Ensure polygon is a shapely object
    if isinstance(polygon, list):
        from shapely.geometry import Polygon
        polygon = Polygon(polygon[0]) if polygon and isinstance(polygon[0], list) else Polygon(polygon)

    lon, lat = polygon.centroid.x, polygon.centroid.y
    zone = int((lon + 180) / 6) + 1
    epsg = 32600 + zone if lat >= 0 else 32700 + zone

    # 5. Reproject
    gdf_utm = gdf.to_crs(epsg=epsg)
    poly_utm = (
        gpd.GeoSeries([polygon], crs="EPSG:4326")
        .to_crs(epsg=epsg)
        .iloc[0]
    )

    # 6. Compute 10 m grid
    minx, miny, maxx, maxy = poly_utm.bounds
    res = float(config.OGM_ND_RESOLUTION)
    width = int(math.ceil((maxx - minx) / res))
    height = int(math.ceil((maxy - miny) / res))
    transform = Affine(res, 0, minx, 0, -res, maxy)

    # 7. Rasterize raw confidence (0–100)
    shapes = (
        (geom, float(props.get("confidence", 0.0)))
        for geom, props in zip(
        gdf_utm.geometry,
        gdf_utm[["confidence"]].to_dict("records")
    )
    )
    raster = rasterize(
        shapes, out_shape=(height, width),
        transform=transform, fill=0.0, dtype="float32"
    )

    # 7a. Normalize to [0,1]
    raster = raster / 100.0

    # 7b. Mask outside the exact polygon
    mask = geometry_mask(
        [poly_utm], out_shape=(height, width),
        transform=transform, invert=True
    )
    raster[~mask] = 0.0

    # 8. Write GeoTIFF
    output_tif = "downloads/satellite_imgs/Fire/cropped_burnt_confidence_10m_norm.tif"

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_tif), exist_ok=True)

    with rasterio.open(
            output_tif, "w", driver="GTiff",
            height=height, width=width, count=1,
            dtype="float32", crs=f"EPSG:{epsg}",
            transform=transform, nodata=0.0
    ) as dst:
        dst.write(raster, 1)
        dst.set_band_description(1, "normalized_burnt-area_confidence (0–1)")

    logger.info(f"Saved normalized burnt-confidence GeoTIFF to {output_tif}")


# def fetch_burnt_area(notification, polygon):
#     """
#     Fetch burnt-area features from the notification's URL,
#     rasterize 'confidence' at 10 m resolution (normalized [0,1]),
#     mask to the input polygon, and save a single-band GeoTIFF.
#     """
#     url = notification.get('data').get('value').get('href')
#     if not url:
#         logger.info("No valid URL found in the notification.")
#         return
#
#     # 2. Download and parse GeoJSON
#     resp = requests.get(url)
#     resp.raise_for_status()
#     features = resp.json().get("features", [])
#     if not features:
#         logger.info("No burnt-area features returned.")
#         return
#
#     # 3. Load into GeoDataFrame (EPSG:4326)
#     gdf = gpd.GeoDataFrame.from_features(features, crs="EPSG:4326")
#
#     # 4. Pick UTM zone from polygon centroid
#     lon, lat = polygon.centroid.x, polygon.centroid.y
#     zone = int((lon + 180) / 6) + 1
#     epsg = 32600 + zone if lat >= 0 else 32700 + zone
#
#     # 5. Reproject
#     gdf_utm = gdf.to_crs(epsg=epsg)
#     poly_utm = (
#         gpd.GeoSeries([polygon], crs="EPSG:4326")
#         .to_crs(epsg=epsg)
#         .iloc[0]
#     )
#
#     # 6. Compute 10 m grid
#     minx, miny, maxx, maxy = poly_utm.bounds
#     res = float(config.OGM_ND_RESOLUTION)
#     width = int(math.ceil((maxx - minx) / res))
#     height = int(math.ceil((maxy - miny) / res))
#     transform = Affine(res, 0, minx, 0, -res, maxy)
#
#     # 7. Rasterize raw confidence (0–100)
#     shapes = (
#         (geom, float(props.get("confidence", 0.0)))
#         for geom, props in zip(
#         gdf_utm.geometry,
#         gdf_utm[["confidence"]].to_dict("records")
#     )
#     )
#     raster = rasterize(
#         shapes, out_shape=(height, width),
#         transform=transform, fill=0.0, dtype="float32"
#     )
#
#     # 7a. Normalize to [0,1]
#     raster = raster / 100.0
#
#     # 7b. Mask outside the exact polygon
#     mask = geometry_mask(
#         [poly_utm], out_shape=(height, width),
#         transform=transform, invert=True
#     )
#     raster[~mask] = 0.0
#
#     # 8. Write GeoTIFF
#     output_tif = "downloads/satellite_imgs/Fire/cropped_burnt_confidence_10m_norm.tif"
#     with rasterio.open(
#             output_tif, "w", driver="GTiff",
#             height=height, width=width, count=1,
#             dtype="float32", crs=f"EPSG:{epsg}",
#             transform=transform, nodata=0.0
#     ) as dst:
#         dst.write(raster, 1)
#         dst.set_band_description(1, "normalized_burnt-area_confidence (0–1)")
#
#     # logger.info(f"Saved normalized burnt-confidence GeoTIFF to {output_tif}")


# def fetch_burnt_area(notification, polygon):
#     """
#     Fetch burnt area data using the parsed URL and save it as a GeoTIFF.
#     """
#     # Parse URL from notification
#     url = notification.get('data').get('value').get('href')
#     if not url:
#         logger.info("No valid URL found in the notification.")
#         return
#
#     # Convert polygon to bounding box
#     minx, miny, maxx, maxy = polygon.bounds
#     bbox = f"{minx},{miny},{maxx},{maxy}"
#
#     # Fetch data from API
#     response = requests.get(url, stream=True)
#     response.raise_for_status()
#     data = response.json()
#
#     # Extract geometries and convert to GeoDataFrame
#     features = data.get("features", [])
#     if not features:
#         logger.info("No burnt area data found for the given parameters.")
#         return
#
#     gdf = gpd.GeoDataFrame.from_features(features)
#     gdf = gdf.set_crs("EPSG:4326")  # Ensure CRS is set
#
#     # Rasterize the data
#     width, height = 500, 500  # Resolution of output raster
#     transform = from_bounds(minx, miny, maxx, maxy, width, height)
#     raster_data = np.zeros((height, width), dtype=np.uint8)
#
#     for geom in gdf.geometry:
#         coords = [(int((x - minx) / (maxx - minx) * width), int((y - miny) / (maxy - miny) * height)) for x, y in
#                   geom.exterior.z_coords]
#         for x, y in coords:
#             if 0 <= x < width and 0 <= y < height:
#                 raster_data[y, x] = 255  # Mark burnt area
#
#     output_tif = 'downloads/satellite_imgs/Fire/cropped_burnt_area.tif'
#     with rasterio.open(
#             output_tif, "w",
#             driver="GTiff",
#             height=height,
#             width=width,
#             count=1,
#             dtype=raster_data.dtype,
#             crs="EPSG:4326",
#             transform=transform
#     ) as dst:
#         dst.write(raster_data, 1)
#
#     print(f"GeoTIFF saved to {output_tif}")

def download_file(entity_type, filename_, bucket):
    file_path = ""

    if entity_type in ["HotspotResult", "SinglePostResult"]:
        file_path = f"downloads/SocialMedia/{filename_['value']}"
    elif entity_type == "UAVTrajectory":
        file_path = f"downloads/drone_planning/{filename_['value']}"
    elif entity_type == "StandardArrivalTime":
        file_path = f"downloads/FireSim/{filename_['value']}"
    elif entity_type == "FloodCalculationResult":
        file_path = f"downloads/FloodSim/{filename_['value']}"

    try:
        if entity_type == "StandardArrivalTime":
            # logger.info(f"Downloading file '{filename_['value']}' from bucket '{bucket['value']}' to '{file_path}'.")
            return minio_client.download_file(bucket, filename_, file_path)
        else:
            # logger.info(f"Downloading file '{filename_['value']}' from bucket '{bucket['value']}' to '{file_path}'.")
            return minio_client.download_file(bucket['value'], filename_['value'], file_path)
    except Exception as e:
        logger.error(f"Error downloading file {filename_['value']} from bucket {bucket['value']}: {e}")
        return None


###################################################################################################################
# Downloading EOFloodExtent
###################################################################################################################
def download_tif_file(entity, polygon):
    """
    Downloads a .tif file from the entity's data URL if the notification corresponds to "EOFloodExtent".
    """
    minio_success = False
    try:
        file_url = entity.get('data').get('value').get('href')
        bucket_dlr = file_url.split('/')[-3]
        file_name_1 = concat(file_url.split('/')[-2],concat('/',file_url.split('/')[-1]))
        if file_url:
            file_name = file_url.split("/")[-1]
            file_path = os.path.join(f'downloads/satellite_imgs/Flood', file_name)
            try:
                downloaded_file_path = minio_client.download_file(bucket_dlr, file_name_1, file_path)

                if downloaded_file_path:
                    logger.info(f"File downloaded successfully to {downloaded_file_path}")
                    minio_success = True
                else:
                    logger.error("File download failed.")

                file_name = os.path.join(f'downloads/satellite_imgs/Flood', file_name)
                ################################################
                try:
                    input_tif = file_name
                    output_tif = file_name
                    ring = polygon[0] if (isinstance(polygon, list) and polygon and isinstance(polygon[0], list)
                                          and polygon and isinstance(polygon[0][0], list)) else polygon
                    if ring[0] != ring[-1]:
                        ring = ring + [ring[0]]

                    poly_ll = Polygon(ring)
                    if not poly_ll.is_valid:
                        poly_ll = poly_ll.buffer(0)

                    with rasterio.open(input_tif) as src:
                        if src.crs is None:
                            raise ValueError("Raster has no CRS. Cannot reproject polygon.")

                        raster_crs = CRS.from_user_input(src.crs)
                        wgs84 = CRS.from_epsg(4326)

                        if raster_crs != wgs84:
                            to_raster = Transformer.from_crs(wgs84, raster_crs, always_xy=True).transform
                            poly_dst = shapely_transform(to_raster, poly_ll)
                        else:
                            poly_dst = poly_ll

                        raster_extent = box(src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top)
                        if not poly_dst.intersects(raster_extent):
                            raise ValueError(
                                f"Polygon does not intersect raster extent. "
                                f"Raster bounds (in {raster_crs.to_string()}): {src.bounds}"
                            )

                        shapes = [mapping(poly_dst)]
                        out_image, out_transform = mask(src, shapes=shapes, crop=True)

                        out_meta = src.meta.copy()
                        out_meta.update({
                            "driver": "GTiff",
                            "height": out_image.shape[1],
                            "width": out_image.shape[2],
                            "transform": out_transform
                        })

                    with rasterio.open(output_tif, "w", **out_meta) as dest:
                        dest.write(out_image)

                    logger.info(f"Cropping complete. Saved as {output_tif}")
                    return output_tif

                except Exception as e:
                    logger.warning(f"Sat image was not cropped due to: {e}")
                    return input_tif

            except requests.exceptions.RequestException as e:
                logger.error(f"Failed to download file: {e}")
                print(f"Failed to download file: {e}")
                return None
        else:
            logger.error("No valid download URL found in entity.")
            print("No valid download URL found in entity.")
    except Exception as e:
            logger.warning(f"MinIO download failed: {e}, falling back to HTTP")
    ##################################################################################
    # VIA HTTP
    ##################################################################################
    if not minio_success:
        try:
            """
                Downloads a .tif file from the entity's data URL if the notification corresponds to "EOFloodExtent".
            """
            file_url = entity.get('data').get('value').get('href')
            if file_url:
                file_name = file_url.split("/")[-1]
                logger.info(f"Downloading {file_name} from {file_url}...")
                try:
                    response = requests.get(file_url, stream=True)
                    response.raise_for_status()  # Raise an error for bad responses
                    file_name = os.path.join(f'downloads/satellite_imgs/Flood', file_name)
                    with open(file_name, "wb") as file:
                        for chunk in response.iter_content(chunk_size=8192):
                            file.write(chunk)
                    logger.info(f"Download completed: {file_name}")
                    ################################################
                    input_tif = file_name
                    output_tif = file_name
                    polygon = Polygon(polygon[0])
                    geojson_polygon = [json.loads(gpd.GeoSeries([polygon]).to_json())['features'][0]['geometry']]
                    with rasterio.open(input_tif) as src:
                        out_image, out_transform = mask(src, geojson_polygon, crop=True)
                        out_meta = src.meta.copy()
                        out_meta.update({
                            "driver": "GTiff",
                            "height": out_image.shape[1],
                            "width": out_image.shape[2],
                            "transform": out_transform
                        })
                        with rasterio.open(output_tif, "w", **out_meta) as dest:
                            dest.write(out_image)
                    ################################################
                    return file_name

                except requests.exceptions.RequestException as e:
                    logger.error(f"Failed to download file: {e}")
                    return None
            else:
                logger.error("No valid download URL found in entity.")
        except Exception as e:
            logger.error(f"Both MinIO and HTTP downloads failed. MinIO error, HTTP error: {e}")
            return None


###################################################################################################################
def download_hotspots_geojson(notification_data):
    print(f"notification_hotspot_data --------------- {notification_data}")
    bucket_name = notification_data.get("bucket")
    object_name = notification_data.get("file_name") or notification_data.get("filename")

    if not bucket_name or not object_name:
        print("Error: Notification data is missing required fields (bucket, file_name or filename)")
        return

    local_file_path = object_name.split('/')[-1]
    if not local_file_path:
        logger.error("Error: Derived local file path is empty")
        return
    download_dir = "downloads/SocialMedia"
    local_file_path = os.path.join(download_dir, local_file_path)  # Save in downloads folder

    try:
        minio_client.download_file(bucket_name, object_name, local_file_path)
        # logger.info(f"File downloaded successfully: {local_file_path}")
    except Exception as e:
        logger.error(f"Unexpected error occurred: {e}")


def process_notification(notification):
    entity_id = notification.get("id")
    entity_type = notification.get("type")

    if not entity_id or not entity_type:
        logger.error(f"Invalid notification received: {notification}")
        return
    try:
        if entity_type == "Alert":
            try:
                logger.info("Triggering Alert entity processing")
                # print("Triggering Alert entity processing")
                process_alert(notification, entity_id)
                alert_event.set()
            except Exception as e:
                logger.error(f"Error processing Alert entity ID {entity_id}: {e}")

        # Notify initialize_processing for non-Alert entities
        if entity_type != "Alert":
            with global_cache_lock:
                alert_was_received = global_cache.get('alert_received', False)
                if alert_was_received:
                    try:
                        logger.info(f"Setting other_entity_event for non-Alert entity: {entity_type}")
                        other_entity_event.set()
                    except Exception as e:
                        logger.error(f"Error setting other_entity_event for entity ID {entity_id}: {e}")
        #######################################################################################################
        if entity_type == "PersonVehicleDetection":
            try:
                parameters = notification.get("parameters", {}).get('value', {})
                if not parameters:
                    raise ValueError(f"Missing parameters in notification: {notification}")
                handle_person_vehicle_detection(notification, parameters)
            except Exception as e:
                logger.error(f"Error processing PersonVehicleDetection for entity ID {entity_id}: {e}", exc_info=True)
        #######################################################################################################
        elif entity_type in ["FireSegmentation", "FloodSegmentation", "BurntSegmentation"]:
            try:
                handle_segmentation(notification)
            except Exception as e:
                logger.error(f"Error handling segmentation for entity ID {entity_id}: {e}", exc_info=True)
        #######################################################################################################
        elif entity_type == "EOFloodExtent":
            try:
                roi_data = global_cache.get('roi')
                download_tif_file(notification, roi_data)
            except Exception as e:
                logger.error(f"Error handling EOFloodExtent for entity ID {entity_id}: {e}")
        #######################################################################################################
        elif entity_type == "EOBurntArea":
            try:
                # logger.info(f"Processing {entity_type} entity: Downloading .tif file")
                roi_data = global_cache.get('roi')
                polygon = Polygon(roi_data[0])
                fetch_burnt_area(notification, polygon)
            except Exception as e:
                logger.error(f"Error handling EOBurntArea for entity ID {entity_id}: {e}", exc_info=True)
        #######################################################################################################
        elif entity_type == "HotspotResult" or entity_type == "SinglePostResult":
            try:
                filename_ = notification.get('filename')
                bucket = notification.get("bucket")
                # logger.info(f"Processing {entity_type} entity: Downloading .geojson file")
                # print(f"Processing {entity_type} entity: Downloading .geojson file")
                ######################################################################################
                if not (isinstance(filename_, dict) and 'value' in filename_ and
                        isinstance(bucket, dict) and 'value' in bucket):
                    logger.warning("Invalid or missing 'filename' or 'bucket' in notification.")
                elif filename_['value'] and bucket['value']:
                    last_part_split = filename_['value'].split("/")[-1]
                    file_path = f"downloads/SocialMedia/{last_part_split}"
                    try:
                        # logger.info(
                        #     f"Downloading file '{filename_['value']}' from bucket '{bucket['value']}' to '{file_path}'.")
                        downloaded_file_path = minio_client.download_file(bucket['value'], filename_['value'],
                                                                          file_path)
                        if downloaded_file_path:
                            logger.info(f"File downloaded successfully to {downloaded_file_path}")
                        else:
                            logger.error("File download failed.")
                    except Exception as e:
                        logger.error(f"Error in downloading file: {e}", exc_info=True)
                ######################################################################################
            except Exception as e:
                logger.error(f"Error in handling of entity ID {entity_id}: {e}")
        #######################################################################################################
        elif entity_type == "FloodCalculationResult":
            try:
                filename_ = notification.get('filename')
                bucket = notification.get("bucket")
                # logger.info(f"Processing {entity_type} entity: Downloading .geotif file")
                ######################################################################################
                if not (isinstance(filename_, dict) and 'value' in filename_ and
                        isinstance(bucket, dict) and 'value' in bucket):
                    logger.warning("Invalid or missing 'filename' or 'bucket' in notification.")
                elif filename_['value'] and bucket['value']:
                    file_path = f"downloads/FloodSim/{filename_['value']}"
                    try:
                        # logger.info(
                        #     f"Downloading file '{filename_['value']}' from bucket '{bucket['value']}' to '{file_path}'.")
                        downloaded_file_path = minio_client.download_file(bucket['value'], filename_['value'],
                                                                          file_path)
                        if downloaded_file_path:
                            logger.info(f"File downloaded successfully to {downloaded_file_path}")
                        else:
                            logger.error("File download failed.")
                    except Exception as e:
                        logger.error(f"Error in downloading file: {e}", exc_info=True)
                ######################################################################################
            except Exception as e:
                logger.error(f"Error in handling of entity ID {entity_id}: {e}")
        #######################################################################################################
        elif entity_type == "StandardArrivalTime":
            try:
                filename_ = notification.get('file_name')
                bucket = notification.get("bucket")
                # logger.info(f"Processing {entity_type} entity: Downloading .geotif file")
                ######################################################################################
                if not (isinstance(filename_, dict) and 'value' in filename_ and
                        isinstance(bucket, dict) and 'value' in bucket):
                    logger.warning("Invalid or missing 'filename' or 'bucket' in notification.")
                elif filename_['value'] and bucket['value']:
                    try:
                        last_part_split = filename_['value'].split("/")[-1]
                        file_path = f"downloads/FireSim/{last_part_split}"
                        # logger.info(
                        #     f"Downloading file '{filename_['value']}' from bucket '{bucket['value']}' to '{file_path}'.")
                        downloaded_file_path = minio_client.download_file(bucket['value'], filename_['value'],
                                                                          file_path)
                        if downloaded_file_path:
                            logger.info(f"File downloaded successfully to {downloaded_file_path}")
                        else:
                            logger.error("File download failed.")
                    except Exception as e:
                        logger.error(f"Error in downloading file: {e}", exc_info=True)
                ######################################################################################
            except Exception as e:
                logger.error(f"Error in handling of entity ID {entity_id}: {e}")
        #######################################################################################################
    except Exception as e:
        logger.error(f"Error processing notification {entity_id} of type {entity_type}: {e}")


########################################################################################################################
def download_opentopography_dem(api_key, output_file, coordinates, dem_dataset):
    """
    Download a DEM GeoTIFF for the given polygon coordinates from OpenTopography.

    Parameters:
        api_key (str): OpenTopography API key.
        output_file (str): Path to save the DEM file.
        coordinates (list): List of (longitude, latitude) tuples defining the polygon.
        dem_dataset (str): Dataset to use (e.g., "SRTMGL1" for 30m DEM, "SRTMGL3" for 90m DEM).
    """
    if isinstance(coordinates[0][0], (list, tuple)):
        coords = coordinates[0]
    else:
        coords = coordinates

    # Create a bounding box from the coordinates
    polygon = Polygon(coords)
    minx, miny, maxx, maxy = polygon.bounds
    #######################
    if dem_dataset in ("SRTMGL1", "SRTMGL3") and (miny < -60 or maxy > 60):
        print("SRTM only covers ±60°. Switching to EU_DTM.")
        dem_dataset = "COP30"
    #######################
    api_url = (
        "https://portal.opentopography.org/API/globaldem"
        f"?demtype={dem_dataset}"
        f"&south={miny}&north={maxy}&west={minx}&east={maxx}"
        "&outputFormat=GTiff"
    )
    # Request parameters
    params = {"API_Key": api_key}

    # Submit the request to OpenTopography
    try:
        print(f"Submitting request to OpenTopography for DEM ({dem_dataset})...")
        response = requests.get(api_url, params=params, stream=True)
        ##############################################
        print("HTTP status:", response.status_code)
        print("Content-Type:", response.headers.get("Content-Type"))
        data = response.content
        with open(output_file, "wb") as f:
            f.write(data)
        ##############################################
        logger.info(f"DEM downloaded and saved to {output_file}")
    except requests.RequestException as e:
        logger.info(f"Failed to download DEM: {e}")


########################################################################################################################
def delete_files_in_directory(dir_path, skip_exts=None):
    """
    Deletes all files in `dir_path` except those whose extensions
    appear in `skip_exts`.
    """
    if skip_exts is None:
        skip_exts = []
    if os.path.isdir(dir_path):
        for file in os.listdir(dir_path):
            # Skip any file whose extension is in skip_exts
            if any(file.endswith(ext) for ext in skip_exts):
                continue

            file_path = os.path.join(dir_path, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
                # logger.info(f"Deleted file: {file_path}")


def initialize_processing():
    global entities_initialized, ogm_flag, polygon_coordinates, ogm_counter

    last_processed_alert_ts = None
    last_processed_alert_bm_id = None

    logger.debug("initialize_processing ➜ thread started")
    try:
        while True:
            with global_cache_lock:
                if not global_cache.get('processing', False):
                    logger.info("Processing thread stopping as requested")
                    break

            with global_cache_lock:
                waiting_for_non_alert = global_cache.get('waiting_for_non_alert', False)

            if waiting_for_non_alert:
                logger.info("Processing thread detected waiting_for_non_alert flag - stopping")
                break

            expiration = global_cache.get('expiration')
            if expiration == "No info":
                time.sleep(0.5)
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

            event_occurred = alert_event.wait(timeout=0.5) or other_entity_event.wait(timeout=0.5)

            if not event_occurred:
                continue

            # logger.info(f"Events detected - alert_event: {alert_event.is_set()}, other_entity_event: {other_entity_event.is_set()}")

            try:
                if alert_event.is_set():
                    current_alert_ts = global_cache.get('alert_timestamp', "No info")
                    current_alert_bm_id = global_cache.get('bm_id', "No info")

                    # Check if this is a new alert
                    if current_alert_ts != last_processed_alert_ts or current_alert_bm_id != last_processed_alert_bm_id:
                        last_processed_alert_ts = current_alert_ts
                        last_processed_alert_bm_id = current_alert_bm_id
                        ogm_flag = False
                        logger.info("New alert received. Resetting initialization process.")

                        alert_event.clear()
                        other_entity_event.clear()

                        time.sleep(0.5)

                        # Clean up directories and logs.
                        dirs_to_cleanup = [
                            "estimated_OGM",
                            "downloads/drone_imgs/Fire",
                            "downloads/drone_imgs/Flood",
                            "georeferenced_drone_images/segmented/burnt",
                            "georeferenced_drone_images/segmented/fire",
                            "georeferenced_drone_images/segmented/flood",
                            "georeferenced_drone_images/detection",
                            "downloads/drone_planning",
                            "downloads/FloodSim",
                            "downloads/FireSim",
                            "downloads/satellite_imgs/Fire",
                            "downloads/satellite_imgs/Flood",
                            "downloads/SocialMedia"
                        ]
                        for path in dirs_to_cleanup:
                            if os.path.exists(path):
                                try:
                                    if "drone_imgs" in path:
                                        delete_files_in_directory(path, skip_exts=[".tif"])
                                    else:
                                        delete_files_in_directory(path)
                                except Exception as e:
                                    pass

                        log_file = 'app_routes.log'
                        if os.path.exists(log_file):
                            try:
                                os.remove(log_file)

                                # Remove existing handlers
                                for handler in logger.handlers[:]:
                                    logger.removeHandler(handler)

                                # Recreate handler
                                handler = logging.FileHandler(log_file, mode='w')
                                formatter = logging.Formatter(
                                    '%(asctime)s - %(levelname)s - %(message)s',
                                    datefmt='%Y-%m-%d %H:%M:%S'
                                )
                                handler.setFormatter(formatter)
                                logger.addHandler(handler)

                                logger.info(f"Deleted and recreated {log_file}")
                            except Exception as e:
                                logger.warning(f"Could not delete/recreate log file: {e}")

                        initialize_entities()
                        entities_initialized = True

                        if not ogm_flag:
                            try:

                                polygon_coordinates = convert_to_polygon()
                                disaster_type = global_cache.get('natural_disaster', 'No disaster info')
                                ogm_path_ND = f"estimated_OGM/occupancy_grid_map_{disaster_type}.tif"

                                if disaster_type in ["Fire", "Flood"]:
                                    os.makedirs(f"downloads/drone_imgs/{disaster_type}", exist_ok=True)
                                    os.makedirs("estimated_OGM", exist_ok=True)

                                    output_dem_file = f"downloads/drone_imgs/{disaster_type}/subset_dem.tif"
                                    try:
                                        download_opentopography_dem(
                                            api_key=config.OpenTopography_api_key,
                                            output_file=output_dem_file,
                                            coordinates=polygon_coordinates,
                                            # dem_dataset="COP30"
                                            # dem_dataset="EU_DTM"
                                            dem_dataset="SRTMGL1"
                                        )
                                    except Exception as e:
                                        logger.warning(f"DEM download fail :{e}")
                                        output_dem_file =None

                                    try:
                                        get_roi(polygon_coordinates, ogm_path_ND, int(config.OGM_ND_RESOLUTION), 4326, 1,
                                                output_dem_file)
                                        result = mask_geotiff_with_polygon_exact(ogm_path_ND,
                                                                                 global_cache.get('roi'),
                                                                                 ogm_path_ND)
                                        if result:
                                            print(f"Masked GeoTIFF saved to {result}")
                                        else:
                                            print("Error occurred while masking the GeoTIFF.")
                                    except Exception as e:
                                        logger.warning(f"Error creating OGM: {e}")

                                    ogm_path_obj = f"estimated_OGM/occupancy_grid_map_{disaster_type}_Objects.tif"

                                    try:
                                        get_roi(polygon_coordinates, ogm_path_obj, int(config.OGM_OBJ_RESOLUTION), 4326, 2,
                                                output_dem_file)
                                        result = mask_geotiff_with_polygon_exact(ogm_path_obj, global_cache.get('roi'),
                                                                                 ogm_path_obj)
                                        if result:
                                            print(f"Masked GeoTIFF saved to {result}")
                                        else:
                                            print("Error occurred while masking the GeoTIFF.")
                                    except Exception as e:
                                        logger.warning(f"Error creating Objects OGM: {e}")

                                    try:
                                        get_geo_dict(f"estimated_OGM/occupancy_grid_map_{disaster_type}_Objects.tif")
                                        logger.info(f"OGM counter --> {ogm_counter}")
                                        ogm_counter += 1
                                        ogm_flag = True
                                    except Exception as e:
                                        logger.warning(f"Error generating geo dictionary: {e}")
                                else:
                                    logger.info(f"Skipping DEM download for disaster type {disaster_type}")
                            except Exception as e:
                                logger.warning(f"Error in OGM initialization: {e}")

                alert_event.clear()
                    # Do not clear the alert_event here if you want it to remain active
                    # alert_event.clear()  <-- Remove this if you rely solely on the timestamp.
                    # alert_event.clear()  <-- Remove this if you rely solely on the timestamp.
                ############################################################################################################
                # Process Non‑Alert Notifications only if no new alert has arrived.
                ############################################################################################################

                if other_entity_event.is_set() and entities_initialized:
                    current_alert_ts = global_cache.get('alert_timestamp', "No info")
                    if current_alert_ts != last_processed_alert_ts:
                        logger.info("A new alert arrived. Skipping non-alert processing this iteration.")
                        other_entity_event.clear()
                    else:
                        # while True:
                        with ThreadPoolExecutor(max_workers=8) as executor:
                            future_nd = executor.submit(estimate_nd_status)
                            future_obj = executor.submit(estimate_objects_status)
                            for future in as_completed([future_nd, future_obj]):
                                try:
                                    future.result()  # Wait for the function to complete.
                                except Exception as e:
                                    if future == future_nd:
                                        logger.info(f"No OGM for ND due to {e}")
                                    else:
                                        logger.info(f"No OGM for objects due to {e}")
            except Exception as e:
                logger.error(f"Error during initialize_processing: {e}")
    except Exception as e:
        logger.error(f"Unexpected error in initialize_processing: {e}")
    finally:
        # CRITICAL: Always reset the processing flag when thread exits
        with global_cache_lock:
            global_cache['processing'] = False
        logger.info("Processing thread EXITED and flag reset")


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
        "Alert": "urn:ngsi-ld:tema:subscription:USE:PDM05:001",  # Tested and able to be notified
        "FloodCalculationResult": "urn:ngsi-ld:tema:subscription:USE:PDM05:002",
        "StandardArrivalTime": "urn:ngsi-ld:tema:subscription:USE:PDM05:003",
        "FireSegmentation": "urn:ngsi-ld:tema:subscription:USE:PDM05:004",  # Tested and able to download
        "BurntSegmentation": "urn:ngsi-ld:tema:subscription:USE:PDM05:005",  # Tested and able to download
        "FloodSegmentation": "urn:ngsi-ld:tema:subscription:USE:PDM05:006",  # Tested and able to download
        "PersonVehicleDetection": "urn:ngsi-ld:tema:subscription:USE:PDM05:007",  # Tested and able to download
        "HotspotResult": "urn:ngsi-ld:tema:subscription:USE:PDM05:008",  # Tested and able to download
        "SinglePostResult": "urn:ngsi-ld:tema:subscription:USE:PDM05:009",  # Tested and able to download
        "EOBurntArea": "urn:ngsi-ld:tema:subscription:USE:PDM05:010",
        # # Tested and able to download (href) cropped tif
        "EOFloodExtent": "urn:ngsi-ld:tema:subscription:USE:PDM05:011",  # Tested and able to download (href)
    }

    try:
        response = requests.get(subscription_url, headers=headers)
        if response.status_code != 200:
            logger.error(f"Error fetching subscriptions. Status: {response.status_code}, Body: {response.text}")
            return

        existing_subscriptions = response.json()
        existing_subscription_ids = {sub.get("id") for sub in existing_subscriptions}

        for entity_type, subscription_id in entity_types_with_ids.items():
            if subscription_id in existing_subscription_ids:
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
            # logger.info(f"Subscription {payload['id']} created successfully.")
        else:
            logger.error(
                f"Failed to create subscription {payload['id']}. Status: {response.status_code}, Body: {response.text}")
    except Exception as e:
        logger.exception("An error occurred while creating the subscription.")
        # print(f"An error occurred while creating the subscription: {e}")


##########################################################################
# Fusing Flood Simulation Results on the OGM
##########################################################################
EPS = 1e-6


def _depth_to_prob(depth, mapping="logistic", params=None):
    """
    Convert water depth (same units as input) to probability in [0,1].
    - 'logistic': p = 1 / (1 + exp(-(d - h50)/s))
        params: {'h50': 0.15, 's': 0.05}  # 50% at 15 cm, slope ~5 cm
    - 'piecewise': 0 below h0, 1 above h1, linear in between
        params: {'h0': 0.05, 'h1': 0.30}  # 5 cm->0, 30 cm->1
    """
    if params is None:
        params = {}
    d = depth.astype(np.float32)

    if mapping == "logistic":
        h50 = float(params.get("h50", 0.15))
        s = float(params.get("s", 0.05))
        # Avoid overflows; clamp to reasonable range
        z = np.clip(-(d - h50) / max(s, 1e-6), -60, 60)
        p = 1.0 / (1.0 + np.exp(z))
    elif mapping == "piecewise":
        h0 = float(params.get("h0", 0.05))
        h1 = float(params.get("h1", 0.30))
        p = (d - h0) / max(h1 - h0, 1e-6)
        p = np.clip(p, 0.0, 1.0)
    else:
        raise ValueError(f"Unknown mapping: {mapping}")
    return p


def _linear_pool(p_prior, p_evidence, alpha=0.5):
    # Convex combination in probability space
    return (1 - alpha) * p_prior + alpha * p_evidence


def _logit(x):
    x = np.clip(x, EPS, 1 - EPS)
    return np.log(x / (1 - x))


def _inv_logit(z):
    # numerically stable sigmoid
    z = np.clip(z, -60, 60)
    return 1.0 / (1.0 + np.exp(-z))


def _logit_pool(p_prior, p_evidence, alpha=0.5):
    # Pool in log-odds space (often better calibrated than linear pool)
    return _inv_logit((1 - alpha) * _logit(p_prior) + alpha * _logit(p_evidence))


def fuse_ogm_with_depth(
        ogm_file: str,
        depth_file: str,
        mapping_: str = "logistic",
        mapping_params: dict | None = None,
        fusion: str = "logit_pool",
        alpha: float = 0.5,
        resampling: Resampling = Resampling.bilinear,
):
    """
    Fuse a water-depth simulation raster into an occupancy grid map (probabilities).

    Parameters
    ----------
    ogm_file : str
        Path to the OGM GeoTIFF (destination grid/CRS). Overwritten in place.
    depth_file : str
        Path to the water-depth GeoTIFF (source grid/CRS).
    mapping_ : {'logistic', 'piecewise'}, optional
        How to convert depth to probability. Default is 'logistic'.
    mapping_params : dict, optional
        Parameters for the mapping. For 'logistic': {'h50': float, 's': float};
        for 'piecewise': {'h0': float, 'h1': float}.
    fusion : {'logit_pool', 'linear_pool'}, optional
        Pooling rule to combine OGM with depth-derived probability. Default 'logit_pool'.
    alpha : float, optional
        Weight of the depth evidence in [0, 1]. Default 0.5.
    resampling : rasterio.enums.Resampling, optional
        Resampling method when reprojecting the depth raster. Default Resampling.bilinear.

    Returns
    -------
    str
        Path to the overwritten OGM file.

    Raises
    ------
    ValueError
        If there are no overlapping valid pixels between the rasters.
    """

    # --- Read OGM (destination grid/CRS) ---
    with rasterio.open(ogm_file) as ogm_ds:
        ogm_profile = ogm_ds.profile.copy()
        ogm = ogm_ds.read(1).astype(np.float32)
        dst_transform = ogm_ds.transform
        dst_crs = ogm_ds.crs
        H, W = ogm_ds.height, ogm_ds.width
        ogm_nodata = ogm_ds.nodata

    if ogm_nodata is None:
        # Keep NaNs as the "nodata"
        ogm_nodata = np.float32(np.nan)
        ogm_profile.update(nodata=ogm_nodata)

    # --- Read depth (source) ---
    with rasterio.open(depth_file) as src_ds:
        depth = src_ds.read(1).astype(np.float32)
        src_transform = src_ds.transform
        src_crs = src_ds.crs
        src_nodata = src_ds.nodata

    # --- Reproject depth -> OGM grid ---
    depth_on_ogm = np.full((H, W), np.nan, dtype=np.float32)
    reproject(
        source=depth,
        destination=depth_on_ogm,
        src_transform=src_transform,
        src_crs=src_crs,
        src_nodata=src_nodata,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        dst_nodata=np.nan,
        resampling=resampling,
    )

    # Valid where we have a finite depth AND OGM is not nodata
    valid_depth = np.isfinite(depth_on_ogm)
    if np.isnan(ogm_nodata):
        valid_ogm = ~np.isnan(ogm)
    else:
        valid_ogm = ogm != ogm_nodata

    valid = valid_depth & valid_ogm
    if not np.any(valid):
        raise ValueError("No overlapping valid pixels between OGM and depth raster.")

    # --- Map depth -> probability ---
    p_from_depth = np.zeros_like(depth_on_ogm, dtype=np.float32)
    p_from_depth[valid_depth] = _depth_to_prob(
        depth_on_ogm[valid_depth], mapping=mapping_, params=mapping_params
    )
    # Clip to [0,1]
    p_from_depth = np.clip(p_from_depth, 0.0, 1.0)

    # --- Fuse with OGM probability ---
    fused = ogm.copy()
    if fusion == "linear_pool":
        fused[valid] = _linear_pool(ogm[valid], p_from_depth[valid], alpha=alpha)
    elif fusion == "logit_pool":
        fused[valid] = _logit_pool(ogm[valid], p_from_depth[valid], alpha=alpha)
    else:
        raise ValueError(f"Unknown fusion rule: {fusion}")

    # Preserve OGM nodata outside valid area
    if np.isnan(ogm_nodata):
        fused[~valid_ogm] = np.nan
    else:
        fused[~valid_ogm] = ogm_nodata

    # --- Write back to the SAME OGM file ---
    ogm_profile.update(dtype=rasterio.float32, count=1)
    with rasterio.open(ogm_file, "w", **ogm_profile) as dst_ds:
        dst_ds.write(fused.astype(np.float32), 1)


##########################################################################

def predict_ogm_flood(ogm_file, pred_file):
    # Open the occupancy grid map (OGM) which is our target grid.
    with rasterio.open(ogm_file) as ogm_ds:
        ogm_profile = ogm_ds.profile
        ogm_data = ogm_ds.read(1)
        dst_transform = ogm_ds.transform
        dst_crs = ogm_ds.crs

    # Open the flood prediction raster.
    with rasterio.open(pred_file) as pred_ds:
        pred_data = pred_ds.read(1)
        src_transform = pred_ds.transform
        src_crs = pred_ds.crs
        pred_nodata = pred_ds.nodata

    # Prepare an array to hold the reprojected flood data with the same shape as the OGM.
    reprojected_pred = np.empty(shape=(ogm_profile['height'], ogm_profile['width']), dtype=np.float32)

    # Reproject the flood data from its CRS (UTM) to the OGM's CRS (EPSG:4326)
    reproject(
        source=pred_data,
        destination=reprojected_pred,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        resampling=Resampling.bilinear
    )

    reprojected_pred = reprojected_pred - ogm_data
    # Create a mask for valid flood data (exclude the NoData values).
    if pred_nodata is not None:
        valid_mask = (reprojected_pred != pred_nodata) & np.isfinite(reprojected_pred)
    else:
        valid_mask = np.isfinite(reprojected_pred)

    # Compute min and max from valid flood data for normalization.
    if np.any(valid_mask):
        pred_min = np.min(reprojected_pred[valid_mask])
        pred_max = np.max(reprojected_pred[valid_mask])
    else:
        raise ValueError("No valid flood data found for normalization.")

    # Normalize the reprojected flood data to the [0, 1] range.

    normalized_pred = np.zeros_like(reprojected_pred, dtype=np.float32)
    normalized_pred[valid_mask] = (reprojected_pred[valid_mask] - pred_min) / (pred_max - pred_min)

    # Invert the normalized flood data:
    # Now the highest flood value becomes 0 and the lowest becomes 1.
    updated_ogm = normalized_pred

    # logger.info(f"updates ogm prediction values  {updated_ogm}")
    # Update the profile to reflect the new data type.
    ogm_profile.update(dtype=rasterio.float32, count=1)

    # Write out the updated occupancy grid map.
    with rasterio.open(ogm_file, "w", **ogm_profile) as dst_ds:
        dst_ds.write(updated_ogm.astype(np.float32), 1)

    logger.info(f"Updated occupancy grid map written to: {ogm_file}")


def predict_ogm_fire(occupancy_path, threshold_path, output_path, method="simple", re_normalize=True):
    """
    Accumulate data from the threshold TIFF into the occupancy TIFF.

    Args:
        occupancy_path (str): Path to the base TIFF file (values between 0 and 1).
        threshold_path (str): Path to the TIFF file whose data will be accumulated (may have values > 1).
        output_path (str): Path where the output TIFF file will be saved.
        method (str): Method to combine data.
                      "simple" adds the images, while "probabilistic" uses:
                        p_combined = 1 - (1 - p1) * (1 - p2)
                      Default is "simple".
        re_normalize (bool): Applicable only for the "simple" method.
                             If True, the combined data is re-scaled to the [0, 1] range;
                             otherwise, values are clipped at 1.

    Returns:
        None. The function writes the output to the given file path.
    """
    # --- Step 1: Open and normalize the threshold (second) TIFF file ---
    with rasterio.open(threshold_path) as src2:
        data2 = src2.read(1).astype(np.float32)
        src2_transform = src2.transform
        src2_crs = src2.crs

    # Normalize data2 to the range [0, 1]
    min_val = np.min(data2)
    max_val = np.max(data2)
    if max_val != min_val:
        norm_data2 = (data2 - min_val) / (max_val - min_val)
    else:
        norm_data2 = np.zeros_like(data2)

    # --- Step 2: Open the occupancy (first) TIFF file to retrieve geospatial metadata ---
    with rasterio.open(occupancy_path) as src1:
        data1 = src1.read(1).astype(np.float32)  # assumed already between 0 and 1
        dst_transform = src1.transform
        dst_crs = src1.crs
        dst_shape = data1.shape
        dst_profile = src1.profile

    # --- Step 3: Reproject normalized threshold data onto the occupancy file's grid ---
    reprojected_data = np.empty(dst_shape, dtype=np.float32)
    reproject(
        source=norm_data2,
        destination=reprojected_data,
        src_transform=src2_transform,
        src_crs=src2_crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        resampling=Resampling.nearest  # Change to bilinear if smoother data is preferred.
    )

    # --- Step 4: Accumulate the data ---
    if method.lower() == "probabilistic":
        # Combine as independent probabilities:
        # p_combined = 1 - (1 - p1) * (1 - p2)
        combined_data = 1 - (1 - data1) * (1 - reprojected_data)
    else:
        # Simple addition
        combined_data = data1 + reprojected_data
        if re_normalize:
            # Re-scale combined data back to [0, 1]
            min_comb = np.min(combined_data)
            max_comb = np.max(combined_data)
            if max_comb != min_comb:
                combined_data = (combined_data - min_comb) / (max_comb - min_comb)
            else:
                combined_data = np.zeros_like(combined_data)
        else:
            # Alternatively, clip values above 1
            combined_data = 1 - np.clip(combined_data, 0, 1)

    # --- Step 5: Save the accumulated result ---
    dst_profile.update(dtype=rasterio.float32)
    with rasterio.open(output_path, "w", **dst_profile) as dst:
        dst.write(combined_data, 1)

    print(f"Accumulated data saved to: {output_path}")


def update_drone_geotransform(old_geotransform, desired_resolution_m=1.0):
    """
    Update the drone geotransform to reflect a new resolution (meters per pixel).

    Args:
        old_geotransform (tuple): The original geotransform in the form
            (origin_x, pixel_width, skew_x, origin_y, skew_y, pixel_height).
        desired_resolution_m (float): Desired resolution in meters per pixel.

    Returns:
        tuple: New geotransform with updated pixel sizes.
    """
    origin_x, _, skew_x, origin_y, skew_y, _ = old_geotransform

    # Approximate meters per degree (latitude is nearly constant; longitude depends on latitude)
    meters_per_degree_lat = 111320.0
    meters_per_degree_lon = 111320.0 * math.cos(math.radians(origin_y))

    # Compute new pixel sizes in degrees
    new_pixel_width = desired_resolution_m / meters_per_degree_lon
    new_pixel_height = -desired_resolution_m / meters_per_degree_lat  # negative because y decreases

    # Create new geotransform (keeping origin and skew the same)
    new_geotransform = (origin_x, new_pixel_width, skew_x, origin_y, skew_y, new_pixel_height)
    return new_geotransform


#################################


def estimate_nd_status():
    global polygon_coordinates, _last_upload
    ############################################################
    # while georeferenced_seg_flag:
    ############################################################
    disaster_type = global_cache.get('natural_disaster', 'No disaster info')
    ogm_path = f"estimated_OGM/occupancy_grid_map_{disaster_type}.tif"
    ###################################################
    # Prediction step
    ###################################################
    if disaster_type == "Flood":
        prediction_path = "downloads/FloodSim"
    else:
        prediction_path = "downloads/FireSim"
    ###################################################
    ###################################################
    try:
        # ############################################################
        # # For copying the FireSim .tif files into the FireSim
        # ############################################################
        # source_folder = "Arrival_time/"
        # destination_folder = "downloads/FireSim/"
        #
        # def copy_files_if_destination_empty(source, destination):
        #     """
        #     Checks if the destination folder is empty.
        #     If it is, copies all files and directories from the source folder.
        #     """
        #     # Check if the destination folder is empty
        #     files = os.listdir(destination)
        #     if '.tif' not in files:
        #         # Iterate over each item in the source folder
        #         for item in os.listdir(source):
        #             source_item = os.path.join(source, item)
        #             destination_item = os.path.join(destination, item)
        #
        #             # Copy directories or files appropriately
        #             if os.path.isdir(source_item):
        #                 # For directories, copy the entire directory tree
        #                 shutil.copytree(source_item, destination_item)
        #             else:
        #                 # For files, preserve metadata with copy2
        #                 shutil.copy2(source_item, destination_item)
        #         # print("Copy process completed successfully.")
        # copy_files_if_destination_empty(source_folder, destination_folder)
        # ############################################################
        for pred_file in sorted(os.listdir(prediction_path),
                                key=lambda x_: os.path.getmtime(os.path.join(prediction_path, x_))):
            if pred_file.lower().endswith('.tif'):
                pred_file = os.path.join(prediction_path, pred_file)
                logger.info(f"prediction file ---> {pred_file}")
                # logger.info(f"OGM file ---> {ogm_path}")
                if disaster_type == 'Flood':
                    # predict_ogm_flood(ogm_path,
                    #                   pred_file)
                    ###########################################################
                    fuse_ogm_with_depth(
                        ogm_path,
                        pred_file,
                        mapping_="logistic",
                        mapping_params={"h50": 0.50, "s": 0.55},
                        fusion="logit_pool",
                        alpha=0.5,
                    )
                    ###########################################################
                    logger.info("prediction is performed")
                    ##################################
                    result = mask_geotiff_with_polygon_exact(ogm_path, global_cache.get('roi'), ogm_path)
                    if result:
                        logger.info(f"Masked GeoTIFF saved to {result}")
                    else:
                        logger.info("Error occurred while masking the GeoTIFF.")
                    ##################################
                    if os.path.exists(pred_file):
                        os.remove(pred_file)
                    break
                if disaster_type == "Fire":
                    predict_ogm_fire(ogm_path,
                                     pred_file,
                                     ogm_path,
                                     "probabilistic",
                                     re_normalize=True)
                    logger.info("prediction is performed")
                    ##################################
                    result = mask_geotiff_with_polygon_exact(ogm_path, global_cache.get('roi'), ogm_path)
                    if result:
                        logger.info(f"Masked GeoTIFF saved to {result}")
                    else:
                        logger.info("Error occurred while masking the GeoTIFF.")
                    ##################################
                    if os.path.exists(pred_file):
                        os.remove(pred_file)
                    break
    except Exception as e:
        logger.info(f"Error in prediction models due to {e}")
    ###################################################
    ogm_data, ogm_gt_, ogm_proj, data_type, observ_nodata = load_image(ogm_path, 0)
    ###################################################
    # Update OGM with drone data
    ###################################################
    try:
        if disaster_type == 'Flood':
            observe_data_drone, observe_gt_drone, observe_proj_drone, data_type, observ_nodata = load_image(
                "georeferenced_drone_images/segmented/flood", 1)
            ############################################################
            if observe_data_drone is not None:
                observe_data_drone = np.asarray(observe_data_drone, dtype=np.float32)
                if observe_data_drone.max() > 1:
                    observe_data_drone = observe_data_drone / 255.0
            ############################################################
            # try:
            if data_type == 'Flood' and observe_data_drone is not None:
                ogm_data = np.asarray(ogm_data)
                ogm_data = update_occupancy_grid_flood(ogm_data,
                                                       ogm_gt_,
                                                       observe_data_drone,
                                                       observe_gt_drone)
                # logger.info(f"OGM done successfully  FOR FLOOD ND {ogm_data.shape}")
    except Exception as e:
        logger.info(f'Error in fusing Segmented Flood images due to {e}')
    ######################################################################
    # Save updated OGM as GeoTIFF
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_tiff_path_timestamp = f"occupancy_grid_map_{disaster_type}_{timestamp}.tif"

        output_tiff_path = f"occupancy_grid_map_{disaster_type}.tif"
        save_geotiff("estimated_OGM",
                     output_tiff_path,
                     ogm_data,
                     ogm_gt_,
                     timestamp,
                     crs_epsg=4326)

        metadata_timestamp = save_geotiff("estimated_OGM",
                                          output_tiff_path_timestamp,
                                          ogm_data,
                                          ogm_gt_,
                                          timestamp,
                                          crs_epsg=4326)

        # with _upload_lock:
        now_1 = time.monotonic()
        if now_1 - _last_upload >= UPLOAD_INTERVAL:
            process_and_upload_ogm(ND_entity_ID, os.path.join("estimated_OGM", output_tiff_path_timestamp),
                                   config.BUCKET_NAME, metadata_timestamp)
            _last_upload = now_1

    except Exception as e:
        logger.error(f"Error saving or uploading OGM: {e}")
    ######################################################################
    ###################################################
    if disaster_type == "Fire":
        # Fuse active fire measurements (segmented fire)
        observe_data_drone, observe_gt_drone, observe_proj_drone, data_type, observ_nodata = load_image(
            "georeferenced_drone_images/segmented/fire", 1)
        ############################################################
        if observe_data_drone is None:
            logger.error("observe_data_drone is None; check your data source or assignment")
        else:
            observe_data_drone = np.asarray(observe_data_drone)
            if observe_data_drone.max() > 1:
                observe_data_drone = observe_data_drone / 255.0
            # logger.info(f"observe_data_drone {observe_data_drone.shape}")
            # logger.info(f"observe_gt_drone {observe_gt_drone}")
            # logger.info(f"observe_proj_drone {observe_proj_drone}")
            # logger.info(f"data_type {data_type}")
            ############################################################
            try:
                if disaster_type == 'Fire' and observe_data_drone is not None:
                    if data_type == 'active_fire':
                        ogm_data = update_occupancy_grid_fire(ogm_data,
                                                              ogm_gt_,
                                                              observe_data_drone,
                                                              observe_gt_drone,
                                                              data_type)
                        logger.info(f"OGM done successfully FOR FIRE ND {ogm_data.shape}")

            except Exception as e:
                logger.info(f"Error in updating OGM with active_fire drone measurement {e}")
        ######################################################################
        # Save updated OGM as GeoTIFF
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_tiff_path_timestamp = f"occupancy_grid_map_{disaster_type}_{timestamp}.tif"

            output_tiff_path = f"occupancy_grid_map_{disaster_type}.tif"
            save_geotiff("estimated_OGM",
                         output_tiff_path,
                         ogm_data,
                         ogm_gt_,
                         timestamp,
                         crs_epsg=4326)

            metadata_timestamp = save_geotiff("estimated_OGM",
                                              output_tiff_path_timestamp,
                                              ogm_data,
                                              ogm_gt_,
                                              timestamp,
                                              crs_epsg=4326)

            # with _upload_lock:
            now_2 = time.monotonic()
            if now_2 - _last_upload >= UPLOAD_INTERVAL:
                logger.info(f" now time and latest_upload ----------> {now_2}   {_last_upload}")
                process_and_upload_ogm(ND_entity_ID,
                                       os.path.join("estimated_OGM", output_tiff_path_timestamp),
                                       config.BUCKET_NAME, metadata_timestamp)
                _last_upload = now_2
            else:
                remaining = UPLOAD_INTERVAL - (now_2 - _last_upload)
                # logger.info("Next upload in %.00f seconds", remaining)
        except Exception as e:
            logger.error(f"Error saving or uploading OGM: {e}")
        ######################################################################
        # Fuse burnt area measurements (segmented burnt area)
        ######################################################################
        observe_data_drone, observe_gt_drone, observe_proj_drone, data_type, observ_nodata = load_image(
            "georeferenced_drone_images/segmented/burnt", 1)

        ############################################################
        if observe_data_drone is None:
            logger.error("observe_data_drone is None; check your data source or assignment")
        else:
            observe_data_drone = np.asarray(observe_data_drone)
            if observe_data_drone.max() > 1:
                observe_data_drone = observe_data_drone / 255.0
            ############################################################
            try:
                if disaster_type == 'Fire' and observe_data_drone is not None:
                    if data_type == 'burnt_area':
                        ogm_data = update_occupancy_grid_fire(ogm_data,
                                                              ogm_gt_,
                                                              observe_data_drone,
                                                              observe_gt_drone,
                                                              data_type)
                        logger.info(f"OGM done successfully FOR FIRE ND {ogm_data.shape}")

            except Exception as e:
                logger.info(f"Error in updating OGM for burnt area {e}")
    ######################################################################
    # Save updated OGM as GeoTIFF
    ######################################################################
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_tiff_path_timestamp = f"occupancy_grid_map_{disaster_type}_{timestamp}.tif"

        output_tiff_path = f"occupancy_grid_map_{disaster_type}.tif"
        metadata = save_geotiff("estimated_OGM",
                                output_tiff_path,
                                ogm_data,
                                ogm_gt_,
                                timestamp,
                                crs_epsg=4326)

        metadata_timestamp = save_geotiff("estimated_OGM",
                                          output_tiff_path_timestamp,
                                          ogm_data,
                                          ogm_gt_,
                                          timestamp,
                                          crs_epsg=4326)

        # with _upload_lock:
        now_3 = time.monotonic()
        if now_3 - _last_upload >= UPLOAD_INTERVAL:
            logger.info(f" now time and latest_upload ----------> {now_3}   {_last_upload}")
            process_and_upload_ogm(ND_entity_ID,
                                   os.path.join("estimated_OGM", output_tiff_path_timestamp),
                                   config.BUCKET_NAME, metadata_timestamp)
            _last_upload = now_3
        else:
            remaining = UPLOAD_INTERVAL - (now_3 - _last_upload)
            # logger.info("Next upload in %.00f seconds", remaining)
    except Exception as e:
        logger.error(f"Error saving or uploading OGM: {e}")
    ######################################################################
    ######################################################################
    # Update OGM with satellite data
    ######################################################################
    try:
        observ_sat_data_, observ_sat_gt_, observ_sat_proj, data_type, observ_nodata = load_image(
            f"downloads/satellite_imgs/{disaster_type}", 3)
        if observ_sat_data_ is not None:
            observ_sat_data_ = np.asarray(observ_sat_data_, dtype=np.float32)
            # if observ_sat_data_.max() > 1:
            #     scale = 255.0
            #     observ_sat_data_ /= scale
            #     if observ_nodata is not None:
            #         observ_nodata = observ_nodata / scale
            if data_type == 'Flood':
                ogm_data = update_occupancy_grid_flood_sat_fixed(
                    ogm_data, ogm_gt_,
                    observ_sat_data_, observ_sat_gt_,
                    ogm_crs='EPSG:4326',
                    observation_crs=observ_sat_proj,
                    obs_nodata=observ_nodata

                )
                ######################################################################
                # Save updated OGM as GeoTIFF
                ######################################################################
                try:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    output_tiff_path_timestamp = f"occupancy_grid_map_{disaster_type}_{timestamp}.tif"

                    output_tiff_path = f"occupancy_grid_map_{disaster_type}.tif"
                    save_geotiff("estimated_OGM",
                                 output_tiff_path,
                                 ogm_data,
                                 ogm_gt_,
                                 timestamp,
                                 crs_epsg=4326)

                    metadata_timestamp = save_geotiff("estimated_OGM",
                                                      output_tiff_path_timestamp,
                                                      ogm_data,
                                                      ogm_gt_,
                                                      timestamp,
                                                      crs_epsg=4326)

                    # with _upload_lock:
                    now_4 = time.monotonic()
                    if now_4 - _last_upload >= UPLOAD_INTERVAL:
                        logger.info(f" now time and latest_upload ----------> {now_4}   {_last_upload}")
                        process_and_upload_ogm(ND_entity_ID, os.path.join("estimated_OGM", output_tiff_path_timestamp),
                                               config.BUCKET_NAME, metadata_timestamp)
                        _last_upload = now_4
                    else:
                        remaining = UPLOAD_INTERVAL - (now_4 - _last_upload)
                        # logger.info("Next upload in %.00f seconds", remaining)
                except Exception as e:
                    logger.error(f"Error saving or uploading OGM: {e}")
                ######################################################################
            elif data_type == 'burnt_area':
                ogm_data = update_occupancy_grid_fire(ogm_data,
                                                      ogm_gt_,
                                                      observ_sat_data_,
                                                      observ_sat_gt_,
                                                      'burnt_area')
                ######################################################################
                # Save updated OGM as GeoTIFF
                try:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    output_tiff_path_timestamp = f"occupancy_grid_map_{disaster_type}_{timestamp}.tif"

                    output_tiff_path = f"occupancy_grid_map_{disaster_type}.tif"
                    save_geotiff("estimated_OGM",
                                 output_tiff_path,
                                 ogm_data,
                                 ogm_gt_,
                                 timestamp,
                                 crs_epsg=4326)

                    metadata_timestamp = save_geotiff("estimated_OGM",
                                                      output_tiff_path_timestamp,
                                                      ogm_data,
                                                      ogm_gt_,
                                                      timestamp,
                                                      crs_epsg=4326)
                    # with _upload_lock:
                    now_5 = time.monotonic()
                    if now_5 - _last_upload >= UPLOAD_INTERVAL:
                        logger.info(f" now time and latest_upload ----------> {now_5}   {_last_upload}")
                        process_and_upload_ogm(ND_entity_ID, os.path.join("estimated_OGM", output_tiff_path_timestamp),
                                               config.BUCKET_NAME, metadata_timestamp)
                        _last_upload = now_5
                    else:
                        remaining = UPLOAD_INTERVAL - (now_5 - _last_upload)
                        # logger.info("Next upload in %.00f seconds", remaining)
                except Exception as e:
                    logger.error(f"Error saving or uploading OGM: {e}")
                ######################################################################
            logger.info(f"OGM has successfully generated and fused sate file")
    except FileNotFoundError:
        logger.warning("Satellite ROI directory not found.")
    except Exception as e:
        logger.info(f"Sat measurements are not available {e}")
    ##################################################################
    # Update OGM with hotspot data from the SocialMedia directory
    ##################################################################
    # Uncomment this section to fuse Geo_social media
    ##################################################################
    '''
        try:
        # ─── Fusion parameters ─────────────────────────────────────────────────────────
        w_c, w_r = 0.5, 0.5  # weights for count vs. ratio
        eps = 1e-6
        count_thr = 10
        ratio_thr = 0.1  # adjust to your “meaningful” ratio cutoff
        # ────────────────────────────────────────────────────────────────────────────────
        with rasterio.open(ogm_path) as src:
            ogm_profile = src.profile
            ogm_data = src.read(1)  # 2D array
            transform = src.transform
            H, W = ogm_data.shape
            crs = src.crs
            ogm_crs_str = crs.to_string()
        # ────────────────────────────────────────────────────────────────────────────────
        aoi_wgs = Polygon(global_cache.get('roi')[0])
        if crs.to_epsg() != 4326:
            tr = Transformer.from_crs("EPSG:4326", ogm_crs_str, always_xy=True)
            aoi_geom = shapely.ops.transform(tr.transform, aoi_wgs)
        else:
            aoi_geom = aoi_wgs

        aoi_mask = rasterize(
            [(mapping(aoi_geom), 1)],
            out_shape=(H, W),
            transform=transform,
            fill=0,
            dtype="uint8"
        ).astype(bool)
        # ────────────────────────────────────────────────────────────────────────────────
        # Sort files by timestamp (assumes the timestamp is part of the filename)
        social_media_dir = "downloads/SocialMedia"  # Path to the SocialMedia directory
        hotspot_files = [os.path.join(social_media_dir, f) for f in os.listdir(social_media_dir)
                         if f.startswith("hotspot_results") and f.endswith(".geojson")]
        hotspot_files.sort()
        for hotspot_file in hotspot_files:
            try:
                # 1) load and filter features
                geoms, counts, ratios = [], [], []
                with open(hotspot_file) as f:
                    gj = json.load(f)
                hot_crs = gj.get("crs", {}).get("properties", {}).get("name", "EPSG:4326")
                tr_h = None
                if hot_crs != ogm_crs_str:
                    tr_h = Transformer.from_crs(hot_crs, ogm_crs_str, always_xy=True)

                for feat in gj["features"]:
                    props = feat["properties"]
                    poly = shape(feat["geometry"])
                    if not poly.is_valid:
                        logger.warning(f"invalid poly in {hotspot_file}")
                        continue
                    c = float(props.get("count", 0))
                    r = float(props.get("ratio", 0))
                    if c < count_thr or r < ratio_thr:
                        continue

                    if tr_h:
                        poly = shapely.ops.transform(tr_h.transform, poly)
                    geoms.append(poly)
                    counts.append(c)
                    ratios.append(r)

                if not geoms:
                    os.remove(hotspot_file)
                    continue

                # 2) normalize metrics
                counts = np.array(counts, dtype=float)
                ratios = np.array(ratios, dtype=float)
                c_norm = counts / counts.max()
                r_norm = ratios / ratios.max()

                # 3) rasterize each metric
                def rasterize_vals(vals):
                    return rasterize(
                        [(mapping(g), v) for g, v in zip(geoms, vals)],
                        out_shape=(H, W),
                        transform=transform,
                        fill=0,
                        dtype="float32"
                    )

                c_r = rasterize_vals(c_norm)
                r_r = rasterize_vals(r_norm)
                hotspot_mask = (c_r > 0) | (r_r > 0)

                # 4) Bayesian‐style fusion
                mask = hotspot_mask & aoi_mask
                if mask.any():
                    p_m = np.zeros_like(ogm_data, dtype=float)
                    p_m[mask] = np.clip(w_c * c_r[mask] + w_r * r_r[mask], eps, 1 - eps)

                    llr = np.zeros_like(ogm_data, dtype=float)
                    llr[mask] = np.log(p_m[mask] / (1 - p_m[mask]))

                    updated = ogm_data.astype(float)
                    updated[mask] = 1 / (1 + np.exp(-llr[mask]))

                    ogm_data = updated

                # remove processed file
                os.remove(hotspot_file)

                # 5) write out
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                out_base = f"occupancy_grid_map_{disaster_type}"
                fn1 = f"{out_base}.tif"
                fn2 = f"{out_base}_{timestamp}.tif"

                for fn in (fn1, fn2):
                    with rasterio.open(
                            os.path.join("estimated_OGM", fn),
                            'w', **ogm_profile
                    ) as dst:
                        dst.write(ogm_data, 1)

                metadata_timestamp = save_geotiff("estimated_OGM",
                                                  fn2,
                                                  ogm_data,
                                                  ogm_gt_,
                                                  timestamp,
                                                  crs_epsg=4326)

                # with _upload_lock:
                now_6 = time.monotonic()
                if now_6 - _last_upload >= UPLOAD_INTERVAL:
                    logger.info(f" now time and latest_upload ----------> {now_6}   {_last_upload}")
                    # upload only the timestamped one
                    process_and_upload_ogm(
                        ND_entity_ID,
                        os.path.join("estimated_OGM", fn2),
                        config.BUCKET_NAME,
                        metadata=metadata_timestamp
                    )
                    _last_upload = now_6
                else:
                    remaining = UPLOAD_INTERVAL - (now_6 - _last_upload)
                    # logger.info("Next upload in %.00f seconds", remaining)
                logger.info(f"Fused & removed {hotspot_file}")

            except Exception as e:
                logger.error(f"Failed on {hotspot_file}: {e}")
    except FileNotFoundError:
        logger.warning("SocialMedia directory not found.")
    except Exception as e:
        logger.error(f"Error integrating hotspot data: {e}")
    '''


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
    x_min, y_min = coordinates_to_pixel_update(geo_transform, bounds[0], bounds[1])
    x_max, y_max = coordinates_to_pixel_update(geo_transform, bounds[2], bounds[3])

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


def haversine_distance(coord1, coord2):
    """
    Calculate the great-circle distance between two points using the Haversine formula.

    :param coord1: Tuple (longitude, latitude) in degrees
    :param coord2: Tuple (longitude, latitude) in degrees
    :return: Distance in meters
    """
    R = 6371000  # Earth's radius in meters

    # Convert latitude and longitude from degrees to radians
    lat1, lon1 = map(math.radians, [coord1[1], coord1[0]])
    lat2, lon2 = map(math.radians, [coord2[1], coord2[0]])

    # Differences
    delta_lat = lat2 - lat1
    delta_lon = lon2 - lon1

    # Haversine formula
    a = math.sin(delta_lat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(delta_lon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    # Compute distance
    distance = R * c  # Distance in meters
    return distance


def save_multi_band_geotiff(output_path, data_3d, transform, proj, crs_epsg=4326):
    height = data_3d.shape[1]
    width = data_3d.shape[2]
    band_count = data_3d.shape[0]

    with rasterio.open(
            output_path,
            'w',
            driver='GTiff',
            height=height,
            width=width,
            count=band_count,
            dtype=data_3d.dtype,
            crs=CRS.from_epsg(crs_epsg),
            transform=Affine.from_gdal(*transform),
    ) as dst:
        for i in range(band_count):
            dst.write(data_3d[i, :, :], i + 1)


def associate_features_with_measurements(ogm_features, new_features, distance_threshold):
    """
    Use the Hungarian algorithm (linear_sum_assignment) to associate OGM features
    with new measurements. Only returns matches with distance < distance_threshold.
    """

    N = len(ogm_features)
    M = len(new_features)
    if N == 0 or M == 0:
        return []

    # Collect (lon, lat) from OGM
    ogm_coords = []
    for feat in ogm_features:
        geom = feat.get("geometry", {})
        if geom.get("type") == "Point":
            coords = geom.get("coordinates", [])
            if len(coords) >= 2:
                ogm_coords.append((coords[0], coords[1]))
            else:
                ogm_coords.append((None, None))
        else:
            ogm_coords.append((None, None))

    # Collect (lon, lat) from new measurements
    new_coords = []
    for feat in new_features:
        geom = feat.get("geometry", {})
        if geom.get("type") == "Point":
            coords = geom.get("coordinates", [])
            if len(coords) >= 2:
                new_coords.append((coords[0], coords[1]))
            else:
                new_coords.append((None, None))
        else:
            new_coords.append((None, None))

    # Build cost matrix (N x M)
    cost_matrix = np.zeros((N, M), dtype=np.float64)
    for i in range(N):
        lon1, lat1 = ogm_coords[i]
        for j in range(M):
            lon2, lat2 = new_coords[j]
            if None in (lon1, lat1, lon2, lat2):
                cost_matrix[i, j] = 1e12  # large cost if invalid geometry
            else:
                dist = haversine_distance((lon1, lat1), (lon2, lat2))
                cost_matrix[i, j] = dist

    # Hungarian assignment
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Filter out pairs with distance above threshold
    matches = []
    for i_ogm, i_new in zip(row_ind, col_ind):
        if cost_matrix[i_ogm, i_new] <= distance_threshold:
            matches.append((i_ogm, i_new))

    return matches


def get_or_create_feature_id(feature: dict) -> str:
    props = feature.setdefault("properties", {})
    feature_id = props.get("id")
    if not feature_id:
        feature_id = str(uuid.uuid4())  # or any other unique ID strategy
        props["id"] = feature_id
    return feature_id


def get_kf_states():
    """
    Retrieve the entire dictionary of ID -> KalmanFilter
    from a global or external cache. If none is found, return an empty dict.
    """
    # Here we return whatever is stored at key "kf_states", or an empty dict if not set
    return global_cache.get("kf_states", {})


def set_kf_states(kf_states: dict):
    """
    Persist the entire dictionary of ID -> KalmanFilter
    into a global or external cache so it can be retrieved later.
    """
    global_cache["kf_states"] = kf_states


def estimate_objects_status():
    """
    1) Load existing OGM as GeoJSON.
    2) Integrate Drone data (with Hungarian data association).
    3) Apply a Kalman Filter to refine positions.
    4) Integrate Social Media data (SinglePostResult) as GeoJSON (also via association).
    5) Save final OGM (overwrite + timestamped, filtering label == -1 only from the timestamped).
    6) Prepare metadata and upload the final result.
    """
    # global georeferenced_obj_flag
    # while georeferenced_obj_flag:
    # (1) Load OGM
    # logger.info("Estimating Maps4Objects in the loop")
    disaster_type = global_cache.get("natural_disaster", "NoDisaster")
    if disaster_type == "NoDisaster":
        logger.info("No valid disaster type. Aborting object status estimation.")
        return

    ogm_metadata_path = f"estimated_OGM/occupancy_grid_map_{disaster_type}_Objects.json"
    if not os.path.exists(ogm_metadata_path):
        logger.info(f"OGM not found at {ogm_metadata_path}. Aborting.")
        return

    if not os.listdir("georeferenced_drone_images/detection"):
        return

    # Attempt to load the JSON file with additional debugging
    try:
        with open(ogm_metadata_path, "r") as f:
            try:
                ogm_metadata = json.load(f)
            except json.JSONDecodeError as e:
                # Seek back a little before the error position to get context
                f.seek(max(0, e.pos - 100))
                snippet = f.read(200)
                logger.error(f"JSON decode error at pos {e.pos}: {e.msg}. Problematic snippet: {snippet}",
                             exc_info=True)
                return
    except Exception as e:
        logger.error(f"Error reading file {ogm_metadata_path}: {e}", exc_info=True)
        return

    ogm_features = ogm_metadata.get("features", [])
    # logger.info(f"Loaded OGM with {len(ogm_features)} features.")

    # (2) Integrate Drone Data (Association + update)
    observ_metadata = get_geotiff_metadata_rasterio("georeferenced_drone_images/detection")
    if observ_metadata and isinstance(observ_metadata, dict):
        drone_features = observ_metadata.get("features", [])
        # logger.info(f"Observ metadata => {len(drone_features)} new Drone features found.")
        distance_threshold = (int(config.OGM_OBJ_RESOLUTION) // 1)
        matches = associate_features_with_measurements(
            ogm_features,
            drone_features,
            distance_threshold
        )
        # logger.info(f"Hungarian matching => {len(matches)} associations (within {distance_threshold}m).")

        # — stash label, score, grid_center & raw_measurement
        for (i_ogm, i_drone) in matches:
            ogm_feat = ogm_features[i_ogm]
            drone_feat = drone_features[i_drone]
            props = ogm_feat.setdefault("properties", {})
            df_props = drone_feat.get("properties", {})

            # preserve original grid‐center
            props["grid_center"] = ogm_feat["geometry"]["coordinates"].copy()
            # record the exact drone measurement
            props["raw_measurement"] = drone_feat["geometry"]["coordinates"].copy()
            # carry over label & score
            props["label"] = df_props.get("label", -1)
            props["score"] = df_props.get("score", 0.0)

        # Overwrite with association results
        with open(ogm_metadata_path, "w") as file:
            json.dump(ogm_metadata, file, indent=4)
        #################################################################
        #################################################################
        # (3) Apply Kalman Filter to each matched feature
        for feat in ogm_features:
            props = feat.get("properties", {})
            meas = props.get("raw_measurement")
            if meas is None:
                continue  # skip unlabeled / unmatched

            # initial state = grid center
            init = np.array(props["grid_center"]).reshape((3, 1))
            z = np.array(meas).reshape((3, 1))

            # configure Kalman Filter
            kf = KalmanFilter(state_dim=3, measurement_dim=3)
            kf.x = init.copy()
            # measurement noise (drone): ~4 cm
            drone_sigma = 0.04
            # process noise (grid spacing): half‐cell variance
            grid_sigma = float(config.OGM_OBJ_RESOLUTION) / 2.0
            kf.R = np.eye(3) * (drone_sigma ** 2)
            kf.Q = np.eye(3) * (grid_sigma ** 2)
            # initial state covariance: allow the filter to learn
            kf.P = np.eye(3) * ((grid_sigma * 2) ** 2)

            # run predict + update
            kf.predict()
            kf.update(z)

            # write back the refined coordinate
            refined = kf.x.flatten().tolist()
            feat["geometry"]["coordinates"] = refined
            props["kf_refined"] = refined

        # Overwrite with KF‐refined results
        with open(ogm_metadata_path, "w") as f:
            json.dump(ogm_metadata, f, indent=4)

        # -- Only filter label == -1 in the timestamped copy, not in the main one
        filtered_metadata = copy.deepcopy(ogm_metadata)
        original_count = len(filtered_metadata["features"])
        filtered_metadata["features"] = [
            ff for ff in filtered_metadata["features"]
            if ff.get("properties", {}).get("label", -1) != -1
        ]
        new_count = len(filtered_metadata["features"])
        # logger.info(
        #     f"KF step => filtered out label == -1 for timestamped version: "
        #     f"old count={original_count}, new count={new_count}."
        # )

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        timestamped_name = f"occupancy_grid_map_{disaster_type}_Objects_{ts}.json"
        with open(os.path.join("estimated_OGM", timestamped_name), "w") as f:
            json.dump(filtered_metadata, f, indent=4)

        # logger.info("Kalman filter applied to OGM.")

        ##################################################################################
        # Save and Upload
        ##################################################################################
        final_ogm_path_obj = timestamped_name
        natural_disaster = global_cache.get('natural_disaster', 'No disaster info')
        metadata = {
            'title': 'Probability-Based Occupancy Grid Map',
            'author': 'GRVC lab, University of Seville',
            'description': (
                f"Estimated occupancy grid map for the {natural_disaster} scenario, "
                "used to track detected persons and vehicles."
            ),
            'creationDate': datetime.now().isoformat(),
            "spatialReference": "EPSG:4326",
            "file_name": f"pdm05/{global_cache.get('bm_id')}/{final_ogm_path_obj}",
            "coordinates": global_cache.get('roi', [None]),
            "XMin": 0,
            "XRes": 0,
            "YMax": 0,
            "YRes": 0,
            "bucket": config.BUCKET_NAME,
            "bm_id": global_cache.get('bm_id'),
            "sent": ts,
            "ignitionPoints": {
                "type": "GeoProperty",
                "value": {
                    "type": "Point",
                    "coordinates": global_cache.get('ignitionPoints', [None])
                }
            },
            "minio_url": (
                f'https://{config.MINIO_ENDPOINT}/'
                f'{config.BUCKET_NAME}/pdm5/{global_cache.get("bm_id")}/{final_ogm_path_obj}'
            )
        }
        output_local_path = os.path.join("estimated_OGM", final_ogm_path_obj)

        logger.info(f"Uploading final OGM to cloud bucket: {config.BUCKET_NAME}")
        process_and_upload_ogm(
            obj_entity_ID,
            output_local_path,
            config.BUCKET_NAME,
            metadata
        )
        logger.info(
            f"OGM successfully processed and uploaded with processed drone image: "
            f"{output_local_path}"
        )

        logger.info("Drone data integrated (with data association).")
    else:
        logger.info("Drone metadata is empty or unavailable.")
    ##################################################################################
    # (4) Integrate Social Media SinglePostResult (Association + update)
    social_media_dir = "downloads/SocialMedia"
    if not os.path.isdir(social_media_dir):
        logger.info("No SocialMedia directory found. Skipping.")
    else:
        single_post_files = [
            f for f in os.listdir(social_media_dir)
            if f.startswith("single_posts_results") and f.endswith(".geojson")
        ]
        single_post_files.sort()  # process in chronological order

        for sm_file in single_post_files:
            sm_path = os.path.join(social_media_dir, sm_file)
            with open(sm_path, "r") as f:
                sm_data = json.load(f)
            sm_features = sm_data.get("features", [])

            # Data association
            matches = associate_features_with_measurements(
                ogm_features, sm_features, distance_threshold=(int(config.OGM_OBJ_RESOLUTION) // 1)
            )
            # logger.info(f"Social Media matching => {len(matches)} associations (within 5m).")

            for (i_ogm, i_sm) in matches:
                ogm_feat = ogm_features[i_ogm]
                sm_feat = sm_features[i_sm]
                ogm_props = ogm_feat.setdefault("properties", {})
                sm_props = sm_feat.get("properties", {})

                existing_score = ogm_props.get("score", 0.0)
                new_prob = sm_props.get("prob", 0.5)
                fused_score = fuse_fire_probability(existing_score, new_prob, weight=0.5)
                ogm_props["score"] = fused_score
                ogm_props["label"] = sm_props.get("label", -1)

            # Remove file after processing
            os.remove(sm_path)
            # logger.info(f"Processed and removed {sm_path}")

        # Overwrite + timestamp after SocialMedia integration
        with open(ogm_metadata_path, "w") as f:
            json.dump(ogm_metadata, f, indent=4)

        # Filter out -1 in the timestamped copy
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        ogm_metadata_path_timestamp = f"occupancy_grid_map_{disaster_type}_Objects_{ts}.json"
        filtered_metadata = copy.deepcopy(ogm_metadata)
        original_count = len(filtered_metadata["features"])
        filtered_metadata["features"] = [
            ff for ff in filtered_metadata["features"]
            if ff.get("properties", {}).get("label", -1) != -1
        ]
        new_count = len(filtered_metadata["features"])
        logger.info(f"Social Media => filtered out label == -1 for timestamped version: "
                    f"old count={original_count}, new count={new_count}.")

        with open(os.path.join("estimated_OGM", ogm_metadata_path_timestamp), "w") as f:
            json.dump(filtered_metadata, f, indent=4)

        logger.info("Social Media data integrated into OGM.")

        # (5) Prepare metadata for upload
        final_ogm_path_obj = ogm_metadata_path_timestamp
        natural_disaster = global_cache.get('natural_disaster', 'No disaster info')
        metadata = {
            'title': 'Probability-Based Occupancy Grid Map',
            'author': 'GRVC lab, University of Seville',
            'description': (
                f"Estimated occupancy grid map for the {natural_disaster} scenario, "
                "used to track detected persons and vehicles."
            ),
            'creationDate': datetime.now().isoformat(),
            "spatialReference": "EPSG:4326",
            "file_name": f"pdm05/{global_cache.get('bm_id')}/{final_ogm_path_obj}",
            "coordinates": global_cache.get('roi', [None]),
            "XMin": 0,
            "XRes": 0,
            "YMax": 0,
            "YRes": 0,
            "bucket": config.BUCKET_NAME,
            "bm_id": global_cache.get('bm_id'),
            "sent": ts,
            "ignitionPoints": {
                "type": "GeoProperty",
                "value": {
                    "type": "Point",
                    "coordinates": global_cache.get('ignitionPoints', [None])
                }
            },
            "minio_url": (
                f'https://{config.MINIO_ENDPOINT}/'
                f'{config.BUCKET_NAME}/pdm05/{global_cache.get("bm_id")}/{final_ogm_path_obj}'
            )
        }
        output_local_path = os.path.join("estimated_OGM", final_ogm_path_obj)

        logger.info(f"Uploading final OGM to cloud bucket: {config.BUCKET_NAME}")
        process_and_upload_ogm(obj_entity_ID, output_local_path, config.BUCKET_NAME, metadata)
        logger.info(f"OGM successfully processed and uploaded for GeoSocial Media: {output_local_path}")


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

    # ---------------------------------------------------------------
    # Normalize the geojson_data into a "FeatureCollection"-like dict
    # ---------------------------------------------------------------
    if isinstance(geojson_data, list):
        # If it's a list, assume it's a list of Features
        geojson_data = {
            "type": "FeatureCollection",
            "features": geojson_data
        }
    elif isinstance(geojson_data, dict):
        # If it's a dict but not a FeatureCollection, try to wrap it
        # in a FeatureCollection if it looks like a single Feature
        if "features" not in geojson_data and geojson_data.get("type") == "Feature":
            geojson_data = {
                "type": "FeatureCollection",
                "features": [geojson_data]
            }
    else:
        raise ValueError("GeoJSON root object is neither a dict nor a list.")

    if "features" not in geojson_data:
        raise ValueError("No 'features' field found in GeoJSON data.")

    # ---------------------------------------------------------------
    # Extract coordinates from each Feature's geometry
    # ---------------------------------------------------------------
    all_coords = []

    for feature in geojson_data.get("features", []):
        geometry = feature.get("geometry", {})
        if not isinstance(geometry, dict):
            # Guard against malformed geometry entries
            continue

        geom_type = geometry.get("type")
        coords = geometry.get("coordinates", [])

        if geom_type == "Point":
            # coords = [lon, lat, (optional elev)]
            _extract_point_coords(coords, all_coords)

        elif geom_type == "MultiPoint":
            # coords = [[lon1, lat1], [lon2, lat2], ...]
            for point_coords in coords:
                _extract_point_coords(point_coords, all_coords)

        elif geom_type == "LineString":
            # coords = [[lon1, lat1], [lon2, lat2], ...]
            for point_coords in coords:
                _extract_point_coords(point_coords, all_coords)

        elif geom_type == "MultiLineString":
            # coords = [ [[lon1, lat1], [lon2, lat2]], [...], ... ]
            for linestring in coords:
                for point_coords in linestring:
                    _extract_point_coords(point_coords, all_coords)

        elif geom_type == "Polygon":
            # coords = [
            #   [ [lon1, lat1], [lon2, lat2], ... ],  # exterior ring
            #   [ [lon1, lat1], [lon2, lat2], ... ],  # hole
            #   ...
            # ]
            for ring in coords:
                for point_coords in ring:
                    _extract_point_coords(point_coords, all_coords)

        elif geom_type == "MultiPolygon":
            # coords = [
            #   [ [ [lon, lat], [lon, lat], ... ], [ring2], ... ],  # poly 1
            #   [ [ [lon, lat], [lon, lat], ... ], ... ],           # poly 2
            # ]
            for polygon in coords:
                for ring in polygon:
                    for point_coords in ring:
                        _extract_point_coords(point_coords, all_coords)

        elif geom_type == "GeometryCollection":
            # geometry: { "type": "GeometryCollection", "geometries": [ ... ] }
            # Each sub-geometry has its own type/coordinates
            sub_geoms = geometry.get("geometries", [])
            for sub_geom in sub_geoms:
                sub_type = sub_geom.get("type")
                sub_coords = sub_geom.get("coordinates", [])
                # Recursively extract
                _extract_by_type(sub_type, sub_coords, all_coords)

        else:
            # If there's some unknown geometry type or empty geometry, skip
            continue

    if not all_coords:
        raise ValueError("No valid coordinates found in GeoJSON.")

    # Separate longitudes and latitudes
    try:
        lons, lats = zip(*all_coords)
    except ValueError:
        raise ValueError("Malformed coordinate structure in GeoJSON.")

    return {
        "min_lon": min(lons),
        "max_lon": max(lons),
        "min_lat": min(lats),
        "max_lat": max(lats)
    }


def _extract_point_coords(coords, all_coords):
    """
    Helper function to safely extract lon/lat from a coordinate array,
    ignoring any 3D altitude dimension.
    """
    if (
            isinstance(coords, list) and len(coords) >= 2
            and isinstance(coords[0], (int, float))
            and isinstance(coords[1], (int, float))
    ):
        # Only keep the first two elements (lon, lat)
        all_coords.append((coords[0], coords[1]))


def _extract_by_type(geom_type, coords, all_coords):
    """Handles geometry extraction for sub-geometries."""
    if geom_type == "Point":
        _extract_point_coords(coords, all_coords)
    elif geom_type == "MultiPoint":
        for pt in coords:
            _extract_point_coords(pt, all_coords)
    elif geom_type == "LineString":
        for pt in coords:
            _extract_point_coords(pt, all_coords)
    elif geom_type == "MultiLineString":
        for line in coords:
            for pt in line:
                _extract_point_coords(pt, all_coords)
    elif geom_type == "Polygon":
        for ring in coords:
            for pt in ring:
                _extract_point_coords(pt, all_coords)
    elif geom_type == "MultiPolygon":
        for poly in coords:
            for ring in poly:
                for pt in ring:
                    _extract_point_coords(pt, all_coords)
    elif geom_type == "GeometryCollection":
        for sg in coords.get("geometries", []):
            st = sg.get("type")
            sc = sg.get("coordinates", [])
            _extract_by_type(st, sc, all_coords)
    # else: ignore unknown geometry types


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


def get_roi(
        new_polygon_coords,
        existing_map_path,
        resolution,
        crs_epsg,
        band_count,
        dem_path  # DEM file path; used only for one-band TIFF
):
    """
    Create a GeoTIFF based on polygon coordinates with a specified resolution.

    For band_count==1:
      - Creates an occupancy grid map (OGM) where each pixel is initialized to 0.5.
      - Additionally, it samples the DEM at the polygon vertices and inserts this
        information (as [lon, lat, elevation]) into the metadata under "georef_data".

    For band_count==2:
      - Creates an OGM with two bands:
          Band 1 (label): default -1
          Band 2 (score): default 0.0
      - No elevation data is inserted in the metadata.

    Args:
        new_polygon_coords (list): Polygon coordinates [(lon, lat), ...].
        existing_map_path (str): Path for saving the GeoTIFF.
        resolution (float): Desired resolution in meters per pixel (converted to degrees if EPSG:4326).
        crs_epsg (int): EPSG code for CRS (default 4326).
        band_count (int): Number of bands. 1 or 2.
        dem_path (str): File path to the DEM GeoTIFF (only used for one-band TIFF).

    Returns:
        data_set (gdal.Dataset): GDAL dataset, or None if an error occurs.
    """
    try:
        if resolution <= 0:
            raise ValueError("Resolution must be a positive value.")

        polygon = Polygon(new_polygon_coords)
        if not polygon.is_valid:
            polygon = polygon.buffer(0)
            if not polygon.is_valid:
                raise ValueError("Polygon is invalid and could not be fixed.")

        minx, miny, maxx, maxy = polygon.bounds

        # Handle coordinate conversions if we're in EPSG:4326 (geographic)
        if crs_epsg == 4326:
            center_lat = (miny + maxy) / 2
            meters_per_degree_lat = 111320.0
            meters_per_degree_lon = 111320.0 * math.cos(math.radians(center_lat))

            lat_res_deg = resolution / meters_per_degree_lat
            lon_res_deg = resolution / meters_per_degree_lon

            width = max(1, int((maxx - minx) / lon_res_deg))
            height = max(1, int((maxy - miny) / lat_res_deg))
        else:
            # For projected CRS (in meters)
            width = max(1, int((maxx - minx) / resolution))
            height = max(1, int((maxy - miny) / resolution))

        transform_ = from_bounds(minx, miny, maxx, maxy, width, height)

        # Prepare the output data arrays
        if band_count == 1:
            data_arrays = [np.full((height, width), 0.5, dtype=np.float32)]
        elif band_count == 2:
            label_band = np.full((height, width), -1, dtype=np.float32)
            score_band = np.zeros((height, width), dtype=np.float32)
            data_arrays = [label_band, score_band]
        else:
            raise ValueError(f"Unsupported band_count={band_count}. Only 1 or 2 are allowed.")

        # Write the new GeoTIFF
        with rasterio.open(
                existing_map_path,
                'w',
                driver='GTiff',
                height=height,
                width=width,
                count=band_count,
                dtype=np.float32,
                crs=CRS.from_epsg(crs_epsg),
                transform=transform_,
        ) as dst:
            for i, band_data in enumerate(data_arrays, start=1):
                dst.write(band_data, i)

            # If one-band TIFF and a DEM is provided, sample elevation at each polygon vertex
            if band_count == 1 and dem_path is not None:
                polygon_with_elev = []
                with rasterio.open(dem_path) as dem_src:
                    # For each (lon, lat) vertex, sample the DEM for elevation.
                    for lon, lat in new_polygon_coords:
                        # Sample returns a generator; get the first value from the first band
                        sample = list(dem_src.sample([(lon, lat)]))
                        if sample and sample[0].size > 0:
                            elev_value = sample[0][0]
                            try:
                                elev_value = int(elev_value)
                            except Exception as conv_e:
                                logger.error(f"Error converting elevation value: {conv_e}", exc_info=True)
                                elev_value = None
                        else:
                            elev_value = None
                        polygon_with_elev.append([lon, lat, elev_value])
                # Update the TIFF metadata with the polygon coordinates including elevation.
                dst.update_tags(georef_data=json.dumps(polygon_with_elev))

        # Open the file with GDAL and return the dataset
        return gdal.Open(existing_map_path, gdal.GA_ReadOnly)

    except Exception as e:
        logger.error(f"Error in get_roi: {e}", exc_info=True)
        return None


# def get_roi(
#     new_polygon_coords,
#     existing_map_path,
#     resolution,
#     crs_epsg,
#     band_count
# ):
#     """
#     Create a GeoTIFF based on polygon coordinates with a specified resolution.
#
#     If band_count=1, we create an OGM with each pixel = 0.5.
#     If band_count=2, we create an OGM with:
#       - Band 1 (label): default -1
#       - Band 2 (score): default 0.0
#
#     Args:
#         new_polygon_coords (list): Polygon coordinates [(lon, lat), ...].
#         existing_map_path (str): Path for saving the GeoTIFF.
#         resolution (float): Desired resolution in meters per pixel (converted to degrees if EPSG:4326).
#         crs_epsg (int): EPSG code for CRS (default 4326).
#         band_count (int): Number of bands. 1 or 2.
#
#     Returns:
#         data_set (gdal.Dataset): GDAL dataset, or None if an error occurs.
#     """
#     try:
#         if resolution <= 0:
#             raise ValueError("Resolution must be a positive value.")
#
#         polygon = Polygon(new_polygon_coords)
#         if not polygon.is_valid:
#             polygon = polygon.buffer(0)
#             if not polygon.is_valid:
#                 raise ValueError("Polygon is invalid and could not be fixed.")
#
#         minx, miny, maxx, maxy = polygon.bounds
#
#         # Handle coordinate conversions if we're in EPSG:4326
#         if crs_epsg == 4326:
#             center_lat = (miny + maxy) / 2
#             meters_per_degree_lat = 111320.0
#             meters_per_degree_lon = 111320.0 * math.cos(math.radians(center_lat))
#
#             lat_res_deg = resolution / meters_per_degree_lat
#             lon_res_deg = resolution / meters_per_degree_lon
#
#             width = max(1, int((maxx - minx) / lon_res_deg))
#             height = max(1, int((maxy - miny) / lat_res_deg))
#         else:
#             # If using a projected coordinate system (in meters)
#             width = max(1, int((maxx - minx) / resolution))
#             height = max(1, int((maxy - miny) / resolution))
#
#         transform_ = from_bounds(minx, miny, maxx, maxy, width, height)
#
#         # Create bands based on band_count
#         if band_count == 1:
#             data_arrays = [
#                 np.full((height, width), 0.5, dtype=np.float32)
#             ]
#         elif band_count == 2:
#             # Two-band OGM:
#             #   Band 1 (labels): default -1
#             #   Band 2 (scores): default 0.0
#             label_band = np.full((height, width), -1, dtype=np.float32)
#             score_band = np.zeros((height, width), dtype=np.float32)
#             data_arrays = [label_band, score_band]
#         else:
#             raise ValueError(f"Unsupported band_count={band_count}. Only 1 or 2 are allowed.")
#
#         with rasterio.open(
#             existing_map_path,
#             'w',
#             driver='GTiff',
#             height=height,
#             width=width,
#             count=band_count,
#             dtype=np.float32,
#             crs=CRS.from_epsg(crs_epsg),
#             transform=transform_,
#         ) as dst:
#             for i, band_data in enumerate(data_arrays, start=1):
#                 dst.write(band_data, i)
#
#         return gdal.Open(existing_map_path, gdal.GA_ReadOnly)
#
#     except Exception as e:
#         logger.error(f"Error in get_roi: {e}")
#         return None

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

        # logger.info(f"Retrieved ROI polygon: {roi_polygon}")

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

        # logger.info(f"Generated polygon coordinates: {polygon_coords_}")
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
        ND_entity_ID = config.ENTITY_Maps4Fire_ID + f'{global_cache.get("bm_id")}'
        # logger.info(f"Set ND_entity_ID for Fire: {ND_entity_ID}")
    elif natural_disaster == 'Flood':
        ND_entity_ID = config.ENTITY_Maps4Flood_ID + f'{global_cache.get("bm_id")}'
        # logger.info(f"Set ND_entity_ID for Flood: {ND_entity_ID}")
    else:
        logger.warning(f"Unrecognized natural disaster type: {natural_disaster}")
        ND_entity_ID = None

    # Set the object entity ID
    obj_entity_ID = config.ENTITY_Maps4Object_ID + f'{global_cache.get("bm_id")}'

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
                    [-180.0, -90.0],
                    [-180.0, 90.0],
                    [180.0, 90.0],
                    [180.0, -90.0],
                    [-180.0, -90.0]
                ]
            ]
        },
        # "location": {
        #     "type": "Polygon",
        #     "coordinates": [
        #         [
        #             [-0.165825, 51.495065],
        #             [-0.165825, 51.513123],
        #             [-0.111065, 51.513123],
        #             [-0.111065, 51.495065],
        #             [-0.165825, 51.495065]
        #         ]
        #     ]
        # },
        "minio_url": {
            "type": "Property",
            "value": f'https://{config.MINIO_ENDPOINT}/{config.BUCKET_NAME}/{global_cache.get("bm_id")}/occupancy_grid_map{natural_disaster}.tif'
        },
        "filename": {
            "type": "Property",
            "value": f"occupancy_grid_map_{natural_disaster}.tif"
        },
        "bucket": {
            "type": "Property",
            "value": "use"
        },
        "sent": {
            "type": "Property",
            "value": time.strftime('%Y-%m-%dT%H:%M:%SZ', time.localtime())
        },
        # "ignitionPoints":{
        #     "type": "GeoProperty",
        #     "value": {
        #         "type": "Point",
        #         "coordinates": [0.0, 0.0]
        #     }
        # },
        "bm_id": global_cache.get("bm_id"),
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
        "bm_id": payload["bm_id"],
        "minio_url": payload["minio_url"],
        "filename": payload["file_name"],
        "bucket": payload["bucket"],
        "sent": payload["sent"],
        # "ignitionPoints": {
        #     "type": "GeoProperty",
        #     "value": {
        #         "type": "Point",
        #         "coordinates": payload["ignitionPoints"]
        #     }
        # },
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
                mask_ = elevation_data != nodata_value
            else:
                mask_ = np.ones_like(elevation_data, dtype=bool)

            # Initialize a transformer for coordinate conversion
            transformer = Transformer.from_crs(dataset_crs, "EPSG:4326", always_xy=True)

            # Get pixel indices
            rows, cols = np.where(mask_)
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

        # logger.info(f"Generated geo dictionary with {len(geo_dict_list)} entries.")
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
        # logger.info(f"Checking for existing geo dict: {json_file}")
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
            ##############################################
            geojson = {
                "type": "FeatureCollection",
                "features": []
            }
            for obj in new_geo_dict:
                coords = obj.get("geo_coords", None)
                if coords and len(coords) >= 2:
                    feature = {
                        "type": "Feature",
                        "geometry": {
                            "type": "Point",
                            "coordinates": coords  # e.g. [lon, lat, elev] is okay
                        },
                        "properties": {
                            "score": obj.get("score", 0.0),
                            "label": obj.get("label", -1)
                        }
                    }
                    geojson["features"].append(feature)
            with open(json_file, 'w') as f:
                json.dump(geojson, f, indent=4)
                # logger.info(f"Geo dict saved to {json_file}")
                print(f"Geo dict saved to ----> {json_file}")
            ##############################################
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
    ##############################################################################
    data, geo_transform, spatial_ref, measurement_type, observ_nodata = None, [None], None, None, None

    ##############################################################################
    # logger.info(f"Loading image from {image_path} with mode {mode}")

    def process_observation(observation_file, temp):
        nonlocal data, geo_transform, spatial_ref, measurement_type
        # logger.info(f"Processing observation: {observation_file}")
        # Open the dataset with rasterio using a context manager.
        if temp != 1:
            with rasterio.open(observation_file) as src:
                # Read the first band of data.
                data = src.read(1).astype(np.float32)
                # Get the geo-transform (affine transform in rasterio)
                geo_transform = src.transform
                geo_transform = geo_transform.to_gdal()
                # Get the spatial reference (CRS)
                spatial_ref = src.crs
                observ_nodata_ = src.nodata
                if observ_nodata_ is None and src.nodatavals:
                    observ_nodata_ = src.nodatavals[0]

            ################################
                # if spatial_ref is None:
                #     logger.warning("WARNING: CRS is None, attempting to determine from file properties")
                #     # Check if this looks like Web Mercator based on coordinates
                #     origin_x = geo_transform[0]
                #     if abs(origin_x) > 180:  # Likely Web Mercator (meters)
                #         spatial_ref = 'EPSG:3857'
                #         logger.info(f"Assuming EPSG:3857 based on coordinate values: {origin_x}")
                #     else:
                #         spatial_ref = 'EPSG:4326'  # Default to geographic
                #         logger.info("Assuming EPSG:4326 as default")
                # else:
                #     logger.info(f"spatial_ref {spatial_ref} for {observation_file}")
                ################################

            return
        elif temp == 1:
            with rasterio.open(observation_file) as src:
                # Read the first band.
                data = src.read(1)
                # Get the affine transform and convert it to GDAL format:
                gt = src.transform.to_gdal()  # (origin_x, pixel_width, skew_x, origin_y, skew_y, pixel_height)
                origin_x, pixel_width, skew_x, origin_y, skew_y, pixel_height = gt

                # --- Correct vertical orientation if needed ---
                # For a north-up image, pixel_height should be negative.
                if pixel_height > 0:
                    # logger.info("Pixel height is positive, flipping data vertically.")
                    data = np.flipud(data)
                    # Adjust the origin_y: new_origin_y = origin_y + (pixel_height * number_of_rows)
                    nrows = data.shape[0]
                    origin_y = origin_y + pixel_height * nrows
                    pixel_height = -pixel_height

                # --- Correct horizontal orientation if needed ---
                # For a north-up image, pixel_width is expected to be positive.
                if pixel_width < 0:
                    # logger.info("Pixel width is negative, flipping data horizontally.")
                    data = np.fliplr(data)
                    ncols = data.shape[1]
                    origin_x = origin_x + pixel_width * ncols
                    pixel_width = -pixel_width

                # Reassemble the geotransform with corrected values.
                geo_transform = (origin_x, pixel_width, skew_x, origin_y, skew_y, pixel_height)
                # Get the spatial reference.
                spatial_ref = src.crs
                observ_nodata_ = src.nodata
                if observ_nodata_ is None and src.nodatavals:
                    observ_nodata_ = src.nodatavals[0]
            return

    if mode == 0:  # Load previous OGM
        process_observation(image_path, 0)
        measurement_type = 'OGM'
    elif mode == 1:  # Load the Observation of geo-referenced segmented drone image TFA-06
        # for observation in sorted(os.listdir(image_path), reverse=False):
        for observation in sorted(os.listdir(image_path), key=lambda x: os.path.getmtime(os.path.join(image_path, x))):
            if observation.endswith("_Segment.tif"):
                if 'Fire' in observation:
                    measurement_type = 'active_fire'
                elif 'Burnt' in observation:
                    measurement_type = 'burnt_area'
                elif 'Flood' in observation:
                    measurement_type = 'Flood'
                logger.info(f"processing segmented drone images {observation}")
                full_path = os.path.join(image_path, observation)
                process_observation(os.path.join(image_path, observation), 0)
                ##############################################################

                ##############################################################
                # Clean up the old temporary file if it exists
                if os.path.exists(full_path):
                    os.remove(full_path)
                break

    elif mode == 2:  # Load the Observation of geo-referenced object drone image TFA-05
        try:
            for observation in sorted(os.listdir(image_path), reverse=False):
                if observation.endswith("_Objects.tif"):
                    measurement_type = 'PersonVehicleDetection'
                    process_observation(os.path.join(image_path, observation), 2)
                    ##############################################################
                    # Clean up the old temporary file if it exists
                    ##############################################################
                    full_path = os.path.join(image_path, observation)
                    if os.path.exists(full_path):
                        os.remove(full_path)
                    break
        except Exception as e:
            logger.info(f"mode 2 error due to {e}")

    elif mode == 3:  # Load the Observation of ROI satellite images TFA-08/09
        try:
            for observation in sorted(os.listdir(image_path), reverse=False):
                if observation.endswith(".tif"):
                    disaster_type = global_cache.get('natural_disaster', 'No disaster info')
                    if disaster_type == 'Flood':
                        measurement_type = 'Flood'
                    elif disaster_type == 'Fire':
                        measurement_type = 'burnt_area'
                    logger.info(f"processing segmented satellite images {observation}")
                    process_observation(os.path.join(image_path, observation), 3)
                    ##############################################################
                    # Clean up the old temporary file if it exists
                    ##############################################################
                    full_path = os.path.join(image_path, observation)
                    if os.path.exists(full_path):
                        os.remove(full_path)
                    break
        except Exception as e:
            logger.info(f"mode 3 error due to {e}")

    elif mode == 4:  # Load the Observation of predictive models
        try:
            for observation in sorted(os.listdir(image_path), reverse=False):
                if observation.endswith((".tif", ".kmz")):
                    process_observation(os.path.join(image_path, observation), 4)
                    ##############################################################
                    # Clean up the old temporary file if it exists
                    ##############################################################
                    full_path = os.path.join(image_path, observation)
                    if os.path.exists(full_path):
                        os.remove(full_path)
                    break
        except Exception as e:
            logger.info(f"mode 4 error due to {e}")

    return data, geo_transform, spatial_ref, measurement_type, observ_nodata


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
    Extract and transform metadata from GeoTIFF files with GDAL,
    returning a GeoJSON FeatureCollection of metadata.

    Parameters:
        geotiff_path (str): Path to a directory containing GeoTIFF files or a single GeoTIFF file.

    Returns:
        dict: A GeoJSON FeatureCollection of the transformed metadata from the first valid file found.
        None: If no valid files are found or no metadata is extracted.
    """
    # 1) Determine which files to process
    if os.path.isdir(geotiff_path):
        file_paths = [
            os.path.join(geotiff_path, f)
            for f in os.listdir(geotiff_path)
            if f.endswith("_Objects.tif")
        ]
    else:
        file_paths = [geotiff_path] if geotiff_path.endswith("_Objects.tif") else []

    if not file_paths:
        logger.info("No valid geo_referenced drone files found.")

        return None

    # 2) Iterate over files in sorted order
    for file_path in sorted(file_paths, reverse=False):
        dataset = gdal.Open(file_path)
        if dataset is None:
            print(f"Unable to open {file_path}")
            continue

        # 3) Extract general metadata from the dataset
        metadata = dataset.GetMetadata()
        if 'georef_data' not in metadata:
            print(f"'georef_data' field missing in metadata for {file_path}")
            continue

        try:
            data = json.loads(metadata['georef_data'])
        except json.JSONDecodeError as e:
            logger.warning(f"Error decoding JSON from georef_data in {file_path}: {e}")
            continue

        # 4) Convert 'data' into a GeoJSON FeatureCollection
        feature_collection = {
            "type": "FeatureCollection",
            "features": []
        }

        # 'data' is presumably a list of dicts
        if not isinstance(data, list):
            logger.warning(f"Unexpected structure for 'georef_data' in {file_path}. Expected a list.")
            continue

        # 5) Build the features
        for item in data:
            # Each item is a dict with one "pixel_coords" key (like "1,114") plus 'score', 'label', etc.
            if not isinstance(item, dict):
                logger.warning(f"Invalid metadata format in {file_path}, skipping item: {item}")
                continue

            # Extract the "pixel coordinate" key (e.g. "1,114")
            pixel_keys = [k for k in item.keys() if "," in k and k not in ("score", "label")]
            if not pixel_keys:
                logger.warning(f"No valid pixel coordinate key in item: {item}")
                continue

            # Let's just take the first such key
            key = pixel_keys[0]
            try:
                row, col = map(int, key.split(","))
                pixel_coords = [row, col]
            except ValueError:
                logger.warning(f"Invalid pixel coordinate format in key: {key}")
                continue

            # Extract geo_coords ([lon, lat, elev]) from the item
            coordinates = item[key]
            if not (isinstance(coordinates, list) and len(coordinates) >= 2):
                logger.warning(f"Invalid geo_coords in item: {item}")
                continue

            # We handle up to 3 coords: [lon, lat, elev]
            # If there's a third, keep it; otherwise set elev=0
            lon = float(coordinates[0])
            lat = float(coordinates[1])
            elev = float(coordinates[2]) if len(coordinates) > 2 else 0.0

            # Extract score/label
            score = item.get('score', 0.0)
            label = item.get('label', -1)

            # Build a GeoJSON Feature
            feature = {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [lon, lat, elev]
                },
                "properties": {
                    "pixel_coords": pixel_coords,
                    "score": score,
                    "label": label
                }
            }
            feature_collection["features"].append(feature)
        # delete the file_path
        ##################################################################################
        if os.path.exists(file_path):
            os.remove(file_path)
        ##################################################################################
        # 6) If we found any features, return the FeatureCollection from the first valid file
        if feature_collection["features"]:
            return feature_collection
        else:
            logger.info(f"No valid features extracted from file: {file_path}")

    # If we reach here, we didn't extract anything from any file
    logger.info("No metadata extracted from valid files.")
    return None


def save_geotiff(output_path_maps_, FileName, data, GTransform, timestamp, crs_epsg=4326):
    """
    Save data as a GeoTIFF file.

    Args:
        output_path_maps_ (str): Directory to save the GeoTIFF file.
        FileName (str): Name of the GeoTIFF file.
        data (ndarray): 2D array of the data to save.
        GTransform (list): GeoTransform values [XMin, XRes, 0, YMax, 0, YRes].
        timestamp: current time
        crs_epsg (int): EPSG code for the CRS.

    Returns:
        dict: Metadata of the saved GeoTIFF.
    """
    try:
        # Ensure data is a numpy array
        data = np.array(data)
        driver = gdal.GetDriverByName('GTiff')
        if not driver:
            logger.error('GTiff driver is not available.')
            return None

        # Validate GeoTransform
        if len(GTransform) != 6:
            raise ValueError("GTransform must contain exactly six elements.")

        logger.debug(f"Input GTransform: {GTransform}")
        logger.debug(f"Input CRS EPSG: {crs_epsg}")

        # Create the dataset
        height, width = data.shape
        output_file_path = os.path.join(output_path_maps_, FileName)
        dataset_ogm = driver.Create(output_file_path, width, height, 1, gdal.GDT_Float32)
        if not dataset_ogm:
            logger.error("Failed to create GeoTIFF dataset.")
            return None

        dataset_ogm.SetGeoTransform(GTransform)

        # Set CRS
        srs = osr.SpatialReference()
        if crs_epsg:
            srs.ImportFromEPSG(crs_epsg)
            dataset_ogm.SetProjection(srs.ExportToWkt())
        else:
            raise ValueError("CRS EPSG code is not defined.")

        logger.debug(f"CRS set to: EPSG:{crs_epsg}")

        # Write data to the raster band
        band_ = dataset_ogm.GetRasterBand(1)
        band_.WriteArray(data)
        band_.SetDescription('Estimated OGM')

        natural_disaster = global_cache.get('natural_disaster', 'No disaster info')
        # Define metadata
        metadata = {
            'title': 'Probabilistic Occupancy Grid Mapping for Dynamic Environments of ND',
            'author': 'GRVC lab, University of Seville',
            'description': f"Probabilistic occupancy grid map for the {natural_disaster} scenario, enabling near real-time estimation and tracking of disaster propagation.",
            'creationDate': datetime.now().isoformat(),  # Current date and time
            "XMin": GTransform[0],
            "XRes": GTransform[1],
            "YMax": GTransform[3],
            "YRes": GTransform[5],
            "spatialReference": f"EPSG:{crs_epsg}",
            "file_name": f'pdm05/{global_cache.get("bm_id")}/{FileName}',
            "coordinates": global_cache.get('roi', [None]),
            "bucket": "use",
            "bm_id": global_cache.get("bm_id"),
            "sent": timestamp,
            "ignitionPoints": global_cache.get('ignitionPoints', [None]),
            "minio_url": f'https://{config.MINIO_ENDPOINT}/{config.BUCKET_NAME}/pdm05/{global_cache.get("bm_id")}/{FileName}'

        }
        dataset_ogm.SetMetadata(metadata)

        logger.debug(f"GeoTIFF metadata: {metadata}")

        # Flush cache and close dataset
        dataset_ogm.FlushCache()
        dataset_ogm = None

        # logger.info(f"GeoTIFF saved successfully at {output_file_path}")
        return metadata

    except Exception as e:
        logger.error(f"Error saving GeoTIFF: {e}")
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
    return x, y


def coordinates_to_pixel_update(geo_transform, x_coord, y_coord):
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


#############################################################################
# def update_occupancy_grid_flood(OGMData,
#                                 OGM_gt_,
#                                 observation_data_,
#                                 observation_gt_):
#     """
#     Update the occupancy grid map using satellite and drone measurements.
#
#     Parameters:
#         OGMData : Occupancy grid data.
#         OGM_gt_ : Geotransform for the occupancy grid.
#         observation_data_ : Observation data (e.g., drone or satellite image).
#         observation_gt_: Geotransform for the observation.
#
#     Returns:
#         np.ndarray: Updated occupancy grid map.
#     """
#
#     epsilon = 1e-9  # Small value to avoid log(0) or division by zero
#     grid_height, grid_width = OGMData.shape
#
#     for y in range(grid_height):
#         for x in range(grid_width):
#             # Convert grid coordinates to geospatial coordinates
#             # y_geo, x_geo = pixel_to_coordinates(OGM_gt_, x, y)
#             x_geo, y_geo = pixel_to_coordinates(OGM_gt_, y, x)
#             # Map geospatial coordinates back to pixel coordinates
#             ogm_pixel = coordinates_to_pixel_update(OGM_gt_, x_geo, y_geo)
#             observation_pixel = coordinates_to_pixel_update(observation_gt_, x_geo, y_geo)
#
#             # Bounds checking
#             if not (0 <= ogm_pixel[0] < grid_width and 0 <= ogm_pixel[1] < grid_height):
#                 continue
#             if not (0 <= observation_pixel[0] < observation_data_.shape[1] and
#                     0 <= observation_pixel[1] < observation_data_.shape[0]):
#                 continue
#
#             ogm_y, ogm_x = int(ogm_pixel[1]), int(ogm_pixel[0])
#             prior_prob = OGMData[ogm_y, ogm_x]
#
#             obs_y, obs_x = int(observation_pixel[1]), int(observation_pixel[0])
#             likelihood = observation_data_[obs_y, obs_x]
#             likelihood = 0.5 + 0.5 * (likelihood - 0.1)
#
#             if 0 < likelihood <= 1:  # Valid probability range
#                 # Compute log-odds for prior and observation
#                 log_odds_prior = math.log((prior_prob + epsilon) / (1 - prior_prob + epsilon))
#                 log_odds_obs = math.log((likelihood + epsilon) / (1 - likelihood + epsilon))
#
#                 # decay_factor = 0.95
#                 log_odds_updated = log_odds_prior + log_odds_obs
#
#                 # Update log-odds
#                 # log_odds_updated = log_odds_prior + log_odds_obs
#
#                 # Clamp log-odds to avoid extreme probabilities
#                 log_odds_clamped = np.clip(log_odds_updated, -5, 5)
#
#                 # Convert log-odds back to probability
#                 updated_prob = 1 - (1 / (1 + np.exp(log_odds_clamped)))
#
#                 # Update the OGM cell
#                 OGMData[ogm_y, ogm_x] = updated_prob
#
#             # Clamp final probabilities to [0, 1] to prevent numerical errors
#             OGMData[ogm_y, ogm_x] = np.clip(OGMData[ogm_y, ogm_x], 0, 1)
#
#     return OGMData

def update_occupancy_grid_flood(OGMData, OGM_gt_,
                                observation_data_, observation_gt_):
    """
    Update the occupancy grid map (OGMData) using the observation (satellite or drone data)
    over the area covered by the observation. This function:
      1. Determines the geographic extent of the observation.
      2. Extracts the corresponding patch from OGMData.
      3. Upsamples the OGM patch to the observation's resolution.
      4. Performs a log-odds update using the observation.
      5. Downsamples the updated patch and reintegrates it into the full OGMData.

    Parameters:
      - OGMData : 2D numpy array (low-res occupancy grid).
      - OGM_gt_ : Geotransform tuple for the OGM.
      - observation_data_ : 2D numpy array (observation, values in [0,1]).
      - observation_gt_ : Geotransform tuple for the observation.

    Returns:
      - Updated OGMData (same dimensions as the input).
    """
    epsilon = 1e-9
    # observation_data_ = np.flip(observation_data_,axis=(0, 1))

    # --- 1. Compute the geographic extent of the observation ---
    obs_rows, obs_cols = observation_data_.shape  # high-res dimensions
    obs_origin_x = observation_gt_[0]
    obs_origin_y = observation_gt_[3]
    obs_pixel_width = observation_gt_[1]
    obs_pixel_height = observation_gt_[5]  # typically negative for north-up images

    # Calculate geographic bounds of the observation:
    obs_x_min = obs_origin_x
    obs_y_max = obs_origin_y  # top (max latitude)
    obs_x_max = obs_origin_x + obs_pixel_width * obs_cols
    obs_y_min = obs_origin_y + obs_pixel_height * obs_rows
    

    # --- 2. Determine the corresponding OGM patch ---
    # Convert the observation's geographic corners to OGM pixel indices.
    top_left_ogm = coordinates_to_pixel_update(OGM_gt_, obs_x_min, obs_y_max)
    bottom_right_ogm = coordinates_to_pixel_update(OGM_gt_, obs_x_max, obs_y_min)

    # Ensure proper ordering (min/max rows and columns)
    ogm_row_min = min(top_left_ogm[0], bottom_right_ogm[0])
    ogm_row_max = max(top_left_ogm[0], bottom_right_ogm[0])
    ogm_col_min = min(top_left_ogm[1], bottom_right_ogm[1])
    ogm_col_max = max(top_left_ogm[1], bottom_right_ogm[1])

    # Extract the patch from OGMData.
    ogm_patch = OGMData[ogm_row_min:ogm_row_max + 1, ogm_col_min:ogm_col_max + 1]
    # logger.info(f"ogm_patch:  {ogm_patch}")

    # --- 3. Upsample the OGM patch to observation resolution ---
    # We want the high-res patch to have the same dimensions as the observation.
    highres_rows = obs_rows
    highres_cols = obs_cols
    ogm_patch_upsampled = cv2.resize(ogm_patch, (highres_cols, highres_rows), interpolation=cv2.INTER_LINEAR)

    # Define a new geotransform for the upsampled patch that exactly covers the observation's geographic area.
    patch_gt = (obs_x_min, obs_pixel_width, 0,
                obs_y_max, 0, obs_pixel_height)

    # --- 4. Fusion update at high resolution ---
    for j in range(highres_rows):  # row index in high-res patch
        for i in range(highres_cols):  # col index in high-res patch
            # Convert the high-res pixel (j, i) into geographic coordinates.
            x_geo, y_geo = pixel_to_coordinates(patch_gt, j, i)

            # Map these geographic coordinates to observation pixel indices.
            obs_pix = coordinates_to_pixel_update(observation_gt_, x_geo, y_geo)
            if not (0 <= obs_pix[0] < obs_rows and 0 <= obs_pix[1] < obs_cols):
                continue

            # Retrieve the observation value.
            # obs_val = observation_data_[obs_pix[0], obs_pix[1]]
            value = observation_data_[obs_pix[0], obs_pix[1]]
            if value <= 0:
                continue
            obs_val = value

            # Compute likelihood based on the given formula:
            likelihood = 0.5 + 0.5 * (obs_val - 0.1)

            # Retrieve current probability from the upsampled patch.
            prior_prob = ogm_patch_upsampled[j, i]

            # Log-odds update, if likelihood is in a valid range.
            if 0 < likelihood <= 1:
                log_odds_prior = math.log((prior_prob + epsilon) / (1 - prior_prob + epsilon))
                log_odds_obs = math.log((likelihood + epsilon) / (1 - likelihood + epsilon))
                log_odds_updated = log_odds_prior + log_odds_obs
                log_odds_clamped = np.clip(log_odds_updated, -5, 5)
                # Flood update formula: note the inversion in the final step.
                updated_prob = 1 - (1 / (1 + np.exp(log_odds_clamped)))
                ogm_patch_upsampled[j, i] = updated_prob

            ogm_patch_upsampled[j, i] = np.clip(ogm_patch_upsampled[j, i], 0, 1)

    # --- 5. Downsample the high-resolution patch back to original OGM patch resolution ---
    orig_patch_rows, orig_patch_cols = ogm_patch.shape
    fused_patch_downsampled = cv2.resize(ogm_patch_upsampled, (orig_patch_cols, orig_patch_rows),
                                         interpolation=cv2.INTER_AREA)

    # --- 6. Replace the corresponding region in the full OGMData ---
    updated_OGM = OGMData.copy()
    updated_OGM[ogm_row_min:ogm_row_max + 1, ogm_col_min:ogm_col_max + 1] = fused_patch_downsampled

    # --- 7. Return the updated full OGMData ---
    return updated_OGM


#############################################################################
# FUSE SAT DATA
#############################################################################
# helpers to convert between GDAL-style geotransform and Affine
def _gt_to_affine(gt):
    """GDAL geotransform (x0, a, b, y0, d, e) -> Affine(a, b, x0, d, e, y0)."""
    return Affine(gt[1], gt[2], gt[0], gt[4], gt[5], gt[3])


def update_occupancy_grid_flood_sat_fixed(
        OGMData,
        OGM_gt_,
        observation_data_,
        observation_gt_,
        ogm_crs: Union[str, RioCRS] = "EPSG:4326",
        observation_crs: Union[str, RioCRS, None] = None,
        obs_nodata=None):

    if observation_crs is None:
        raise ValueError("observation_crs is None. Pass src.crs or an EPSG string (e.g., 'EPSG:3857').")

    """
    Reproject the observation *onto the OGM grid* and fuse in log-odds space.
    Returns an updated grid with the exact same shape as OGMData.
    """
    epsilon = 1e-9
    # logger.info(f"Starting fusion (simple): OGM CRS={ogm_crs}, Observation CRS={observation_crs}")
    # logger.info(f"OGM shape: {OGMData.shape}, Observation shape: {observation_data_.shape}")

    try:
        # Normalize CRS inputs
        ogm_crs = RioCRS.from_user_input(ogm_crs)
        observation_crs = RioCRS.from_user_input(observation_crs)

        # 1) Prepare observation as float and mask its NoData to NaN
        obs = observation_data_.astype(np.float32, copy=True)
        if obs_nodata is not None and np.isfinite(obs_nodata):
            nodata_mask = (observation_data_ == obs_nodata)
            if nodata_mask.any():
                obs[nodata_mask] = np.nan
            logger.info(f"[FUSE] NoData in OBS: {int(nodata_mask.sum())}/{observation_data_.size} (value={obs_nodata})")
        # else:
        #     logger.info("[FUSE] obs_nodata=None (no numeric NoData masked before reprojection)")

        # 2) Reproject observation *directly to OGM grid* (same transform, CRS, and shape)
        obs_on_ogm = np.full_like(OGMData, np.nan, dtype=np.float32)
        reproject(
            source=obs,
            destination=obs_on_ogm,
            src_transform=_gt_to_affine(observation_gt_),
            src_crs=observation_crs,
            dst_transform=_gt_to_affine(OGM_gt_),
            dst_crs=ogm_crs,
            resampling=Resampling.bilinear,
            src_nodata=obs_nodata,
            dst_nodata=np.nan,
        )

        # 3) Bayesian fusion on the aligned grid
        prior = np.clip(OGMData.astype(np.float32), 0.0, 1.0)

        # Your likelihood mapping
        likelihood = 0.5 + 0.5 * (obs_on_ogm - 0.1)
        likelihood = np.clip(likelihood, epsilon, 1.0 - epsilon)

        # Keep original gating behavior: update only where obs > 0 and finite
        mask_valid = np.isfinite(obs_on_ogm) & (obs_on_ogm > 0)
        if not np.any(mask_valid):
            logger.info("No valid observation pixels after re-projection; returning OGM unchanged.")
            return prior  # ensure dtype/clip

        # log-odds update (clamped like your code)
        log_odds_prior = np.log((prior[mask_valid] + epsilon) / (1.0 - prior[mask_valid] + epsilon))
        log_odds_obs   = np.log(likelihood[mask_valid] / (1.0 - likelihood[mask_valid]))
        log_odds_upd   = np.clip(log_odds_prior + log_odds_obs, -5.0, 5.0)
        updated_prob   = 1.0 / (1.0 + np.exp(-log_odds_upd))  # sigmoid

        # 4) Write back to full OGM grid and return
        updated_OGM = prior.copy()
        updated_OGM[mask_valid] = np.clip(updated_prob, 0.0, 1.0)
        logger.info("Fusion completed successfully (SAT measurement).")
        return updated_OGM

    except Exception as e:
        logger.info(f"Error in flood satellite fusion: {e}")
        return OGMData
#############################################################################
#############################################################################
def update_occupancy_grid_fire(OGMData, OGM_gt_,
                               measurement_data, measurement_gt_,
                               measurement_type):
    """
    Fuses ONE drone measurement into OGMData.

    This version only increases the spatial resolution of the OGM in the
    geographic area covered by the drone measurement:
      1. Compute the drone measurement geographic extent.
      2. Extract the corresponding OGM patch.
      3. Upsample that patch to the drone measurement resolution.
      4. Fuse the drone measurement using the log-odds update (performed at high resolution).
      5. Downsample the fused patch back to the original OGM patch resolution.
      6. Replace the corresponding area in OGMData.

    Parameters:
      - OGMData: 2D numpy array for the low-res occupancy grid.
      - OGM_gt_: GeoTransform for the OGM (tuple of 6 numbers).
      - measurement_data: 2D numpy array (drone measurement, values in [0,1]).
      - measurement_gt_: GeoTransform for the drone measurement image.
      - measurement_type: either "active_fire" or "burnt_area".

    Returns:
      - Updated OGMData (of the same shape as the input) with the fused measurement.
    """
    epsilon = 1e-9

    # --- 1. Compute the geographic extent of the drone measurement ---
    meas_rows, meas_cols = measurement_data.shape  # drone image dimensions
    # measurement_gt_ is assumed to be: (origin_x, pixel_width, skew_x, origin_y, skew_y, pixel_height)
    meas_origin_x = measurement_gt_[0]
    meas_origin_y = measurement_gt_[3]
    meas_pixel_width = measurement_gt_[1]
    meas_pixel_height = measurement_gt_[5]  # likely negative (north-up)

    # Compute the geographic bounds of the drone image:
    # Top-left corner:
    meas_x_min = meas_origin_x
    meas_y_max = meas_origin_y
    # Bottom-right corner:
    meas_x_max = meas_origin_x + meas_pixel_width * meas_cols
    meas_y_min = meas_origin_y + meas_pixel_height * meas_rows

    # --- 2. Determine the corresponding OGM patch in OGMData ---
    # Use your coordinates_to_pixel_update function (which returns (row, col)) on the corners.
    top_left_ogm = coordinates_to_pixel_update(OGM_gt_, meas_x_min, meas_y_max)
    bottom_right_ogm = coordinates_to_pixel_update(OGM_gt_, meas_x_max, meas_y_min)

    # Ensure proper ordering (rows and cols):
    ogm_row_min = min(top_left_ogm[0], bottom_right_ogm[0])
    ogm_row_max = max(top_left_ogm[0], bottom_right_ogm[0])
    ogm_col_min = min(top_left_ogm[1], bottom_right_ogm[1])
    ogm_col_max = max(top_left_ogm[1], bottom_right_ogm[1])

    # Extract the patch from OGMData.
    ogm_patch = OGMData[ogm_row_min:ogm_row_max + 1, ogm_col_min:ogm_col_max + 1]

    # --- 3. Upsample the OGM patch to drone resolution ---
    # The high-res patch dimensions will match the drone measurement dimensions.
    highres_rows = meas_rows
    highres_cols = meas_cols
    ogm_patch_upsampled = cv2.resize(ogm_patch, (highres_cols, highres_rows), interpolation=cv2.INTER_LINEAR)

    # Define a new geotransform for the upsampled patch.
    # Here we use the drone measurement's geographic extent.
    patch_gt = (meas_x_min, meas_pixel_width, 0,
                meas_y_max, 0, meas_pixel_height)

    # --- 4. Fusion update at high resolution ---
    # For each pixel in the high-res patch, update based on the drone measurement.
    for j in range(highres_rows):  # row index in upsampled patch
        for i in range(highres_cols):  # col index in upsampled patch
            # Convert high-res pixel (j,i) to geographic coordinates using patch_gt.
            x_geo, y_geo = pixel_to_coordinates(patch_gt, j, i)

            # Map geographic coordinates to drone measurement pixel indices.
            meas_pix = coordinates_to_pixel_update(measurement_gt_, x_geo, y_geo)
            # Check bounds in the drone measurement:
            if not (0 <= meas_pix[0] < meas_rows and 0 <= meas_pix[1] < meas_cols):
                continue
            sensor_val = measurement_data[meas_pix[0], meas_pix[1]]  # in [0,1]

            # Get current probability from upsampled OGM patch.
            if sensor_val > 0:
                prior_prob = ogm_patch_upsampled[j, i]

                # Convert sensor value to likelihood.
                if measurement_type == "active_fire":
                    likelihood = 0.05 + 0.9 * sensor_val
                elif measurement_type == "burnt_area":
                    likelihood = 1.0 - 0.9 * sensor_val
                else:
                    raise ValueError(f"Unknown measurement_type: {measurement_type}")
                likelihood = np.clip(likelihood, 0.0, 1.0)

                # Log-odds update.
                if 0 < likelihood < 1:
                    log_odds_prior = math.log((prior_prob + epsilon) / (1 - prior_prob + epsilon))
                    log_odds_obs = math.log((likelihood + epsilon) / (1 - likelihood + epsilon))
                    log_odds_sum = log_odds_prior + log_odds_obs
                    log_odds_clamped = np.clip(log_odds_sum, -5, 5)
                    updated_prob = 1.0 / (1.0 + np.exp(-log_odds_clamped))
                    ogm_patch_upsampled[j, i] = updated_prob

                ogm_patch_upsampled[j, i] = np.clip(ogm_patch_upsampled[j, i], 0.0, 1.0)

    # --- 5. Downsample the fused high-resolution patch back to original OGM patch resolution ---
    orig_patch_rows, orig_patch_cols = ogm_patch.shape
    fused_patch_downsampled = cv2.resize(ogm_patch_upsampled, (orig_patch_cols, orig_patch_rows),
                                         interpolation=cv2.INTER_AREA)

    # --- 6. Replace the corresponding region in the full OGMData ---
    updated_OGM = OGMData.copy()  # avoid modifying the original in-place
    updated_OGM[ogm_row_min:ogm_row_max + 1, ogm_col_min:ogm_col_max + 1] = fused_patch_downsampled

    # --- 7. Return the updated full OGMData ---
    return updated_OGM


# def update_occupancy_grid_fire(OGMData,
#                                OGM_gt_,
#                                measurement_data,
#                                measurement_gt_,
#                                measurement_type # "active_fire"  # or "burnt_area"
# ):
#     """
#     Fuses ONE measurement into OGMData, where OGMData represents
#     the probability of *active fire* in each cell.
#
#     measurement_type can be:
#       - "active_fire"
#       - "burnt_area"
#
#     For "active_fire": we interpret high measurement -> high probability of active fire
#     For "burnt_area": we interpret high measurement -> NOT actively burning
#
#     So a burnt_area measurement *lowers* the OGM's probability of active fire.
#     """
#
#     logger.info(f'ogm_geo -- {OGM_gt_}')
#     logger.info(f"measurement_gt_ --{measurement_gt_}")
#     # measurement_gt_ = update_drone_geotransform(measurement_gt_, desired_resolution_m=1.0)
#     # -- 3) Dimensions and constants --
#     height, width = OGMData.shape
#     epsilon = 1e-9
#     ogm_geo = OGM_gt_
#     meas_geo = measurement_gt_
#     # -- 4) Iterate over each cell of OGMData --
#     for y in range(height):
#         for x in range(width):
#             # (y,x) -> geo coords
#             # y_geo, x_geo = pixel_to_coordinates(ogm_geo, y, x)
#             x_geo, y_geo = pixel_to_coordinates(ogm_geo, y, x)
#
#             # geo coords -> pixel in OGM (sanity check)
#             ogm_pixel = coordinates_to_pixel_update(ogm_geo, x_geo, y_geo)
#             # geo coords -> pixel in measurement
#             meas_pixel = coordinates_to_pixel_update(meas_geo, x_geo, y_geo)
#             ##################################################################
#
#             # Check OGM bounds
#             if not (0 <= ogm_pixel[0] < width and 0 <= ogm_pixel[1] < height):
#                 continue
#             # Check measurement bounds
#             if not (0 <= meas_pixel[0] < measurement_data.shape[1] and
#                     0 <= meas_pixel[1] < measurement_data.shape[0]):
#                 continue
#
#             # Indices
#             ogm_y, ogm_x = int(ogm_pixel[1]), int(ogm_pixel[0])
#             prior_prob = OGMData[ogm_y, ogm_x]
#
#             obs_y, obs_x = int(meas_pixel[1]), int(meas_pixel[0])
#             sensor_val = measurement_data[obs_y, obs_x]  # in [0..1]
#
#             # ----------------------------------------------------
#             # 5) Convert sensor_val -> likelihood(cell is actively on fire)
#             #    depending on measurement_type
#             # ----------------------------------------------------
#             if measurement_type == "active_fire":
#                 # High sensor_val => definitely on fire
#                 # Low sensor_val => probably not on fire
#                 # Example: directly use sensor_val in [0..1], with a small offset
#                 likelihood = 0.05 + 0.9 * sensor_val
#
#             elif measurement_type == "burnt_area":
#                 # High sensor_val => cell is burnt => likely NOT actively on fire
#                 # So we invert to get the probability of active fire
#                 # e.g. if sensor_val=1.0 => definitely burnt => 0% chance of still on fire
#                 # if sensor_val=0.0 => no burn => possibly on fire
#                 # We might do:
#                 likelihood = 1.0 - 0.8 * sensor_val
#                 # This means a fully burnt cell => likelihood=0.2 or 0.0, your choice
#
#             else:
#                 # Possibly other measurement types, or error
#                 raise ValueError(f"Unknown measurement_type: {measurement_type}")
#
#             # Clip to valid probability
#             likelihood = np.clip(likelihood, 0.0, 1.0)
#
#             # ----------------------------------------------------
#             # 6) Standard log-odds update
#             # ----------------------------------------------------
#             if 0 < likelihood < 1:
#                 log_odds_prior = math.log((prior_prob + epsilon)/(1 - prior_prob + epsilon))
#                 log_odds_obs   = math.log((likelihood + epsilon)/(1 - likelihood + epsilon))
#                 log_odds_sum   = log_odds_prior + log_odds_obs
#
#                 # clamp log-odds
#                 log_odds_clamped = np.clip(log_odds_sum, -5, 5)
#
#                 updated_prob = 1.0 / (1.0 + np.exp(-log_odds_clamped))
#                 OGMData[ogm_y, ogm_x] = updated_prob
#
#             # final clamp
#             OGMData[ogm_y, ogm_x] = np.clip(OGMData[ogm_y, ogm_x], 0.0, 1.0)
#
#     return OGMData

#############################################################################


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
    Update the occupancy grid using geo-referenced dictionaries.

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


def process_and_upload_ogm(entity_id, file_path_, bucket_name, metadata):
    """
        Process the occupancy grid, update the entity and upload the OGM to MinIO.
    """

    # logger.info(f"inside process_and_upload_ogm entity_id ---> {entity_id}")
    print(f"inside process_and_upload_ogm entity_id ---> {entity_id}")
    # {config.BUCKET_NAME}/{global_cache.get("bm_id")}/{FileName}
    FileName = file_path_.split("/")[-1]
    # object_name = (file_path_.split("/")[-1])
    object_name = (f'pdm05/'
                   f'{global_cache.get("bm_id")}/'
                   f'{FileName}')
    # logger.info(f"Object name {object_name}")
    # print(f"file path {file_path_}")
    try:
        minio_client.upload_file(bucket_name, object_name, file_path_)
        # logger.info(f"File '{file_path_}' uploaded to bucket '{bucket_name}' successfully.")
        # print(f"File '{file_path_}' uploaded to bucket '{bucket_name}' successfully.")
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

