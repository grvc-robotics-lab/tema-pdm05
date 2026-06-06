# Standard libraries
import copy
import logging
import tempfile
import threading
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
# import cv2
import numpy as np
import requests
import geopandas as gpd
import rasterio
import shapely
from rasterio.crs import CRS
from rasterio.transform import Affine
from rasterio.features import rasterize, geometry_mask
from rasterio.transform import from_bounds, rowcol
from rasterio.mask import mask
from rasterio.warp import calculate_default_transform, reproject, Resampling
from requests.auth import HTTPBasicAuth
from scipy.ndimage import gaussian_filter
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
# GlobalCache class to replace the global_cache dictionary
class GlobalCache:
    def __init__(self):
        self._lock = threading.Lock()
        self.processing = False
        self.natural_disaster = "No info"
        self.roi = []
        self.expiration = "No info"
        self.ignitionPoints = "No info"
        self.alert_timestamp = "No info"
        self.bm_id = "No info"
        self.kf_states = {}
        self.alert_received = False
        self.waiting_for_non_alert = False
        self.ogm_flag = False
        self.ogm_counter = 0
        self.initial_nd_ogm_metadata = None

        self.objects_tracking_started = False
    def get(self, key, default=None):
        with self._lock:
            return getattr(self, key, default)

    def set(self, key, value):
        with self._lock:
            setattr(self, key, value)

    def update(self, **kwargs):
        with self._lock:
            for key, value in kwargs.items():
                setattr(self, key, value)


global_cache = GlobalCache()
# --------------------------------------------------------
alert_event = threading.Event()  # Event triggered by Alert notifications
other_entity_event = threading.Event()  # Event triggered by other notifications
work_event = threading.Event()  # Unified wakeup: set whenever alert_event or other_entity_event is set
reset_done_event = threading.Event()  # Acknowledges completion of Alert-driven reset
reset_lock = threading.Lock()  # Guards reset vs file I/O (short critical sections)
notification_queue = queue.Queue()  # Thread-safe queue for non-Alert notifications
entities_initialized = False  # Tracks whether initialize_entities has run
is_processing = False
processing_lock = Lock()
ogm_flag = False
ogm_counter = 0
# --------------------------------------------------------
entity_creation_lock = threading.Lock()
processing_thread = None
processing_thread_lock = threading.Lock()
nd_file_locks = {}
nd_file_locks_lock = threading.Lock()
objects_file_locks = {}
objects_file_locks_lock = threading.Lock()
cleanup_lock = threading.Lock()
# --------------------------------------------------------
# initialize at module level
UPLOAD_INTERVAL = float(config.OGM_UPLOAD_INTERVAL_SEC)
_last_upload = time.monotonic() - UPLOAD_INTERVAL
_upload_lock = threading.Lock()
# --------------------------------------------------------
MAX_WORKERS = min(8, (os.cpu_count() or 4) * 2)


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

# ---------------------------------------------------------------
def start_processing_thread_safe():
    global processing_thread

    with processing_thread_lock:
        current_processing = global_cache.get('processing', False)
        if current_processing:
            return False  # Already running

        # Atomic state update
        global_cache.update(processing=True, waiting_for_non_alert=False)

        # Check if previous thread is still alive
        if processing_thread and processing_thread.is_alive():
            global_cache.set('processing', False)
            return False

        processing_thread = threading.Thread(target=initialize_processing, daemon=True)
        processing_thread.start()
        return True
# ---------------------------------------------------------------
@app.route(f'/{config.BASE_PATH}/{config.API_ENDPOINT}', methods=['POST'])
def notify():
    global processing_thread
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

        # -----------------------------------------------------------
        alerts = [n for n in data_list if n.get("type") == "Alert"]
        others = [n for n in data_list if n.get("type") != "Alert"]
        has_alerts_in_this_request = bool(alerts)
        alerts_processed = 0
        others_processed = 0
        started_init = False
        reset_completed = False

        # -----------------------------------------------------------
        # PHASE 1+2: Alerts have highest priority and must trigger an immediate reset.
        # We do NOT wait for non-Alert entities to start/reset processing.
        # -----------------------------------------------------------
        if has_alerts_in_this_request:
            logger.info(f"Alert(s) received ({len(alerts)}). Triggering immediate reset.")

            # Mark scenario active (do not stop the processing loop)
            global_cache.update(
                alert_received=True,
                waiting_for_non_alert=False
            )

            # Drop any stale non-alert fuse signal from previous scenario
            other_entity_event.clear()
            reset_done_event.clear()

            # Prevent the processing thread from resetting against a partially-updated cache
            # while we process multiple Alert entities in this same request.
            with reset_lock:
                alert_event.clear()
                other_entity_event.clear()

                for alert_notification in alerts:
                    try:
                        if "type" not in alert_notification:
                            logger.error(f"Missing 'type' in Alert notification: {alert_notification}", exc_info=True)
                            continue
                        logger.info(f"Processing HIGH-PRIORITY Alert: {alert_notification.get('id', 'Unknown ID')}")
                        process_notification(alert_notification)  # process_notification sets alert_event
                        alerts_processed += 1
                    except Exception as e:
                        logger.error(f"Error processing Alert notification: {e}", exc_info=True)

                # Ensure the reset is triggered even if upstream alert processing didn't set the event for some reason
                alert_event.set()
                work_event.set()

            # Ensure the processing thread is running immediately (Alert-only requests must still reset)
            if start_processing_thread_safe():
                started_init = True
                logger.info("Processing thread started (Alert-triggered).")
            else:
                logger.info("Processing thread already running (Alert-triggered).")

            # Wait (briefly) for reset to actually complete to guarantee 'instant reset' semantics
            reset_completed = reset_done_event.wait(timeout=30.0)
            if not reset_completed:
                logger.warning("Alert reset did not complete within 30s; continuing request processing anyway.")

        # -----------------------------------------------------------
        # PHASE 3: Process non-Alert entities (optionally after reset completion).
        # -----------------------------------------------------------
        disaster_type = global_cache.get('natural_disaster', 'No disaster info')
        expiration = global_cache.get('expiration', "No info")
        alert_was_received = global_cache.get('alert_received', False)

        # Process others only if we are in an active scenario (after an alert) and the disaster type is valid.
        should_process_others = (alert_was_received and disaster_type in ["Fire", "Flood"])

        not_expired = True
        if expiration not in (None, "No info"):
            try:
                expiration_time = datetime.strptime(expiration, "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=timezone.utc)
                current_time = datetime.now(timezone.utc)
                not_expired = current_time < expiration_time
            except Exception as e:
                logger.warning(f"Could not parse expiration '{expiration}': {e}. Treating as not expired.")


        if should_process_others and others and not_expired:
            logger.info(f"Processing {len(others)} non-Alert entity(ies)")
            try:
                # Block non-alert file I/O while a reset is in progress
                with reset_lock:
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
                logger.warning(f"Issues in non-Alert entities processing: {e}", exc_info=True)
        # If we staged any non-Alert inputs, trigger one fusion pass.
        if others_processed > 0:
            other_entity_event.set()
            work_event.set()

        # -----------------------------------------------------------
        # PHASE 4: Ensure processing is running when scenario is active
        # (do not gate start on non-Alert arrival).
        # -----------------------------------------------------------
        current_processing = global_cache.get('processing', False)
        if alert_was_received and disaster_type in ["Fire", "Flood"] and not current_processing:
            if start_processing_thread_safe():
                started_init = True
                logger.info("Processing thread started (post-notify check).")

        return jsonify({
            "status": "Notifications processed",
            "alerts_processed": alerts_processed,
            "others_processed": others_processed,
            "started_initialization": started_init,
            "disaster_type": disaster_type,
            "alert_received": alert_was_received,
            "reset_completed": reset_completed,
            "priority_handling": "Alerts trigger immediate reset (no waiting for non-alert entities)"
        }), 200

    except Exception as e:
        logger.error(f"Unexpected error in notify function: {e}", exc_info=True)
        return jsonify({"error": "Internal server error"}), 500


def handle_person_vehicle_detection(notification, parameters):
    auth_filename = notification.get('parameters', {}).get('value', {}).get('FileName', None)
    detection = notification.get("detection", {}).get('value', {})
    bm_id_from_notification = (notification.get('BM_id') or notification.get('bm_id') or {}).get('value')
    if bm_id_from_notification != global_cache.get('bm_id'):
        logger.warning(f"Not in the same BM {bm_id_from_notification}, File: {auth_filename}")
        return

    if not auth_filename:
        logger.warning("No filename found for PersonVehicleDetection.")
        return

    if "." not in auth_filename:
        logger.error("Invalid filename format received: %s", auth_filename)
        return

    disaster_info = global_cache.get('natural_disaster', 'No disaster info')
    if disaster_info == 'No disaster info':
        logger.warning("No disaster information found in the cache.")
        return

    scaling_factor = float(config.SCALING_FACTOR)
    for bbox in detection.get("boxes", []):
        current_bbox = bbox.get("bbox", [])
        try:
            upscale_bbox = [int(float(coord) * scaling_factor) for coord in current_bbox]
        except (ValueError, TypeError) as e:
            raise ValueError("One or more bbox coordinates could not be converted to a number.") from e
        bbox["bbox"] = upscale_bbox

    auth_filename = auth_filename.split("/")[-1]
    metadata_file_base = auth_filename.split('.')[0]
    detection_file = f"{disaster_info}_{metadata_file_base}_obj.json"
    metadata_file = f"{disaster_info}_{metadata_file_base}_objects_metadata.json"
    download_path = None
    if 'Flood' in metadata_file:
        download_path = os.path.join("downloads", "drone_imgs", 'Flood')
    elif 'Fire' in metadata_file:
        download_path = os.path.join("downloads", "drone_imgs", 'Fire')
    os.makedirs(download_path, exist_ok=True)  # Ensure directory exists
    json_file_path_metadata = os.path.join(download_path, metadata_file)
    json_file_path_detection = os.path.join(download_path, detection_file)

    with open(json_file_path_metadata, 'w', encoding='utf-8') as json_file:
        json.dump(parameters, json_file, indent=4)

    with open(json_file_path_detection, 'w', encoding='utf-8') as json_file:
        json.dump(detection, json_file, indent=4)

    try:
        main(disaster_info, "bbox", ground_resolution=1.00,
             target_base_name=f"{disaster_info}_{metadata_file_base}")
    except Exception as e:
        logger.error(f"Issue in geo-referencing due to {e}")
        return


def handle_segmentation(notification):
    bucket = notification.get("bucket", {}).get('value')
    auth_filename = notification.get('segmentation', {}).get('value')
    disaster = global_cache.get('natural_disaster', 'No disaster info')
    bm_id_from_notification = (notification.get('BM_id') or notification.get('bm_id') or {}).get('value')
    if bm_id_from_notification != global_cache.get('bm_id'):
        logger.warning(f"Not in the same BM {bm_id_from_notification}, File: {auth_filename}")
        return

    if not auth_filename or not isinstance(auth_filename, dict) or not bucket:
        logger.warning("Invalid or missing 'segmentation' value in notification.")
        return

    mask_id = auth_filename.get('mask_id')
    mask_id_temp = mask_id.split('/')[-1]
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

    try:
        downloaded_file_path = minio_client.download_file(bucket, mask_id, file_path)
        if downloaded_file_path:
            try:
                main(disaster, "segmented", ground_resolution=1.00, target_base_name=metadata_file_base)
            except Exception as e:
                logger.error(f"Error in Geo referencing due to {e}", exc_info=True)
        else:
            logger.error("File download failed.")
    except Exception as e:
        logger.error(f"Error downloading file for mask_id {mask_id}: {e}")
        return


def process_alert(notification, entity_id):
    location = notification.get("area", {}).get("value", {})
    event_ = notification.get("event", {}).get("value", None)
    expiration = notification.get("expires", {}).get("value")
    ignitionPoints = notification.get("ignitionPoints", {}).get("value", {})
    alert_timestamp = notification.get("sent", {}).get("value")
    bm_id = notification.get("bm_id", {}).get("value", None)

    updates = {}

    if isinstance(location, dict) and "coordinates" in location:
        updates['roi'] = location['coordinates']
    else:
        updates['roi'] = [None]
        logger.warning(f"Unexpected 'location' format for entity ID {entity_id}: {location}")

    if event_:
        updates['natural_disaster'] = event_
    else:
        updates['natural_disaster'] = "Unknown event"

    if expiration:
        updates['expiration'] = expiration
    else:
        updates['expiration'] = "No info"

    if isinstance(ignitionPoints, dict) and "coordinates" in ignitionPoints:
        updates['ignitionPoints'] = ignitionPoints['coordinates']
    else:
        updates['ignitionPoints'] = "No info"

    if alert_timestamp:
        updates['alert_timestamp'] = alert_timestamp
    else:
        updates['alert_timestamp'] = "No info"

    if bm_id:
        updates['bm_id'] = bm_id
    else:
        updates['bm_id'] = "Unknown event"

    global_cache.update(**updates)

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

    bm_id_from_notification = (notification.get('BM_id') or notification.get('bm_id') or {}).get('value')
    if bm_id_from_notification != global_cache.get('bm_id'):
        logger.warning(f"Not in the same BM {bm_id_from_notification}, BurntArea")
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

# ---------------------------------------------------------------
# Downloading EOFloodExtent
# ---------------------------------------------------------------
def download_tif_file(entity, polygon):
    """
    Downloads a .tif file from the entity's data URL if the notification corresponds to "EOFloodExtent".
    """
    minio_success = False
    try:
        data_value = entity.get('data', {}).get('value', {})
        file_url = data_value.get('href', {})

        if not file_url:
            logger.error("No valid download URL found in entity.")
            return None

        bucket_dlr = file_url.split('/')[-4]
        file_name_minio = f"{file_url.split('/')[-3]}/{file_url.split('/')[-2]}/{file_url.split('/')[-1]}"
        file_name = file_url.split("/")[-1]
        file_path = os.path.join(f'downloads/satellite_imgs/Flood', file_name)

        try:
            downloaded_file_path = minio_client.download_file(bucket_dlr, file_name_minio, file_path)

            if downloaded_file_path:
                logger.info(f"File downloaded successfully to {downloaded_file_path}")
                minio_success = True
            else:
                logger.error("File download failed. via MINIO")

            input_tif = os.path.join(f'downloads/satellite_imgs/Flood', file_name)
            output_tif = file_name

            try:
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

                    raster_extent = box(src.bounds.left,
                                        src.bounds.bottom,
                                        src.bounds.right,
                                        src.bounds.top)

                    if not poly_dst.intersects(raster_extent):
                        logger.warning(
                            f"Polygon does not intersect raster extent. "
                            f"Raster bounds (in {raster_crs.to_string()}): {src.bounds}"
                        )
                        return None

                    shapes = [mapping(poly_dst)]
                    out_image, out_transform = mask(src,
                                                    shapes=shapes,
                                                    crop=True,
                                                    filled=False)

                    out_meta = src.meta.copy()
                    out_meta.update({
                        "driver": "GTiff",
                        "height": out_image.shape[1],
                        "width": out_image.shape[2],
                        "transform": out_transform,
                        "nodata": src.nodata if src.nodata is not None else 0
                    })

                with rasterio.open(output_tif, "w", **out_meta) as dest:
                    dest.write(out_image)

                logger.info(f"Cropping complete. Saved as {output_tif}")
                return output_tif

            except Exception as e:
                logger.warning(f"Sat image was not cropped due to: {e}")
                return None

        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to download file: {e}")
            return None
    except Exception as e:
        logger.warning(f"MinIO download failed: {e}, falling back to HTTP")
    # ---------------------------------------------------------------
    # VIA HTTP
    # ---------------------------------------------------------------
    if not minio_success:
        try:
            """
                Downloads a .tif file from the entity's data URL if the notification corresponds to "EOFloodExtent".
            """
            data_value = entity.get('data', {}).get('value', {})
            file_url = data_value.get('href')
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
                    # ---------------------------------------------------------------
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
                            "transform": out_transform,
                            "nodata": src.nodata if src.nodata is not None else 0
                        })
                        with rasterio.open(output_tif, "w", **out_meta) as dest:
                            dest.write(out_image)
                    # ---------------------------------------------------------------
                    return output_tif

                except requests.exceptions.RequestException as e:
                    logger.error(f"Failed to download file via hyperref: {e}")
                    return None
            else:
                logger.error("No valid download URL found in entity.")
        except Exception as e:
            logger.error(f"Both MinIO and HTTP downloads failed. MinIO error, HTTP error: {e}")
            return None

# ---------------------------------------------------------------
def process_notification(notification):
    entity_id = notification.get("id")
    entity_type = notification.get("type")

    if not entity_id or not entity_type:
        logger.error(f"Invalid notification received: {notification}")
        return
    try:
        # --------------------------------------------------------------------------#
        # Alert
        # --------------------------------------------------------------------------#
        if entity_type == "Alert":
            try:
                logger.info("Triggering Alert entity processing")
                process_alert(notification, entity_id)
                alert_event.set()
                work_event.set()
            except Exception as e:
                logger.error(f"Error processing Alert entity ID {entity_id}: {e}")
            return
        # --------------------------------------------------------------------------#
        # PersonVehicleDetection
        # --------------------------------------------------------------------------#
        if entity_type == "PersonVehicleDetection":
            try:
                parameters = notification.get("parameters", {}).get('value', {})
                if not parameters:
                    logger.warning(f"Missing parameters in notification for entity {entity_id}")
                    return
                handle_person_vehicle_detection(notification, parameters)
            except Exception as e:
                logger.error(f"Error processing PersonVehicleDetection for entity {entity_id}: {e}", exc_info=True)
        # --------------------------------------------------------------------------#
        # FireSegmentation, FloodSegmentation, BurntSegmentation
        # --------------------------------------------------------------------------#
        elif entity_type in ["FireSegmentation", "FloodSegmentation", "BurntSegmentation"]:
            try:
                handle_segmentation(notification)
            except Exception as e:
                logger.error(f"Error handling segmentation for entity {entity_id}: {e}", exc_info=True)
        # --------------------------------------------------------------------------#
        # EOFloodExtent
        # --------------------------------------------------------------------------#
        elif entity_type == "EOFloodExtent":
            try:
                roi_data = global_cache.get('roi')
                bm_id_from_notification = (notification.get('BM_id') or notification.get('bm_id') or {}).get('value')
                if bm_id_from_notification != global_cache.get('bm_id'):
                    logger.warning(f"Not in the same BM {bm_id_from_notification}, Entity: {entity_type}")
                    return
                download_tif_file(notification, roi_data)
            except Exception as e:
                logger.error(f"Error handling EOFloodExtent for entity {entity_id}: {e}")
        # --------------------------------------------------------------------------#
        # EOBurntArea
        # --------------------------------------------------------------------------#
        elif entity_type == "EOBurntArea":
            try:
                roi_data = global_cache.get('roi')
                polygon = Polygon(roi_data[0])
                fetch_burnt_area(notification, polygon)
            except Exception as e:
                logger.error(f"Error handling EOBurntArea for entity ID {entity_id}: {e}", exc_info=True)
        # --------------------------------------------------------------------------#
        # Geo Social media analysis
        # --------------------------------------------------------------------------#
        elif entity_type in ["HotspotResult", "SinglePostResult"]:
            try:
                filename_ = notification.get('filename')
                bucket = notification.get("bucket")
                bm_id_from_notification = (notification.get('BM_id') or notification.get('bm_id') or {}).get('value')
                if bm_id_from_notification != global_cache.get('bm_id'):
                    logger.warning(f"Not in the same BM {bm_id_from_notification}, Entity: {entity_type}")
                    return
                if not (isinstance(filename_, dict) and 'value' in filename_ and
                        isinstance(bucket, dict) and 'value' in bucket):
                    logger.warning("Invalid or missing 'filename' or 'bucket' in notification.")
                    return
                elif filename_['value'] and bucket['value']:
                    last_part_split = filename_['value'].split("/")[-1]
                    file_path = f"downloads/SocialMedia/{last_part_split}"
                    try:
                        downloaded_file_path = minio_client.download_file(bucket['value'], filename_['value'],
                                                                          file_path)
                        if downloaded_file_path:
                            logger.info(f"File downloaded successfully to {downloaded_file_path}")
                        else:
                            logger.error("File download failed.")
                    except Exception as e:
                        logger.error(f"Error in downloading file: {e}", exc_info=True)
            except Exception as e:
                logger.error(f"Error in handling of entity ID {entity_id}: {e}")
        # --------------------------------------------------------------------------#
        # FloodCalculationResult
        # --------------------------------------------------------------------------#
        elif entity_type == "FloodCalculationResult":
            try:
                filename_ = notification.get('filename')
                bucket = notification.get("bucket")
                bm_id_from_notification = (notification.get('BM_id') or notification.get('bm_id') or {}).get('value')
                if bm_id_from_notification != global_cache.get('bm_id'):
                    logger.warning(f"Not in the same BM {bm_id_from_notification}, Entity: {entity_type}")
                    return
                if not all([
                    isinstance(filename_, dict) and filename_.get('value'),
                    isinstance(bucket, dict) and bucket.get('value'),
                    bm_id_from_notification]):
                    logger.warning(
                        "Invalid or missing required fields in notification: 'filename', 'bucket', or 'bm_id'")
                    return
                elif filename_['value'] and bucket['value']:
                    file_path = f"downloads/FloodSim/{filename_['value']}"
                    try:
                        downloaded_file_path = minio_client.download_file(bucket['value'], filename_['value'],
                                                                          file_path)
                        if downloaded_file_path:
                            logger.info(f"File downloaded successfully to {downloaded_file_path}")
                        else:
                            logger.error("File download failed.")
                    except Exception as e:
                        logger.error(f"Error in downloading file: {e}", exc_info=True)
            except Exception as e:
                logger.error(f"Error in handling of entity ID {entity_id}: {e}")
        # --------------------------------------------------------------------------#
        # StandardArrivalTime
        # --------------------------------------------------------------------------#
        elif entity_type == "StandardArrivalTime":
            try:
                filename_ = notification.get('file_name')
                bucket = notification.get("bucket")
                bm_id_from_notification = (notification.get('BM_id') or notification.get('bm_id') or {}).get('value')
                if bm_id_from_notification != global_cache.get('bm_id'):
                    logger.warning(f"Not in the same BM {bm_id_from_notification}, Entity: {entity_type}")
                    return
                if not (isinstance(filename_, dict) and 'value' in filename_ and
                        isinstance(bucket, dict) and 'value' in bucket):
                    logger.warning("Invalid or missing 'filename' or 'bucket' in notification.")
                    return
                elif filename_['value'] and bucket['value']:
                    try:
                        last_part_split = filename_['value'].split("/")[-1]
                        file_path = f"downloads/FireSim/{last_part_split}"
                        downloaded_file_path = minio_client.download_file(bucket['value'], filename_['value'],
                                                                          file_path)
                        if downloaded_file_path:
                            logger.info(f"File downloaded successfully to {downloaded_file_path}")
                        else:
                            logger.error("File download failed.")
                    except Exception as e:
                        logger.error(f"Error in downloading file: {e}", exc_info=True)
            except Exception as e:
                logger.error(f"Error in handling of entity ID {entity_id}: {e}")
        # --------------------------------------------------------------------------#
    except Exception as e:
        logger.error(f"Error processing notification {entity_id} of type {entity_type}: {e}")


# ------------------------------------------------------------------------------
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
    # ---------------------------------------------------------------
    if dem_dataset in ("SRTMGL1", "SRTMGL3") and (miny < -60 or maxy > 60):
        print("SRTM only covers ±60°. Switching to EU_DTM.")
        dem_dataset = "COP30"
    # ---------------------------------------------------------------
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
        response = requests.get(api_url, params=params, stream=True)
        data = response.content
        with open(output_file, "wb") as f:
            f.write(data)
        logger.info(f"DEM downloaded and saved to {output_file}")
    except requests.RequestException as e:
        logger.info(f"Failed to download DEM: {e}")


# ------------------------------------------------------------------------------
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


def initialize_processing():
    """
    Worker thread behavior:
      - Reset immediately whenever an Alert arrives (alert_event set).
      - Run fusion (+ upload inside estimate_* functions) whenever non-Alert inputs have been staged
        (other_entity_event set) AND the current Alert scenario is valid/initialized.

    Design notes:
      - Uses work_event as a unified wakeup to avoid polling and the double-wait pattern.
      - Uses reset_lock to block non-Alert file I/O during reset (short critical sections).
      - Uses cleanup_lock to prevent directory cleanup racing with fusion writes.
      - Does NOT hold reset_lock during long fusion computations, so alert ingestion cannot be blocked
        behind fusion.
    """
    global entities_initialized, ogm_flag, polygon_coordinates, ogm_counter, _last_upload

    # Track the last alert context that has been fully reset/initialized.
    # We use (bm_id, alert_timestamp) as the scenario key.
    last_reset_alert_key = None

    # Predict-only publishing for objects when evidence is missing (armed after first evidence)
    last_obj_predict_ts = 0.0
    obj_predict_interval_sec = float(getattr(config, "OBJ_PREDICT_INTERVAL_SEC", 3.0))

    logger.debug("initialize_processing ➜ thread started")
    try:
        while True:
            if not global_cache.get('processing', False):
                logger.info("Processing thread stopping as requested")
                break

            # ---------------------------------------------------------------
            # Wait for work (Alert or Non-Alert staging). Timeout only to allow
            # periodic expiration checks; no log spam on idle.
            # ---------------------------------------------------------------
            woke = work_event.wait(timeout=0.5)
            if woke:
                work_event.clear()
            else:
                # Idle tick: allow predict-only tracking *only after* first evidence has started tracking.
                if (entities_initialized
                        and global_cache.get('alert_received', False)
                        and global_cache.get('objects_tracking_started', False)
                        and not alert_event.is_set()):
                    now_ts = time.time()
                    if (now_ts - last_obj_predict_ts) >= obj_predict_interval_sec:
                        try:
                            estimate_objects_status(predict_only=True)
                        except Exception as e:
                            logger.warning(f"Objects: predict-only tick failed: {e}")
                        last_obj_predict_ts = now_ts

            # ---------------------------------------------------------------
            # Highest priority: handle Alert reset.
            # ---------------------------------------------------------------
            if alert_event.is_set():
                # Block non-Alert file I/O while resetting.
                with reset_lock:
                    reset_done_event.clear()
                    try:
                        current_alert_ts = global_cache.get('alert_timestamp', "No info")
                        current_alert_bm_id = global_cache.get('bm_id', "No info")
                        alert_key = (current_alert_bm_id, current_alert_ts)

                        logger.info(f"Alert received. Performing immediate full reset. key={alert_key}")

                        # Clear events first to avoid stale triggers.
                        alert_event.clear()
                        other_entity_event.clear()

                        # Reset upload throttling so the new scenario can upload immediately.
                        with _upload_lock:
                            _last_upload = time.monotonic() - UPLOAD_INTERVAL

                        # Reset in-memory state.
                        entities_initialized = False
                        polygon_coordinates = None
                        ogm_flag = False
                        ogm_counter = 0
                        global_cache.update(
                            ogm_flag=False,
                            ogm_counter=0,
                            initial_nd_ogm_metadata=None,
                            kf_states={},
                            objects_tracking_started=False
                        )

                        # Clean up directories (prevent races with fusion writes).
                        with cleanup_lock:
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
                                        logger.warning(f"Could not clean directory {path}: {e}")

                        # Log rotation (best effort; do not fail reset if it fails).
                        log_file = 'app_routes.log'
                        if os.path.exists(log_file):
                            try:
                                os.remove(log_file)
                                for handler in logger.handlers[:]:
                                    logger.removeHandler(handler)
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

                        # Initialize OGM/ROI for the new alert scenario (once).
                        if not global_cache.get('ogm_flag'):
                            try:
                                polygon_coordinates = convert_to_polygon()
                                disaster_type = global_cache.get('natural_disaster', 'No disaster info')

                                if disaster_type in ["Fire", "Flood"]:
                                    os.makedirs(f"downloads/drone_imgs/{disaster_type}", exist_ok=True)
                                    os.makedirs("estimated_OGM", exist_ok=True)

                                    output_dem_file = f"downloads/drone_imgs/{disaster_type}/subset_dem.tif"
                                    try:
                                        download_opentopography_dem(
                                            api_key=config.OpenTopography_api_key,
                                            output_file=output_dem_file,
                                            coordinates=polygon_coordinates,
                                            dem_dataset="COP30" #"SRTMGL1"
                                        )
                                    except Exception as e:
                                        logger.warning(f"DEM download failed: {e}")
                                        output_dem_file = None

                                    # ND OGM
                                    ogm_path_ND = f"estimated_OGM/occupancy_grid_map_{disaster_type}.tif"
                                    try:
                                        get_roi(polygon_coordinates, ogm_path_ND, int(config.OGM_ND_RESOLUTION), 4326, 1, output_dem_file)
                                        result = mask_geotiff_with_polygon_exact(ogm_path_ND, global_cache.get('roi'), ogm_path_ND)
                                        if result:
                                            logger.info(f"Masked GeoTIFF saved to {result}")
                                            publish_initial_nd_ogm(disaster_type, ogm_path_ND)
                                        else:
                                            logger.warning("Masking ND OGM failed.")
                                    except Exception as e:
                                        logger.warning(f"Error creating ND OGM: {e}")

                                    # Objects OGM
                                    ogm_path_obj = f"estimated_OGM/occupancy_grid_map_{disaster_type}_Objects.tif"
                                    try:
                                        get_roi(polygon_coordinates, ogm_path_obj, int(config.OGM_OBJ_RESOLUTION), 4326, 2, output_dem_file)
                                        result = mask_geotiff_with_polygon_exact(ogm_path_obj, global_cache.get('roi'), ogm_path_obj)
                                        if result:
                                            logger.info(f"Masked GeoTIFF saved to {result}")
                                        else:
                                            logger.warning("Masking Objects OGM failed.")
                                    except Exception as e:
                                        logger.warning(f"Error creating Objects OGM: {e}")

                                    # Geo dict (expected empty at init)
                                    try:
                                        get_geo_dict(ogm_path_obj)
                                        global_cache.set('ogm_counter', global_cache.get('ogm_counter', 0) + 1)
                                        ogm_counter = global_cache.get('ogm_counter', 0)
                                        ogm_flag = True
                                        global_cache.set('ogm_flag', True)
                                    except Exception as e:
                                        logger.warning(f"Error generating geo dictionary: {e}")
                                else:
                                    logger.info(f"Skipping OGM init for disaster type {disaster_type}")
                            except Exception as e:
                                logger.warning(f"Error in OGM initialization: {e}")

                        # Initialize NGSI-LD entities after the initial OGM is ready, so the
                        # Maps4* creation payload can point to an uploaded map file.
                        initialize_entities()
                        disaster_type = global_cache.get('natural_disaster', 'No disaster info')
                        if disaster_type in ["Fire", "Flood"] and not global_cache.get('initial_nd_ogm_metadata'):
                            entities_initialized = False
                            logger.error(
                                "Scenario initialization is incomplete: initial ND OGM metadata is missing. "
                                "Skipping fusion until a valid Alert reset creates and uploads the initial OGM."
                            )
                        else:
                            entities_initialized = True

                        # Record which alert we reset for.
                        last_reset_alert_key = alert_key

                    except Exception as e:
                        logger.error(f"Error during Alert reset: {e}", exc_info=True)
                    finally:
                        reset_done_event.set()

                # After reset, continue loop (fusion waits for other_entity_event).
                continue

            # ---------------------------------------------------------------
            # Optional expiration check (never blocks Alert reset).
            # ---------------------------------------------------------------
            expiration = global_cache.get('expiration', "No info")
            if expiration not in (None, "No info"):
                try:
                    expiration_time = datetime.strptime(expiration, "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=timezone.utc)
                    if datetime.now(timezone.utc) >= expiration_time:
                        logger.info(f"Expiration time reached: {expiration}. Exiting the loop.")
                        break
                except Exception as e:
                    logger.warning(f"Error parsing expiration time '{expiration}': {e}. Ignoring expiration check.")

            # ---------------------------------------------------------------
            # Fusion: only when non-Alert inputs have been staged.
            # ---------------------------------------------------------------
            if not other_entity_event.is_set():
                continue

            # Must have a valid initialized scenario.
            if not entities_initialized or not global_cache.get('alert_received', False):
                other_entity_event.clear()
                continue

            # Ensure we fuse against the same alert scenario we last reset for.
            current_key = (global_cache.get('bm_id', "No info"), global_cache.get('alert_timestamp', "No info"))
            if last_reset_alert_key is None or current_key != last_reset_alert_key:
                # Do NOT clear: keep pending until reset completes for this scenario.
                logger.info(f"Non-alert work arrived but scenario mismatch "
                            f"(current={current_key}, last_reset={last_reset_alert_key}). "
                            f"Waiting for reset.")
                continue

            # Prevent cleanup races with fusion writes.
            with cleanup_lock:
                # If an Alert arrived while waiting, let next loop reset.
                if alert_event.is_set():
                    continue

                try:
                    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                        future_nd = executor.submit(estimate_nd_status)
                        future_obj = executor.submit(estimate_objects_status)
                        for future in as_completed([future_nd, future_obj]):
                            if alert_event.is_set():
                                break
                            try:
                                future.result()
                            except Exception as e:
                                if future == future_nd:
                                    logger.info(f"ND estimation skipped/failed: {e}")
                                else:
                                    logger.info(f"Objects estimation skipped/failed: {e}")
                finally:
                    # Critical: clear after a pass to avoid infinite retrigger loops.
                    # other_entity_event.clear()
                    pass


    except Exception as e:
        logger.error(f"Unexpected error in initialize_processing: {e}", exc_info=True)
    finally:
        global_cache.set('processing', False)
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

            logger.info(f"Creating subscription for {entity_type} with ID {subscription_id}")

            subscription_payload = subscription_payload_template.copy()
            subscription_payload["id"] = subscription_id
            subscription_payload["entities"] = [{"type": entity_type}]

            create_subscription(subscription_url, subscription_payload, headers)

    except Exception as e:
        logger.exception(f"An error occurred during subscription management: {e}")


def create_subscription(subscription_url, payload, headers):
    try:
        response = requests.post(subscription_url, json=payload, headers=headers)
        if response.status_code in (200, 201):
            logger.info(f"Subscription {payload['id']} created successfully.")
        else:
            logger.error(
                f"Failed to create subscription {payload['id']}. Status: {response.status_code}, Body: {response.text}")
    except Exception as e:
        logger.error(f"Error creating subscription {payload['id']}: {e}")


# ---------------------------------------------------------------
# Fusing Flood Simulation Results on the OGM
# ---------------------------------------------------------------
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


def _linear_pool(p_prior, p_evidence, alpha=0.8):
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
        alpha: float = 0.70,
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


# ---------------------------------------------------------------

def predict_ogm_fire(
        occupancy_path,
        arrival_time_path,
        output_path,
        horizon_hours=1.0,
        method="probabilistic",
        sim_weight=0.35,
        transition_hours=0.1,
        state_path=None):
    """
    Fuse FireSim Arrival Time output into the current-state fire OGM.

    This function is designed for FireSim rasters like
    `OutputStandardArrivalTime_*.tif`, where:
    - there is a single raster band
    - each pixel value is the predicted fire arrival time in hours
    - lower values mean earlier arrival
    - nodata marks cells outside the simulated domain

    The first FireSim notification initializes the forecast sequence but does
    not update the current-state OGM. From the second notification onward, only
    the perimeter of the selected FireSim isochrone is fused. Older isochrones
    are treated as already passed by the front and therefore act as inverse
    evidence for current active fire.

    Args:
        occupancy_path (str): Path to the current occupancy OGM GeoTIFF.
                              Expected values in [0, 1].
        arrival_time_path (str): Path to FireSim Arrival Time GeoTIFF.
                                 Values are expected in hours.
        output_path (str): Path to save the fused OGM.
        horizon_hours (float): FireSim notification step in hours. The first
                               FireSim notification initializes the sequence
                               at hour 0, the second uses hour 1, the third
                               uses hour 2, and so on.
                               Default: 1.0 hour.
        method (str): Fusion method.
                      - "probabilistic": weighted probabilistic union
                      - "simple": weighted additive fusion
                      Default: "probabilistic"
        sim_weight (float): Weight of simulation evidence in [0, 1].
                            Since this is forecast information, it should
                            usually be weaker than direct observations.
                            Recommended range: 0.2 to 0.5
        transition_hours (float): Total temporal width of the perimeter band
                                  centered on the selected hour isochrone.
                                  Default: 0.25 hours.
        state_path (str | None): Optional JSON file storing how many FireSim
                                 notifications have already been processed.

    Returns:
        None. Writes the fused raster to `output_path`.
    """
    sim_weight = float(np.clip(sim_weight, 0.0, 1.0))
    horizon_hours = float(max(horizon_hours, 0.0))
    transition_hours = float(max(transition_hours, 0.0))
    firesim_step = 1

    if state_path and os.path.exists(state_path):
        try:
            with open(state_path, "r", encoding="utf-8") as fh:
                state = json.load(fh)
            firesim_step = int(state.get("notification_count", 0)) + 1
        except Exception as e:
            logger.warning(f"Failed to load FireSim notification state from {state_path}: {e}")
            firesim_step = 1

    target_hour = max(horizon_hours * (firesim_step - 1), 0.0)
    band_half_width = max(transition_hours / 2.0, 0.0)
    passed_burnt_cap = 0.30

    if firesim_step == 1:
        if output_path != occupancy_path:
            with rasterio.open(occupancy_path) as src1:
                data1 = src1.read(1).astype(np.float32)
                dst_profile = src1.profile
            dst_profile.update(dtype=rasterio.float32, count=1)
            with rasterio.open(output_path, "w", **dst_profile) as dst:
                dst.write(data1, 1)

        if state_path:
            try:
                os.makedirs(os.path.dirname(state_path), exist_ok=True)
                with open(state_path, "w", encoding="utf-8") as fh:
                    json.dump(
                        {
                            "notification_count": firesim_step,
                            "current_hour": 0.0,
                            "next_fused_hour": horizon_hours
                        },
                        fh
                    )
            except Exception as e:
                logger.warning(f"Failed to save FireSim notification state to {state_path}: {e}")

        logger.info(
            f"FireSim notification initialized without OGM fusion: {arrival_time_path}, "
            f"firesim_step={firesim_step}, next_fused_hour={horizon_hours}"
        )
        return

    # ------------------------------------------------------------------
    # Step 1: Read FireSim Arrival Time raster
    # ------------------------------------------------------------------
    with rasterio.open(arrival_time_path) as src2:
        if src2.count < 1:
            raise ValueError(f"No bands found in arrival-time raster: {arrival_time_path}")

        arrival = src2.read(1).astype(np.float32)
        src2_transform = src2.transform
        src2_crs = src2.crs
        src2_nodata = src2.nodata

    valid2 = np.isfinite(arrival)
    if src2_nodata is not None:
        valid2 &= (arrival != src2_nodata)

    # ------------------------------------------------------------------
    # Step 2: Convert arrival time into current-front support and passed-area mask
    # ------------------------------------------------------------------
    sim_prob = np.zeros_like(arrival, dtype=np.float32)
    passed_mask_src = np.zeros_like(arrival, dtype=np.float32)

    if np.any(valid2):
        if target_hour > 0.0:
            passed_mask_src[valid2 & (arrival < target_hour)] = 1.0

        if band_half_width > 0.0:
            band_distance = np.abs(arrival - target_hour)
            # Keep FireSim active support on the current isochrone/front edge.
            # Cells inside the isochrone are already passed/burnt.
            front_mask = valid2 & (arrival >= target_hour) & (band_distance <= band_half_width)
            sim_prob[front_mask] = 1.0 - (band_distance[front_mask] / band_half_width)
        else:
            front_mask = valid2 & np.isclose(arrival, target_hour, atol=1e-6)
            sim_prob[front_mask] = 1.0

    sim_prob = np.where(valid2, sim_prob, 0.0).astype(np.float32)
    sim_prob = np.clip(sim_prob, 0.0, 1.0)
    passed_mask_src = np.where(valid2, passed_mask_src, 0.0).astype(np.float32)

    # ------------------------------------------------------------------
    # Step 3: Read current-state occupancy OGM
    # ------------------------------------------------------------------
    with rasterio.open(occupancy_path) as src1:
        data1 = src1.read(1).astype(np.float32)
        dst_transform = src1.transform
        dst_crs = src1.crs
        dst_shape = data1.shape
        dst_profile = src1.profile

    data1 = np.clip(data1, 0.0, 1.0)

    # ------------------------------------------------------------------
    # Step 4: Reproject current-front support and passed-area mask onto the OGM grid
    # ------------------------------------------------------------------
    reprojected_sim = np.full(dst_shape, np.nan, dtype=np.float32)
    reprojected_passed = np.full(dst_shape, np.nan, dtype=np.float32)

    reproject(
        source=sim_prob,
        destination=reprojected_sim,
        src_transform=src2_transform,
        src_crs=src2_crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        src_nodata=None,
        dst_nodata=np.nan,
        resampling=Resampling.nearest
    )
    reproject(
        source=passed_mask_src,
        destination=reprojected_passed,
        src_transform=src2_transform,
        src_crs=src2_crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        src_nodata=None,
        dst_nodata=np.nan,
        resampling=Resampling.nearest
    )

    valid_sim = np.isfinite(reprojected_sim)
    reprojected_sim = np.where(valid_sim, np.clip(reprojected_sim, 0.0, 1.0), 0.0)
    passed_mask = np.isfinite(reprojected_passed) & (reprojected_passed > 0.5)

    # Forecast evidence should be weaker than direct observations
    weighted_sim = sim_weight * reprojected_sim

    # ------------------------------------------------------------------
    # Step 5: Fuse with current occupancy
    # ------------------------------------------------------------------
    combined_data = data1.copy()

    if passed_mask.any():
        combined_data[passed_mask] = np.minimum(combined_data[passed_mask], passed_burnt_cap)

    if method.lower() == "probabilistic":
        combined_data[valid_sim] = 1.0 - (
                (1.0 - combined_data[valid_sim]) * (1.0 - weighted_sim[valid_sim])
        )
    else:
        combined_data[valid_sim] = combined_data[valid_sim] + weighted_sim[valid_sim]
        combined_data = np.clip(combined_data, 0.0, 1.0)

    combined_data = np.clip(combined_data, 0.0, 1.0).astype(np.float32)

    # ------------------------------------------------------------------
    # Step 6: Save fused OGM
    # ------------------------------------------------------------------
    dst_profile.update(dtype=rasterio.float32, count=1)

    with rasterio.open(output_path, "w", **dst_profile) as dst:
        dst.write(combined_data, 1)

    if state_path:
        try:
            os.makedirs(os.path.dirname(state_path), exist_ok=True)
            with open(state_path, "w", encoding="utf-8") as fh:
                json.dump(
                    {
                        "notification_count": firesim_step,
                        "current_hour": target_hour
                    },
                    fh
                )
        except Exception as e:
            logger.warning(f"Failed to save FireSim notification state to {state_path}: {e}")

    logger.info(
        f"Fire OGM fused with FireSim isochrone perimeter: {arrival_time_path}, "
        f"firesim_step={firesim_step}, target_hour={target_hour}, "
        f"band_width_hours={transition_hours}, sim_weight={sim_weight}, "
        f"passed_burnt_cap={passed_burnt_cap}, method={method}"
    )

# -----------------------------------------------------------------
def should_upload_now():
    global _last_upload
    with _upload_lock:
        now = time.monotonic()
        if now - _last_upload >= UPLOAD_INTERVAL:
            _last_upload = now
            return True
        return False
# -----------------------------------------------------------------
def publish_initial_nd_ogm(disaster_type, ogm_path):
    """
    Upload the neutral OGM generated during alert initialization so the newly
    created Maps4* entity points to a real map file from the start.
    """
    global _last_upload

    if not os.path.exists(ogm_path):
        logger.warning(f"Initial OGM upload skipped; file does not exist: {ogm_path}")
        return

    try:
        with rasterio.open(ogm_path) as src:
            ogm_data = src.read(1).astype(np.float32)
            transform = src.transform
            crs = src.crs

        ogm_gt = (
            transform.c,
            transform.a,
            transform.b,
            transform.f,
            transform.d,
            transform.e,
        )
        crs_epsg = crs.to_epsg() if crs else 4326
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        initial_file_name = f"occupancy_grid_map_{disaster_type}_initial_{timestamp}.tif"

        metadata = save_geotiff(
            "estimated_OGM",
            initial_file_name,
            ogm_data,
            ogm_gt,
            timestamp,
            crs_epsg=crs_epsg or 4326,
        )

        if not metadata:
            logger.warning("Initial OGM upload skipped; metadata generation failed.")
            return

        if metadata.get('coordinates') != global_cache.get('roi'):
            logger.warning("Initial OGM upload skipped; metadata ROI does not match current alert ROI.")
            return

        file_path = os.path.join("estimated_OGM", initial_file_name)
        object_name = f'pdm05/{global_cache.get("bm_id")}/{initial_file_name}'
        minio_client.upload_file(config.BUCKET_NAME, object_name, file_path)
        global_cache.set('initial_nd_ogm_metadata', metadata)

        with _upload_lock:
            _last_upload = time.monotonic()

        logger.info(f"Initial neutral OGM uploaded and staged for entity creation: {initial_file_name}")

    except Exception as e:
        logger.error(f"Error publishing initial OGM: {e}", exc_info=True)

# -----------------------------------------------------------------
def get_nd_lock(disaster_type):
    """Get a lock for a specific disaster type's OGM files"""
    with nd_file_locks_lock:
        if disaster_type not in nd_file_locks:
            nd_file_locks[disaster_type] = threading.RLock()
        return nd_file_locks[disaster_type]
# -----------------------------------------------------------------
def estimate_nd_status():
    global polygon_coordinates, _last_upload
    # ----------------------------------------------------------
    # while georeferenced_seg_flag:
    # ----------------------------------------------------------
    disaster_type = global_cache.get('natural_disaster', 'No disaster info')
    # ----------------------------------------------------------
    if disaster_type == 'No disaster info':
        return
    # ----------------------------------------------------------
    with get_nd_lock(disaster_type):
        ogm_path = f"estimated_OGM/occupancy_grid_map_{disaster_type}.tif"
        firesim_state_path = os.path.join("downloads", "FireSim", "firesim_notification_state.json")
        # ----------------------------------------------------------
        # Prediction step
        # ----------------------------------------------------------
        prediction_path = "downloads/FloodSim" if disaster_type == "Flood" else "downloads/FireSim"
        # ----------------------------------------------------------
        try:
            for pred_file in sorted(os.listdir(prediction_path),
                                    key=lambda x_: os.path.getmtime(os.path.join(prediction_path, x_))):
                if pred_file.lower().endswith('.tif'):
                    pred_file_path = os.path.join(prediction_path, pred_file)
                    logger.info(f"prediction file ---> {pred_file_path}")
                    if disaster_type == 'Flood':
                        fuse_ogm_with_depth(
                            ogm_path,
                            pred_file_path,
                            mapping_="logistic",
                            mapping_params={"h50": 0.50, "s": 0.50},
                            fusion="logit_pool",
                            alpha=0.75,
                        )
                        # ----------------------------------------------------------
                        logger.info("prediction is performed")
                        # ----------------------------------------------------------
                        result = mask_geotiff_with_polygon_exact(ogm_path, global_cache.get('roi'), ogm_path)
                        if result:
                            logger.info(f"Masked GeoTIFF saved to {result}")
                        else:
                            logger.info("Error occurred while masking the GeoTIFF.")
                        # ----------------------------------------------------------
                        if os.path.exists(pred_file_path):
                            os.remove(pred_file_path)
                        break
                    elif disaster_type == "Fire":
                        predict_ogm_fire(
                            ogm_path,
                            pred_file_path,
                            ogm_path,
                            horizon_hours=1.0,
                            method="probabilistic",
                            sim_weight=0.35,
                            transition_hours=0.25,
                            state_path=firesim_state_path)
                        logger.info("prediction is performed")
                        # ----------------------------------------------------------
                        result = mask_geotiff_with_polygon_exact(ogm_path, global_cache.get('roi'), ogm_path)
                        if result:
                            logger.info(f"Masked GeoTIFF saved to {result}")
                        else:
                            logger.info("Error occurred while masking the GeoTIFF.")
                        # ----------------------------------------------------------
                        if os.path.exists(pred_file_path):
                            os.remove(pred_file_path)
                        break
        except Exception as e:
            logger.info(f"Error in prediction models due to {e}")
        # ----------------------------------------------------------
        ogm_data, ogm_gt_, ogm_proj, data_type, observ_nodata = load_image(ogm_path, 0)
        # ----------------------------------------------------------
        # Update OGM with drone data
        # ----------------------------------------------------------
        try:
            if disaster_type == 'Flood':
                segment_payload, consumed_segment = load_segment_georef_data(
                    "georeferenced_drone_images/segmented/flood",
                    return_consumed=True,
                    delete_on_success=True,
                    newest_first=False,
                )
                if segment_payload is not None and segment_payload.get("measurement_type") == "Flood":
                    ogm_data = update_occupancy_grid_from_georef_segments(
                        np.asarray(ogm_data, dtype=np.float32),
                        ogm_gt_,
                        segment_payload.get("samples", []),
                        "Flood"
                    )
                    logger.info(
                        f"Fused flood UAV segmentation using georef_data from "
                        f"{os.path.basename(consumed_segment) if consumed_segment else 'unknown file'}."
                    )
        except Exception as e:
            logger.info(f'Error in fusing Segmented Flood images due to {e}')
        # ----------------------------------------------------------
        # Save updated OGM as GeoTIFF
        # ----------------------------------------------------------
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
            if metadata_timestamp.get('coordinates') == global_cache.get('roi'):
                if should_upload_now():
                    process_and_upload_ogm(ND_entity_ID,
                                           os.path.join("estimated_OGM", output_tiff_path_timestamp),
                                           config.BUCKET_NAME, metadata_timestamp)
        except Exception as e:
            logger.error(f"Error saving or uploading OGM: {e}")
        # ----------------------------------------------------------
        if disaster_type == "Fire":
            # Fuse active fire measurements (segmented fire)
            try:
                segment_payload, consumed_segment = load_segment_georef_data(
                    "georeferenced_drone_images/segmented/fire",
                    return_consumed=True,
                    delete_on_success=True,
                    newest_first=False,
                )
                if segment_payload is not None and segment_payload.get("measurement_type") == "active_fire":
                    ogm_data = update_occupancy_grid_from_georef_segments(
                        np.asarray(ogm_data, dtype=np.float32),
                        ogm_gt_,
                        segment_payload.get("samples", []),
                        "active_fire"
                    )
                    logger.info(
                        f"OGM updated successfully for fire UAV segmentation using georef_data "
                        f"from {os.path.basename(consumed_segment) if consumed_segment else 'unknown file'}."
                    )
            except Exception as e:
                logger.info(f"Error in updating OGM with active_fire drone measurement {e}")
            # ----------------------------------------------------------
            # Save updated OGM as GeoTIFF
            # ----------------------------------------------------------
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
                if metadata_timestamp.get('coordinates') == global_cache.get('roi'):
                    if should_upload_now():
                        process_and_upload_ogm(ND_entity_ID,
                                               os.path.join("estimated_OGM", output_tiff_path_timestamp),
                                               config.BUCKET_NAME, metadata_timestamp)
            except Exception as e:
                logger.error(f"Error saving or uploading OGM: {e}")
            # ----------------------------------------------------------
            # Fuse burnt area measurements (segmented burnt area)
            # ----------------------------------------------------------
            try:
                segment_payload, consumed_segment = load_segment_georef_data(
                    "georeferenced_drone_images/segmented/burnt",
                    return_consumed=True,
                    delete_on_success=True,
                    newest_first=False,
                )
                if segment_payload is not None and segment_payload.get("measurement_type") == "burnt_area":
                    ogm_data = update_occupancy_grid_from_georef_segments(
                        np.asarray(ogm_data, dtype=np.float32),
                        ogm_gt_,
                        segment_payload.get("samples", []),
                        "burnt_area"
                    )
                    logger.info(
                        f"OGM updated successfully for burnt-area UAV segmentation using georef_data "
                        f"from {os.path.basename(consumed_segment) if consumed_segment else 'unknown file'}."
                    )
            except Exception as e:
                logger.info(f"Error in updating OGM for burnt area {e}")
        # ----------------------------------------------------------
        # Save updated OGM as GeoTIFF
        # ----------------------------------------------------------
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

            if metadata_timestamp.get('coordinates') == global_cache.get('roi'):
                if should_upload_now():
                    process_and_upload_ogm(ND_entity_ID,
                                           os.path.join("estimated_OGM", output_tiff_path_timestamp),
                                           config.BUCKET_NAME, metadata_timestamp)
        except Exception as e:
            logger.error(f"Error saving or uploading OGM: {e}")
        # ----------------------------------------------------------
        # Update OGM with satellite data
        # ----------------------------------------------------------
        try:
            observ_sat_data_, observ_sat_gt_, observ_sat_proj, data_type, observ_nodata = load_image(
                f"downloads/satellite_imgs/{disaster_type}", 3)
            if observ_sat_data_ is not None:
                observ_sat_data_ = np.asarray(observ_sat_data_, dtype=np.float32)
                if data_type == 'Flood':
                    ogm_data = update_occupancy_grid_flood_sat_fixed(
                        ogm_data, ogm_gt_,
                        observ_sat_data_, observ_sat_gt_,
                        ogm_crs='EPSG:4326',
                        observation_crs=observ_sat_proj,
                        obs_nodata=observ_nodata
                    )
                    # ----------------------------------------------------------
                    # Save updated OGM as GeoTIFF
                    # ----------------------------------------------------------
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

                        if metadata_timestamp.get('coordinates') == global_cache.get('roi'):
                            if should_upload_now():
                                process_and_upload_ogm(ND_entity_ID,
                                                       os.path.join("estimated_OGM", output_tiff_path_timestamp),
                                                       config.BUCKET_NAME, metadata_timestamp)
                    except Exception as e:
                        logger.error(f"Error saving or uploading OGM: {e}")
                    # ----------------------------------------------------------
                """
                commented for SAR trial
                """
                # elif data_type == 'burnt_area':
                #     ogm_data = update_occupancy_grid_fire_sat_fixed(
                #         ogm_data,
                #         ogm_gt_,
                #         observ_sat_data_,
                #         observ_sat_gt_,
                #         'burnt_area',
                #         ogm_crs='EPSG:4326',
                #         observation_crs=observ_sat_proj,
                #         obs_nodata=observ_nodata)
                #     # ----------------------------------------------------------
                #     # Save updated OGM as GeoTIFF
                #     # ----------------------------------------------------------
                #     try:
                #         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                #         output_tiff_path_timestamp = f"occupancy_grid_map_{disaster_type}_{timestamp}.tif"
                #
                #         output_tiff_path = f"occupancy_grid_map_{disaster_type}.tif"
                #         save_geotiff("estimated_OGM",
                #                      output_tiff_path,
                #                      ogm_data,
                #                      ogm_gt_,
                #                      timestamp,
                #                      crs_epsg=4326)
                #
                #         metadata_timestamp = save_geotiff("estimated_OGM",
                #                                           output_tiff_path_timestamp,
                #                                           ogm_data,
                #                                           ogm_gt_,
                #                                           timestamp,
                #                                           crs_epsg=4326)
                #         if metadata_timestamp.get('coordinates') == global_cache.get('roi'):
                #             if should_upload_now():
                #                 process_and_upload_ogm(ND_entity_ID,
                #                                        os.path.join("estimated_OGM", output_tiff_path_timestamp),
                #                                        config.BUCKET_NAME, metadata_timestamp)
                #     except Exception as e:
                #         logger.error(f"Error saving or uploading OGM: {e}")
                #     # ----------------------------------------------------------
                # logger.info(f"OGM has successfully generated and fused sate file")
        except FileNotFoundError:
            logger.warning("Satellite ROI directory not found.")
        except Exception as e:
            logger.info(f"Sat measurements are not available {e}")
        # ----------------------------------------------------------
        # Update OGM with hotspot data from the SocialMedia directory
        # ----------------------------------------------------------
        # try:
        #     eps = 1e-6
        #     social_weight = float(getattr(config, "HOTSPOT_SOCIAL_WEIGHT", 0.15))
        #     min_hotspot_score = float(getattr(config, "HOTSPOT_MIN_SCORE", 0.18))
        #
        #     with rasterio.open(ogm_path) as src:
        #         ogm_profile = src.profile
        #         ogm_data = src.read(1).astype(np.float32)
        #         transform = src.transform
        #         H, W = ogm_data.shape
        #         crs = src.crs
        #         bounds = src.bounds
        #         ogm_crs_str = crs.to_string()
        #
        #     aoi_wgs = Polygon(global_cache.get('roi')[0])
        #     if crs.to_epsg() != 4326:
        #         tr = Transformer.from_crs("EPSG:4326", ogm_crs_str, always_xy=True)
        #         aoi_geom = shapely_transform(tr.transform, aoi_wgs)
        #     else:
        #         aoi_geom = aoi_wgs
        #
        #     aoi_mask = rasterize(
        #         [(mapping(aoi_geom), 1)],
        #         out_shape=(H, W),
        #         transform=transform,
        #         fill=0,
        #         dtype="uint8"
        #     ).astype(bool)
        #
        #     social_media_dir = "downloads/SocialMedia"
        #     hotspot_files = [
        #         os.path.join(social_media_dir, f)
        #         for f in os.listdir(social_media_dir)
        #         if f.startswith("hotspot_results") and f.endswith(".geojson")
        #     ]
        #     hotspot_files.sort()
        #
        #     for hotspot_file in hotspot_files:
        #         try:
        #             with open(hotspot_file) as f:
        #                 gj = json.load(f)
        #
        #             hot_crs = gj.get("crs", {}).get("properties", {}).get("name", "EPSG:4326")
        #             if hot_crs == "urn:ogc:def:crs:OGC:1.3:CRS84":
        #                 hot_crs = "EPSG:4326"
        #
        #             tr_h = None
        #             if hot_crs != ogm_crs_str:
        #                 tr_h = Transformer.from_crs(hot_crs, ogm_crs_str, always_xy=True)
        #
        #             features = gj.get("features", [])
        #             if not features:
        #                 os.remove(hotspot_file)
        #                 continue
        #
        #             # Use only the latest hotspot snapshot contained in the file
        #             latest_date = max(
        #                 (feat.get("properties", {}) or {}).get("date", "")
        #                 for feat in features
        #             )
        #             if latest_date:
        #                 features = [
        #                     feat for feat in features
        #                     if (feat.get("properties", {}) or {}).get("date", "") == latest_date
        #                 ]
        #
        #             logger.info(
        #                 f"Hotspot file {os.path.basename(hotspot_file)} -> "
        #                 f"using latest date {latest_date!r} with {len(features)} features"
        #             )
        #
        #             hotspot_support = np.zeros((H, W), dtype=np.float32)
        #             used_hotspots = []
        #
        #             for feat in features:
        #                 props = feat.get("properties", {})
        #                 geom = feat.get("geometry")
        #                 if not geom:
        #                     continue
        #
        #                 poly = shape(geom)
        #                 if not poly.is_valid:
        #                     poly = poly.buffer(0)
        #                 if poly.is_empty or not poly.is_valid:
        #                     logger.warning(f"Invalid polygon in {hotspot_file}")
        #                     continue
        #
        #                 hotspot_score = compute_hotspot_score(props)
        #                 if hotspot_score < min_hotspot_score:
        #                     continue
        #
        #                 if tr_h:
        #                     poly = shapely_transform(tr_h.transform, poly)
        #
        #                 poly_mask = rasterize(
        #                     [(mapping(poly), hotspot_score)],
        #                     out_shape=(H, W),
        #                     transform=transform,
        #                     fill=0.0,
        #                     dtype="float32",
        #                     all_touched=True
        #                 )
        #                 hotspot_support = np.maximum(hotspot_support, poly_mask)
        #                 used_hotspots.append({
        #                     "h3": props.get("h3"),
        #                     "score": round(hotspot_score, 4)
        #                 })
        #
        #             if not used_hotspots:
        #                 logger.info(
        #                     f"No hotspot features passed score threshold in "
        #                     f"{os.path.basename(hotspot_file)} (min_hotspot_score={min_hotspot_score})"
        #                 )
        #                 os.remove(hotspot_file)
        #                 continue
        #
        #             sigma_pixels = sigma_meters_to_pixels(transform, bounds)
        #             hotspot_influence = gaussian_filter(hotspot_support, sigma=sigma_pixels, mode="nearest")
        #             if hotspot_influence.max() > 0:
        #                 hotspot_influence = hotspot_influence / hotspot_influence.max()
        #
        #             hotspot_influence = np.clip(hotspot_influence, 0.0, 1.0)
        #             mask = (hotspot_influence > 0) & aoi_mask
        #
        #             if mask.any():
        #                 p_prev = np.clip(ogm_data, 0.0, 1.0)
        #                 p_soc = np.clip(social_weight * hotspot_influence, eps, 1.0 - eps)
        #
        #                 p_new = p_prev.copy()
        #                 p_new[mask] = 1.0 - ((1.0 - p_prev[mask]) * (1.0 - p_soc[mask]))
        #                 ogm_data = np.clip(p_new, 0.0, 1.0)
        #             else:
        #                 logger.info(
        #                     f"No hotspot influence intersected AOI in {os.path.basename(hotspot_file)}"
        #                 )
        #
        #             os.remove(hotspot_file)
        #
        #             timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        #             out_base = f"occupancy_grid_map_{disaster_type}"
        #             fn1 = f"{out_base}.tif"
        #             fn2 = f"{out_base}_{timestamp}.tif"
        #
        #             for fn in (fn1, fn2):
        #                 with rasterio.open(os.path.join("estimated_OGM", fn), "w", **ogm_profile) as dst:
        #                     dst.write(ogm_data, 1)
        #
        #             metadata_timestamp = save_geotiff(
        #                 "estimated_OGM",
        #                 fn2,
        #                 ogm_data,
        #                 ogm_gt_,
        #                 timestamp,
        #                 crs_epsg=4326
        #             )
        #
        #             now_6 = time.monotonic()
        #             if now_6 - _last_upload >= UPLOAD_INTERVAL:
        #                 logger.info(f"now time and latest_upload ----------> {now_6}   {_last_upload}")
        #                 process_and_upload_ogm(
        #                     ND_entity_ID,
        #                     os.path.join("estimated_OGM", fn2),
        #                     config.BUCKET_NAME,
        #                     metadata=metadata_timestamp
        #                 )
        #                 _last_upload = now_6
        #
        #             logger.info(
        #                 f"Fused & removed {hotspot_file} using polygon-support hotspot influence "
        #                 f"(used_hotspots={len(used_hotspots)}, sigma_pixels={sigma_pixels:.2f}, "
        #                 f"top_h3={[h['h3'] for h in used_hotspots[:5]]})"
        #             )
        #
        #         except Exception as e:
        #             logger.error(f"Failed on {hotspot_file}: {e}")
        #
        # except FileNotFoundError:
        #     logger.warning("SocialMedia directory not found.")
        # except Exception as e:
        #     logger.error(f"Error integrating hotspot data: {e}")

        try:
            eps = 1e-9
            min_hotspot_score = float(getattr(config, "HOTSPOT_MIN_SCORE", 0.40))
            hotspot_gate_threshold = float(getattr(config, "HOTSPOT_OGM_GATE_THRESHOLD", 0.50))
            hotspot_kernel_sigma_m = float(getattr(config, "HOTSPOT_KERNEL_SIGMA_METERS", 40.0))
            hotspot_radius_m = float(getattr(config, "HOTSPOT_SUPPORT_RADIUS_METERS", 50.0))
            hotspot_pmin = float(getattr(config, "HOTSPOT_LOGODDS_PMIN", 0.501))
            hotspot_pmax = float(getattr(config, "HOTSPOT_LOGODDS_PMAX", 0.525))

            with rasterio.open(ogm_path) as src:
                ogm_profile = src.profile
                ogm_data = src.read(1).astype(np.float32)
                transform = src.transform
                H, W = ogm_data.shape
                crs = src.crs
                bounds = src.bounds
                ogm_crs_str = crs.to_string()

            aoi_wgs = Polygon(global_cache.get('roi')[0])
            if crs.to_epsg() != 4326:
                tr = Transformer.from_crs("EPSG:4326", ogm_crs_str, always_xy=True)
                aoi_geom = shapely_transform(tr.transform, aoi_wgs)
            else:
                aoi_geom = aoi_wgs

            aoi_mask = rasterize(
                [(mapping(aoi_geom), 1)],
                out_shape=(H, W),
                transform=transform,
                fill=0,
                dtype="uint8"
            ).astype(bool)

            social_media_dir = "downloads/SocialMedia"
            hotspot_files = [
                os.path.join(social_media_dir, f)
                for f in os.listdir(social_media_dir)
                if f.startswith("hotspot_results") and f.endswith(".geojson")
            ]
            hotspot_files.sort()

            for hotspot_file in hotspot_files:
                try:
                    with open(hotspot_file) as f:
                        gj = json.load(f)

                    hot_crs = gj.get("crs", {}).get("properties", {}).get("name", "EPSG:4326")
                    if hot_crs == "urn:ogc:def:crs:OGC:1.3:CRS84":
                        hot_crs = "EPSG:4326"

                    tr_h = None
                    if hot_crs != ogm_crs_str:
                        tr_h = Transformer.from_crs(hot_crs, ogm_crs_str, always_xy=True)

                    features = gj.get("features", [])
                    if not features:
                        os.remove(hotspot_file)
                        continue

                    # Use only the latest hotspot snapshot contained in the file
                    latest_date = max(
                        (feat.get("properties", {}) or {}).get("date", "")
                        for feat in features
                    )
                    if latest_date:
                        features = [
                            feat for feat in features
                            if (feat.get("properties", {}) or {}).get("date", "") == latest_date
                        ]

                    logger.info(
                        f"Hotspot file {os.path.basename(hotspot_file)} -> "
                        f"using latest date {latest_date!r} with {len(features)} features"
                    )

                    mid_lat = (bounds.bottom + bounds.top) / 2.0
                    xres_deg = abs(transform.a)
                    yres_deg = abs(transform.e)
                    mx = xres_deg * 111320.0 * math.cos(math.radians(mid_lat))
                    my = yres_deg * 111320.0
                    meters_per_pixel = max((mx + my) / 2.0, 1e-3)
                    sigma_pixels = max(hotspot_kernel_sigma_m / meters_per_pixel, 1e-3)
                    radius_pixels = max(int(math.ceil(hotspot_radius_m / meters_per_pixel)), 1)

                    hotspot_support = np.zeros((H, W), dtype=np.float32)
                    used_hotspots = []

                    for feat in features:
                        props = feat.get("properties", {})
                        geom = feat.get("geometry")
                        if not geom:
                            continue

                        poly = shape(geom)
                        if not poly.is_valid:
                            poly = poly.buffer(0)
                        if poly.is_empty or not poly.is_valid:
                            logger.warning(f"Invalid polygon in {hotspot_file}")
                            continue

                        hotspot_score = compute_hotspot_score(props)
                        if hotspot_score < min_hotspot_score:
                            continue

                        if tr_h:
                            poly = shapely_transform(tr_h.transform, poly)

                        if not aoi_geom.intersects(poly):
                            continue

                        centroid = poly.centroid
                        row, col = rasterio.transform.rowcol(transform, centroid.x, centroid.y)
                        if not (0 <= row < H and 0 <= col < W):
                            continue

                        row_min = max(0, row - radius_pixels)
                        row_max = min(H, row + radius_pixels + 1)
                        col_min = max(0, col - radius_pixels)
                        col_max = min(W, col + radius_pixels + 1)

                        rr, cc = np.ogrid[row_min:row_max, col_min:col_max]
                        d2 = (rr - row) ** 2 + (cc - col) ** 2
                        local_mask = d2 <= (radius_pixels ** 2)
                        kernel = np.zeros((row_max - row_min, col_max - col_min), dtype=np.float32)
                        kernel[local_mask] = hotspot_score * np.exp(-0.5 * d2[local_mask] / (sigma_pixels ** 2))
                        hotspot_support[row_min:row_max, col_min:col_max] = np.maximum(
                            hotspot_support[row_min:row_max, col_min:col_max],
                            kernel
                        )
                        used_hotspots.append({
                            "h3": props.get("h3"),
                            "score": round(hotspot_score, 4)
                        })

                    if not used_hotspots:
                        logger.info(
                            f"No hotspot features passed score threshold in "
                            f"{os.path.basename(hotspot_file)} (min_hotspot_score={min_hotspot_score})"
                        )
                        os.remove(hotspot_file)
                        continue

                    hotspot_influence = np.clip(hotspot_support, 0.0, 1.0)
                    p_prev = np.clip(ogm_data, 0.0, 1.0)
                    mask = (hotspot_influence > 0) & aoi_mask & (p_prev >= hotspot_gate_threshold)

                    if mask.any():
                        likelihood = hotspot_pmin + (hotspot_pmax - hotspot_pmin) * hotspot_influence[mask]
                        likelihood = np.clip(likelihood, eps, 1.0 - eps)
                        prior_valid = np.clip(p_prev[mask], eps, 1.0 - eps)
                        log_odds_prior = np.log(prior_valid / (1.0 - prior_valid))
                        log_odds_obs = np.log(likelihood / (1.0 - likelihood))
                        log_odds_post = np.clip(log_odds_prior + log_odds_obs, -5.0, 5.0)

                        p_new = p_prev.copy()
                        p_new[mask] = 1.0 / (1.0 + np.exp(-log_odds_post))
                        ogm_data = np.clip(p_new, 0.0, 1.0)
                    else:
                        logger.info(
                            f"No physically plausible hotspot overlap intersected AOI in "
                            f"{os.path.basename(hotspot_file)}"
                        )

                    os.remove(hotspot_file)

                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    out_base = f"occupancy_grid_map_{disaster_type}"
                    fn1 = f"{out_base}.tif"
                    fn2 = f"{out_base}_{timestamp}.tif"

                    for fn in (fn1, fn2):
                        with rasterio.open(os.path.join("estimated_OGM", fn), "w", **ogm_profile) as dst:
                            dst.write(ogm_data, 1)

                    metadata_timestamp = save_geotiff(
                        "estimated_OGM",
                        fn2,
                        ogm_data,
                        ogm_gt_,
                        timestamp,
                        crs_epsg=4326
                    )

                    now_6 = time.monotonic()
                    if now_6 - _last_upload >= UPLOAD_INTERVAL:
                        logger.info(f"now time and latest_upload ----------> {now_6}   {_last_upload}")
                        process_and_upload_ogm(
                            ND_entity_ID,
                            os.path.join("estimated_OGM", fn2),
                            config.BUCKET_NAME,
                            metadata=metadata_timestamp
                        )
                        _last_upload = now_6

                    logger.info(
                        f"Fused & removed {hotspot_file} using gated weak local geosocial log-odds "
                        f"(used_hotspots={len(used_hotspots)}, sigma_pixels={sigma_pixels:.2f}, "
                        f"radius_pixels={radius_pixels}, gate={hotspot_gate_threshold}, "
                        f"top_h3={[h['h3'] for h in used_hotspots[:5]]})"
                    )

                except Exception as e:
                    logger.error(f"Failed on {hotspot_file}: {e}")

        except FileNotFoundError:
            logger.warning("SocialMedia directory not found.")
        except Exception as e:
            logger.error(f"Error integrating hotspot data: {e}")


"""
WITH THE PREVSIOUS FUSION OF THE SOCIAL MEDIA
"""
# def compute_hotspot_score(props):
#     """
#     Convert a HotspotResult cell's attributes into a bounded confidence score.
#
#     This is intentionally conservative: count/ratio drive the score, while the
#     significance-related terms only help when the anomaly direction is positive.
#     """
#     count = float(props.get("count", 0) or 0)
#     ratio = float(props.get("ratio", 0) or 0)
#     z_score = float(props.get("z_score", 0) or 0)
#     p_value = float(props.get("p_value", 1) or 1)
#     hotspot_bin = float(props.get("bin", 0) or 0)
#     count_related = float(props.get("count_related", 0) or 0)
#
#     count_score = min(count / 20.0, 1.0)
#     ratio_score = np.clip(ratio, 0.0, 1.0)
#     z_score_norm = np.clip(max(z_score, 0.0) / 3.0, 0.0, 1.0)
#     p_score = (1.0 - np.clip(p_value, 0.0, 1.0)) if z_score > 0 else 0.0
#     bin_score = np.clip(hotspot_bin / 3.0, 0.0, 1.0)
#     related_score = np.clip(count_related / max(count, 1.0), 0.0, 1.0) if count > 0 else 0.0
#
#     return float(np.clip(
#         0.30 * count_score +
#         0.25 * ratio_score +
#         0.20 * z_score_norm +
#         0.10 * p_score +
#         0.10 * bin_score +
#         0.05 * related_score,
#         0.0,
#         1.0
#     ))

def compute_hotspot_score(props):
    """
    Convert a HotspotResult cell's attributes into a bounded confidence score.

    Geosocial hotspots are indirect evidence. This score emphasizes event
    specificity over raw traffic volume and rejects cells with no sign of
    event-relevant activity.
    """
    count = float(props.get("count", 0) or 0)
    ratio = float(props.get("ratio", 0) or 0)
    z_score = float(props.get("z_score", 0) or 0)
    p_value = float(props.get("p_value", 1) or 1)
    hotspot_bin = float(props.get("bin", 0) or 0)
    count_related = float(props.get("count_related", 0) or 0)

    if not (ratio > 0.0 or z_score > 0.0 or hotspot_bin > 0.0 or count_related > 0.0):
        return 0.0

    count_score = np.clip(np.log1p(count) / np.log1p(100.0), 0.0, 1.0)
    ratio_score = np.clip(ratio, 0.0, 1.0)
    z_score_norm = np.clip(max(z_score, 0.0) / 3.0, 0.0, 1.0)
    p_score = (1.0 - np.clip(p_value, 0.0, 1.0)) if z_score > 0 else 0.0
    bin_score = np.clip(hotspot_bin / 3.0, 0.0, 1.0)
    related_score = np.clip(count_related / max(count, 1.0), 0.0, 1.0) if count > 0 else 0.0

    return float(np.clip(
        0.15 * count_score +
        0.30 * ratio_score +
        0.25 * z_score_norm +
        0.10 * p_score +
        0.15 * bin_score +
        0.05 * related_score,
        0.0,
        1.0
    ))


def sigma_meters_to_pixels(transform, bounds):
    """
    Convert a metric smoothing scale into raster pixels.

    The OGM is usually EPSG:4326, so approximate metres-per-pixel from the
    geotransform at the raster mid-latitude.
    """
    sigma_meters = float(getattr(config, "HOTSPOT_SMOOTH_SIGMA_METERS", 200.0))
    mid_lat = (bounds.bottom + bounds.top) / 2.0
    xres_deg = abs(transform.a)
    yres_deg = abs(transform.e)
    mx = xres_deg * 111320.0 * math.cos(math.radians(mid_lat))
    my = yres_deg * 111320.0
    meters_per_pixel = max((mx + my) / 2.0, 1e-3)
    return sigma_meters / meters_per_pixel


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


def associate_features_with_measurements(ogm_features, new_features, r_gate_m,
                                        alpha=0.7, beta=0.3, default_conf=0.5):
    N, M = len(ogm_features), len(new_features)
    if N == 0 or M == 0:
        return []

    INF = 1e12
    cost = np.full((N, M), INF, dtype=np.float64)

    for i, feat in enumerate(ogm_features):
        c1 = feat.get("geometry", {}).get("coordinates", [])
        if len(c1) < 2:
            continue
        lon1, lat1 = c1[0], c1[1]

        for j, det in enumerate(new_features):
            c2 = det.get("geometry", {}).get("coordinates", [])
            if len(c2) < 2:
                continue
            lon2, lat2 = c2[0], c2[1]

            d = haversine_distance((lon1, lat1), (lon2, lat2))
            if d > r_gate_m:
                continue  # stays INF (not a candidate)

            conf = det.get("properties", {}).get("score", default_conf)
            conf = float(conf) if conf is not None else default_conf

            cost[i, j] = alpha * (d / r_gate_m) + beta * (1.0 - conf)

    # Add dummies so solver can leave items unmatched
    size = max(N, M)
    UNMATCH = 1.0  # slightly above best possible match (0..1)
    padded = np.full((size, size), UNMATCH, dtype=np.float64)
    padded[:N, :M] = cost

    row_ind, col_ind = linear_sum_assignment(padded)

    matches = []
    for r, c in zip(row_ind, col_ind):
        if r < N and c < M and padded[r, c] < UNMATCH:
            matches.append((r, c))
    return matches


# ----------------------------------------------------------------------
def get_objects_lock(disaster_type):
    """Get a lock for a specific disaster type's Objects files"""
    with objects_file_locks_lock:
        if disaster_type not in objects_file_locks:
            objects_file_locks[disaster_type] = threading.RLock()
        return objects_file_locks[disaster_type]
# ----------------------------------------------------------------------


def estimate_objects_status(predict_only: bool = False):
    """Multi-object fusion / tracking for Persons + Vehicles using a 6D CV KF.

    State (6D):  [x, y, z, vx, vy, vz] in LOCAL METRES (x east, y north, z up).
    Meas  (3D):  [x, y, z] in LOCAL METRES.
    Geometry persisted/published as GeoJSON lon/lat/elev (EPSG:4326).

    Processing order per call
    ─────────────────────────
    Phase 0 – Predict
        For every existing track build the KF from its stored state, call
        kf.predict(), write x̄_{t|t-1} into feat["geometry"]["coordinates"],
        and store the predicted KF in predicted_kfs[track_id].

    Phase 1 – Associate (drone) → inline KF update
        associate_features_with_measurements() runs against the predicted
        positions written in Phase 0.  For every confirmed match the already-
        predicted KF is updated inline with kf.update(z), where kf.R is set
        to the confidence-scaled measurement covariance (Paper Eq. 8):
            R_{t,j} = R_base / max(c^det_{t,j}, ε_c)
        so high-confidence detections exert a stronger pull than weak ones.
        Deletions and appends use track_id throughout — no index-shift risk.

    Phase 2 – Social media score/label fusion
        Score and label are Bayesian-fused into matched tracks.
        No KF position update (social positions are too coarse).

    Phase 3 – Finalise KF state
        Write kf_x / kf_P / kf_t back for every existing track.
        Brand-new tracks (created in Phase 1) skip this — their KF state was
        set in _new_track_from_measurement().

    Predict-only path
    ─────────────────
    Runs Phase 0 (predict all tracks), increments miss counters, writes the
    predicted state back, then falls through to snapshot/upload.
    Only active after objects_tracking_started == True.
    """

    # -------------------------------------------------------------------------
    # Hyperparameters
    # -------------------------------------------------------------------------
    max_step_distance_m    = float(getattr(config, "MAX_STEP_DISTANCE_M",    7.0))
    reset_threshold_m      = float(getattr(config, "RESET_THRESHOLD_M",     10.0))
    max_misses             = int(getattr(config,   "MAX_MISSES",              5))
    confirm_updates        = int(getattr(config,   "CONFIRM_UPDATES",         2))
    tentative_max_misses   = int(getattr(config,   "TENTATIVE_MAX_MISSES",    max(1, confirm_updates)))
    max_no_evidence_ticks  = int(getattr(config,   "MAX_NO_EVIDENCE_TICKS",  10))
    max_misses             = max(max_misses, max_no_evidence_ticks)

    meas_sigma_m           = float(getattr(config, "OBJ_MEAS_SIGMA_M",       2.0))
    accel_sigma_m_s2       = float(getattr(config, "OBJ_ACCEL_SIGMA_M_S2",   0.025))
    init_vel_sigma_m_s     = float(getattr(config, "OBJ_INIT_VEL_SIGMA_M_S", 1.5))
    eps_c                  = float(getattr(config, "OBJ_MEAS_CONF_EPS",       0.3))  # Paper ε_c
    c_init                 = float(getattr(config, "C_INIT",       0.3))

    disaster_type = global_cache.get("natural_disaster", "NoDisaster")
    if disaster_type == "NoDisaster":
        logger.info("Objects: No valid disaster type. Aborting.")
        return
    if alert_event.is_set():
        return

    ogm_metadata_path = f"estimated_OGM/occupancy_grid_map_{disaster_type}_Objects.json"
    if not os.path.exists(ogm_metadata_path):
        logger.info(f"Objects: base OGM JSON not found at {ogm_metadata_path}. Aborting.")
        return

    detection_dir    = "georeferenced_drone_images/detection"
    social_media_dir = "downloads/SocialMedia"

    now_iso = datetime.now().isoformat()
    now_t   = time.time()

    # -------------------------------------------------------------------------
    # Local tangent-plane helpers
    # -------------------------------------------------------------------------
    R_EARTH = 6378137.0

    def _roi_centroid_lonlat():
        roi    = global_cache.get("roi", None)
        coords = roi.get("coordinates") if isinstance(roi, dict) else roi
        pts    = None
        if isinstance(coords, list) and coords:
            if isinstance(coords[0], list) and isinstance(coords[0][0], (list, tuple)):
                pts = coords[0]
            else:
                pts = coords
        if not pts:
            return None
        xs, ys = [], []
        for p in pts:
            if not isinstance(p, (list, tuple)) or len(p) < 2:
                continue
            try:
                xs.append(float(p[0])); ys.append(float(p[1]))
            except Exception:
                continue
        return (sum(xs) / len(xs), sum(ys) / len(ys)) if xs else None

    origin  = _roi_centroid_lonlat()
    if origin is None:
        ig = global_cache.get("ignitionPoints", None)
        if isinstance(ig, (list, tuple)) and len(ig) >= 2:
            try:    origin = (float(ig[0]), float(ig[1]))
            except: origin = None
    if origin is None:
        origin = (0.0, 0.0)
    lon0, lat0 = origin
    cos_lat0   = max(0.1, math.cos(math.radians(lat0)))

    def _llz_to_local_m(llz):
        llz = _ensure_3d(llz)
        if llz is None:
            return None
        return [
            math.radians(float(llz[0]) - lon0) * cos_lat0 * R_EARTH,
            math.radians(float(llz[1]) - lat0) * R_EARTH,
            float(llz[2]),
        ]

    def _local_m_to_llz(xyz):
        if xyz is None or len(xyz) < 3:
            return None
        return [
            lon0 + math.degrees(float(xyz[0]) / (R_EARTH * cos_lat0)),
            lat0 + math.degrees(float(xyz[1]) / R_EARTH),
            float(xyz[2]),
        ]

    def _ensure_3d(coords):
        if coords is None:
            return None
        if len(coords) == 2:
            return [float(coords[0]), float(coords[1]), 0.0]
        return [float(coords[0]), float(coords[1]),
                float(coords[2]) if coords[2] is not None else 0.0]

    # -------------------------------------------------------------------------
    # Confidence-scaled measurement covariance  (Paper Eq. 8)
    #   R_{t,j} = R_base / max(c^det_{t,j}, ε_c)
    # High confidence → small R → strong pull on filter.
    # Low  confidence → large R → weak  pull on filter.
    # -------------------------------------------------------------------------
    def _confidence_scaled_R(score: float) -> np.ndarray:
        scale = 1.0 / max(float(score), eps_c)
        return np.diag([meas_sigma_m ** 2 * scale] * 3)

    # -------------------------------------------------------------------------
    # Social media helpers
    # -------------------------------------------------------------------------
    def _sm_prob(props: dict) -> float:
        if not isinstance(props, dict):
            return 0.5
        for key in ("emotion_label_probability", "prob"):
            if key in props:
                try:    return float(props[key])
                except: pass
        if "related" in props:
            try:    return 0.75 if int(props["related"]) == 1 else 0.25
            except: pass
        return 0.5

    def _sm_label(props: dict) -> int:
        if not isinstance(props, dict):
            return -1
        if "label" in props:
            try:    return int(props["label"])
            except: return -1
        if "related" in props:
            try:    return 1 if int(props["related"]) == 1 else -1
            except: return -1
        return -1

    # -------------------------------------------------------------------------
    # KF helpers
    # -------------------------------------------------------------------------
    def _cv_F_Q(dt: float, sigma_a: float):
        dt = max(1e-3, float(dt))
        F  = np.eye(6)
        F[0, 3] = dt; F[1, 4] = dt; F[2, 5] = dt
        q        = float(sigma_a) ** 2
        dt2, dt3, dt4 = dt ** 2, dt ** 3, dt ** 4
        Q = np.zeros((6, 6))
        for i in range(3):
            Q[i,   i  ] = dt4 / 4.0 * q
            Q[i,   i+3] = dt3 / 2.0 * q
            Q[i+3, i  ] = dt3 / 2.0 * q
            Q[i+3, i+3] = dt2       * q
        return F, Q

    def _build_kf(x6, P6, dt):
        """Build and return a KF with default (unscaled) R — caller overrides R before update."""
        F, Q = _cv_F_Q(dt, accel_sigma_m_s2)
        kf   = KalmanFilter(state_dim=6, measurement_dim=3, F=F, H=np.eye(3, 6))
        kf.x = x6;  kf.P = P6;  kf.Q = Q
        kf.R = np.diag([meas_sigma_m ** 2] * 3)   # default; overridden per-detection in Phase 1
        return kf

    def _upgrade_kf_state(props: dict, llz_geom):
        xyz = _llz_to_local_m(llz_geom)
        if xyz is None:
            return None, None
        x6 = np.array(xyz + [0., 0., 0.], dtype=float).reshape(6, 1)
        P6 = np.diag([meas_sigma_m ** 2] * 3 + [init_vel_sigma_m_s ** 2] * 3).astype(float)
        kx = props.get("kf_x")
        kP = props.get("kf_P")
        try:
            if isinstance(kx, (list, tuple)):
                if len(kx) == 6:
                    x6 = np.array(kx, dtype=float).reshape(6, 1)
                elif len(kx) == 3:
                    a, b, c = float(kx[0]), float(kx[1]), float(kx[2])
                    if -180. <= a <= 180. and -90. <= b <= 90.:
                        xyz2 = _llz_to_local_m([a, b, c])
                        if xyz2:
                            x6 = np.array(xyz2 + [0., 0., 0.], dtype=float).reshape(6, 1)
                    else:
                        x6 = np.array([a, b, c, 0., 0., 0.], dtype=float).reshape(6, 1)
        except Exception:
            pass
        try:
            if isinstance(kP, (list, tuple)):
                if len(kP) == 36:
                    P6 = np.array(kP, dtype=float).reshape(6, 6)
                elif len(kP) == 9:
                    P6 = np.diag([meas_sigma_m ** 2] * 3 +
                                 [init_vel_sigma_m_s ** 2] * 3).astype(float)
                    P6[0:3, 0:3] = np.array(kP, dtype=float).reshape(3, 3)
        except Exception:
            pass
        return x6, P6

    def _new_track_from_measurement(meas_llz, meas_props, source: str):
        llz = _ensure_3d(meas_llz)
        if llz is None:
            return None
        xyz = _llz_to_local_m(llz)
        if xyz is None:
            return None
        try:    score = float(meas_props.get("score", 0.5)) if isinstance(meas_props, dict) else 0.5
        except: score = 0.5
        try:    label = int(meas_props.get("label", -1))    if isinstance(meas_props, dict) else -1
        except: label = -1

        # Initial position uncertainty is confidence-scaled (Paper Eq. 8):
        # a low-confidence first detection starts with larger position variance.
        r_init = meas_sigma_m ** 2 / max(float(score), eps_c)

        return {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": llz},
            "properties": {
                "score":  score,
                "label":  label,
                "source": source,
                "creation_time": now_iso,
                "last_update":   now_iso,
                "update_count":  1,
                "track_id":      str(uuid.uuid4()),
                "confirmed":     confirm_updates <= 1,
                "miss_count":    0,
                "grid_center":           llz.copy(),
                "raw_measurement":       llz.copy(),
                "raw_measurement_local": xyz.copy(),
                "kf_refined": llz.copy(),
                # Initial P: confidence-scaled position block + default velocity block
                "kf_x": xyz + [0., 0., 0.],
                "kf_P": np.diag([r_init] * 3 +
                                [init_vel_sigma_m_s ** 2] * 3).reshape(-1).tolist(),
                "kf_t":      now_t,
                "kf_origin": [lon0, lat0],
            }
        }

    def _ensure_lifecycle_fields(feats):
        for feat in feats:
            p = feat.setdefault("properties", {})
            p.setdefault("creation_time", now_iso)
            p.setdefault("last_update",   now_iso)
            p.setdefault("update_count",  0)
            p.setdefault("miss_count",    0)
            p.setdefault("confirmed",     False)
            p.setdefault("track_id",      str(uuid.uuid4()))
            p.setdefault("kf_t",          now_t)
            p.setdefault("kf_origin",     [lon0, lat0])

    # -------------------------------------------------------------------------
    # Early exit
    # -------------------------------------------------------------------------
    has_drone  = (os.path.isdir(detection_dir) and
                  any(f.endswith("_Objects.tif") for f in os.listdir(detection_dir)))
    has_social = (os.path.isdir(social_media_dir) and
                  any(f.startswith("single_posts_results") and f.endswith(".geojson")
                      for f in os.listdir(social_media_dir)))

    if not has_drone and not has_social and not predict_only:
        logger.debug("Objects: no new evidence and not predict_only -> nothing to do.")
        return

    bm_id         = global_cache.get("bm_id")
    obj_entity_ID = f"urn:ngsi-ld:USE:PDM-05:Maps4Object:{bm_id}" if bm_id else None
    did_update = drone_updated = social_updated = False

    with get_objects_lock(disaster_type):

        # ---------------------------------------------------------------------
        # Load persistent track list
        # ---------------------------------------------------------------------
        try:
            with open(ogm_metadata_path, "r") as f:
                ogm_metadata = json.load(f)
        except Exception as e:
            logger.error(f"Objects: failed reading {ogm_metadata_path}: {e}", exc_info=True)
            return

        ogm_features = ogm_metadata.get("features", []) or []
        _ensure_lifecycle_fields(ogm_features)

        # =====================================================================
        # PHASE 0 — PREDICT ALL EXISTING TRACKS
        # Keyed by track_id so deletions/appends in Phase 1 cannot corrupt cache.
        # Geometry is advanced to x̄_{t|t-1} so association uses predicted pos.
        # =====================================================================
        predicted_kfs: dict[str, KalmanFilter] = {}

        for feat in ogm_features:
            props   = feat.get("properties", {})
            tid     = props.get("track_id")
            if not tid:
                continue

            llz_geom = _ensure_3d(feat["geometry"]["coordinates"])
            x6, P6   = _upgrade_kf_state(props, llz_geom)
            if x6 is None:
                continue

            last_t = float(props.get("kf_t", now_t))
            dt     = max(1e-3, now_t - last_t)

            kf = _build_kf(x6, P6, dt)
            try:
                kf.predict()
            except Exception as e:
                logger.warning(f"Objects: predict failed for track {tid}: {e}")
                continue

            # Advance geometry to predicted position for association.
            pred_llz = _local_m_to_llz(kf.x.flatten()[:3].tolist())
            if pred_llz is not None:
                feat["geometry"]["coordinates"] = pred_llz

            predicted_kfs[tid] = kf   # already predicted; ready for optional update

        # =====================================================================
        # PREDICT-ONLY PATH
        # =====================================================================
        if predict_only and not has_drone and not has_social:
            if not global_cache.get("objects_tracking_started", False):
                logger.debug("Objects: predict_only ignored (tracking not started).")
                return
            if not ogm_features:
                logger.debug("Objects: predict_only tick but no tracks exist -> stop.")
                return

            tracks_to_delete = set()
            for feat in ogm_features:
                props = feat.setdefault("properties", {})
                tid   = props.get("track_id")
                props["raw_measurement"]       = None
                props["raw_measurement_local"] = None
                props["miss_count"] = int(props.get("miss_count", 0)) + 1
                props["last_update"] = datetime.now().isoformat()

                kf = predicted_kfs.get(tid)
                if kf is not None:
                    x_new   = kf.x.flatten().tolist()
                    llz_new = _local_m_to_llz(x_new[:3])
                    if llz_new is not None:
                        feat["geometry"]["coordinates"] = llz_new
                        props["kf_refined"] = llz_new
                    props["kf_x"] = x_new
                    props["kf_P"] = kf.P.reshape(-1).tolist()
                    props["kf_t"] = now_t

                if props["miss_count"] >= max_no_evidence_ticks:
                    tracks_to_delete.add(id(feat))

            ogm_features = [f for f in ogm_features
                            if id(f) not in tracks_to_delete]

            if not ogm_features:
                ogm_metadata["features"] = []
                try:
                    with open(ogm_metadata_path, "w") as f:
                        json.dump(ogm_metadata, f, indent=4)
                except Exception as e:
                    logger.warning(f"Objects: failed writing empty state: {e}")
                global_cache.set("objects_tracking_started", False)
                logger.info("Objects: all tracks expired after predict-only ticks -> stop tracking.")
                return

            did_update = True
            # Fall through to snapshot/upload.

        # =====================================================================
        # PHASE 1 — DRONE: ASSOCIATE against predicted positions → INLINE UPDATE
        # kf.R is set per-detection using the confidence-scaled covariance.
        # =====================================================================
        if not predict_only and has_drone:
            while True:
                fc, consumed_path = get_geotiff_metadata_rasterio(
                    detection_dir,
                    return_consumed=True,
                    delete_on_success=True,
                    newest_first=False,
                )
                if fc is None:
                    break

                drone_features = fc.get("features", []) or []
                logger.info(f"Objects: consumed tif={os.path.basename(consumed_path)}"
                            f" features={len(drone_features)}")
                if not drone_features:
                    continue

                # ── Cold start ───────────────────────────────────────────────
                if not ogm_features:
                    for df in drone_features:
                        new_feat = _new_track_from_measurement(
                            df["geometry"].get("coordinates"),
                            df.get("properties") or {},
                            source="drone",
                        )
                        if new_feat is not None:
                            ogm_features.append(new_feat)
                    drone_updated = True
                    continue

                # ── Associate against PREDICTED positions (Phase 0 output) ──
                matches_raw = associate_features_with_measurements(
                    ogm_features, drone_features, float(max_step_distance_m)
                )

                matched_track_ids = set()
                matched_det_idxs  = set()
                tracks_force_del  = set()
                valid_matches     = []

                for (i_trk, i_det) in matches_raw:
                    trk_llz = _ensure_3d(ogm_features[i_trk]["geometry"]["coordinates"])
                    det_llz = _ensure_3d(drone_features[i_det]["geometry"].get("coordinates"))
                    if trk_llz is None or det_llz is None:
                        continue
                    d_m = haversine_distance((trk_llz[0], trk_llz[1]),
                                            (det_llz[0], det_llz[1]))
                    tid = ogm_features[i_trk]["properties"].get("track_id")
                    if d_m > reset_threshold_m:
                        if tid:
                            tracks_force_del.add(tid)
                        continue
                    valid_matches.append((i_trk, i_det))
                    if tid:
                        matched_track_ids.add(tid)
                    matched_det_idxs.add(i_det)

                # ── Inline KF update with confidence-scaled R ────────────────
                for (i_trk, i_det) in valid_matches:
                    feat      = ogm_features[i_trk]
                    props     = feat.setdefault("properties", {})
                    tid       = props.get("track_id")
                    det_feat  = drone_features[i_det]
                    det_props = det_feat.get("properties") or {}
                    det_llz   = _ensure_3d(det_feat["geometry"].get("coordinates"))
                    if det_llz is None:
                        continue
                    det_xyz = _llz_to_local_m(det_llz)

                    kf = predicted_kfs.get(tid)
                    if kf is not None and det_xyz is not None:
                        z = np.array(det_xyz, dtype=float).reshape(3, 1)
                        try:
                            # Scale R by detection confidence before updating (Paper Eq. 8).
                            score = float(det_props.get("score", 0.5))
                            kf.R  = _confidence_scaled_R(score)
                            kf.update(z)
                        except Exception as e:
                            logger.warning(f"Objects: KF update failed track {tid}: {e}")
                        post_llz = _local_m_to_llz(kf.x.flatten()[:3].tolist())
                        if post_llz is not None:
                            feat["geometry"]["coordinates"] = post_llz

                    props["source"]                 = "drone"
                    props["grid_center"]            = _ensure_3d(feat["geometry"]["coordinates"]).copy()
                    props["raw_measurement"]        = det_llz.copy()
                    props["raw_measurement_local"]  = det_xyz.copy() if det_xyz else None
                    try:    props["label"] = int(det_props.get("label", props.get("label", -1)))
                    except: pass
                    try:    props["score"] = float(det_props.get("score", props.get("score", 0.5)))
                    except: pass
                    props["last_update"]  = now_iso
                    props["update_count"] = int(props.get("update_count", 0)) + 1
                    props["miss_count"]   = 0
                    if not props.get("confirmed") and props["update_count"] >= confirm_updates:
                        props["confirmed"] = True

                # ── Missed-detection handling ─────────────────────────────────
                for feat in ogm_features:
                    props = feat.get("properties", {})
                    tid   = props.get("track_id")
                    if tid in matched_track_ids or tid in tracks_force_del:
                        continue
                    props["raw_measurement"]       = None
                    props["raw_measurement_local"] = None
                    props["miss_count"] = int(props.get("miss_count", 0)) + 1
                    if ((not props.get("confirmed") and
                             props["miss_count"] >= tentative_max_misses) or
                            props["miss_count"] > max_misses):
                        tracks_force_del.add(tid)

                # ── Delete by track_id (index-safe) ───────────────────────────
                if tracks_force_del:
                    ogm_features = [
                        f for f in ogm_features
                        if f["properties"].get("track_id") not in tracks_force_del
                    ]

                # ── New tracks from unmatched detections ──────────────────────
                for j, det_feat in enumerate(drone_features):
                    if j in matched_det_idxs:
                        continue

                    ###################################
                    det_props = det_feat.get("properties") or {}
                    score = float(det_props.get("score", 0.0))

                    if score < c_init:
                        continue  # DO NOT spawn a track for weak detections
                    ###################################

                    new_feat = _new_track_from_measurement(
                        det_feat["geometry"].get("coordinates"),
                        det_feat.get("properties") or {},
                        source="drone",
                    )
                    if new_feat is not None:
                        ogm_features.append(new_feat)

                drone_updated = True

            if drone_updated:
                did_update = True
                global_cache.set("objects_tracking_started", True)
                logger.info("Objects: drone integration done")

        # =====================================================================
        # PHASE 2 — SOCIAL MEDIA: score/label fusion only (no KF position update)
        # =====================================================================
        # if not predict_only and has_social:
        #     sm_files = sorted(
        #         f for f in os.listdir(social_media_dir)
        #         if f.startswith("single_posts_results") and f.endswith(".geojson")
        #     )
        #     for sm_file in sm_files:
        #         sm_path = os.path.join(social_media_dir, sm_file)
        #         try:
        #             with open(sm_path, "r") as f:
        #                 sm_data = json.load(f)
        #         except Exception as e:
        #             logger.warning(f"Objects: could not parse {sm_path}: {e}")
        #             continue
        #
        #         sm_features = sm_data.get("features", []) or []
        #         if not sm_features:
        #             try: os.remove(sm_path)
        #             except Exception: pass
        #             continue
        #
        #         for sf in sm_features:
        #             sp = sf.setdefault("properties", {})
        #             sp["prob"]  = _sm_prob(sp)
        #             sp["label"] = _sm_label(sp)
        #             try:    sp["score"] = float(sp.get("score", sp["prob"]))
        #             except: sp["score"] = float(sp["prob"])
        #
        #         if not ogm_features:
        #             for sf in sm_features:
        #                 new_feat = _new_track_from_measurement(
        #                     sf["geometry"].get("coordinates"),
        #                     sf.get("properties") or {},
        #                     source="social",
        #                 )
        #                 if new_feat is not None:
        #                     ogm_features.append(new_feat)
        #             social_updated = did_update = True
        #             global_cache.set("objects_tracking_started", True)
        #             try: os.remove(sm_path)
        #             except Exception: pass
        #             continue
        #
        #         matches        = associate_features_with_measurements(
        #             ogm_features, sm_features, float(max_step_distance_m)
        #         )
        #         matched_sm_idxs = set()
        #
        #         for (i_trk, i_sm) in matches:
        #             trk_props = ogm_features[i_trk].setdefault("properties", {})
        #             sm_props  = sm_features[i_sm].get("properties") or {}
        #             trk_props["score"] = float(
        #                 fuse_fire_probability(
        #                     float(trk_props.get("score", 0.5)),
        #                     float(sm_props.get("prob",  0.5)),
        #                     weight=0.5,
        #                 )
        #             )
        #             sm_label = int(sm_props.get("label", -1))
        #             if sm_label != -1:
        #                 trk_props["label"] = sm_label
        #             trk_props["last_update"]  = now_iso
        #             trk_props["update_count"] = int(trk_props.get("update_count", 0)) + 1
        #             trk_props["miss_count"]   = 0
        #             if not trk_props.get("confirmed") and trk_props["update_count"] >= confirm_updates:
        #                 trk_props["confirmed"] = True
        #             matched_sm_idxs.add(i_sm)
        #
        #         for j, sf in enumerate(sm_features):
        #             if j in matched_sm_idxs:
        #                 continue
        #             new_feat = _new_track_from_measurement(
        #                 sf["geometry"].get("coordinates"),
        #                 sf.get("properties") or {},
        #                 source="social",
        #             )
        #             if new_feat is not None:
        #                 ogm_features.append(new_feat)
        #
        #         try: os.remove(sm_path)
        #         except Exception: pass
        #
        #         social_updated = did_update = True
        #         global_cache.set("objects_tracking_started", True)

        if not did_update:
            logger.debug("Objects: no updates applied; skipping write/upload.")
            return

        # =====================================================================
        # PHASE 3 — FINALISE KF STATE FOR ALL EXISTING TRACKS
        # Look up by track_id; brand-new tracks (not in predicted_kfs) skip.
        # =====================================================================
        for feat in ogm_features:
            props = feat.get("properties", {})
            tid   = props.get("track_id")
            kf    = predicted_kfs.get(tid)
            if kf is None:
                continue   # brand-new track; already initialised

            x_new   = kf.x.flatten().tolist()
            llz_new = _local_m_to_llz(x_new[:3])
            if llz_new is not None:
                feat["geometry"]["coordinates"] = llz_new
                props["kf_refined"] = llz_new
            props["kf_x"] = x_new
            props["kf_P"] = kf.P.reshape(-1).tolist()
            props["kf_t"] = now_t

        # =====================================================================
        # Track management
        # =====================================================================
        logger.info(f"Objects: pre-management track_count={len(ogm_features)}")

        ogm_features = remove_stale_objects(ogm_features, max_age_minutes=30)
        ogm_features = _keep_best_within_threshold(ogm_features, threshold_m=1.0)

        logger.info(f"Objects: post-management track_count={len(ogm_features)}")

        # =====================================================================
        # Persistence + snapshot + upload
        # =====================================================================
        ogm_metadata["features"] = ogm_features
        try:
            with open(ogm_metadata_path, "w") as f:
                json.dump(ogm_metadata, f, indent=4)
        except Exception as e:
            logger.error(f"Objects: failed writing {ogm_metadata_path}: {e}", exc_info=True)
            return

        if not ogm_features:
            global_cache.set("objects_tracking_started", False)
            logger.info("Objects: no tracks remain -> nothing to publish.")
            return

        ts            = datetime.now().strftime("%Y%m%d_%H%M%S")
        snapshot_name = f"occupancy_grid_map_{disaster_type}_Objects_{ts}.json"
        snapshot_path = os.path.join("estimated_OGM", snapshot_name)

        filtered_metadata             = copy.deepcopy(ogm_metadata)
        filtered_metadata["features"] = [
            ff for ff in filtered_metadata.get("features", [])
            if (ff.get("properties", {}).get("label", -1) != -1)
            or (ff.get("properties", {}).get("source") == "social")
        ]

        try:
            with open(snapshot_path, "w") as f:
                json.dump(filtered_metadata, f, indent=4)
        except Exception as e:
            logger.error(f"Objects: failed writing snapshot {snapshot_path}: {e}", exc_info=True)
            return

        logger.info(
            f"Objects: snapshot features={len(filtered_metadata.get('features', []))} "
            f"(drone={drone_updated}, social={social_updated}, predict_only={predict_only})"
        )

        metadata = {
            "title":            "Probability-Based Occupancy Grid Map",
            "author":           "GRVC lab, University of Seville",
            "description":      (f"Estimated occupancy grid map for the {disaster_type} scenario, "
                                 "used to track detected persons and vehicles."),
            "creationDate":     datetime.now().isoformat(),
            "spatialReference": "EPSG:4326",
            "file_name":        f"pdm05/{global_cache.get('bm_id')}/{snapshot_name}",
            "coordinates":      global_cache.get("roi", [None]),
            "XMin": 0, "XRes": 0, "YMax": 0, "YRes": 0,
            "bucket":           config.BUCKET_NAME,
            "bm_id":            global_cache.get("bm_id"),
            "sent":             ts,
            "ignitionPoints": {
                "type":  "GeoProperty",
                "value": {"type": "Point",
                          "coordinates": global_cache.get("ignitionPoints", [None])},
            },
            "minio_url": (
                f"https://{config.MINIO_ENDPOINT}/"
                f"{config.BUCKET_NAME}/pdm05/{global_cache.get('bm_id')}/{snapshot_name}"
            ),
        }

        if metadata.get("coordinates") == global_cache.get("roi"):
            try:
                logger.info(f"Objects: uploading -> {snapshot_name}")
                process_and_upload_ogm(obj_entity_ID, snapshot_path,
                                       config.BUCKET_NAME, metadata)
                logger.info("Objects: upload done")
            except Exception as e:
                logger.error(f"Objects: upload failed: {e}", exc_info=True)
        else:
            logger.warning("Objects: skipping upload (ROI mismatch).")


def _keep_best_within_threshold(features: list, threshold_m: float = 1.0) -> list:
    """
    For each cluster of tracks within threshold_m of one another, keep only
    the track with the highest (score, update_count).  The survivor's KF state
    is preserved intact, avoiding the covariance corruption that weighted-
    average merging would cause.
    """
    if len(features) <= 1:
        return features

    surviving = set(range(len(features)))

    for i in range(len(features)):
        if i not in surviving:
            continue
        ci = features[i]["geometry"]["coordinates"]
        pi = features[i]["properties"]

        for j in range(i + 1, len(features)):
            if j not in surviving:
                continue
            cj  = features[j]["geometry"]["coordinates"]
            pj  = features[j]["properties"]
            d_m = haversine_distance((ci[0], ci[1]), (cj[0], cj[1]))
            if d_m > threshold_m:
                continue
            score_i = (float(pi.get("score", 0.0)), int(pi.get("update_count", 0)))
            score_j = (float(pj.get("score", 0.0)), int(pj.get("update_count", 0)))
            if score_j > score_i:
                surviving.discard(i)
                break
            else:
                surviving.discard(j)

    kept    = [features[i] for i in sorted(surviving)]
    removed = len(features) - len(kept)
    if removed:
        logger.info(f"Objects: deduplicated {removed} close tracks (kept best survivor)")
    return kept


def remove_stale_objects(features, max_age_minutes=2, min_updates=1):
    """
    Remove objects that haven't been updated recently or have too few updates
    """
    current_time = datetime.now()
    filtered_features = []

    for feat in features:
        props = feat.get("properties", {})

        # Skip if missing essential properties
        if "creation_time" not in props or "last_update" not in props:
            continue

        try:
            # Calculate object age
            creation_time = datetime.fromisoformat(props["creation_time"])
            last_update = datetime.fromisoformat(props["last_update"])
            update_count = props.get("update_count", 0)

            object_age = (current_time - creation_time).total_seconds() / 60  # in minutes
            time_since_update = (current_time - last_update).total_seconds() / 60  # in minutes

            # Keep object if:
            # 1. It has sufficient updates AND hasn't been stale for too long, OR
            # 2. It's a recently created object (even with few updates)
            if (update_count >= min_updates and time_since_update <= max_age_minutes) or \
                    (object_age <= max_age_minutes / 2):  # Keep new objects for at least half the max age
                filtered_features.append(feat)
            else:
                logger.debug(f"Removing stale object: age={object_age:.1f}m, "
                             f"last_update={time_since_update:.1f}m, updates={update_count}")

        except (ValueError, KeyError) as e:
            logger.warning(f"Error processing object lifecycle: {e}, keeping object")
            filtered_features.append(feat)

    removed_count = len(features) - len(filtered_features)
    if removed_count > 0:
        logger.info(f"Removed {removed_count} stale objects")

    return filtered_features

# ------------------------------------------------------------------------------
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
        return polygon_coords_

    except ValueError as ve:
        logger.error(f"Validation error in convert_to_polygon: {ve}")
    except Exception as e:
        logger.exception(f"Unexpected error in convert_to_polygon: {e}")

    # Return None in case of failure
    return None


def initialize_entities():
    """Initialize entities based on a natural disaster type and objects."""
    global ND_entity_ID, obj_entity_ID, entity_creation_lock

    with entity_creation_lock:
        # Log the natural disaster info from the cache
        natural_disaster = global_cache.get('natural_disaster', 'No disaster info')
        logger.info(f"Natural disaster info from global_cache: {natural_disaster}")

        # Determine ND_entity_ID based on a natural disaster type
        if natural_disaster == 'Fire':
            ND_entity_ID = config.ENTITY_Maps4Fire_ID + f'{global_cache.get("bm_id")}'
        elif natural_disaster == 'Flood':
            ND_entity_ID = config.ENTITY_Maps4Flood_ID + f'{global_cache.get("bm_id")}'
        else:
            logger.warning(f"Unrecognized natural disaster type: {natural_disaster}")
            ND_entity_ID = None

        # Set the object entity ID
        obj_entity_ID = config.ENTITY_Maps4Object_ID + f'{global_cache.get("bm_id")}'

        # List of entities to process. The ND map entity must only be created
        # after the initial neutral OGM has been uploaded and staged.
        entity_ids = [obj_entity_ID]
        if ND_entity_ID:
            if global_cache.get('initial_nd_ogm_metadata'):
                entity_ids.insert(0, ND_entity_ID)
            else:
                logger.error(
                    f"Skipping ND entity creation for {ND_entity_ID}: "
                    "initial OGM upload metadata is missing."
                )

        for entity_id in entity_ids:
            if not check_entity_exists(entity_id):
                # logger.info(f"Entity {entity_id} does not exist, attempting to create it.")
                try:
                    parts = entity_id.split(":")
                    entity_type = parts[-2]
                    response = create_entity(entity_id, entity_type)

                    if isinstance(response, dict):
                        status = response.get("status")
                        if status == "Entity already exists":
                            logger.info(f"Entity {entity_id} already exists, no action needed.")
                        elif status == "created":
                            pass
                        elif status == "Error":
                            logger.error(f"Failed to create entity {entity_id}: {response}")
                    else:
                        logger.error(f"Unexpected response type for entity creation: {response}")
                except Exception as e:
                    logger.exception(f"Exception occurred while creating entity {entity_id}: {e}")
            else:
                logger.info(f"Entity {entity_id} already exists, skipping creation.")


# ------------------------------------------------------------------------------
# Check Entity Existence
# ------------------------------------------------------------------------------
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


# ------------------------------------------------------------------------------
# Create Entity
# ------------------------------------------------------------------------------
def _valid_lonlat_point(coordinates):
    if not isinstance(coordinates, (list, tuple)) or len(coordinates) < 2:
        return False
    try:
        float(coordinates[0])
        float(coordinates[1])
        return True
    except (TypeError, ValueError):
        return False


def _valid_polygon_coordinates(coordinates):
    if not isinstance(coordinates, list) or not coordinates:
        return False
    ring = coordinates[0]
    if not isinstance(ring, list) or len(ring) < 4:
        return False
    return all(_valid_lonlat_point(point) for point in ring)


def create_entity(entity_ID, entity_type_):
    natural_disaster = global_cache.get('natural_disaster', 'No disaster info')
    logger.info("Attempting to create entity...")  # Log for debugging
    url = f'{config.BROKER_URL}/ngsi-ld/v1/entities/'

    headers = {
        'Content-Type': 'application/ld+json',
        'Accept': 'application/ld+json'
    }
    time.sleep(0.1)
    # Check if entity exists
    if check_entity_exists(entity_ID):
        logger.info(f"Entity {entity_ID} already exists.")
        return {"status": "Entity already exists"}, 409

    ignition_points = global_cache.get('ignitionPoints', None)
    roi_coordinates = global_cache.get('roi', None)
    if not _valid_polygon_coordinates(roi_coordinates):
        roi_coordinates = [
            [
                [-180.0, -90.0],
                [-180.0, 90.0],
                [180.0, 90.0],
                [180.0, -90.0],
                [-180.0, -90.0]
            ]
        ]

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
            "coordinates": roi_coordinates
        },
        "bm_id": global_cache.get("bm_id"),
        "@context": [
            "https://uri.etsi.org/ngsi-ld/v1/ngsi-ld-core-context.jsonld"
        ]
    }

    if natural_disaster == 'Fire' and _valid_lonlat_point(ignition_points):
        data["ignitionPoints"] = {
            "type": "GeoProperty",
            "value": {
                "type": "Point",
                "coordinates": ignition_points
            }
        }

    initial_nd_metadata = global_cache.get('initial_nd_ogm_metadata')
    if entity_ID == ND_entity_ID and initial_nd_metadata:
        data["minio_url"] = {
            "type": "Property",
            "value": initial_nd_metadata["minio_url"]
        }
        data["filename"] = {
            "type": "Property",
            "value": initial_nd_metadata["file_name"]
        }
        data["bucket"] = {
            "type": "Property",
            "value": initial_nd_metadata["bucket"]
        }
        data["sent"] = {
            "type": "Property",
            "value": initial_nd_metadata["sent"]
        }

    try:
        response_ = requests.post(url, json=data, headers=headers)
        response_.raise_for_status()  # Raise exception for HTTP errors

        if response_.status_code == 201:
            logger.info(f"Entity {entity_ID} created successfully.")
            return {"status": "created", "response": response_}
        elif response_.status_code == 409:
            logger.info(f"Entity {entity_ID} already exists.")
            return {"status": "Entity already exists"}, 409
    except requests.exceptions.RequestException as e:
        logger.error(f"Error creating entity {entity_ID}: {e}")
        return {"status": "Connection Error"}, 500

    logger.warning(f"Failed to create entity: {response_.status_code} - {response_.text}")
    return {"status": "Error", "message": response_.text}, response_.status_code


# ------------------------------------------------------------------------------
# Update Entity
# ------------------------------------------------------------------------------
def update_entity(entity_id_, payload):
    response = None
    url_ = f'{config.BROKER_URL}/ngsi-ld/v1/entities/{entity_id_}/attrs'
    headers = {'Content-Type': 'application/ld+json'}
    natural_disaster = global_cache.get('natural_disaster', 'No disaster info')
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
        "location": {
            "type": "Polygon",
            "coordinates": payload["coordinates"]
        }
    }

    ignition_points = payload.get("ignitionPoints")
    if natural_disaster == 'Fire' and _valid_lonlat_point(ignition_points):
        data_to_send["ignitionPoints"] = {
            "type": "GeoProperty",
            "value": {
                "type": "Point",
                "coordinates": ignition_points
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


# ------------------------------------------------------------------------------
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


def generate_geo_dict_from_tiff(tiff_path, background_label=-1, min_score=0.0):
    """
    Generates a geographic dictionary from a GeoTIFF file.

    Behavior:
      - If the TIFF has >=2 bands, treat it as an Objects OGM:
          Band 1: label
          Band 2: score
        Emit ONLY "active" cells: (label != background_label) OR (score > min_score)

      - If the TIFF has 1 band, keep your previous behavior.

    Returns:
        list of dicts: [{"pixel_coords": (r,c), "geo_coords": [lon,lat,alt],
                         "score": float, "label": int}, ...]
    """
    if not os.path.exists(tiff_path):
        logger.error(f"GeoTIFF file not found: {tiff_path}")
        return None

    try:
        geo_dict_list = []

        logger.debug(f"Opening GeoTIFF file: {tiff_path}")
        with rasterio.open(tiff_path) as dataset:
            logger.debug(f"CRS: {dataset.crs}")
            logger.debug(f"Width: {dataset.width}, Height: {dataset.height}, Bands: {dataset.count}")

            if dataset.count < 1:
                raise ValueError(f"GeoTIFF file {tiff_path} contains no bands.")

            # Handle CRS
            if dataset.crs is None or str(dataset.crs).startswith("LOCAL_CS"):
                logger.warning("CRS is non-standard (LOCAL_CS/None). Assuming EPSG:3857 for transformations.")
                dataset_crs = "EPSG:3857"
            else:
                dataset_crs = dataset.crs

            # Transformer to WGS84
            transformer = Transformer.from_crs(dataset_crs, "EPSG:4326", always_xy=True)

            nodata_value = dataset.nodata

            if dataset.count >= 2:
                # --- Objects OGM mode: band1=label, band2=score ---
                label_band = dataset.read(1)
                score_band = dataset.read(2).astype(np.float32)

                # Valid mask (nodata handling)
                if nodata_value is not None and np.isfinite(nodata_value):
                    valid = (label_band != nodata_value) & np.isfinite(score_band)
                else:
                    valid = np.isfinite(score_band)

                # Active-object mask (sparse initialization)
                active = valid & ((label_band != background_label) | (score_band > float(min_score)))

                rows, cols = np.where(active)
                if len(rows) == 0:
                    return []  # No active objects (common at initialization)

                xs, ys = dataset.xy(rows, cols)
                lons, lats = transformer.transform(xs, ys)

                for i in range(len(rows)):
                    r = int(rows[i]); c = int(cols[i])
                    lbl = int(label_band[r, c])
                    scr = float(score_band[r, c])

                    geo_dict_list.append({
                        "pixel_coords": (r, c),
                        # Objects OGM doesn't store elevation; keep 0.0 here.
                        "geo_coords": [float(lons[i]), float(lats[i]), 0.0],
                        "score": scr,
                        "label": lbl
                    })

            else:
                # --- Single-band fallback (your original behavior) ---
                elevation_data = dataset.read(1).astype(np.float32)
                logger.debug(f"Band-1 data shape: {elevation_data.shape}")

                if nodata_value is not None and np.isfinite(nodata_value):
                    mask_ = elevation_data != nodata_value
                else:
                    mask_ = np.isfinite(elevation_data)

                rows, cols = np.where(mask_)
                if len(rows) == 0:
                    return []

                xs, ys = dataset.xy(rows, cols)
                lons, lats = transformer.transform(xs, ys)

                for i in range(len(rows)):
                    pixel_coords = (int(rows[i]), int(cols[i]))
                    val = float(elevation_data[rows[i], cols[i]])
                    geo_dict = {
                        "pixel_coords": pixel_coords,
                        "geo_coords": [float(lons[i]), float(lats[i]), val],
                        "score": 0.0,
                        "label": -1
                    }
                    geo_dict_list.append(geo_dict)

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
        if new_geo_dict is None:
            logger.error(f"Failed to generate geo dict for {tiff_file}.")
            print(f"Failed to generate geo dict for {tiff_file}.")
            return None

        if len(new_geo_dict) == 0:
            logger.info(f"No active objects found in {tiff_file} (expected at initialization).")

        # Save the generated geo dictionary to JSON
        try:
            current_time = datetime.now().isoformat()

            geojson = {
                "type": "FeatureCollection",
                "features": []
            }

            for obj in new_geo_dict:
                coords = obj.get("geo_coords", None)
                if not coords or len(coords) < 2:
                    continue

                # Ensure 3D [lon, lat, alt]
                if len(coords) == 2:
                    coords = [float(coords[0]), float(coords[1]), 0.0]
                else:
                    coords = [float(coords[0]), float(coords[1]), float(coords[2]) if coords[2] is not None else 0.0]

                # Prefer a stable ID based on pixel coords if present; otherwise fall back to lon/lat
                px = obj.get("pixel_coords", None)
                if px is not None:
                    track_id_seed = f"{json_file}:{int(px[0])}:{int(px[1])}"
                else:
                    track_id_seed = f"{json_file}:{coords[0]:.10f}:{coords[1]:.10f}:{coords[2]:.3f}"
                track_id = str(uuid.uuid5(uuid.NAMESPACE_URL, track_id_seed))

                # Preserve any score/label coming from the TIFF-derived object
                score = float(obj.get("score", 0.0))
                label = int(obj.get("label", -1))

                # Minimal persistent KF state (3x3 covariance, flattened)
                kf_x = coords.copy()
                kf_P = (np.eye(3) * 1.0).reshape(-1).tolist()

                feature = {
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": coords
                    },
                    "properties": {
                        # Existing fields
                        "score": score,
                        "label": label,

                        # Lifecycle (used by remove_stale_objects)
                        "creation_time": current_time,
                        "last_update": current_time,
                        "update_count": 0,

                        # Track status
                        "track_id": track_id,
                        "confirmed": False,
                        "miss_count": 0,

                        # Your bookkeeping
                        "grid_center": coords.copy(),
                        "raw_measurement": None,
                        "kf_refined": coords.copy(),

                        # Persistent KF state
                        "kf_x": kf_x,
                        "kf_P": kf_P,
                    }
                }

                geojson["features"].append(feature)

            with open(json_file, "w") as f:
                json.dump(geojson, f, indent=4)
            print(f"Geo dict saved to ----> {json_file}")

        except IOError as e:
            logger.error(f"Failed to save geo dict to {json_file}: {e}")
            print(f"Failed to save geo dict to {json_file}: {e}")
            return None

        # Save the generated geo dictionary to JSON
        # try:
        #     # ---------------------------------------------------------------
        #     current_time = datetime.now().isoformat()
        #     geojson = {
        #         "type": "FeatureCollection",
        #         "features": []
        #     }
        #     for obj in new_geo_dict:
        #         coords = obj.get("geo_coords", None)
        #         if not coords or len(coords) < 2:
        #             continue
        #         if len(coords) ==2:
        #             coords = [float(coords[0]), float(coords[1]), 0.0]
        #         else:
        #             coords = [float(coords[0]), float(coords[1]), float(coords[2]) if coords[2] is not None else 0.0]
        #
        #         lon, lat, alt = coords
        #         tracker_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{json_file}:{lon:.10f}:{lat:.10f}:{alt:.3f}"))
        #         kf_x = coords.copy()
        #         kf_p = (np.eye(3) * 1.0).reshape(-1).tolist()
        #         feature = {
        #             "type": "Feature",
        #             "geometry": {
        #                 "type": "Point",
        #                 "coordinates": coords
        #             },
        #             "properties": {
        #                 "score": obj.get("score", 0.0),
        #                 "label": obj.get("label", -1),
        #                 # Lifecycle fields
        #                 "creation_time": current_time,
        #                 "last_update": current_time,
        #                 "update_count": 0,
        #                 # Track identity + status
        #                 "track_id": tracker_id,
        #                 "confirmed": False,
        #                 "miss_count": 0,
        #                 # Cell info
        #                 "grid_center": coords.copy(),
        #                 "raw_measurement": None,
        #                 "kf_refined": coords.copy(),
        #                 # Kalman filter fields to prevent re-initializing every cycle
        #                 "kf_x": kf_x,
        #                 "kf_P": kf_p,
        #             }
        #         }
        #         geojson["features"].append(feature)
        #     with open(json_file, 'w') as f:
        #         json.dump(geojson, f, indent=4)
        #         print(f"Geo dict saved to ----> {json_file}")
        #     # ---------------------------------------------------------------
        # except IOError as e:
        #     logger.error(f"Failed to save geo dict to {json_file}: {e}")
        #     print(f"Failed to save geo dict to {json_file}: {e}")
        #     return None

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


# ------------------------------------------------------------------------------
def load_image(image_path, mode):
    """
    Load the image and return its pixel values, geo-transform, and CRS.
    Supports multiple modes for different input types.
    """

    """
    Load the image and return its pixel values, geo-transform, and CRS
    """
    # ------------------------------------------------------------------------------
    data, geo_transform, spatial_ref, measurement_type, observ_nodata = None, [None], None, None, None

    # ------------------------------------------------------------------------------
    def process_observation(observation_file, temp):
        nonlocal data, geo_transform, spatial_ref, measurement_type
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

            # ---------------------------------------------------------------
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
            # ---------------------------------------------------------------

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
                    data = np.flipud(data)
                    # Adjust the origin_y: new_origin_y = origin_y + (pixel_height * number_of_rows)
                    nrows = data.shape[0]
                    origin_y = origin_y + pixel_height * nrows
                    pixel_height = -pixel_height

                # --- Correct horizontal orientation if needed ---
                # For a north-up image, pixel_width is expected to be positive.
                if pixel_width < 0:
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
                # ---------------------------------------------------------------
                # Clean up the old temporary file if it exists
                # ---------------------------------------------------------------
                if os.path.exists(full_path):
                    os.remove(full_path)
                break

    elif mode == 2:  # Load the Observation of geo-referenced object drone image TFA-05
        try:
            for observation in sorted(os.listdir(image_path), reverse=False):
                if observation.endswith("_Objects.tif"):
                    measurement_type = 'PersonVehicleDetection'
                    process_observation(os.path.join(image_path, observation), 2)
                    # ---------------------------------------------------------------
                    # Clean up the old temporary file if it exists
                    # ---------------------------------------------------------------
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
                    # ---------------------------------------------------------------
                    # Clean up the old temporary file if it exists
                    # ---------------------------------------------------------------
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
                    # ---------------------------------------------------------------
                    # Clean up the old temporary file if it exists
                    # ---------------------------------------------------------------
                    full_path = os.path.join(image_path, observation)
                    if os.path.exists(full_path):
                        os.remove(full_path)
                    break
        except Exception as e:
            logger.info(f"mode 4 error due to {e}")

    return data, geo_transform, spatial_ref, measurement_type, observ_nodata


def get_geotiff_metadata_rasterio(
    geotiff_path,
    *,
    return_consumed: bool = False,
    delete_on_success: bool = False,
    newest_first: bool = False,
    quarantine_dir: str = "georeferenced_drone_images/detection/_bad"
):
    """
    Extract and transform GeoTIFF tag metadata (key: 'georef_data') from one or more GeoTIFF files
    ending with *_Objects.tif into a GeoJSON FeatureCollection.

    Designed for streaming / directory-polling pipelines:
      - Processes a directory or a single GeoTIFF path.
      - Optionally returns the consumed file path (return_consumed=True).
      - Optionally deletes the consumed file when at least one feature is extracted.
      - Invalid/unparseable files are quarantined to avoid blocking reprocessing.

    Expected tag format (dataset-level tag):
      georef_data = JSON list of dicts, e.g.:
        [{"32,66": [lon, lat, elev?], "score": 0.8, "label": 1}, ...]

    Returns
    -------
    dict | None  OR  (dict|None, str|None)
        FeatureCollection for the first successfully-parsed file (with at least one feature),
        optionally with the consumed file path.
    """
    import os
    import json
    import re
    import shutil
    import rasterio

    pixel_key_re = re.compile(r"^\s*\d+\s*,\s*\d+\s*$")

    # 1) Collect candidate files
    file_paths = []
    if os.path.isdir(geotiff_path):
        file_paths = [
            os.path.join(geotiff_path, f)
            for f in os.listdir(geotiff_path)
            if f.endswith("_Objects.tif")
        ]
    else:
        if isinstance(geotiff_path, str) and geotiff_path.endswith("_Objects.tif") and os.path.exists(geotiff_path):
            file_paths = [geotiff_path]

    if not file_paths:
        logger.info("No valid geo_referenced drone files found.")
        return (None, None) if return_consumed else None

    # 2) Sort by mtime (arrival-ish)
    try:
        file_paths.sort(key=os.path.getmtime, reverse=newest_first)
    except Exception:
        file_paths.sort(reverse=newest_first)

    os.makedirs(quarantine_dir, exist_ok=True)

    def _quarantine(path: str, reason: str):
        try:
            base = os.path.basename(path)
            dst = os.path.join(quarantine_dir, base)
            # Avoid overwrite loops
            if os.path.abspath(path) == os.path.abspath(dst):
                logger.warning(f"Quarantine skipped (already in quarantine): {path}. Reason: {reason}")
                return
            shutil.move(path, dst)
            logger.warning(f"Quarantined bad detection tif: {path} -> {dst}. Reason: {reason}")
        except Exception as e:
            logger.warning(f"Failed to quarantine {path}: {e}. Reason: {reason}")

    def _ensure_3d_local(coords):
        if coords is None:
            return None
        if not isinstance(coords, (list, tuple)) or len(coords) < 2:
            return None
        try:
            lon = float(coords[0])
            lat = float(coords[1])
            elev = float(coords[2]) if len(coords) > 2 and coords[2] is not None else 0.0
            return [lon, lat, elev]
        except Exception:
            return None

    # 3) Iterate until we find the first valid file with >=1 feature
    for file_path in file_paths:
        try:
            with rasterio.open(file_path) as ds:
                tags = ds.tags() or {}

            if "georef_data" not in tags:
                _quarantine(file_path, "missing georef_data tag")
                continue

            try:
                data = json.loads(tags["georef_data"])
            except Exception as e:
                _quarantine(file_path, f"invalid JSON in georef_data: {e}")
                continue

            if not isinstance(data, list):
                _quarantine(file_path, "georef_data is not a list")
                continue

            feature_collection = {"type": "FeatureCollection", "features": []}

            for item in data:
                if not isinstance(item, dict):
                    continue

                # pixel coordinate key: "row,col"
                pixel_keys = [k for k in item.keys() if pixel_key_re.match(str(k))]
                if not pixel_keys:
                    continue

                # Allow multiple pixel keys in one item (rare but safe)
                for key in pixel_keys:
                    try:
                        row, col = [int(x.strip()) for x in str(key).split(",")]
                    except Exception:
                        continue

                    coordinates = item.get(key, None)
                    coords3 = _ensure_3d_local(coordinates)
                    if coords3 is None:
                        continue

                    # Extract score/label robustly
                    try:
                        score = float(item.get("score", 0.0))
                    except Exception:
                        score = 0.0
                    try:
                        label = int(item.get("label", -1))
                    except Exception:
                        label = -1

                    feature_collection["features"].append({
                        "type": "Feature",
                        "geometry": {"type": "Point", "coordinates": coords3},
                        "properties": {
                            "pixel_coords": [row, col],
                            "score": score,
                            "label": label
                        }
                    })

            if not feature_collection["features"]:
                _quarantine(file_path, "parsed but extracted 0 valid features")
                continue

            if delete_on_success:
                try:
                    os.remove(file_path)
                    logger.info(f"Deleted consumed detection tif: {file_path}")
                except Exception as e:
                    logger.warning(f"Failed to delete consumed tif {file_path}: {e}")

            if return_consumed:
                return feature_collection, file_path
            return feature_collection

        except Exception as e:
            _quarantine(file_path, f"unexpected exception: {e}")
            continue

    # None of the files yielded valid features
    return (None, None) if return_consumed else None


def load_segment_georef_data(
    geotiff_path,
    *,
    return_consumed: bool = False,
    delete_on_success: bool = False,
    newest_first: bool = False,
    quarantine_dir: str = "georeferenced_drone_images/segmented/_bad"
):
    """
    Load per-pixel georeferenced segmentation samples from the GeoTIFF tag
    ``georef_data`` written by the georeferencing module.

    Expected tag format for segmentation masks:
      {
        "row,col": [lon, lat, elev, value],
        ...
      }

    Returns
    -------
    dict | None  OR  (dict|None, str|None)
        {
          "measurement_type": "active_fire" | "burnt_area" | "Flood",
          "samples": [{"pixel_coords": [row, col], "coords": [lon, lat, elev], "value": float}, ...]
        }
    """
    import shutil

    file_paths = []
    if os.path.isdir(geotiff_path):
        file_paths = [
            os.path.join(geotiff_path, f)
            for f in os.listdir(geotiff_path)
            if f.endswith("_Segment.tif")
        ]
    elif isinstance(geotiff_path, str) and geotiff_path.endswith("_Segment.tif") and os.path.exists(geotiff_path):
        file_paths = [geotiff_path]

    if not file_paths:
        return (None, None) if return_consumed else None

    try:
        file_paths.sort(key=os.path.getmtime, reverse=newest_first)
    except Exception:
        file_paths.sort(reverse=newest_first)

    os.makedirs(quarantine_dir, exist_ok=True)

    def _quarantine(path: str, reason: str):
        try:
            dst = os.path.join(quarantine_dir, os.path.basename(path))
            if os.path.abspath(path) == os.path.abspath(dst):
                logger.warning(f"Segmentation quarantine skipped (already in quarantine): {path}. Reason: {reason}")
                return
            shutil.move(path, dst)
            logger.warning(f"Quarantined bad segmentation tif: {path} -> {dst}. Reason: {reason}")
        except Exception as e:
            logger.warning(f"Failed to quarantine segmentation tif {path}: {e}. Reason: {reason}")

    for file_path in file_paths:
        try:
            with rasterio.open(file_path) as ds:
                tags = ds.tags() or {}

            georef_data_raw = tags.get("georef_data")
            if georef_data_raw is None:
                _quarantine(file_path, "missing georef_data tag")
                continue

            try:
                georef_data = json.loads(georef_data_raw)
            except Exception as e:
                _quarantine(file_path, f"invalid JSON in georef_data: {e}")
                continue

            if not isinstance(georef_data, dict):
                _quarantine(file_path, "segmentation georef_data is not a dict")
                continue

            base_name = os.path.basename(file_path)
            if "Burnt" in base_name:
                measurement_type = "burnt_area"
            elif "Fire" in base_name:
                measurement_type = "active_fire"
            elif "Flood" in base_name:
                measurement_type = "Flood"
            else:
                _quarantine(file_path, "could not infer segmentation measurement type from filename")
                continue

            samples = []
            for pixel_key, payload in georef_data.items():
                if not isinstance(pixel_key, str) or "," not in pixel_key:
                    continue
                if not isinstance(payload, (list, tuple)) or len(payload) < 4:
                    continue
                try:
                    row_str, col_str = pixel_key.split(",", 1)
                    row = int(row_str.strip())
                    col = int(col_str.strip())
                    lon = float(payload[0])
                    lat = float(payload[1])
                    elev = float(payload[2]) if payload[2] is not None else 0.0
                    value = float(payload[3])
                except Exception:
                    continue

                if not np.isfinite(lon) or not np.isfinite(lat) or not np.isfinite(value):
                    continue

                samples.append({
                    "pixel_coords": [row, col],
                    "coords": [lon, lat, elev],
                    "value": value
                })

            if not samples:
                _quarantine(file_path, "parsed but extracted 0 valid segmentation samples")
                continue

            if delete_on_success:
                try:
                    os.remove(file_path)
                    logger.info(f"Deleted consumed segmentation tif: {file_path}")
                except Exception as e:
                    logger.warning(f"Failed to delete consumed segmentation tif {file_path}: {e}")

            payload = {"measurement_type": measurement_type, "samples": samples}
            if return_consumed:
                return payload, file_path
            return payload

        except Exception as e:
            _quarantine(file_path, f"unexpected exception: {e}")

    return (None, None) if return_consumed else None

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


def update_occupancy_grid_from_georef_segments(OGMData, OGM_gt_, samples, measurement_type):
    """
    Fuse UAV segmentation evidence into the OGM using per-pixel georeferenced
    samples from the ``georef_data`` tag instead of the GeoTIFF affine transform.

    Each sample is mapped directly to an OGM cell via its lon/lat coordinates.
    Samples landing in the same OGM cell are averaged before a single Bayesian
    update is applied to avoid overweighting dense image patches.
    """
    if OGMData is None or samples is None:
        return OGMData

    if not isinstance(samples, list) or len(samples) == 0:
        return OGMData

    epsilon = 1e-9
    height, width = OGMData.shape
    sums = np.zeros_like(OGMData, dtype=np.float32)
    counts = np.zeros_like(OGMData, dtype=np.int32)

    for sample in samples:
        if not isinstance(sample, dict):
            continue
        coords = sample.get("coords")
        try:
            lon = float(coords[0])
            lat = float(coords[1])
            value = float(sample.get("value", 0.0))
        except Exception:
            continue

        if not np.isfinite(lon) or not np.isfinite(lat) or not np.isfinite(value):
            continue

        try:
            row, col = coordinates_to_pixel_update(OGM_gt_, lon, lat)
        except Exception:
            continue

        if not (0 <= row < height and 0 <= col < width):
            continue

        sums[row, col] += value
        counts[row, col] += 1

    valid = counts > 0
    if not np.any(valid):
        logger.info(f"No valid OGM overlap found for segmentation measurement type {measurement_type}.")
        return OGMData

    mean_values = np.zeros_like(OGMData, dtype=np.float32)
    mean_values[valid] = sums[valid] / counts[valid]
    mean_values = np.clip(mean_values, 0.0, 1.0)

    updated_ogm = np.asarray(OGMData, dtype=np.float32).copy()
    prior = np.clip(updated_ogm[valid], epsilon, 1.0 - epsilon)
    obs = mean_values[valid]

    if measurement_type == "active_fire":
        likelihood = 0.5 + 0.45 * obs
    elif measurement_type == "burnt_area":
        p_ba_max = 0.30
        p_ba_min = 0.05
        likelihood = p_ba_max - (p_ba_max - p_ba_min) * obs
    elif measurement_type == "Flood":
        pmin_uav = 0.05
        pmax_uav = 0.95
        likelihood = pmin_uav + (pmax_uav - pmin_uav) * obs
    else:
        raise ValueError(f"Unknown georef segmentation measurement_type: {measurement_type}")

    likelihood = np.clip(likelihood, epsilon, 1.0 - epsilon)
    log_odds_prior = np.log(prior / (1.0 - prior))
    log_odds_obs = np.log(likelihood / (1.0 - likelihood))
    log_odds_post = np.clip(log_odds_prior + log_odds_obs, -5.0, 5.0)
    updated_ogm[valid] = 1.0 / (1.0 + np.exp(-log_odds_post))

    logger.info(
        f"Fused {int(np.count_nonzero(valid))} OGM cells from {int(np.sum(counts[valid]))} "
        f"georeferenced UAV segmentation samples ({measurement_type})."
    )
    return np.clip(updated_ogm, 0.0, 1.0)

# ---------------------------------------------------------------
# FUSE SAT DATA
# ---------------------------------------------------------------
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
        """
        likelihood = 0.5 + 0.5 * (obs_on_ogm - 0.1)
        likelihood = np.clip(likelihood, epsilon, 1.0 - epsilon)
        """
        likelihood = np.clip(obs_on_ogm, epsilon, 1.0 - epsilon)

        # Keep original gating behavior: update only where obs > 0 and finite
        """
        mask_valid = np.isfinite(obs_on_ogm) & (obs_on_ogm > 0)
        """
        mask_valid = np.isfinite(obs_on_ogm)
        if not np.any(mask_valid):
            logger.info("No valid observation pixels after re-projection; returning OGM unchanged.")
            return prior  # ensure dtype/clip

        # log-odds update (clamped like your code)
        """log_odds_prior = np.log((prior[mask_valid] + epsilon) / (1.0 - prior[mask_valid] + epsilon))"""
        log_odds_prior = np.log((np.clip(prior[mask_valid], epsilon, 1 - epsilon)) /
                                (1.0 - np.clip(prior[mask_valid], epsilon, 1 - epsilon)))
        log_odds_obs = np.log(likelihood[mask_valid] / (1.0 - likelihood[mask_valid]))
        log_odds_post = np.clip(log_odds_prior + log_odds_obs, -5.0, 5.0)
        updated_prob = 1.0 / (1.0 + np.exp(-log_odds_post))  # sigmoid

        # 4) Write back to full OGM grid and return
        updated_OGM = prior.copy()
        updated_OGM[mask_valid] = np.clip(updated_prob, 0.0, 1.0)
        logger.info("Fusion completed successfully (SAT measurement).")
        return updated_OGM

    except Exception as e:
        logger.info(f"Error in flood satellite fusion: {e}")
        return OGMData


# ---------------------------------------------------------------
def update_occupancy_grid_fire_sat_fixed(
        OGMData,
        OGM_gt_,
        observation_data_,
        observation_gt_,
        measurement_type,
        ogm_crs: Union[str, RioCRS] = "EPSG:4326",
        observation_crs: Union[str, RioCRS, None] = None,
        obs_nodata=None):
    """
    Reproject a Fire satellite observation onto the OGM grid and fuse it in
    log-odds space. This avoids the CRS mismatch in the older patch-based
    update path, which assumes the observation and OGM already share the same
    coordinate reference system.
    """
    if observation_crs is None:
        raise ValueError("observation_crs is None. Pass src.crs or an EPSG string (e.g., 'EPSG:32632').")

    epsilon = 1e-9
    burnt_area_threshold = 0.50
    try:
        ogm_crs = RioCRS.from_user_input(ogm_crs)
        observation_crs = RioCRS.from_user_input(observation_crs)

        obs = observation_data_.astype(np.float32, copy=True)
        if obs_nodata is not None and np.isfinite(obs_nodata):
            nodata_mask = (observation_data_ == obs_nodata)
            if nodata_mask.any():
                obs[nodata_mask] = np.nan

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

        prior = np.clip(OGMData.astype(np.float32), 0.0, 1.0)
        # Burnt-area rasters are cropped to the AOI and use 0 for background.
        # Ignore weak styled WMS pixels and only fuse meaningful burnt support.
        if measurement_type == "burnt_area":
            mask_valid = np.isfinite(obs_on_ogm) & (obs_on_ogm >= burnt_area_threshold)
        else:
            mask_valid = np.isfinite(obs_on_ogm)
        if not np.any(mask_valid):
            logger.info("No valid burnt-area pixels after re-projection; returning OGM unchanged.")
            return prior

        obs_values = np.clip(obs_on_ogm[mask_valid], 0.0, 1.0)
        if measurement_type == "burnt_area":
            p_ba_sat_max = 0.30
            p_ba_sat_min = 0.05
            likelihood = p_ba_sat_max - (p_ba_sat_max - p_ba_sat_min) * obs_values
        elif measurement_type == "active_fire":
            likelihood = 0.05 + 0.9 * obs_values
        else:
            raise ValueError(f"Unknown fire satellite measurement_type: {measurement_type}")

        likelihood = np.clip(likelihood, epsilon, 1.0 - epsilon)
        prior_valid = np.clip(prior[mask_valid], epsilon, 1.0 - epsilon)
        log_odds_prior = np.log(prior_valid / (1.0 - prior_valid))
        log_odds_obs = np.log(likelihood / (1.0 - likelihood))
        log_odds_post = np.clip(log_odds_prior + log_odds_obs, -5.0, 5.0)

        updated_ogm = prior.copy()
        updated_ogm[mask_valid] = 1.0 / (1.0 + np.exp(-log_odds_post))
        logger.info(f"Fusion completed successfully (Fire SAT measurement: {measurement_type}).")
        return np.clip(updated_ogm, 0.0, 1.0)

    except Exception as e:
        logger.info(f"Error in fire satellite fusion: {e}")
        return OGMData


def process_and_upload_ogm(entity_id, file_path_, bucket_name, metadata):
    """
        Process the occupancy grid, update the entity and upload the OGM to MinIO.
    """

    FileName = file_path_.split("/")[-1]
    object_name = (f'pdm05/'
                   f'{global_cache.get("bm_id")}/'
                   f'{FileName}')
    try:
        minio_client.upload_file(bucket_name, object_name, file_path_)
    except Exception as e:
        logger.error(f"Error uploading file: {e}")
    try:
        response = update_entity(entity_id, metadata)
        if response:
            logger.info(f"Payload published successfully for entity: {entity_id}")
            print(f"Payload published successfully for entity: {entity_id}")
        else:
            logger.error(f"Failed to publish payload for entity: {entity_id}")
    except Exception as e:
        logger.error(f"Error updating NGSI-LD entity: {e}")
