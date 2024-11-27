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

from flask import Flask, request, jsonify
from flask_socketio import SocketIO
from osgeo import gdal, osr, ogr
from rasterio.features import geometry_mask, rasterize
from rasterio.transform import array_bounds, from_origin
from Kalman_filter_estimating_Objects import KalmanFilter
from georeferencing_module import main
from minio_client import MinIOClient
from logging_config import logger
from pyproj import CRS
from rasterio.windows import Window
from datetime import datetime
import json
from shapely.geometry import Polygon
from shapely.geometry import box

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

global_cache = {"processing": False, "natural_disaster": "No disaster info", "roi": "No disaster info"}
global_cache_lock = threading.Lock()
start_event = threading.Event()


@app.route(f'{config.BASE_PATH}')
# @app.route('/')
def index():
    return jsonify({"status": "App is running", "message": "Occupancy Grid Estimation System"})


def handle_person_vehicle_detection(notification, parameters):
    auth_filename = notification.get('parameters', {}).get('value', {}).get('FileName', None)
    detection = notification.get("detection", {}).get('value', {})

    if not auth_filename:
        logger.warning("No filename found for PersonVehicleDetection.")
        return

    metadata_file_base = auth_filename.split('.')[0]
    detection_file = f"{metadata_file_base}_obj.json"
    metadata_file = f"{metadata_file_base}_metadata.json"

    json_file_path_metadata = f"downloads/drone_imgs/{global_cache.get('natural_disaster', 'No disaster info')}/{metadata_file}"
    json_file_path_detection = f"downloads/drone_imgs/{global_cache.get('natural_disaster', 'No disaster info')}/{detection_file}"
    if global_cache.get('natural_disaster', 'No disaster info') != 'No disaster info':
        try:
            # Write to a metadata file
            with open(json_file_path_metadata, 'w') as json_file:
                json_file.write(json.dumps(parameters, indent=4))

            # Write to a detection file
            with open(json_file_path_detection, 'w') as json_file:
                json_file.write(json.dumps(detection, indent=4))

        except Exception as e:
            logger.error(f"Error writing detection files: {e}")


def handle_segmentation(notification):
    bucket = notification.get("bucket", {}).get('value')
    auth_filename = notification.get('segmentation', {}).get('value')
    mask_id = auth_filename.get('mask_id')

    if not mask_id:
        logger.warning("No mask_id found in auth_filename.")
        return

    file_path = f"downloads/drone_imgs/{global_cache.get('natural_disaster', 'No disaster info')}/{mask_id}"
    logger.info(f"path of download {file_path}")
    try:
        # Synchronous file download simulation
        downloaded_file_path = minio_client.download_file(bucket, mask_id, file_path)
        if downloaded_file_path:
            logger.info(f"File downloaded successfully to {downloaded_file_path}")
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


def process_notification(notification):
    entity_id = notification.get("id")
    entity_type = notification.get("type")
    logger.info(f"Processing notification for entity ID: {entity_id}, Entity type: {entity_type}")

    # Process different entity types accordingly
    if entity_type == "Alert":
        process_alert(notification, entity_id)
        start_event.set()
    elif entity_type == "PersonVehicleDetection":
        parameters = notification.get("parameters", {}).get('value', {})
        handle_person_vehicle_detection(notification, parameters)
    elif entity_type in ["BurntSegmentation", "FloodSegmentation"]:
        handle_segmentation(notification)
    else:
        handle_file_download(notification)


def process_alert(notification, entity_id):
    area = notification.get("location", {})
    if isinstance(area, dict) and "value" in area:
        global_cache['roi'] = area['value'].get("coordinates")
    else:
        global_cache['roi'] = "No disaster info"
        logger.warning(f"Unexpected 'location' format for entity ID {entity_id}: {area}")

    event_ = notification.get("event", {})
    if isinstance(event_, dict) and "value" in event_:
        global_cache['natural_disaster'] = event_['value']
    else:
        global_cache['natural_disaster'] = "Unknown event"
        logger.warning(f"Unexpected 'event' format for entity ID {entity_id}: {event_}")


def handle_file_download(notification):
    filename_ = notification.get('filename')
    bucket = notification.get("bucket", {})

    if isinstance(filename_, dict) and 'value' in filename_ and isinstance(bucket, dict) and 'value' in bucket:
        try:
            downloaded_file_path = download_file(notification['type'], filename_, bucket)
            if downloaded_file_path:
                logger.info(f"File downloaded successfully to {downloaded_file_path}")
            else:
                logger.error("File download failed.")
        except Exception as e:
            logger.error(f"Error in downloading file: {e}")


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

    try:
        return minio_client.download_file(bucket['value'], filename_['value'], file_path)
    except Exception as e:
        logger.error(f"Error downloading file {filename_['value']} from bucket {bucket['value']}: {e}")
        return None


@app.route(f"{config.BASE_PATH}/{config.API_ENDPOINT}", methods=['POST'])
# @app.route('/notify', methods=['POST'])
def notify():
    notification_data = request.get_json()
    logger.info(f"Notification data received: {notification_data}")

    for notification in notification_data.get("data", []):
        logger.info(f"Preparing to process notification: {notification}")
        process_notification(notification)

    with global_cache_lock:
        # Ensure that initialize_processing() is not already running
        if global_cache['natural_disaster'] != 'No disaster info':
            if not global_cache['processing']:
                global_cache['processing'] = True
                try:
                    logger.info("Starting initialize_processing() in a new thread")
                    processing_thread = threading.Thread(target=initialize_processing)
                    processing_thread.start()
                except Exception as e:
                    logger.error(f"Error in initialize_processing: {e}")
                    global_cache['processing'] = False  # Reset on failure
            else:
                logger.info("initialize_processing() is already running, skipping.")
        else:
            logger.info("No relevant natural disaster data, skipping processing.")

    return jsonify({"status": "Notifications processed successfully"}), 200


def initialize_processing():
    while True:
        if start_event.is_set():
            global polygon_coordinates
            initialize_entities()
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


def subscribe_to_entities():
    subscription_url = f"{config.BROKER_URL}/ngsi-ld/v1/subscriptions"

    headers = {
        "Content-Type": "application/ld+json",
        "Accept": "application/ld+json"
    }
    subscription_payload = {
        "type": "Subscription",
        "entities": [
            {"type": "Alert"},  # Alert entity by END USER
            {"type": "FloodCalculationResults"},  # NS PDM-tech-02 for Flood prediction ---> GeoTIFF
            {"type": "StandardArrivalTime"},
            {"type": "BurntSegmentation"},
            {"type": "FloodSegmentation"},  # AUTH TFA-tech-06 ---> image
            {"type": "PersonVehicleDetection"},  # AUTH TFA-tech-05 ---> JSON
            {"type": "HotspotResult"},  # PLUS TFA-tech-11 ---> GeoJson
            {"type": "SinglePostResult"},  # PLUS TFA-tech-11 ---> GeoJson
        ],
        "watchedAttributes": [
            "minio_url",
            "filename",
            "bucket",
            "parameters",
            "segmentation",
            "detection",
            "location",
            "event",
            "effective",
        ],
        "notification": {
            "attributes":
                [
                    "minio_url",
                    "filename",
                    "bucket",
                    "parameters",
                    "segmentation",
                    "detection",
                    "location",
                    "event",
                    "effective",
                ],
            "endpoint": {
                "uri": config.CALLBACK_URL,
                "accept": "application/json"
            }
        },
        '@context': ['https://uri.etsi.org/ngsi-ld/v1/ngsi-ld-core-context.jsonld']
    }
    # Perform the POST request to create the subscription
    response_ = requests.post(subscription_url,
                              json=subscription_payload,
                              headers=headers)

    # Check the response
    if response_.status_code == 201:
        logger.info("Subscription created successfully.")
        print("Subscription created successfully.")
    else:
        print(f"Failed to create subscription. Status code: {response_.status_code}")
        logger.info(f"Failed to create subscription. Status code: {response_.status_code}")


def estimate_ND_status():
    global polygon_coordinates
    polygon_coordinates = convert_to_polygon()
    disaster_type = global_cache.get('natural_disaster', 'No disaster info')
    ogm_path = f"estimated_OGM/occupancy_grid_map_{disaster_type}.tif"
    roi_data = global_cache.get('roi', None)

    get_roi(polygon_coordinates, ogm_path)
    ogm_data, ogm_gt_, ogm_proj = load_image(ogm_path, 0)
    if os.path.isdir("segmented_sat_roi_imgs"):
        if len(os.listdir("segmented_sat_roi_imgs")) == 0:
            get_sate_roi(PolygonCoords=roi_data[0])

    # Update OGM with drone data
    try:
        observe_data_drone, observe_gt_drone, observe_proj_drone = load_image(
            "georeferenced_drone_images/", 1)
        ogm_data = update_occupancy_grid(ogm_data, ogm_gt_, observe_data_drone, observe_gt_drone)
    except FileNotFoundError:
        logger.warning("Drone images directory not found.")
    except Exception as e:
        logger.info(f"Drone measurements are not available: {e}")

    # Update OGM with satellite data
    try:
        observ_sat_data_, observ_sat_gt_, observ_sat_proj = load_image("segmented_sat_roi_imgs/", 3)
        ogm_data = update_occupancy_grid(ogm_data, ogm_gt_, observ_sat_data_, observ_sat_gt_)
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
                    properties = feature.get("properties", {})
                    geometry = feature.get("geometry", {})
                    if geometry.get("type") == "Polygon":
                        # Extract polygon coordinates and calculate grid cells it covers
                        polygon = Polygon(geometry.get("coordinates")[0])
                        if polygon.is_valid:
                            # Iterate through grid cells covered by the polygon
                            for x, y in polygon_to_pixels(polygon, ogm_gt_, ogm_data.shape):
                                if 0 <= x < ogm_data.shape[1] and 0 <= y < ogm_data.shape[0]:
                                    # Fuse hotspot data with existing OGM value
                                    hotspot_count = properties.get("count", 0)
                                    hotspot_ratio = properties.get("ratio", 0)
                                    hotspot_value = max(hotspot_count, hotspot_ratio)  # Use count or ratio as value
                                    ogm_data[y, x] = fuse_fire_probability(ogm_data[y, x], hotspot_value)

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
        metadata = save_geotiff("estimated_OGM", output_tiff_path, ogm_data, ogm_gt_, ogm_proj)
        logger.info(f"{ND_entity_ID}")

        process_and_upload_ogm(ND_entity_ID, os.path.join("estimated_OGM", output_tiff_path),
                               config.BUCKET_NAME, metadata)
    except Exception as e:
        logger.error(f"Error saving or uploading OGM: {e}")
    ###################################################################################################################


def polygon_to_pixels(polygon, geo_transform, grid_shape):
    pixel_indices = []
    bounds = polygon.bounds
    x_min, y_min = coordinates_to_pixel(geo_transform, bounds[0], bounds[1])
    x_max, y_max = coordinates_to_pixel(geo_transform, bounds[2], bounds[3])
    x_min, x_max = max(0, x_min), min(grid_shape[1] - 1, x_max)
    y_min, y_max = max(0, y_min), min(grid_shape[0] - 1, y_max)
    for y in range(y_min, y_max + 1):
        for x in range(x_min, x_max + 1):
            cell_polygon = box(*pixel_to_coordinates(geo_transform, x, y),
                               *pixel_to_coordinates(geo_transform, x + 1, y + 1))
            if polygon.intersects(cell_polygon):
                pixel_indices.append((x, y))
    return pixel_indices


# fuse_fire_probability: Fuses the existing fire probability with the hotspot value.
def fuse_fire_probability(existing_prob, hotspot_value, weight=0.5):
    return (1 - weight) * existing_prob + weight * hotspot_value


def estimate_Objects_status():
    """Estimate objects status and process the occupancy grid map."""
    global polygon_coordinates

    polygon_coordinates = convert_to_polygon()
    disaster_type = global_cache.get('natural_disaster', 'No disaster info')
    ogm_path = f"estimated_OGM/occupancy_grid_map_{disaster_type}_Objects.tif"
    # Load or generate the ROI
    get_roi(polygon_coordinates, ogm_path)
    try:
        get_geo_dict(
            f"estimated_OGM/occupancy_grid_map_{disaster_type}_Objects")
    except Exception as e:
        logger.info(f"Dict was not created due to {e}")

    if disaster_type != 'No disaster info':
        ogm_metadata = load_existing_geo_dict(
            f"estimated_OGM/occupancy_grid_map_{disaster_type}_Objects")
        #######################################################################################
        # Integrate Drone data
        #######################################################################################
        try:
            observ_metadata = get_geotiff_metadata_rasterio("georeferenced_drone_images")
            if not observ_metadata:
                logger.info("Drone metadata is empty or not available.")
            ogm_metadata = get_observations_in_FOV(ogm_metadata, observ_metadata)

            # Initialize the Kalman filter
            state_dim = 3  # For position (x, y, z)
            measurement_dim = 3  # For measurements (x, y, z)
            kf = KalmanFilter(state_dim, measurement_dim)
            # Loop through each observation
            for entry in ogm_metadata:
                # Check if the label is not zero
                label = entry.get("label", 0)
                if label != 0:
                    # Extract the position (observation)
                    position = list(entry.values())[0]  # Get the position data from the first (and only) key
                    z = np.array(position).reshape((measurement_dim, 1))  # Reshape for the update step
                    # Kalman filter prediction
                    kf.predict()
                    # Kalman filter update with the observation
                    kf.update(z)
                    # Update the entry with the new state (updated position)
                    updated_position = kf.x.flatten().tolist()  # Get the updated position
                    entry[list(entry.keys())[0]] = updated_position  # Update the position in the entry
        except Exception as e:
            logger.info(f"Object measurements are not available {e}")
        #######################################################################################
        # Save updated metadata to JSON
        #######################################################################################
        try:
            metadata_path = f"estimated_OGM/occupancy_grid_map_{disaster_type}_Objects.json"
            with open(metadata_path, 'w') as f:
                json.dump(ogm_metadata, f, indent=4)
            logger.info(f"OGM metadata saved to {metadata_path}")
        except Exception as e:
            logger.error(f"Failed to save OGM metadata: {e}")
        geojson = {"type": "FeatureCollection", "features": []}

        # Iterate over the custom data and convert to GeoJSON features, skipping zero score/label
        for entry in ogm_metadata:
            for key, coordinates in entry.items():
                if isinstance(coordinates, list):  # Coordinates are a list of [lon, lat, alt]
                    score = entry.get("score", 0.0)
                    label = entry.get("label", 0)
                    # Only include features with non-zero score OR label
                    if score != 0.0 or label != 0:
                        feature = {
                            "type": "Feature",
                            "geometry": {
                                "type": "Point",
                                "coordinates": [coordinates[0], coordinates[1], coordinates[2]]  # Use only lon, lat
                            },
                            "properties": {
                                "score": score,
                                "label": label
                            }
                        }
                        geojson["features"].append(feature)

        geojson_path = f"estimated_OGM/occupancy_grid_map_{disaster_type}_Objects_filtered.json"
        # Write the result to a GeoJSON file
        try:
            with open(geojson_path, "w") as f:
                json.dump(geojson, f, indent=4)
            logger.info(f"GeoJSON saved to {geojson_path}")
        except Exception as e:
            logger.error(f"Failed to save GeoJSON: {e}")

        try:
            geotiff_path = f"estimated_OGM/occupancy_grid_map_{disaster_type}_Objects_filtered.tif"
            geojson_to_multi_band_geotiff(
                geojson_path,
                geotiff_path,
                pixel_size_lat=0.000008983,
                pixel_size_lon=0.00001405
            )

            output_tiff_path = f"occupancy_grid_map_{disaster_type}_Objects_filtered.tif"
            ogm_data, ogm_gt_, ogm_proj = load_image(geotiff_path, 0)
            metadata = save_geotiff("estimated_OGM", output_tiff_path, ogm_data, ogm_gt_, ogm_proj)
            process_and_upload_ogm(obj_entity_ID, output_tiff_path, config.BUCKET_NAME, metadata)
            logger.info(f"OGM successfully processed and uploaded: {geotiff_path}")
            ########################################################################################
        except Exception as e:
            logger.info(f"Failed to process AOI GeoTIFF: {e}")

        #######################################################################################
        # Integrate SinglePostResult data with Kalman filter
        #######################################################################################
        try:
            ogm_data, ogm_gt_, ogm_proj = load_image(ogm_path, 0)
            social_media_dir = "downloads/SocialMedia"  # Directory containing SinglePostResult files
            single_post_files = [os.path.join(social_media_dir, f) for f in os.listdir(social_media_dir)
                                 if f.startswith("single_posts_results") and f.endswith(".geojson")]
            single_post_files.sort()  # Process files in chronological order
            logger.info(f"Found {len(single_post_files)} SinglePostResult files.")
            # Initialize the Kalman filter
            state_dim = 3  # For position (x, y, z)
            measurement_dim = 3  # For measurements (x, y, z)
            kf = KalmanFilter(state_dim, measurement_dim)
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
                                logger.debug(f"Mapping post coordinates: {lon}, {lat}")

                                # Construct the observation (x, y, z)
                                z = np.array([lon, lat, 0]).reshape((measurement_dim, 1))  # Assuming no altitude

                                # Kalman filter prediction and update
                                try:
                                    kf.predict()
                                    kf.update(z)

                                    # Get the refined position
                                    refined_position = kf.x.flatten().tolist()

                                    # Map the refined position to pixel coordinates
                                    try:
                                        x, y = coordinates_to_pixel(ogm_gt_, refined_position[0],
                                                                    refined_position[1])
                                        if 0 <= x < ogm_data.shape[1] and 0 <= y < ogm_data.shape[0]:
                                            # Update OGM metadata for the corresponding grid cell
                                            emotion_prob = properties.get("emotion_label_probability", 0.0)
                                            logger.debug(f"Emotion probability: {emotion_prob}")
                                            emotion_label = properties.get("emotion_label", "person")
                                            existing_score = ogm_metadata[y][x].get("score", 0.0)

                                            # Fuse data into the grid cell
                                            ogm_metadata[y][x] = {
                                                "score": max(existing_score, emotion_prob),  # Combine probabilities
                                                "label": emotion_label,
                                                "coordinates": refined_position  # Update with refined position
                                            }
                                    except Exception as e:
                                        logger.warning(
                                            f"Error mapping refined coordinates ({refined_position}) to pixel: {e}")
                                except Exception as e:
                                    logger.warning(
                                        f"Kalman filter update failed for coordinates ({lon}, {lat}): {e}")

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
            with open(metadata_path, 'w') as f:
                json.dump(ogm_metadata, f, indent=4)
            logger.info(f"OGM metadata saved to {metadata_path}")
        except Exception as e:
            logger.error(f"Failed to save OGM metadata: {e}")
        geojson = {"type": "FeatureCollection", "features": []}
        # Iterate over the custom data and convert to GeoJSON features, skipping zero score/label
        for entry in ogm_metadata:
            for key, coordinates in entry.items():
                if isinstance(coordinates, list):  # Coordinates are a list of [lon, lat, alt]
                    score = entry.get("score", 0.0)
                    label = entry.get("label", 0)
                    # Only include features with non-zero score OR label
                    if score != 0.0 or label != 0:
                        feature = {
                            "type": "Feature",
                            "geometry": {
                                "type": "Point",
                                "coordinates": [coordinates[0], coordinates[1], coordinates[2]]  # Use only lon, lat
                            },
                            "properties": {
                                "score": score,
                                "label": label
                            }
                        }
                        geojson["features"].append(feature)

        geojson_path = f"estimated_OGM/occupancy_grid_map_{disaster_type}_Objects_filtered.json"
        # Write the result to a GeoJSON file
        try:
            with open(geojson_path, "w") as f:
                json.dump(geojson, f, indent=4)
            logger.info(f"GeoJSON saved to {geojson_path}")
        except Exception as e:
            logger.error(f"Failed to save GeoJSON: {e}")

        try:
            geotiff_path = f"estimated_OGM/occupancy_grid_map_{disaster_type}_Objects_filtered.tif"
            geojson_to_multi_band_geotiff(
                geojson_path,
                geotiff_path,
                pixel_size_lat=0.000008983,
                pixel_size_lon=0.00001405
            )

            output_tiff_path = f"occupancy_grid_map_{disaster_type}_Objects_filtered.tif"
            ogm_data, ogm_gt_, ogm_proj = load_image(geotiff_path, 0)
            metadata = save_geotiff("estimated_OGM", output_tiff_path, ogm_data, ogm_gt_, ogm_proj)
            process_and_upload_ogm(obj_entity_ID, output_tiff_path, config.BUCKET_NAME, metadata)
            logger.info(f"OGM successfully processed and uploaded: {geotiff_path}")
            ########################################################################################
        except Exception as e:
            logger.info(f"Failed to process AOI GeoTIFF: {e}")


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


def get_roi(new_polygon_coords, existing_map_path):
    """
    Create an empty occupancy grid map based on new polygon coordinates.
    Transfer intersected data from an existing map if there is an intersection.

    Args:
        new_polygon_coords (list): A list of coordinates defining a polygon.
        existing_map_path (str): Path to the existing occupancy grid map.

    Returns:
        data_set: GDAL dataset of the created occupancy grid.
    """
    minx, miny = np.min(new_polygon_coords, axis=0)
    maxx, maxy = np.max(new_polygon_coords, axis=0)

    pixel_width = (maxx - minx) / 1414
    pixel_height = (maxy - miny) / 1414
    occupancy_grid_data, transform_ = create_empty_grid(new_polygon_coords, pixel_width, pixel_height)

    if os.path.exists(existing_map_path):
        try:
            existing_data, existing_transform, existing_crs = load_raster(existing_map_path)

            # Create existing polygon bounds
            existing_bounds = array_bounds(existing_data.shape[0], existing_data.shape[1], existing_transform)
            existing_polygon = Polygon([
                (existing_bounds[0], existing_bounds[1]),
                (existing_bounds[2], existing_bounds[1]),
                (existing_bounds[2], existing_bounds[3]),
                (existing_bounds[0], existing_bounds[3]),
                (existing_bounds[0], existing_bounds[1])
            ])
            new_polygon = Polygon(new_polygon_coords)
            print(f"new_polygon: {new_polygon}")
            print(f"existing_polygon: {existing_polygon}")
            reordered_polygon2 = reorder_polygon(list(new_polygon.exterior.coords),
                                                 list(existing_polygon.exterior.coords))
            if reordered_polygon2:
                print("ROI did not change")
                logger.info("ROI did not change")
                return None
            if new_polygon.intersects(existing_polygon):
                intersected_data = get_intersected_data(existing_data, existing_transform, new_polygon_coords)
                occupancy_grid_data += intersected_data

        except Exception as e:
            print(f"Error processing existing map: {e}")
            logger.info(f"Error processing existing map: {e}")

    # Save the new occupancy grid map
    try:
        output_filename = existing_map_path
        with rasterio.open(
                output_filename,
                'w',
                driver='GTiff',
                height=occupancy_grid_data.shape[0],
                width=occupancy_grid_data.shape[1],
                count=1,
                dtype=occupancy_grid_data.dtype,
                crs=CRS.from_epsg(4326),
                transform=transform_,
        ) as dst:
            dst.write(occupancy_grid_data, 1)

        # Load the new map using GDAL
        data_set = gdal.Open(output_filename, gdal.GA_ReadOnly)
        if data_set is None:
            print(f"GDAL failed to open {output_filename}")
            logger.info(f"GDAL failed to open {output_filename}")
            return None
        return data_set

    except Exception as e:
        print(f"An error occurred while creating the new dataset: {e}")
        logger.info(f"An error occurred while creating the new dataset: {e}")
        return None


def convert_to_polygon():
    logger.info(f"ROI ---> {global_cache.get('roi', 'No disaster info')}")
    print(f"ROI ---> {global_cache.get('roi', 'No disaster info')}")
    try:
        polygon = global_cache.get('roi', 'No disaster info')[0]
        # Separate x and y coordinates (longitudes and latitudes)
        x_coords = [coord[0] for coord in polygon]  # Longitudes (minx, maxx)
        y_coords = [coord[1] for coord in polygon]  # Latitudes (miny, maxy)

        # Calculate minx, maxx, miny, maxy
        minx = min(x_coords)
        maxx = max(x_coords)
        miny = min(y_coords)
        maxy = max(y_coords)

        # Create the polygon coordinates
        polygon_coords_ = [
            [minx, maxy],  # Top-left
            [maxx, maxy],  # Top-right
            [maxx, miny],  # Bottom-right
            [minx, miny],  # Bottom-left
            [minx, maxy]  # Closing the polygon (same as a top-left)
        ]
        return polygon_coords_
    except Exception as e:
        logger.info(f'No roi is available now {e}')


########################################################################################################################
# Initialize Entities
########################################################################################################################
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
                        "XMin":
                            {
                                "type": "Property",
                                "value": 0.0
                            },
                        "XRes":
                            {
                                "type": "Property",
                                "value": 0.0
                            },
                        "YMax":
                            {
                                "type": "Property",
                                "value": 0.0
                            },
                        "YRes":
                            {
                                "type": "Property",
                                "value": 0.0
                            }
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
                    [
                        -0.165825,
                        51.495065
                    ],
                    [
                        -0.165825,
                        51.513123
                    ],
                    [
                        -0.111065,
                        51.513123
                    ],
                    [
                        -0.111065,
                        51.495065
                    ],
                    [
                        -0.165825,
                        51.495065
                    ]
                ]
            ]
        },
        "minio_url": {
            "type": "Property",
            "value": f'https://{config.MINIO_ENDPOINT}/{config.BUCKET_NAME}/occupancy_grid_map{natural_disaster}'
        },
        "filename": {
            "type": "Property",
            "value": f"occupancy_grid_map_{global_cache.get('natural_disaster', 'No disaster info')}.tif"
        },

        "bucket": {
            "type": "Property",
            "value": "naples"
        },
        "@context": [
            "https://uri.etsi.org/ngsi-ld/v1/ngsi-ld-core-context.jsonld"
        ]
    }

    response_ = requests.post(url, json=data, headers=headers)

    if response_.status_code == 201:
        logger.info(f"Entity {entity_ID} created successfully.")
        return response_
    elif response_.status_code == 409:
        logger.info(f"Entity {entity_ID} already exists.")
        return {"status": "Entity already exists"}, 409
    else:
        logger.warning(f"Failed to create entity: {response_.status_code} - {response_.text}")
        return {"status": "Error"}, 500


########################################################################################################################
# Update Entity
########################################################################################################################
# @app.route(f'{config.BASE_PATH}/update_entity', methods=['POST'])
def update_entity(entity_id_, payload):
    response = None
    url_ = f'{config.BROKER_URL}/ngsi-ld/v1/entities/{entity_id_}/attrs'
    headers = {'Content-Type': 'application/ld+json'}
    data_to_send = {
        "description": payload["description"],
        "creationDate": payload["creationDate"],
        "geotransform":
            {
                "type": "Property",
                "value":
                    {
                        "XMin":
                            {
                                "type": "Property",
                                "value": payload["XMin"]
                            },
                        "XRes":
                            {
                                "type": "Property",
                                "value": payload["XRes"]
                            },
                        "YMax":
                            {
                                "type": "Property",
                                "value": payload["YMax"]
                            },
                        "YRes":
                            {
                                "type": "Property",
                                "value": payload["YRes"]
                            }
                    }
            },
        "minio_url": f'https://{config.MINIO_ENDPOINT}/{config.BUCKET_NAME}/{payload["file_name"]}',
        "filename": payload["file_name"],
        "bucket": payload["bucket"],
        "location": {
            "type": "Polygon",
            "coordinates": global_cache.get('roi', 'No disaster info')
        }
    }
    # Ensure @context is included in the payload
    payload_with_context = {
        "@context": "https://uri.etsi.org/ngsi-ld/v1/ngsi-ld-core-context.jsonld",
        **data_to_send
    }

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
########################################################################################################################
def load_existing_geo_dict(tiff_path):
    geo_dict_path = f"{tiff_path}.json"  # Using the same base name with .json extension
    if os.path.exists(geo_dict_path):
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
    return None


# Function to generate a new geo dict from a GeoTIFF file
def generate_geo_dict_from_tiff(tiff_path):
    geo_dict_list = []
    try:
        # Open the GeoTIFF file
        with rasterio.open(tiff_path) as dataset:
            # Read the elevation data from the first band (assuming its elevation)
            elevation_data = dataset.read(1)

            # Loop through each pixel (row, col)
            for row in range(dataset.height):
                for col in range(dataset.width):
                    # Get pixel coordinates as a string (row,col)
                    pixel_coords = f"{row},{col}"

                    # Get the geographic coordinates (longitude, latitude) for the current pixel
                    lon, lat = dataset.xy(row, col)

                    # Get the elevation value for this pixel and convert to float
                    elevation = float(elevation_data[row, col])  # Convert numpy.float32 to native Python float

                    # Create a dictionary entry for this pixel
                    geo_dict = {
                        pixel_coords: [float(lon), float(lat), elevation],  # Ensure lon, lat are also native floats
                        "score": 0.0,  # Placeholder for score
                        "label": 0  # Placeholder for label
                    }

                    # Add the entry to the list of dictionaries
                    geo_dict_list.append(geo_dict)

        return geo_dict_list
    except Exception as e:
        logger.info(f"Unable to generate Dict due to {e}")


# Main function to either load or generate the geo dict
def get_geo_dict(tiff_path):
    # Check if the geo dict already exists
    existing_geo_dict = load_existing_geo_dict(tiff_path)
    if existing_geo_dict:
        print("Using existing geo dict.")
        return existing_geo_dict
    else:
        try:
            print("Generating new geo dict from the GeoTIFF.")
            new_geo_dict = generate_geo_dict_from_tiff(f"{tiff_path}.tif")
            tiff_path = tiff_path.split('.')[0]
            # Save the newly created geo dict for future use
            with open(f"{tiff_path}.json", 'w') as f:
                json.dump(new_geo_dict, f, indent=4)  # Save with indentation for readability
        except Exception as e:
            logger.info(f"could not generate new dict due to {e}")


########################################################################################################################
def load_image(image_path, mode):
    try:
        main(global_cache.get('natural_disaster', 'No disaster info'))
    except Exception as e:
        logger.error(f"Issue in geo-referencing due to {e}")
    """
    Load the image and return its pixel values, geo-transform, and CRS
    """
    data, geo_transform, spatial_ref = None, None, None
    print(f"Loading image from {image_path} with mode {mode}")

    def process_observation(observation_file):
        nonlocal data, geo_transform, spatial_ref
        print(f"observation {observation_file}")
        dataset__ = gdal.Open(observation_file, gdal.GA_ReadOnly)
        if dataset__ is None:
            raise FileNotFoundError(f"Could not open file {observation_file}")
        band__ = dataset__.GetRasterBand(1)
        data = band__.ReadAsArray()
        geo_transform = dataset__.GetGeoTransform()
        spatial_ref = dataset__.GetProjection()

    if mode == 0:  # Load previous OGM
        dataset_ = gdal.Open(image_path, gdal.GA_ReadOnly)
        band_ = dataset_.GetRasterBand(1)
        data = band_.ReadAsArray()
        geo_transform = dataset_.GetGeoTransform()
        spatial_ref = dataset_.GetProjection()

    elif mode == 1:  # Load the Observation of geo-referenced segmented drone image TFA-06
        for observation in sorted(os.listdir(image_path), reverse=False):
            if observation.endswith("_Segment.tif"):
                logger.info(f"processing segmented drone images {observation}")
                process_observation(os.path.join(image_path, observation))
                # Clean up the old temporary file if it exists
                full_path = os.path.join(image_path, observation)
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
        for observation in sorted(os.listdir(image_path), reverse=False):
            if observation.endswith(".tif"):
                process_observation(os.path.join(image_path, observation))
                full_path = os.path.join(image_path, observation)
                if os.path.exists(full_path):
                    os.remove(full_path)
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

    # Define the geotransform: [top-left x, pixel width (lon), 0, top-left y, 0, -pixel height (lat)]
    geotransform = (x_min, pixel_size_lon, 0, y_max, 0, -pixel_size_lat)
    target_ds.SetGeoTransform(geotransform)

    # Set the projection (spatial reference) of the raster to match the source layer
    target_ds.SetProjection(spatial_ref.ExportToWkt())

    # Rasterize the 'label' attribute into Band 1
    gdal.RasterizeLayer(target_ds, [1], source_layer, options=["ATTRIBUTE=label"])

    # Rasterize the 'score' attribute into Band 2
    gdal.RasterizeLayer(target_ds, [2], source_layer, options=["ATTRIBUTE=score"])

    print(f"Rasterization complete. Output saved as: {geotiff_file}")


def get_geotiff_metadata_rasterio(geotiff_path):
    for file_path in sorted(os.listdir(geotiff_path)):
        if file_path.endswith("_Objects.tif"):
            dataset = gdal.Open(os.path.join(geotiff_path, file_path))
            if dataset is None:
                print(f"Unable to open {geotiff_path}")
                return None

            metadata = dataset.GetMetadata()  # Extract general metadata

            # geo_transform = dataset.GetGeoTransform()  # Get georeferencing information
            # projection = dataset.GetProjection()  # Get projection information

            data = json.loads(metadata['georef_data'])

            # Initialize a new list for the transformed data
            transformed_data = []

            # Transform the data
            for index_, item in enumerate(data):
                key = list(item.keys())[0]  # Get the first key (like "1,114")
                coordinates = item[key]  # Get the coordinates
                score = item['score']  # Get the score
                label = item['label']  # Get the label

                # Create the new format
                new_item = {
                    f"{index_ // 10},{index_ % 10}": coordinates,
                    "score": score,
                    "label": label
                }

                transformed_data.append(new_item)
            return transformed_data


def save_geotiff(output_path_maps_, FileName, data, GTransform, projection):
    """
        Save data as a GeoTIFF file
    """
    data = np.array(data)
    driver = gdal.GetDriverByName('GTiff')
    if not driver:
        logging.error('GTiff driver is not available.')
        return
    height, width = data.shape

    dataset_ogm = driver.Create(os.path.join(output_path_maps_, FileName), width, height, 1, gdal.GDT_Float32)
    dataset_ogm.SetGeoTransform(GTransform)
    dataset_ogm.SetProjection(projection)

    # Define the CRS using EPSG code (EPSG:4326 for WGS 84)
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    dataset_ogm.SetProjection(srs.ExportToWkt())
    band_ = dataset_ogm.GetRasterBand(1)
    band_.WriteArray(data)
    band_.SetDescription('Estimated OGM')  # Band description

    metadata = {
        'title': 'OGM',
        'author': 'GRVC lab, University of Seville',
        'description': f"Estimated OGM for {global_cache.get('natural_disaster', 'No disaster info')}.",
        'creationDate': datetime.now().isoformat(),  # Current date and time
        "XMin": GTransform[0],
        "XRes": GTransform[1],
        "YMax": GTransform[3],
        "YRes": GTransform[5],
        "spatialReference": 'EPSG:4326 for WGS 84',
        "file_name": FileName,
        "bucket": config.BUCKET_NAME
    }
    dataset_ogm.SetMetadata(metadata)
    dataset_ogm.FlushCache()  # Write to disk
    return metadata


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
        raise ValueError("GeoTransform must be a tuple of six elements.")

    if not isinstance(row, (int, float)) or not isinstance(col, (int, float)):
        raise TypeError("Row and column must be integers or floats.")

    # Extract GeoTransform parameters
    origin_x = geo_transform[0]
    pixel_width = geo_transform[1]
    skew_x = geo_transform[2]
    origin_y = geo_transform[3]
    skew_y = geo_transform[4]
    pixel_height = geo_transform[5]

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


def update_occupancy_grid(OGMData, OGM_gt_, observation_data_, observation_gt_):
    """
    Update the occupancy grid map using satellite, drone, and geo social-media measurements.

    Parameters:
        OGMData (np.ndarray): Occupancy grid data.
        OGM_gt_ (dict): Geotransform for the occupancy grid.
        observation_data_ (np.ndarray): Observation data (e.g., drone or satellite image).
        observation_gt_ (dict): Geotransform for the observation.

    Returns:
        np.ndarray: Updated occupancy grid map.
    """
    # # Determine the size of the grid
    # epsilon = 1e-9  # Small value to avoid log(0) or division by zero
    # grid_height, grid_width = OGMData.shape
    #
    # for y in range(grid_height):
    #     for x in range(grid_width):
    #         # Get the center pixel coordinates of the OGM
    #         y_geo, x_geo = pixel_to_coordinates(OGM_gt_, x + 0.5, y + 0.5)  # Use the cell center instead of the corner
    #         # Convert geospatial coordinates to pixel coordinates in satellite image
    #         ogm_x_axis, ogm_y_axis = coordinates_to_pixel(OGM_gt_, x_geo, y_geo)
    #         # Convert geospatial coordinates to pixel coordinates in measurement image
    #         observation_x_axis, observation_y_axis = coordinates_to_pixel(observation_gt_, x_geo, y_geo)
    #         # Check if the pixel falls within the perceptual field of both images
    #         if 0 <= ogm_x_axis < OGMData.shape[1] and 0 <= ogm_y_axis < OGMData.shape[0] and \
    #                 0 <= observation_x_axis < observation_data_.shape[1] and 0 <= observation_y_axis < \
    #                 observation_data_.shape[0]:
    #             # Calculate the posterior probability
    #             # OGMData[ogm_y_axis, ogm_x_axis] is the prior
    #             prev_x = OGMData[ogm_y_axis, ogm_x_axis]
    #             # the likelihood is the measurement at this pixel coordinate
    #             P_z_given_x = observation_data_[observation_y_axis, observation_x_axis]
    #             if P_z_given_x > 0:
    #                 P_z_given_not_x = 1 - P_z_given_x
    #                 P_not_x = 1 - prev_x
    #                 # Calculate the marginal likelihood P(z)
    #                 P_z = (P_z_given_x * prev_x) + (P_z_given_not_x * P_not_x)
    #                 if P_z > 0:
    #                     # Calculate the posterior probability P(H|z) using Bayes' Theorem
    #                     posterior_ND = (P_z_given_x * prev_x) / P_z
    #
    #                     l_t_i_1 = math.log((prev_x + epsilon) / (1 - prev_x + epsilon))
    #
    #                     if posterior_ND > 0:
    #                         inverse_sensor_model = math.log((posterior_ND + epsilon) / (1 - posterior_ND + epsilon))
    #                         l_t_i = l_t_i_1 + inverse_sensor_model
    #                         # Clamp the log-odds to prevent extreme values
    #                         l_t_i = np.clip(l_t_i, -10, 10)
    #                         # Update OGMData
    #                         OGMData[ogm_y_axis, ogm_x_axis] = 1 - (1 / (1 + np.exp(l_t_i)))
    #                     # else:
    #                     #     prior = OGMData[ogm_y_axis, ogm_x_axis]
    #                     #     l_t_i_1 = math.log((prior + epsilon) / (1 - prior + epsilon))
    #                     #     # Update OGMData
    #                     #     OGMData[ogm_y_axis, ogm_x_axis] = 1 - (1 / (1 + np.exp(l_t_i_1)))
    #                     #############################
    #             else:
    #                 prior = OGMData[ogm_y_axis, ogm_x_axis]
    #                 l_t_i_1 = math.log((prior + epsilon) / (1 - prior + epsilon))
    #                 # OGMData[ogm_y_axis, ogm_x_axis] = l_t_i_1
    #                 #############################
    #                 OGMData[ogm_y_axis, ogm_x_axis] = 1 - (1 / (1 + np.exp(l_t_i_1)))
    #                 #############################
    #         # Clamp probabilities to ensure they are within [0, 1]
    #         OGMData[ogm_y_axis, ogm_x_axis] = np.clip(OGMData[ogm_y_axis, ogm_x_axis], 0, 1)
    # return OGMData
    epsilon = 1e-9  # Small value to avoid log(0) or division by zero

    grid_height, grid_width = OGMData.shape

    for y in range(grid_height):
        for x in range(grid_width):
            # Get the center pixel coordinates in geospatial terms
            y_geo, x_geo = pixel_to_coordinates(OGM_gt_, x + 0.5, y + 0.5)

            # Convert geospatial coordinates back to pixel coordinates
            ogm_pixel = coordinates_to_pixel(OGM_gt_, x_geo, y_geo)
            observation_pixel = coordinates_to_pixel(observation_gt_, x_geo, y_geo)

            # Bounds checking
            if not (0 <= ogm_pixel[0] < grid_width and 0 <= ogm_pixel[1] < grid_height):
                continue
            if not (0 <= observation_pixel[0] < observation_data_.shape[1] and
                    0 <= observation_pixel[1] < observation_data_.shape[0]):
                continue

            # Retrieve prior probability from OGM
            ogm_y_axis, ogm_x_axis = int(ogm_pixel[1]), int(ogm_pixel[0])
            prev_prob = OGMData[ogm_y_axis, ogm_x_axis]

            # Measurement (likelihood) at the observation pixel
            obs_y_axis, obs_x_axis = int(observation_pixel[1]), int(observation_pixel[0])
            P_z_given_x = observation_data_[obs_y_axis, obs_x_axis]

            # Compute posterior probability if valid observation
            if P_z_given_x > 0:
                P_z_given_not_x = 1 - P_z_given_x
                P_not_x = 1 - prev_prob
                P_z = P_z_given_x * prev_prob + P_z_given_not_x * P_not_x

                if P_z > 0:
                    posterior_prob = (P_z_given_x * prev_prob) / P_z
                    log_odds_prev = math.log((prev_prob + epsilon) / (1 - prev_prob + epsilon))
                    log_odds_obs = math.log((posterior_prob + epsilon) / (1 - posterior_prob + epsilon))
                    log_odds_updated = log_odds_prev + log_odds_obs

                    # Clamp log-odds and convert back to probability
                    log_odds_clamped = np.clip(log_odds_updated, -10, 10)
                    OGMData[ogm_y_axis, ogm_x_axis] = 1 - (1 / (1 + np.exp(log_odds_clamped)))
            else:
                # No valid measurement: retain prior
                log_odds_prev = math.log((prev_prob + epsilon) / (1 - prev_prob + epsilon))
                OGMData[ogm_y_axis, ogm_x_axis] = 1 - (1 / (1 + np.exp(log_odds_prev)))

            # Clamp probabilities to [0, 1]
            OGMData[ogm_y_axis, ogm_x_axis] = np.clip(OGMData[ogm_y_axis, ogm_x_axis], 0, 1)

    return OGMData


def is_within_fov(ogm_values, observation_values):
    """
    Check if the observation is within the field of view of the OGM values.
    Here, we check if the longitude and latitude match or are within a small tolerance.
    """
    ogm_lon, ogm_lat, _ = ogm_values[:3]  # Extract longitude and latitude
    obs_lon, obs_lat, _ = observation_values[:3]

    # Define a small tolerance to account for minor differences in coordinate values
    tolerance = 1e-5

    # Check if the longitude and latitude are within the tolerance
    return (abs(ogm_lon - obs_lon) <= tolerance) and (abs(ogm_lat - obs_lat) <= tolerance)


def get_observations_in_FOV(georef_ogm, georef_observation):
    """
    Update the occupancy grid using georeferenced dictionaries.

    Parameters:
        georef_ogm (list): List of dictionaries representing the occupancy grid map (OGM).
            Each dictionary contains pixel coordinates as keys and geospatial information,
            along with "score" and "label".
        georef_observation (list): List of dictionaries representing observations.
            Similar structure to georef_ogm but represents updated information.

    Returns:
        list: Updated occupancy grid with modified "score" and "label" where applicable.
    """
    # epsilon = 1e-9  # Small value to avoid log(0) or division by zero
    updated_ogm = []

    # Iterate through each OGM entry
    for ogm in georef_ogm:
        # Create a new dictionary to hold the updated values
        updated_entry = ogm.copy()  # Copy existing entry

        # Extract the pixel key and OGM values
        # Extract pixel key and values
        pixel_key = next((key for key in ogm if key not in ["score", "label"]), None)
        if not pixel_key:
            # Skip malformed entries
            continue

        # pixel_key = [key for key in ogm if key not in ["score", "label"]][0]  # e.g., "0,0"

        ogm_values = ogm[pixel_key]  # The [lon, lat, elev] values
        prior_probability = ogm.get("score", 0)  # Default to 0 if "score" is missing
        prior_label = ogm.get("label", 0)  # Default to 0 if "label" is missing

        # Initialize updated score
        updated_score = prior_probability
        updated_label = prior_label
        observation_found = False

        # Check if this pixel key exists in the observation
        for obs in georef_observation:
            # Extract the pixel coordinate from the observation (e.g., "1,114")
            # obs_pixel_key = [key for key in obs if key not in ["score", "label"]][0]
            obs_pixel_key = next((key for key in obs if key not in ["score", "label"]), None)
            if not obs_pixel_key:
                # Skip malformed observation entries
                continue

            obs_values = obs[obs_pixel_key]  # The [lon, lat, elev] values
            obs_score = obs.get("score", 0)  # Likelihood from observation
            obs_label = obs.get("label", 0)  # Label from observation

            # Ensure the pixel keys match and the observation is within the field of view
            if is_within_fov(ogm_values, obs_values):
                observation_found = True
                if obs_score > updated_score:  # Update only if observation score is higher
                    # if obs_score > 0:
                    updated_score = obs_score
                    updated_label = obs_label

        # Update the entry only if a relevant observation was found
        if observation_found:
            updated_entry["score"] = np.clip(updated_score, 0, 1)
            updated_entry["label"] = updated_label
        else:
            # Retain original values if no observation was found
            updated_entry["score"] = prior_probability
            updated_entry["label"] = prior_label

        # Append the updated entry to the updated occupancy grid
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
