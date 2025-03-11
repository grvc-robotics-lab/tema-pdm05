import numpy as np
import requests
import logging
import os
import geopandas as gpd
from osgeo import gdal
from pyproj import Transformer
from rasterio.mask import mask
import json
import rasterio
from shapely.geometry import Polygon

from Kalman_filter_estimating_Objects import KalmanFilter

# Initialize logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def download_tif_file(entity, polygon):
    """
    Downloads a .tif file from the entity's data URL if the notification corresponds to "EOFloodExtent".
    """
    file_url = entity.get("href")

    if file_url:
        file_name = file_url.split("/")[-1]  # Extract filename from URL

        logger.info(f"Downloading {file_name} from {file_url}...")
        print(f"Downloading {file_name} from {file_url}...")

        try:
            response = requests.get(file_url, stream=True)
            disaster_type = "Flood"
            response.raise_for_status()  # Raise an error for bad responses
            file_name = os.path.join(f'/home/grvc/Documents/GitHub/tema-pdm05/downloads/satellite_imgs/{disaster_type}', file_name)
            with open(file_name, "wb") as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)

            logger.info(f"Download completed: {file_name}")
            print(f"Download completed: {file_name}")
            ################################################
            input_tif = file_name
            output_tif =file_name

            polygon = Polygon(polygon[0])
            geojson_polygon = [json.loads(gpd.GeoSeries([polygon]).to_json())['features'][0]['geometry']]

            with rasterio.open(input_tif) as src:
                # Crop the raster
                out_image, out_transform = mask(src, geojson_polygon, crop=True)

                # Update metadata
                out_meta = src.meta.copy()
                out_meta.update({
                    "driver": "GTiff",
                    "height": out_image.shape[1],
                    "width": out_image.shape[2],
                    "transform": out_transform
                })

                # Save the cropped raster
                with rasterio.open(output_tif, "w", **out_meta) as dest:
                    dest.write(out_image)

            print("Cropping complete. Saved as", output_tif)

            ################################################

            return file_name  # Return the filename for further processing if needed

        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to download file: {e}")
            print(f"Failed to download file: {e}")
            return None
    else:
        logger.error("No valid download URL found in entity.")
        print("No valid download URL found in entity.")

# Corrected JSON
# entity_test = json.loads("""
# {
#   "creator": "DLR/EOC Georisks and Civil Security",
#   "data": {
#     "description": "Identifies all water pixels in the satellite image. Water segmentation is done with a pre-trained convolutional neural network. Invalid pixels are the result of no-data values in the satellite image and/or atmospheric obstructions such as clouds or cloud-shadows.",
#     "href": "https://download.geoservice.dlr.de/SWIM_WE/files//2021/07/15/SWIM_WE_S1A_IW_GRDH_1SDV_20210715T055052_20210715T055117_038784_049389_2074.SAFE/SWIM_WE_S1A_IW_GRDH_1SDV_20210715T055052_20210715T055117_038784_049389_2074.SAFE_DATA.tif",
#     "title": "Surface Water Extent",
#     "type": "image/tiff; application=geotiff; profile=cloud-optimized"
#   },
#   "dateCreated": "2025-03-03T14:57:59",
#   "geometry": {
#     "coordinates": [
#       [
#         [
#           6.95202,
#           49.21365
#         ],
#         [
#           7.426398,
#           50.707546
#         ],
#         [
#           3.721971,
#           51.117897
#         ],
#         [
#           3.361044,
#           49.622269
#         ],
#         [
#           6.95202,
#           49.21365
#         ]
#       ]
#     ],
#     "type": "Polygon"
#   },
#   "id": "urn:ngsi-ld:tema:DLR:TFA-08:EOFloodExtent:S3:latest-ahrtal-2021",
#   "overview-mask": {
#     "description": "Provides a visual overview of analysis results.",
#     "href": "https://download.geoservice.dlr.de/SWIM_WE/files//2021/07/15/SWIM_WE_S1A_IW_GRDH_1SDV_20210715T055052_20210715T055117_038784_049389_2074.SAFE/SWIM_WE_S1A_IW_GRDH_1SDV_20210715T055052_20210715T055117_038784_049389_2074.SAFE_OVERVIEW_MASK.tif",
#     "title": "Overview Mask (downsampled)",
#     "type": "image/tiff; application=geotiff; profile=cloud-optimized"
#   },
#   "processing:software": {
#     "onnx_model": "1.2.1-s1water",
#     "proc_hr_semseg": "2.1.0"
#   },
#   "type": "EOFloodExtent"
# }
# """)


# # Example usage
# polygon_coords =[
#       [
#         [
#           6.996817,
#           50.521793
#         ],
#         [
#           6.98238,
#           50.517756
#         ],
#         [
#           6.976085,
#           50.513527
#         ],
#         [
#           6.980425,
#           50.50647
#         ],
#         [
#           6.995879,
#           50.511398
#         ],
#         [
#           6.996817,
#           50.521793
#         ]
#       ]
#     ]
#
# # # Run the download function
# # download_tif_file(entity_test['data'], polygon_coords)


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
        print("No valid '_Objects.tif' files found.")
        return None

    # 2) Iterate over files in sorted order
    for file_path in sorted(file_paths):
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
        #    Each entry in 'data' is expected to be something like:
        #        { "1,114": [lon, lat, elevation], "score": ..., "label": ... }
        #    or a similar structure.
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

        # 6) If we found any features, return the FeatureCollection from the first valid file
        if feature_collection["features"]:
            return feature_collection
        else:
            logger.info(f"No valid features extracted from file: {file_path}")

    # If we reach here, we didn't extract anything from any file
    logger.info("No metadata extracted from valid files.")
    return None

import math

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


transformed_data = get_geotiff_metadata_rasterio(f"georeferenced_drone_images/DJI_20211023131953_0063_W_Objects.tif")

################
# dataset = gdal.Open(f"georeferenced_drone_images/DJI_20211023131953_0063_W_Objects.tif")
# if dataset is None:
#     print(f"Unable to open georeferenced_drone_images/DJI_20211023131953_0063_W_Objects.tif")
#
# # Extract general metadata
# metadata = dataset.GetMetadata()
# if 'georef_data' not in metadata:
#     print(f"'georef_data' field missing in metadata for georeferenced_drone_images/DJI_20211023131953_0063_W_Objects.tif")
# data = json.loads(metadata['georef_data'])
#         # Transform the data
# transformed_data = []
# for index_, item in enumerate(data):
#     key = list(item.keys())[0]  # Extract the first key (e.g., "1,114")
#     coordinates = item[key]  # Extract geographic coordinates [lon, lat, elevation]
#     score = item.get('score', 0.0)  # Extract score, default to 0.0 if missing
#     label = item.get('label', -1)  # Extract label, default to -1 if missing
#     print(f"coordinates  ----- {coordinates}")
#     print(f"score  ----- {score}")
#     print(f"label  ----- {label}")
#     breakpoint()
################



with open(f"georeferenced_drone_images/DJI_20211023131953_0063_W_Objects.json", 'w') as file:
    json.dump(transformed_data, file, indent=4)

with open('georeferenced_drone_images/DJI_20211023131953_0063_W_Objects.json', 'r') as file:
    measurement = json.load(file)

with open('estimated_OGM/occupancy_grid_map_Flood_Objects.json', 'r') as file:
    ogm = json.load(file)

ogm = measurement
# for m in ogm:
#     if m['label'] !=-1:
#         print(m)


# observ_metadata = get_geotiff_metadata_rasterio("georeferenced_drone_images")
# if observ_metadata:
#     ##########################################################
#     if observ_metadata:
#         # 'observ_metadata' should be a dict with "type" and "features" keys
#         features_list = observ_metadata.get("features", [])
#         ogm_list = ogm.get("features", [])
#         # Loop over your OGM metadata
#         for n in ogm_list:
#             for feat in features_list:
#                 # Extract the geometry (coordinates) and properties
#                 z_coords = feat.get("geometry", {}).get("coordinates", [])
#                 ogm_coords = n.get("geometry", {}).get("coordinates", [])
#                 props = feat.get("properties", {})
#
#                 if not z_coords or len(z_coords) < 2:
#                     continue  # Skip if no valid coordinate set
#
#                 # Perform distance check (assuming n['coordinates'] is [lon, lat, (elev)])
#                 distance_th = haversine_distance(ogm_coords, z_coords)
#                 if distance_th < 5:
#                     # print("yes\n")
#                     # Update your OGM metadata with the drone's score/label
#                     n['score'] = props.get("score", 0.0)
#                     n['label'] = props.get("label", -1)
#
#                     # If both have an elevation coordinate, update
#                     if len(ogm_coords) > 2 and len(z_coords) > 2:
#                         ogm_coords[-1] = z_coords[-1]  # Transfer elev
#
#     # Write updated metadata to JSON
#     with open(f'estimated_OGM/occupancy_grid_map_Flood_Objects.json', 'w') as file:
#         json.dump(ogm, file, indent=4)
#     logger.info("Correctly updating the OGM with measurements")
#
##########################################################
# Initialize a 3D Kalman filter for [lon, lat, elev]
state_dim = 3
measurement_dim = 3
kf = KalmanFilter(state_dim, measurement_dim)
print()
ogm_list = ogm.get("features", [])
for entry in ogm_list:
    # print(ogm_cords)
    try:

        label = entry.get("properties", {}).get("label")
        if label == -1:
            continue
        coords = entry.get("geometry", {}).get("coordinates", [])
        print(coords)
        if not isinstance(coords, list) or len(coords) != 3:
            logger.warning(f"Skipping entry with invalid coordinate length: {coords}")
            continue
        # coords -> [lon, lat, elev]
        z = np.array(coords).reshape((measurement_dim, 1))
        kf.predict()
        kf.update(z)

        updated_position = kf.x.flatten().tolist()
        print(f"updated_position ====== {updated_position}")
        # Overwrite the coordinates in the feature geometry
        entry["geometry"]["coordinates"] = updated_position
        print(f"entry['geometry']['coordinates'] ====== {entry['geometry']['coordinates']}")
    except Exception as e:
        logger.error(f"Error updating Kalman filter for entry: {e}")


with open('georeferenced_drone_images/DJI_20211023131953_0063_W_Objects.json', 'w') as file:
    json.dump(ogm, file, indent=4)





# get_geo_dict(f"/home/grvc/Documents/GitHub/tema-pdm05/georeferenced_drone_images/DJI_20211023131953_0063_W_Objects.tif")



# def print_structure(obj, indent=0):
#     if isinstance(obj, dict):
#         for key, value in obj.items():
#             print("  " * indent + f"{key}: {type(value).__name__}")
#             print_structure(value, indent + 1)
#     elif isinstance(obj, list) and obj:
#         print("  " * indent + f"List[{type(obj[0]).__name__}]")
#         print_structure(obj[0], indent + 1)


# print_structure(data)

# Measurement
# List[dict]
#   pixel_coords: list
#     List[int]
#   geo_coords: list
#     List[float]
#   score: float
#   label: int



# OGM
# List[dict]
#   pixel_coords: list
#     List[int]
#   geo_coords: list
#     List[float]
#   score: float
#   label: float