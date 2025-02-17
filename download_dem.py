import geopandas as gpd
import requests
from shapely.geometry import Polygon
import os


def download_opentopography_dem(api_key, output_file, coordinates, dem_dataset="SRTMGL1"):
    """
    Download a DEM GeoTIFF for the given polygon coordinates from OpenTopography.

    Parameters:
        api_key (str): OpenTopography API key.
        output_file (str): Path to save the DEM file.
        coordinates (list): List of (longitude, latitude) tuples defining the polygon.
        dem_dataset (str): Dataset to use (e.g., "SRTMGL1" for 30m DEM, "SRTMGL3" for 90m DEM).
    """
    # Create a bounding box from the coordinates
    polygon = Polygon(coordinates)
    minx, miny, maxx, maxy = polygon.bounds

    # OpenTopography API endpoint
    api_url = f"https://portal.opentopography.org/API/globaldem?demtype={dem_dataset}&south={miny}&north={maxy}&west={minx}&east={maxx}&outputFormat=GTiff"

    # Request parameters
    params = {
        "API_Key": api_key
    }

    # Submit the request to OpenTopography
    try:
        print(f"Submitting request to OpenTopography for DEM ({dem_dataset})...")
        response = requests.get(api_url, params=params, stream=True)
        response.raise_for_status()

        # Save the DEM file
        with open(output_file, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        print(f"DEM downloaded and saved to {output_file}")
    except requests.RequestException as e:
        print(f"Failed to download DEM: {e}")


# Example Usage
# Define your OpenTopography API key
opentopo_api_key = "56da0f69ae202d4d9414278b0f6537bd"  # Replace with your OpenTopography API key
polygon_coords =[
      [
        [
          8.646228,
          40.145055
        ],
        [
          8.625344,
          40.144059
        ],
        [
          8.620159,
          40.135551
        ],
        [
          8.624378,
          40.129896
        ],
        [
          8.636082,
          40.127167
        ],
        [
          8.651879,
          40.129928
        ],
        [
          8.650204,
          40.139981
        ],
        [
          8.646228,
          40.145055
        ]
      ]
    ][0]


# Output file path

output_dem_file = "subset_dem.tif"

# Download the DEM
download_opentopography_dem(
    api_key=opentopo_api_key,
    output_file=output_dem_file,
    coordinates=polygon_coords,
    dem_dataset="SRTMGL1"  # Choose "SRTMGL1" (30m resolution) or "SRTMGL3" (90m resolution)
)


#################################################################################
# import rasterio
# import json
# from geojson import Feature, FeatureCollection, dump
#
# # Specify the GeoTIFF file
# geotiff_file = "your_file.tif"
#
# # Read metadata from the GeoTIFF
# with rasterio.open(geotiff_file) as dataset:
#     # Extract metadata
#     metadata = dataset.meta
#
#     # Extract bounds as a GeoJSON geometry
#     bounds = dataset.bounds
#     bbox_geometry = {
#         "type": "Polygon",
#         "coordinates": [[
#             [bounds.left, bounds.bottom],
#             [bounds.left, bounds.top],
#             [bounds.right, bounds.top],
#             [bounds.right, bounds.bottom],
#             [bounds.left, bounds.bottom]
#         ]]
#     }
#
#     # Create a GeoJSON feature
#     feature = Feature(
#         geometry=bbox_geometry,
#         properties={key: metadata[key] for key in metadata if key != 'transform'}
#     )
#
# # Create a FeatureCollection
# feature_collection = FeatureCollection([feature])
#
# # Save the FeatureCollection to a GeoJSON file
# output_file = "output_metadata.geojson"
# with open(output_file, 'w') as f:
#     dump(feature_collection, f)
#
# print(f"Metadata has been saved to {output_file}")

# from datetime import datetime, timezone
# import time
#
# # Example global_cache dictionary
# global_cache = {
#     "expiration": "2025-01-15T13:11:37.899Z"  # Initial expiration value
# }
#
#
# def monitor_expiration():
#     while True:
#         # Get the expiration from global_cache
#         expiration = global_cache.get('expiration')
#
#         if expiration:
#             # Parse the expiration timestamp into a datetime object
#             try:
#                 expiration_time = datetime.strptime(expiration, "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=timezone.utc)
#                 current_time = datetime.now(timezone.utc)
#                 print(current_time)
#
#                 # Check if the current time has reached or surpassed the expiration time
#                 if current_time >= expiration_time:
#                     print(f"Expiration time reached: {expiration}. Exiting the loop.")
#                     break
#             except ValueError as e:
#                 print(f"Invalid expiration format: {expiration}. Error: {e}")
#                 break  # Exit the loop if expiration is malformed
#         else:
#             print("No valid expiration found in global_cache. Exiting the loop.")
#             break
#
#         # Sleep for a short interval to avoid busy waiting
#         time.sleep(1)
#
#
# # Call the function to monitor expiration
# monitor_expiration()