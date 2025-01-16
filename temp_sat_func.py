# # import geopandas as gpd
# # import requests
# # from shapely.geometry import Polygon
# # import os
# #
# #
# # def download_opentopography_dem(api_key, output_file, coordinates, dem_dataset="SRTMGL1"):
# #     """
# #     Download a DEM GeoTIFF for the given polygon coordinates from OpenTopography.
# #
# #     Parameters:
# #         api_key (str): OpenTopography API key.
# #         output_file (str): Path to save the DEM file.
# #         coordinates (list): List of (longitude, latitude) tuples defining the polygon.
# #         dem_dataset (str): Dataset to use (e.g., "SRTMGL1" for 30m DEM, "SRTMGL3" for 90m DEM).
# #     """
# #     # Create a bounding box from the coordinates
# #     polygon = Polygon(coordinates)
# #     minx, miny, maxx, maxy = polygon.bounds
# #
# #     # OpenTopography API endpoint
# #     api_url = f"https://portal.opentopography.org/API/globaldem?demtype={dem_dataset}&south={miny}&north={maxy}&west={minx}&east={maxx}&outputFormat=GTiff"
# #
# #     # Request parameters
# #     params = {
# #         "API_Key": api_key
# #     }
# #
# #     # Submit the request to OpenTopography
# #     try:
# #         print(f"Submitting request to OpenTopography for DEM ({dem_dataset})...")
# #         response = requests.get(api_url, params=params, stream=True)
# #         response.raise_for_status()
# #
# #         # Save the DEM file
# #         with open(output_file, "wb") as f:
# #             for chunk in response.iter_content(chunk_size=8192):
# #                 f.write(chunk)
# #
# #         print(f"DEM downloaded and saved to {output_file}")
# #     except requests.RequestException as e:
# #         print(f"Failed to download DEM: {e}")
# #
# #
# # # Example Usage
# # # Define your OpenTopography API key
# # opentopo_api_key = "56da0f69ae202d4d9414278b0f6537bd"  # Replace with your OpenTopography API key
# # polygon_coords =[
# #       [
# #         [
# #           8.659286,
# #           40.15841
# #         ],
# #         [
# #           8.659286,
# #           40.210343
# #         ],
# #         [
# #           8.685379,
# #           40.210343
# #         ],
# #         [
# #           8.685379,
# #           40.15841
# #         ],
# #         [
# #           8.659286,
# #           40.15841
# #         ]
# #       ]
# #     ][0]
# #
# #
# # # Output file path
# #
# # output_dem_file = "subset_dem_opentopo.tif"
# #
# # # Download the DEM
# # download_opentopography_dem(
# #     api_key=opentopo_api_key,
# #     output_file=output_dem_file,
# #     coordinates=polygon_coords,
# #     dem_dataset="SRTMGL1"  # Choose "SRTMGL1" (30m resolution) or "SRTMGL3" (90m resolution)
# # )
# #
# #
# # #################################################################################
# # import rasterio
# # import json
# # from geojson import Feature, FeatureCollection, dump
# #
# # # Specify the GeoTIFF file
# # geotiff_file = "your_file.tif"
# #
# # # Read metadata from the GeoTIFF
# # with rasterio.open(geotiff_file) as dataset:
# #     # Extract metadata
# #     metadata = dataset.meta
# #
# #     # Extract bounds as a GeoJSON geometry
# #     bounds = dataset.bounds
# #     bbox_geometry = {
# #         "type": "Polygon",
# #         "coordinates": [[
# #             [bounds.left, bounds.bottom],
# #             [bounds.left, bounds.top],
# #             [bounds.right, bounds.top],
# #             [bounds.right, bounds.bottom],
# #             [bounds.left, bounds.bottom]
# #         ]]
# #     }
# #
# #     # Create a GeoJSON feature
# #     feature = Feature(
# #         geometry=bbox_geometry,
# #         properties={key: metadata[key] for key in metadata if key != 'transform'}
# #     )
# #
# # # Create a FeatureCollection
# # feature_collection = FeatureCollection([feature])
# #
# # # Save the FeatureCollection to a GeoJSON file
# # output_file = "output_metadata.geojson"
# # with open(output_file, 'w') as f:
# #     dump(feature_collection, f)
# #
# # print(f"Metadata has been saved to {output_file}")
# #
# # #################################################################################
# #
# #
# #
# #
# #
# # ##############################################
# #
# # from datetime import datetime, timezone
# # import time
# #
# # # Example global_cache dictionary
# # global_cache = {
# #     "expiration": "2025-01-15T13:11:37.899Z"  # Initial expiration value
# # }
# #
# #
# # def monitor_expiration():
# #     while True:
# #         # Get the expiration from global_cache
# #         expiration = global_cache.get('expiration')
# #
# #         if expiration:
# #             # Parse the expiration timestamp into a datetime object
# #             try:
# #                 expiration_time = datetime.strptime(expiration, "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=timezone.utc)
# #                 current_time = datetime.now(timezone.utc)
# #                 print(current_time)
# #
# #                 # Check if the current time has reached or surpassed the expiration time
# #                 if current_time >= expiration_time:
# #                     print(f"Expiration time reached: {expiration}. Exiting the loop.")
# #                     break
# #             except ValueError as e:
# #                 print(f"Invalid expiration format: {expiration}. Error: {e}")
# #                 break  # Exit the loop if expiration is malformed
# #         else:
# #             print("No valid expiration found in global_cache. Exiting the loop.")
# #             break
# #
# #         # Sleep for a short interval to avoid busy waiting
# #         time.sleep(1)
# #
# #
# # # Call the function to monitor expiration
# # monitor_expiration()
#
#
#
# import rasterio
#
# # Path to the GeoTIFF file
# geotiff_file_path = "estimated_OGM/occupancy_grid_map_Fire_Objects.tif"
#
# # Read and display metadata of the GeoTIFF file
# try:
#     with rasterio.open(geotiff_file_path) as dataset:
#         metadata = dataset.meta
#         bounds = dataset.bounds
#         crs = dataset.crs
#         transform = dataset.transform
#
#         # Display relevant metadata
#         geo_metadata = {
#             "Width": metadata.get("width"),
#             "Height": metadata.get("height"),
#             "CRS": str(crs),
#             "Bounds": {
#                 "Left": bounds.left,
#                 "Bottom": bounds.bottom,
#                 "Right": bounds.right,
#                 "Top": bounds.top,
#             },
#             "Driver": metadata.get("driver"),
#             "DType": metadata.get("dtype"),
#             "Count": metadata.get("count"),
#             "Transform": str(transform),
#         }
#         print(geo_metadata)
# except Exception as e:
#     geo_metadata = {"Error": str(e)}
#
#

import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_bounds
from shapely.geometry import Polygon, mapping
import numpy as np

def clip_raster_to_polygon(raster_path, polygon_coords, output_path):
    """
    Clip raster data to a polygon and set values outside the polygon to zero.

    Args:
        raster_path (str): Path to the input raster.
        polygon_coords (list): List of coordinates defining the polygon.
        output_path (str): Path to save the clipped raster.

    Returns:
        None
    """
    try:
        # Open the raster file
        with rasterio.open(raster_path) as src:
            raster_data = src.read(1)  # Read the first band
            raster_meta = src.meta.copy()

            # Create a polygon object
            polygon = Polygon(polygon_coords)

            # Generate a mask for the polygon
            mask = rasterize(
                [(mapping(polygon), 1)],  # Geometry and value for inside polygon
                out_shape=raster_data.shape,
                transform=src.transform,
                fill=0,  # Values outside the polygon
                dtype=np.uint8
            )

            # Apply the mask to the raster
            clipped_data = np.where(mask == 1, raster_data, 0)

            # Update metadata for the output raster
            raster_meta.update({
                "dtype": clipped_data.dtype,
                "count": 1
            })

            # Write the clipped raster to the output path
            with rasterio.open(output_path, "w", **raster_meta) as dst:
                dst.write(clipped_data, 1)

        print(f"Clipped raster saved to {output_path}")
    except Exception as e:
        print(f"Error: {e}")

# Example usage
polygon_coords = [
      [
        [
          8.632164,
          40.172315
        ],
        [
          8.685722,
          40.169692
        ],
        [
          8.645897,
          40.121141
        ],
        [
          8.588905,
          40.115627
        ],
        [
          8.583412,
          40.147126
        ],
        [
          8.588219,
          40.171004
        ],
        [
          8.632164,
          40.172315
        ]
      ]
    ]

raster_path = "estimated_OGM/occupancy_grid_map_Fire.tif"  # Replace with your raster file path
output_path = "clipped_output.tif"  # Replace with desired output path

clip_raster_to_polygon(raster_path, polygon_coords[0], output_path)
