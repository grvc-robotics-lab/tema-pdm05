import requests
import logging
import os
import geopandas as gpd
from rasterio.mask import mask
import json
import rasterio
from shapely.geometry import Polygon, mapping

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
entity_test = json.loads("""
{
  "creator": "DLR/EOC Georisks and Civil Security",
  "data": {
    "description": "Identifies all water pixels in the satellite image. Water segmentation is done with a pre-trained convolutional neural network. Invalid pixels are the result of no-data values in the satellite image and/or atmospheric obstructions such as clouds or cloud-shadows.",
    "href": "https://download.geoservice.dlr.de/SWIM_WE/files//2021/07/15/SWIM_WE_S1A_IW_GRDH_1SDV_20210715T055052_20210715T055117_038784_049389_2074.SAFE/SWIM_WE_S1A_IW_GRDH_1SDV_20210715T055052_20210715T055117_038784_049389_2074.SAFE_DATA.tif",
    "title": "Surface Water Extent",
    "type": "image/tiff; application=geotiff; profile=cloud-optimized"
  },
  "dateCreated": "2025-03-03T14:57:59",
  "geometry": {
    "coordinates": [
      [
        [
          6.95202,
          49.21365
        ],
        [
          7.426398,
          50.707546
        ],
        [
          3.721971,
          51.117897
        ],
        [
          3.361044,
          49.622269
        ],
        [
          6.95202,
          49.21365
        ]
      ]
    ],
    "type": "Polygon"
  },
  "id": "urn:ngsi-ld:tema:DLR:TFA-08:EOFloodExtent:S3:latest-ahrtal-2021",
  "overview-mask": {
    "description": "Provides a visual overview of analysis results.",
    "href": "https://download.geoservice.dlr.de/SWIM_WE/files//2021/07/15/SWIM_WE_S1A_IW_GRDH_1SDV_20210715T055052_20210715T055117_038784_049389_2074.SAFE/SWIM_WE_S1A_IW_GRDH_1SDV_20210715T055052_20210715T055117_038784_049389_2074.SAFE_OVERVIEW_MASK.tif",
    "title": "Overview Mask (downsampled)",
    "type": "image/tiff; application=geotiff; profile=cloud-optimized"
  },
  "processing:software": {
    "onnx_model": "1.2.1-s1water",
    "proc_hr_semseg": "2.1.0"
  },
  "type": "EOFloodExtent"
}
""")



# Example usage
polygon_coords =[
      [
        [
          6.996817,
          50.521793
        ],
        [
          6.98238,
          50.517756
        ],
        [
          6.976085,
          50.513527
        ],
        [
          6.980425,
          50.50647
        ],
        [
          6.995879,
          50.511398
        ],
        [
          6.996817,
          50.521793
        ]
      ]
    ]

# Run the download function
download_tif_file(entity_test['data'], polygon_coords)
