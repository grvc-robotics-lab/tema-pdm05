import geopandas as gpd
import rasterio
from rasterio.mask import mask
from shapely.geometry import Polygon
import requests
import os
import math
import rasterio
from rasterio.merge import merge
import glob


def merge_tiles(tile_files, output_file):
    """
    Merge multiple raster tiles into a single file.
    tile_files: List of raster file paths.
    output_file: Path to save the merged output.
    """
    rasters = [rasterio.open(f) for f in tile_files]
    mosaic, out_trans = merge(rasters)

    # Update metadata for the mosaic
    out_meta = rasters[0].meta.copy()
    out_meta.update({
        "driver": "GTiff",
        "height": mosaic.shape[1],
        "width": mosaic.shape[2],
        "transform": out_trans
    })

    # Save the mosaic to the output file
    with rasterio.open(output_file, "w", **out_meta) as dest:
        dest.write(mosaic)

    # Close raster files
    for raster in rasters:
        raster.close()

    print(f"Merged DEM saved to {output_file}")


def deg2num(lat_deg, lon_deg, zoom):
    """Convert latitude/longitude to tile numbers."""
    lat_rad = math.radians(lat_deg)
    n = 2.0 ** zoom
    x = int((lon_deg + 180.0) / 360.0 * n)
    y = int((1.0 - math.log(math.tan(lat_rad) + 1 / math.cos(lat_rad)) / math.pi) / 2.0 * n)
    return x, y


def num2deg(x, y, zoom):
    """Convert tile numbers to latitude/longitude."""
    n = 2.0 ** zoom
    lon_deg = x / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * y / n)))
    lat_deg = math.degrees(lat_rad)
    return lat_deg, lon_deg


def get_tiles_from_bounds(bounds, zoom):
    """
    Get tile indices (x, y, z) for a bounding box at a given zoom level.
    Bounds: (min_lon, min_lat, max_lon, max_lat)
    """
    min_lon, min_lat, max_lon, max_lat = bounds
    x_min, y_min = deg2num(max_lat, min_lon, zoom)
    x_max, y_max = deg2num(min_lat, max_lon, zoom)

    tiles = []
    for x in range(x_min, x_max + 1):
        for y in range(y_min, y_max + 1):
            tiles.append({'x': x, 'y': y, 'z': zoom})
    return tiles


# Define the polygon coordinates
polygon_coords = [
    (-120.1, 35.1),
    (-120.1, 35.3),
    (-119.9, 35.3),
    (-119.9, 35.1),
    (-120.1, 35.1),
]
polygon = Polygon(polygon_coords)

# Create a GeoDataFrame
gdf = gpd.GeoDataFrame(index=[0], crs="EPSG:4326", geometry=[polygon])

# URL for AWS Terrain Tiles (you can replace this with another DEM source)
DEM_SOURCE = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"


# Download DEM data using the polygon bounds
def download_dem(bounds, zoom=12, output_file="dem.tif"):
    # Generate tile indices (this example uses a generic URL template for simplicity)
    tiles = get_tiles_from_bounds(bounds, zoom)
    dem_files = []

    for tile in tiles:
        url = DEM_SOURCE.format(z=tile['z'], x=tile['x'], y=tile['y'])
        response = requests.get(url)
        if response.status_code == 200:
            tile_path = f"tile_{tile['x']}_{tile['y']}.png"
            with open(tile_path, "wb") as f:
                f.write(response.content)
            dem_files.append(tile_path)

    # Merge tiles into one DEM file
    if dem_files:
        merge_tiles(dem_files, output_file)
        for file in dem_files:
            os.remove(file)
        print(f"DEM saved to {output_file}")


# A helper function to get tile indices and merge raster tiles can be added.

# Use rasterio to mask and crop the DEM
def clip_dem(input_dem, output_dem, polygon):
    with rasterio.open(input_dem) as src:
        out_image, out_transform = mask(src, [polygon], crop=True)
        out_meta = src.meta.copy()
        out_meta.update({"driver": "GTiff", "height": out_image.shape[1],
                         "width": out_image.shape[2], "transform": out_transform})
        with rasterio.open(output_dem, "w", **out_meta) as dest:
            dest.write(out_image)


# Example usage
bounds = gdf.geometry.total_bounds
download_dem(bounds)
clip_dem("dem.tif", "clipped_dem.tif", polygon)
