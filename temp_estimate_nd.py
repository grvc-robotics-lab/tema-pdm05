# import rasterio
# from rasterio.warp import reproject, Resampling
# import numpy as np
#
# # --- Step 1: Open and normalize the second TIFF file ---
# with rasterio.open("downloads/FireSim/threshold_5.tif") as src2:
#     data2 = src2.read(1).astype(np.float32)
#     src2_transform = src2.transform
#     src2_crs = src2.crs
#
# # Normalize data2 to the range [0, 1]
# min_val = np.min(data2)
# max_val = np.max(data2)
# if max_val != min_val:
#     norm_data2 = (data2 - min_val) / (max_val - min_val)
# else:
#     norm_data2 = np.zeros_like(data2)
#
# # --- Step 2: Open the first TIFF file (destination) ---
# with rasterio.open("estimated_OGM/occupancy_grid_map_Fire_20250408_134327.tif") as src1:
#     data1 = src1.read(1).astype(np.float32)  # Values already between 0 and 1
#     dst_transform = src1.transform
#     dst_crs = src1.crs
#     dst_shape = data1.shape
#     dst_profile = src1.profile
#
# # --- Step 3: Reproject the normalized second data onto the grid of the first file ---
# reprojected_data = np.empty(dst_shape, dtype=np.float32)
#
# reproject(
#     source=norm_data2,
#     destination=reprojected_data,
#     src_transform=src2_transform,
#     src_crs=src2_crs,
#     dst_transform=dst_transform,
#     dst_crs=dst_crs,
#     resampling=Resampling.nearest  # Change to Resampling.bilinear if smoother resampling is desired.
# )
#
# # --- Step 4: Accumulate (Add) the data ---
# # Simple addition: sum the first file's data and the reprojected normalized second data
# combined_data = data1 + reprojected_data
#
# # Option A: Clip values above 1 (if you prefer clipping)
# combined_clipped = np.clip(combined_data, 0, 1)
#
# # Option B: Re-normalize the combined data so that the min maps to 0 and max to 1
# combined_min = np.min(combined_data)
# combined_max = np.max(combined_data)
# if combined_max != combined_min:
#     combined_norm = (combined_data - combined_min) / (combined_max - combined_min)
# else:
#     combined_norm = np.zeros_like(combined_data)
#
# # --- Step 5: Save the accumulated result ---
# # Choose whether to save the clipped result or the re-normalized result.
# # For example, here we use the re-normalized result.
# dst_profile.update(dtype=rasterio.float32)
#
# output_filename = "occupancy_grid_map_Fire_updated.tif"
# with rasterio.open(output_filename, "w", **dst_profile) as dst:
#     dst.write(combined_norm, 1)
#
# print(f"Accumulated data have been saved to {output_filename}")

import rasterio
from rasterio.warp import reproject, Resampling
import numpy as np


def accumulate_tiffs(occupancy_path, threshold_path, output_path, method="probabilistic", re_normalize=True):
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
            combined_data = np.clip(combined_data, 0, 1)

    # --- Step 5: Save the accumulated result ---
    dst_profile.update(dtype=rasterio.float32)
    with rasterio.open(output_path, "w", **dst_profile) as dst:
        dst.write(combined_data, 1)

    print(f"Accumulated data saved to: {output_path}")

# Example of how to call the function:
accumulate_tiffs("estimated_OGM/occupancy_grid_map_Fire.tif",
                 "downloads/FireSim/threshold_6.tif",
                 "estimated_OGM/occupancy_grid_map_Fire.tif")

