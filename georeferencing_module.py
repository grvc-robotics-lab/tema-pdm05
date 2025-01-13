import numpy as np
import os
from PIL import Image
from skimage.transform import downscale_local_mean
from math import sqrt, radians, sin, cos, tan, pi
from osgeo import gdal, osr
import rasterio
import json
from pyproj import Transformer
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

from logging_config import logger

path = None


def main(natural_disaster, flag):
    global path
    disaster = natural_disaster
    if natural_disaster == 'Flood':
        path = './downloads/drone_imgs/Flood/'
    elif natural_disaster == 'Fire':
        path = './downloads/drone_imgs/Fire/'
    output_path = './georeferenced_drone_images/'
    DEM_path = f'{path}subset_dem.tif'

    # Parameter selection
    downsampling = True
    downsampling_factors = (4, 4)  # (height, width)
    coordinate_system = 4326  # For GeoTIFF use coordinate system EPSG:4326 or EPSG:3857
    terrain_model = 'uneven'  # Select between 'flat' and 'uneven'
    parallel_processing = True

    # Camera coordinate system configuration
    '''
    In this project the camera plane coordinate system (c) is related to
    the drone body coordinate system (b) as follows:

                               ^  Y_c = -Z_b
                               | 
                               |
                      __ __ __ |__ __ __ 
                     |         |        |
                     |         |        |
                     |         |        |--------> X_c == Y_b
                     | camera plane     |
                     |__ __ __ __ __ __ |

    In DJI three perpendicular axes are defined for the drone body coordinate system
    such that the origin is the center of mass, and the X axis is directed through
    the front of the drone and the Y axis through the right of the drone. Using the
    coordinate right hand rule, the Z axis is then through the bottom of the drone.
    So there is an initial fixed rotation transformation from body to camera
    which is needed to pre multiplied by the varient rotation matrixt (roll, pitch, yaw)
    '''
    R_x__pi_2 = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
    R_y__pi_2 = np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]])
    Rot_b_c_fixed = np.dot(R_x__pi_2, R_y__pi_2)  # Rotation from Body to Camera
    '''
    On the other hand, the X and Y axis of the world coordinate system are in the
    positive directions of East and North respectively in our convention. So the
    body coordinate system (b) is related to the world coordinate system (w) as follows:

                               ^  Y_w = X_b
                               | 
                               |
                               | 
                               |
                               |
                               |-------------> X_w == Y_b


    Then there is another initial fixed rotation transformation from world to camera
    '''
    R_z_pi_2 = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    R_x_pi = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    Rot_w_b_fixed = np.dot(R_z_pi_2, R_x_pi)
    '''
    Given that the Euler angles roll, pitch and yaw are measured in the body frame,
    we can calculate a dynamic rotation and rotations between frames:
    Rot_dyn = b_to_cam_rot_matrix(roll, pitch, yaw)
    rot_b2c = np.dot(Rot_dyn, Rot_b_c_fixed)
    rot_w2c = np.dot(Rot_w_b_fixed, rot_b2c)
    and then the orientation of an array in the camera plane (r_c) in the world
    frame:
    r_w = np.dot(rot_w2c, r_c)
    '''

    # Retrieve DEM metadata
    with rasterio.open(DEM_path) as dem:
        dem_elevation_data = dem.read(1)
        dem_metadata = dem.meta
        dem_dim = np.array([dem_metadata['width'], dem_metadata['height']])
        dem_res_geo = np.array([dem_metadata['transform'][0], dem_metadata['transform'][4]])
        dem_res_meter = np.array([lon_or_lat_to_meter(dem_metadata['transform'][5], dem_res_geo[0], 0),
                                  lon_or_lat_to_meter(dem_metadata['transform'][5], dem_res_geo[1], 1)])
        dem_tlc = np.array([dem_metadata['transform'][2], dem_metadata['transform'][5]])
        dem_brc = dem_tlc + dem_dim * dem_res_geo
        dem_bounds = ((dem_tlc[0], dem_brc[0]), (dem_brc[1], dem_tlc[1]))
        max_elevation = np.max(dem_elevation_data)
        min_elevation = np.min(dem_elevation_data)

    file_name = 'none'
    for file in sorted(os.listdir(path), reverse=False):
        if flag == "segmented":
            if file.endswith('.png'):
                file_name = os.path.splitext(file)[0]
                if file_name == 'none':
                    logger.info('Error: No .JPG nor .png files found in the specified directory.')
                    return
                else:
                    run_geo(output_path, path, disaster, file_name, Rot_b_c_fixed, Rot_w_b_fixed, dem_elevation_data,
                            max_elevation,
                            min_elevation, dem_bounds,
                            dem_res_geo, dem_res_meter, downsampling, downsampling_factors, coordinate_system,
                            terrain_model, parallel_processing, flag)

        elif flag == "bbox":
            if file.endswith('_obj.json'):
                file_name = "_".join((file.split("_"))[:-1])
                if file_name == 'none':
                    logger.info('Error: No .JPG nor .png files found in the specified directory.')
                    return
                else:
                    run_geo(output_path, path, disaster, file_name, Rot_b_c_fixed, Rot_w_b_fixed, dem_elevation_data,
                            max_elevation,
                            min_elevation, dem_bounds,
                            dem_res_geo, dem_res_meter, downsampling, downsampling_factors, coordinate_system,
                            terrain_model, parallel_processing, flag)

        #         break
        # if file_name == 'none':
        #     logger.info('Error: No .JPG nor .png files found in the specified directory.')
        #     return
        #
        # run_goe(output_path, path, disaster, file_name, Rot_b_c_fixed, Rot_w_b_fixed, dem_elevation_data, max_elevation,
        #         min_elevation, dem_bounds,
        #         dem_res_geo, dem_res_meter, downsampling, downsampling_factors, coordinate_system,
        #         terrain_model, parallel_processing)


def run_geo(output_path, path, disaster, file_name,
            Rot_b_c_fixed, Rot_w_b_fixed, dem_elevation_data, max_elevation, min_elevation, dem_bounds, dem_res_geo,
            dem_res_meter,
            downsampling, downsampling_factors, coordinate_systeme, terrain_model, parallel_processing, flag):
    # Inputs
    image_path = path + file_name

    segmented_image = f'{image_path}.png'
    segmented_metadata_file = f'{image_path}_metadata.json'
    start_pixel = (0, 0)
    # Georeferencing the segmented image
    if flag == "segmented":
        # Retrieve the camera state from image metadata
        with open(segmented_metadata_file, 'r') as met_file:
            original_metadata = json.load(met_file)
        metadata = create_metadata(original_metadata)
        camera_position = [value for key, value in metadata['drone_location'].items()]
        camera_orientation = [value for key, value in metadata['gimbal_parameters'].items()]
        DFOV = metadata['camera_parameters']['fov']
        roll = camera_orientation[0]
        with Image.open(segmented_image) as img:
            image = np.array(img)
            if len(image.shape) == 3:
                image = np.mean(image, axis=2)
            if downsampling:
                image = downscale_local_mean(image, downsampling_factors)
            image_dim = image.shape

            # Put the values of four corners and keys 1 (which is != 0) to be georeferenced in the ray-tracing function
            image[0, 0] = image[image.shape[0] - 1, image.shape[1] - 1] = image[image.shape[0] - 1, 0] = image[
                0, image.shape[1] - 1] = 1

        camera_focal_length = compute_focal_length(image_dim, DFOV)  # In pixels
        start_pixel = (0, 0)
        end_pixel = (image_dim[0] - 1, image_dim[1] - 1)

        if parallel_processing:
            georef_dic_seg = parallel_ray_tracing(terrain_model, disaster, image, image_dim, camera_position,
                                                  camera_orientation,
                                                  Rot_b_c_fixed, Rot_w_b_fixed, roll, camera_focal_length,
                                                  dem_elevation_data,
                                                  max_elevation, min_elevation, dem_bounds, dem_res_geo, dem_res_meter,
                                                  start_pixel, end_pixel)
        else:
            georef_dic_seg = ray_tracing(terrain_model, disaster, image, image_dim, camera_position, camera_orientation,
                                         Rot_b_c_fixed,
                                         Rot_w_b_fixed, roll, camera_focal_length, dem_elevation_data, max_elevation,
                                         min_elevation,
                                         dem_bounds, dem_res_geo, dem_res_meter, start_pixel, end_pixel)

        create_geotif(output_path, file_name, '_Segment', image, 'none', georef_dic_seg, camera_orientation,
                      coordinate_systeme)

        # Clean up an old temporary files
        os.remove(segmented_image)
        os.remove(segmented_metadata_file)

    # Georeferencing the objects of the image
    if flag == 'bbox':
        bounding_boxes_file = f'{image_path}_obj.json'
        if os.path.exists(bounding_boxes_file):
            # Retrieve the camera state from image metadata
            object_metadata_file = f'{image_path}_objects_metadata.json'
            with open(object_metadata_file, 'r') as met_file:
                original_metadata = json.load(met_file)
            metadata = create_metadata(original_metadata)
            camera_position = [value for key, value in metadata['drone_location'].items()]
            camera_orientation = [value for key, value in metadata['gimbal_parameters'].items()]
            DFOV = metadata['camera_parameters']['fov']
            img_height = metadata["camera_parameters"]['height'] // downsampling_factors[0]
            img_width = metadata["camera_parameters"]['width'] // downsampling_factors[1]
            camera_focal_length = compute_focal_length((img_height, img_width), DFOV)  # In pixels

            roll = camera_orientation[0]

            # Retreive bounding box JSON file information
            with open(bounding_boxes_file, 'r') as file:
                bounding_data = json.load(file)
                bounding_boxes = []
                scores = []
                labels = []
                if len(bounding_data)>0:
                    for _ in range(len(bounding_data['value'])):
                        bounding_boxes.append([int(i) for i in bounding_data['value'][_]['bbox']])
                        scores.append(bounding_data['value'][_]['confidence'])
                        labels.append(bounding_data['value'][_]['category_id'])

                    #bounding_boxes = [[int(i) for i in bounding_box] for bounding_box in bounding_boxes]
                    if downsampling:
                        down_bounding_box = []
                        for bounding_box in bounding_boxes:
                            down_bbx = [bounding_box[0] // downsampling_factors[1],
                                        bounding_box[1] // downsampling_factors[0],
                                        bounding_box[2] // downsampling_factors[1],
                                        bounding_box[3] // downsampling_factors[0]]
                            down_bounding_box.append(down_bbx)
                        bounding_boxes = down_bounding_box
            image_emp = np.zeros((img_height, img_width))

            # Put the values of four corners as 1 (which is != 0) to be georeferenced in the ray-tracing function
            image_emp[0, 0] = image_emp[img_height - 1, img_width - 1] = image_emp[img_height - 1, 0] = image_emp[0, img_width - 1] = 1

            end_pixel = (img_height - 1, img_width - 1)

            # The corners of the empty image should be georeferenced to be used in the GEOTIFF creation
            crn_dic = ray_tracing(terrain_model, disaster, image_emp, (img_height, img_width), camera_position, camera_orientation,
                                  Rot_b_c_fixed,
                                  Rot_w_b_fixed, roll, camera_focal_length, dem_elevation_data, max_elevation,
                                  min_elevation,
                                  dem_bounds, dem_res_geo, dem_res_meter, start_pixel, end_pixel)
            image_emp = np.zeros((img_height, img_width))
            boxes_georeferencing = []
            for _ in range(len(bounding_boxes)):
                # Find the center of mass pixel
                cm_pixel = (int((bounding_boxes[_][1] + bounding_boxes[_][3]) / 2),
                            int((bounding_boxes[_][0] + bounding_boxes[_][2]) / 2))
                georef_dic_obj = ray_tracing(terrain_model, disaster, image_emp, image_dim, camera_position,
                                             camera_orientation, Rot_b_c_fixed,
                                             Rot_w_b_fixed, roll, camera_focal_length, dem_elevation_data,
                                             max_elevation,
                                             min_elevation, dem_bounds, dem_res_geo, dem_res_meter, cm_pixel, cm_pixel)

                image_emp[cm_pixel] = scores[_]
                georef_dic_obj['score'] = scores[_]
                georef_dic_obj['label'] = labels[_]
                boxes_georeferencing.append(georef_dic_obj)

            create_geotif(output_path, file_name, '_Objects', image_emp, crn_dic, boxes_georeferencing,
                          camera_orientation,
                          coordinate_systeme)

            # Clean up an old temporary files
            os.remove(bounding_boxes_file)
            os.remove(object_metadata_file)


def create_metadata(json_data):
    '''
    This function returns camera position, orientation, and field of view by reading
    the image metadata
    '''
    lon_dms = json_data['GPSLongitude'].split()
    camera_lon = dms_to_decimal(lon_dms[0], lon_dms[2][:-1], lon_dms[3][:-2], lon_dms[4])
    lat_dms = json_data['GPSLatitude'].split()
    camera_lat = dms_to_decimal(lat_dms[0], lat_dms[2][:-1], lat_dms[3][:-2], lat_dms[4])
    camera_alt = float(json_data['GPSAltitude'].split()[0])
    camera_alt_rel = float(json_data['RelativeAltitude'])
    gimbal_roll = radians(float(json_data['GimbalRollDegree']))
    gimbal_pitch = radians(float(json_data['GimbalPitchDegree']))
    gimbal_yaw = radians(float(json_data['GimbalYawDegree']))
    DFOV = float(json_data['FOV'].split()[0])
    img_height = int(json_data['ImageHeight'])
    img_width = int(json_data['ImageWidth'])
    ImageMetadata = {
        "drone_location": {"longitude": camera_lon, "latitude": camera_lat, "altitude": camera_alt,
                           "altitude_rel": camera_alt_rel},
        "camera_parameters": {"fov": DFOV, "height": img_height, "width": img_width},
        "gimbal_parameters": {"roll": gimbal_roll, "pitch": gimbal_pitch, "yaw": gimbal_yaw},
    }
    return ImageMetadata


def dms_to_decimal(degrees, minutes, seconds, direction):
    '''
    This function converts degrees, minutes, seconds to decimal degrees
    '''
    decimal_degrees = float(degrees) + float(minutes) / 60 + float(seconds) / (60 * 60)

    if direction in ['S', 'W']:
        decimal_degrees *= -1

    return decimal_degrees


def compute_focal_length(iamge_dimensions, DFOV):
    '''
    This function computes a camera focal length in pixel, given the image
    width and height in pixel and diametric field of view in degrees
    '''
    height, width = iamge_dimensions[:2]
    diameter = sqrt(width ** 2 + height ** 2)
    focal_pixel = (diameter / 2) / tan(radians(DFOV / 2))

    return focal_pixel


def find_ray_tip(ray, dem_bounds, dem_res_geo, dem_elevation_data, min_elevation):
    '''
    This function returns the location of the ray tip in the DEM grid
    i and j are indexes of the cell on which the ray tip is.
    i is the latitude index and starts from the top boundary of the grid.
    j is the longitude index and starts from the left boundary of the grid.
    The value of -1 for each i or j, means the ray tip is not in DEM
    '''
    # The left longitude = dem_bounds[0][0] and The upper latitude = dem_bounds[1][1]
    i = j = -1
    if ray[0] > dem_bounds[0][0] and ray[0] < dem_bounds[0][1]:
        j = int((ray[0] - dem_bounds[0][0]) / dem_res_geo[0])
    if ray[1] > dem_bounds[1][0] and ray[1] < dem_bounds[1][1]:
        i = int((ray[1] - dem_bounds[1][1]) / dem_res_geo[1])
    if i != -1 and j != -1:
        return i, j, dem_elevation_data[i, j]
    else:
        return i, j, min_elevation


def find_raycasting_origin(camera_origin, ray_dir, dem_bounds, max_elevation):
    '''
    This function returns the point from which we starts ray casting as a distance
    from the camera origion. It finds the location in which an adge of the DEM
    cuts the ray casted from the camera and consider it as the rey tracing start
    point. To this end, all possible locations of the camera related to DEM are
    considered
    '''
    cutting_index = 2
    cutting_adge = max_elevation

    # The component of distance between the camera origion and the cutting adge:
    displacement_component = cutting_adge - camera_origin[cutting_index]

    # Return R, the magnitude of the ray at the begining of the ray-casting
    return abs(displacement_component / ray_dir[cutting_index])


def b_to_cam_rot_matrix(pitch, yaw):
    # Compute individual rotation matrices
    R_pitch = np.array([[cos(pitch), 0, sin(pitch)],
                        [0, 1, 0],
                        [-sin(pitch), 0, cos(pitch)]])
    R_yaw = np.array([[cos(yaw), -sin(yaw), 0],
                      [sin(yaw), cos(yaw), 0],
                      [0, 0, 1]])

    # Combine rotation matrices
    Rotation = np.dot(R_yaw, R_pitch)

    return Rotation


def meterTo_lon_lat(dx, dy, longitude, latitude):
    r_earth = 6371000
    new_longitude = longitude + (dx / r_earth) * (180 / pi) / cos(latitude * pi / 180)
    new_latitude = latitude + (dy / r_earth) * (180 / pi)

    return new_longitude, new_latitude


def lon_or_lat_to_meter(origin_lat, displacement, direction):
    r_earth = 6371000
    if direction == 0:  # Longitude direction
        return r_earth * displacement * (pi / 180) * cos(origin_lat * pi / 180)
    elif direction == 1:  # Latitude direction
        return r_earth * displacement * (pi / 180)


def ray_tracing(terrain_type, disaster, image, image_dimensions, camera_position, camera_orientation, Rot_b_c_fixed,
                Rot_w_b_fixed, roll, focal_length, dem_elevation_data, max_elevation, min_elevation, dem_bounds,
                dem_res_geo, dem_res_meter, start_pixel, end_pixel, camera_range=2000, epsilon=0.001):
    # Rotation from world to camera:
    Rot_dyn = b_to_cam_rot_matrix(camera_orientation[1], camera_orientation[2])
    rot_b2c = np.dot(Rot_dyn, Rot_b_c_fixed)
    rot_w2c = np.dot(Rot_w_b_fixed, rot_b2c)
    rot_w2c = np.dot(rot_w2c, np.array([[cos(roll), -sin(roll), 0], [sin(roll), cos(roll), 0], [0, 0, 1]]))

    # Initializing geo referencing dictionary
    Georef = {}
    dz = - focal_length
    for i in [_ for _ in range(start_pixel[0], end_pixel[0])] + [end_pixel[0]]:  # In image height diraction
        for j in [_ for _ in range(start_pixel[1], end_pixel[1])] + [end_pixel[1]]:  # In image width direction
            if image[i, j] != 0:
                # Calculation of the direction of the ray which is going to be cast through the pixel
                dx = j - image_dimensions[1] / 2 + 0.5  # Add 0.5 to be in the center of each pixel
                dy = image_dimensions[0] / 2 - i - 0.5
                r_dir = np.array([dx, dy, dz])
                r_dir /= np.linalg.norm(r_dir)  # Normalization
                r_dir = np.dot(rot_w2c, r_dir)  # Transfering to the world coordinate system
                if terrain_type == 'flat':
                    lz = -camera_position[3]
                    R = lz / r_dir[2]
                    dem_lon, dem_lat = meterTo_lon_lat(R * r_dir[0], R * r_dir[1], camera_position[0],
                                                       camera_position[1])
                    Georef[str(i) + ',' + str(j)] = [round(item, 6) for item in [dem_lon, dem_lat]]
                elif terrain_type == 'uneven':
                    # The ray tracing begins from this distance from the camera origin:
                    if disaster == 'Flood': R = find_raycasting_origin(camera_position, r_dir, dem_bounds,
                                                                       max_elevation)
                    if disaster == 'Fire': R = 0
                    R = 0
                    base_step = np.min(abs(dem_res_meter / (r_dir[:2] + 0.0001)))
                    tin_step = False
                    sum_idxs = 0  # For chacking the ray doesn't pass more then one cell in each step
                    while R < camera_range:
                        ray_lon, ray_lat = meterTo_lon_lat(R * r_dir[0], R * r_dir[1], camera_position[0],
                                                           camera_position[1])
                        ray_alt = camera_position[2] + R * r_dir[2]

                        # The vector of the ray tip with respect to the World origin:
                        ray = np.array([ray_lon, ray_lat, ray_alt])

                        # Elevation in DEM at the location of the ray tip:
                        lat_idx, lon_idx, dem_alt = find_ray_tip(ray, dem_bounds, dem_res_geo, dem_elevation_data,
                                                                 min_elevation)

                        # This measures the step by which the ray hits the level of the currant cell:
                        hit_step = abs((ray_alt - dem_alt) / r_dir[2])
                        step = min(base_step, hit_step) + epsilon

                        # It is needed for chacking the ray doesn't pass more than one cell in each step:
                        new_sum_idxs = lat_idx + lon_idx

                        # Check for intersection:
                        if ray_alt <= dem_alt and lat_idx != -1 and lon_idx != -1:
                            dem_lon = dem_bounds[0][0] + (lon_idx + 0.5) * dem_res_geo[0]
                            dem_lat = dem_bounds[1][1] + (lat_idx + 0.5) * dem_res_geo[1]

                            # Chack the ray doesn't pass more then one cell in each step
                            if abs(new_sum_idxs - sum_idxs) != 2 or tin_step:
                                Georef[str(i) + ',' + str(j)] = [round(item, 6) for item in
                                                                 [ray_lon, ray_lat, dem_alt.astype(float)]]
                                break  # Intersection detected
                            else:
                                R -= step
                                step /= 2
                                tin_step = True

                        sum_idxs = new_sum_idxs
                        R += step

    return Georef


def process_chunk(terrain_type, disaster, image, image_dimensions, camera_position, camera_orientation,
                  Rot_b_c_fixed, Rot_w_b_fixed, roll, focal_length, dem_elevation_data, max_elevation,
                  min_elevation, dem_bounds, dem_res_geo, dem_res_meter, chunk_start_pixel, chunk_end_pixel,
                  camera_range=2000, epsilon=0.001):
    """This function processes a subset (chunk) of the image."""
    return ray_tracing(terrain_type, disaster, image, image_dimensions, camera_position, camera_orientation,
                       Rot_b_c_fixed, Rot_w_b_fixed, roll, focal_length, dem_elevation_data, max_elevation,
                       min_elevation, dem_bounds, dem_res_geo, dem_res_meter, chunk_start_pixel, chunk_end_pixel,
                       camera_range, epsilon)


def parallel_ray_tracing(terrain_type, disaster, image, image_dimensions, camera_position, camera_orientation,
                         Rot_b_c_fixed, Rot_w_b_fixed, roll, focal_length, dem_elevation_data, max_elevation,
                         min_elevation, dem_bounds, dem_res_geo, dem_res_meter, start_pixel, end_pixel,
                         camera_range=2000, epsilon=0.001, n_chunks=multiprocessing.cpu_count()):
    # Split the range of pixels (start_pixel to end_pixel) into chunks
    pixel_height_range = end_pixel[0] - start_pixel[0]
    chunk_size = pixel_height_range // n_chunks

    # Create chunk ranges for parallel processing
    chunk_ranges = []
    for i in range(n_chunks):
        chunk_start_pixel = (start_pixel[0] + i * chunk_size, start_pixel[1])
        if i == n_chunks - 1:  # Last chunk includes the remainder
            chunk_end_pixel = end_pixel
        else:
            chunk_end_pixel = (start_pixel[0] + (i + 1) * chunk_size - 1, end_pixel[1])
        chunk_ranges.append((chunk_start_pixel, chunk_end_pixel))

    # Use ProcessPoolExecutor for parallel processing
    georef = {}
    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(process_chunk, terrain_type, disaster, image, image_dimensions, camera_position,
                            camera_orientation, Rot_b_c_fixed, Rot_w_b_fixed, roll, focal_length,
                            dem_elevation_data, max_elevation, min_elevation, dem_bounds, dem_res_geo,
                            dem_res_meter, chunk_start, chunk_end, camera_range, epsilon)
            for chunk_start, chunk_end in chunk_ranges
        ]

        # Combine the results of each chunk
        for future in futures:
            chunk_georef = future.result()  # Get the result of each chunk
            georef.update(chunk_georef)

    return georef


def create_geotif(output, file_name, subject, image, crn_dic, georef_data, camera_orientation, coordinate_system):
    img_height, img_width = image.shape
    orig_dic = crn_dic
    if crn_dic == 'none': orig_dic = georef_data

    # Create a new GeoTIFF file
    driver = gdal.GetDriverByName('GTiff')
    dataset = driver.Create(output + file_name + subject + '.tif', img_width, img_height, 1,
                            gdal.GDT_UInt16, options=['COMPRESS=LZW', 'TILED=YES'])

    # Write the numpy image data to the GeoTIFF file band by band
    dataset.GetRasterBand(1).WriteArray(image)

    # Convert georef_dic to JSON string
    json_metadata = json.dumps(georef_data)

    # Attach JSON metadata to the image
    dataset.SetMetadataItem('georef_data', json_metadata)

    # Assume the image is in WGS84 coordinate system (EPSG:4326)
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(coordinate_system)

    # Set the spatial reference
    dataset.SetProjection(srs.ExportToWkt())

    rotation = -camera_orientation[2]  # Rotation angle in degrees
    cos_rotation = cos(rotation)
    sin_rotation = sin(rotation)

    if coordinate_system == 4326:
        tl_lon, tl_lat = orig_dic['0,0'][:2]
        tr_lon, tr_lat = orig_dic['0,' + str(img_width - 1)][:2]
        bl_lon, bl_lat = orig_dic[str(img_height - 1) + ',0'][:2]

    elif coordinate_system == 3857:
        tl_lon, tl_lat = Transformer.from_crs("EPSG:4326", "EPSG:3857").transform(orig_dic['0,0'][1],
                                                                                  orig_dic['0,0'][0])
        tr_lon, tr_lat = Transformer.from_crs("EPSG:4326", "EPSG:3857").transform(
            orig_dic['0,' + str(img_width - 1)][1],
            orig_dic['0,' + str(img_width - 1)][0])
        bl_lon, bl_lat = Transformer.from_crs("EPSG:4326", "EPSG:3857").transform(
            orig_dic[str(img_height - 1) + ',0'][1],
            orig_dic[str(img_height - 1) + ',0'][0])

    delta_x = sqrt((tl_lon - tr_lon) ** 2 + (tl_lat - tr_lat) ** 2)
    delta_y = sqrt((tl_lon - bl_lon) ** 2 + (tl_lat - bl_lat) ** 2)
    x_res = delta_x / img_width
    y_res = -delta_y / img_height

    geotransform = (
        tl_lon, x_res * cos_rotation, x_res * sin_rotation, tl_lat, y_res * -sin_rotation,
        y_res * cos_rotation)
    dataset.SetGeoTransform(geotransform)

    return image, geotransform, srs.ExportToWkt()
