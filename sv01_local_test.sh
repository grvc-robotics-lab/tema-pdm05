docker run -it --rm --name sv01 --network host \
  -e HOST=0.0.0.0 \
  -e PORT=5505 \
  -e BROKER_URL=https://orion.tema.digital-enabler.eng.it \
  -e BROKER_ENTITY_ImageMetaData=urn:ngsi-ld:USE:SV-01:ImageMetadata:01 \
  -e BROKER_ENTITY_DroneImages=urn:ngsi-ld:USE:SV-01:DroneImages:01 \
  -e OBJECT_NAME=fireflies-uav-trajectories \
  -e BROKER_TYPE_ID=ImageMetadata \
  -e MINIO_ENDPOINT=storage.tema.digital-enabler.eng.it:443 \
  -e MINIO_ACCESS_KEY=AUMFK4CGDFORW7PC9URA \
  -e MINIO_SECRET_KEY=v9L6zs+G8Qu0UKgfMi8FNIncXtZ+ASMJrAXQwpTB \
  -e BUCKET_NAME=use \
  -e DEBUG=True \
  -e ROS_MASTER_URI=http://192.168.2.10:11311 \
  -e ROS_IP=192.168.2.10 \
  -e NODE=image_publisher.py \
  -e BATCH_SIZE=10 \
  -e SLEEP=5 \
  -e PUBLIC_IP_ADDRESS=localhost:5505 \
  -e BASE_PATH=/sv01 \
  -e API_ENDPOINT=notify \
  -v /mnt/storage/home/abdalraheem/Documents/DRONE_DATA/synthetic_Fire/DJI_202411051053_synthetic_fire:/home/ros/catkin_ws/src/drone_img_transmission/img_src \
  -v /home/abdalraheem/Documents/GitHub/catkin_ws_TEMA/downloads:/home/ros/catkin_ws/downloads \
  sv_tech_01:compressed_stall

    
    
# Montiferro Fire Images (synthetic)
# -v /mnt/storage/home/abdalraheem/Documents/DRONE_DATA/synthetic_Fire/DJI_202411051053_synthetic_fire:/home/ros/catkin_ws/src/drone_img_transmission/img_src \

# Montiferro Fire Images
# -v /mnt/storage/home/abdalraheem/Documents/DRONE_DATA/11_06_2024/DJI_202406111116_015_TEMA-31tema:/home/ros/catkin_ws/src/drone_img_transmission/img_src \

# Ahrtal Flood Images
# -v /mnt/storage/home/abdalraheem/Documents/DRONE_DATA/Ahrthal_Germany/DroneSurvey_DLR_20211023_Images/W:/home/ros/catkin_ws/src/drone_img_transmission/img_src \

# Munich
# -v /home/grvc/Desktop/Munich_IMG_FIRE/W:/home/ros/catkin_ws/src/drone_img_transmission/img_src \    
