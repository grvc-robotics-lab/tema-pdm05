# Flask Web Application

This repository contains a Flask web application designed for PDM-tech-05 (Information fusion). 

## How to use

### 1. pull the docker image
```bash
docker pull ghcr.io/he-tema/inf_fusion_v02:latest
```
### 2. Environment Variables

The application uses several environment variables
that can be passed as arguments when running the Docker container.
Here are the available variables:

- **HOST**: The host address for the application. Default is `0.0.0.0`.
- **PORT**: The port on which the application listens. Default is `5505`.
- **DEBUG**: Default is `True`.
- **BROKER_URL**: The URL for the Orion Context Broker. Default is `https://orion.tema.digital-enabler.eng.it`.
- **BROKER_ENTITY_Maps4Flood_ID**: The ID for the Maps4Flood entity.
- **BROKER_ENTITY_Maps4Fire_ID**: The ID for the Maps4Fire entity.
- **BROKER_ENTITY_Maps4Object_ID**: The ID for the Maps4Object entity.
- **BROKER_TYPE_ID**: The type of the broker entity. Default is `GeoTIFF`.
- **BROKER_SUBSCRIPTION_ID**: The subscription ID for the service. Default is `subscription123`.
- **MINIO_ENDPOINT**: The MinIO service endpoint. Default is `storage.tema.digital-enabler.eng.it:443`.
- **MINIO_ACCESS_KEY**: The access key for MinIO storage.
- **MINIO_SECRET_KEY**: The secret key for MinIO storage.
- **OBJECT_NAME**: The name of the object to be processed. Default is `estimated_ogm_ND.tif`.
- **BUCKET_NAME**: The MinIO bucket name. Default is `naples`.
- **PROCESSING_UNIT**: The unit used for processing. Default is `cpu`.

### New Callback URL Configuration:

In addition to the existing `CALLBACK_URL` variable,
the following variables are used to construct the callback URL dynamically:

- **PUBLIC_IP_ADDRESS**: The public IP address of your cluster or host (e.g., `your.public.ip`). 
- **BASE_PATH**: The base path for the callback URL (e.g., `/pdm05`). 
- **API_ENDPOINT**: The final API endpoint for the callback URL (e.g., `notify`). 

### Example:

Here’s how the **CALLBACK_URL** is constructed at runtime:

```python
CALLBACK_URL = f"https://{PUBLIC_IP_ADDRESS}:{PORT}{BASE_PATH}/{API_ENDPOINT}"
```

### 3. Run the Docker Container

After pulling the image, you can run the application using the following command:

```bash
docker run -it -p 5505:5505 \
  -e HOST="0.0.0.0" \
  -e PORT="5505" \
  -e DEBUG="True" \
  -e BROKER_URL="https://orion.tema.digital-enabler.eng.it" \
  -e BROKER_ENTITY_Maps4Flood_ID="urn:ngsi-ld:USE:PDM-05:Maps4Flood:01" \
  -e BROKER_ENTITY_Maps4Fire_ID="urn:ngsi-ld:USE:PDM-05:Maps4Fire:01" \
  -e BROKER_ENTITY_Maps4Object_ID="urn:ngsi-ld:USE:PDM-05:Maps4Object:01" \
  -e BROKER_TYPE_ID="GeoTIFF" \
  -e BROKER_SUBSCRIPTION_ID="subscription123" \
  -e MINIO_ENDPOINT="storage.tema.digital-enabler.eng.it:443" \
  -e MINIO_ACCESS_KEY="D4xMAQylbJML0ppbLMtt" \
  -e MINIO_SECRET_KEY="rTV2pwa2PMApAzgV3tssGf7NKNVobM3MalAaSXpY" \
  -e OBJECT_NAME="estimated_ogm_ND.tif" \
  -e BUCKET_NAME="naples" \
  -e PROCESSING_UNIT="cpu" \
  -e PUBLIC_IP_ADDRESS="your.public.ip" \
  -e BASE_PATH="/pdm05" \
  -e API_ENDPOINT="notify" \
  ghcr.io/he-tema/inf_fusion_v02
```
