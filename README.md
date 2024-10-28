# Flask Web Application

This repository contains a Flask web application designed for PDM-tech-05 (Information fusion). 

## How to use

### 1. Pull the Docker Image


```bash
docker pull ghcr.io/he-tema/inf_fusion:latest
```
### 2. Environment Variables

The application uses several environment variables that can be defined in the `.env` file. Here are the available variables:

- **HOST**: The host address for the application  is `0.0.0.0`.
- **PORT**: The port on which the application listens  is `5505`.
- **DEBUG**: Default is `True`.
- **BROKER_URL**: Default is `https://orion.tema.digital-enabler.eng.it`.
- **CALLBACK_URL**: URL for notifications. Default is `https://informationfusion.pagekite.me/notify`.
The call back can be updated (the provided callback is for ngrok)


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
  -e CALLBACK_URL="https://informationfusion.pagekite.me/notify" \
  -e MINIO_ENDPOINT="storage.tema.digital-enabler.eng.it:443" \
  -e MINIO_ACCESS_KEY="D4xMAQylbJML0ppbLMtt" \
  -e MINIO_SECRET_KEY="rTV2pwa2PMApAzgV3tssGf7NKNVobM3MalAaSXpY" \
  -e OBJECT_NAME="estimated_ogm_ND.tif" \
  -e BUCKET_NAME="naples" \
  -e PROCESSING_UNIT="cpu" \
  ghcr.io/he-tema/inf_fusion:latest

```

**Explanation of the command:**
- `-d`: Runs the container in detached mode (in the background).
- `-p 5505:5505`: Maps port 5505 on your host to port 5505 on the container.
- `--env-file .env`: Passes the environment variables specified in the `.env` file.

### 3. Access the Application

Once the container is running, you can access the application by navigating to:

```
http://localhost:5505
```

### 4. Application Structure

The main components of the application include:

- `app.py`: The main Flask application file.
- `requirements.txt`: Lists the Python packages required to run the application.
- `.env`: Contains environment variable configurations for the application.

