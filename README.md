# Flask Web Application — PDM‑tech‑05 (Information Fusion)

This repository contains a Flask web application for the **PDM‑tech‑05 (Information Fusion)** use case. It integrates with an **Orion Context Broker** and **MinIO** object storage to process GeoTIFF inputs (e.g., Occupancy Grid Maps) and exposes a callback endpoint for notifications.

---

## How to use

### 1) Pull the image

```bash
# Use :latest or pin to a specific version (e.g., :6.03)
docker pull ghcr.io/he-tema/inf_fusion_v02:latest
```
### 2. Environment Variables

The application uses several environment variables
that can be passed as arguments when running the Docker container.
Here are the available variables:

## Environment Variables

| Variable | Description | Default |
|---|---|---|
| `HOST` | Host interface the web app binds to. | `0.0.0.0` |
| `PORT` | Port the web app listens on. | `5505` |
| `DEBUG` | Flask debug flag. | `True` |
| `BROKER_URL` | Orion Context Broker base URL. | `https://orion.tema.digital-enabler.eng.it` |
| `BROKER_ENTITY_Maps4Flood_ID` | NGSI-LD entity id for Maps4Flood. | — |
| `BROKER_ENTITY_Maps4Fire_ID` | NGSI-LD entity id for Maps4Fire. | — |
| `BROKER_ENTITY_Maps4Object_ID` | NGSI-LD entity id for Maps4Object. | — |
| `BROKER_TYPE_ID` | Broker entity type. | `GeoTIFF` |
| `BROKER_SUBSCRIPTION_ID` | Broker subscription id. | `subscription123` |
| `MINIO_ENDPOINT` | MinIO host:port. | `storage.tema.digital-enabler.eng.it:443` |
| `MINIO_ACCESS_KEY` | MinIO access key. | — |
| `MINIO_SECRET_KEY` | MinIO secret key. | — |
| `BUCKET_NAME` | MinIO bucket to read from. | `naples` |
| `OBJECT_NAME` | Object to process. | `estimated_ogm_ND.tif` |
| `PROCESSING_UNIT` | Processing backend. Use `cpu`. | `cpu` |
| `OpenTopography_api_key` | API key for OpenTopography. | — |
| `OGM_OBJ_RESOLUTION` | Resolution (obj) in meters/pixel. | `20` |
| `OGM_ND_RESOLUTION` | Resolution (nd) in meters/pixel. | `20` |
| `SCALING_FACTOR` | Scaling factor for processing. | `1` |
| `CALLBACK_URL` | *(Optional)* Full callback URL override (see below). | — |
| `PUBLIC_IP_ADDRESS` | Public FQDN/IP used to construct `CALLBACK_URL`. | — |
| `BASE_PATH` | Base path (no trailing slash). Example: `/pdm05` | — |
| `API_ENDPOINT` | Final path segment (no leading slash). Example: `notify` | — |

> **Important:** `PROCESSING_UNIT` must be `cpu` (not `cpus`).

---

## Callback URL Configuration

You can either:

1. **Provide `CALLBACK_URL` directly**, or
2. **Let the app construct it** from `PUBLIC_IP_ADDRESS`, `PORT`, `BASE_PATH`, and `API_ENDPOINT`:

```python
CALLBACK_URL = f"https://{PUBLIC_IP_ADDRESS}:{PORT}{BASE_PATH}/{API_ENDPOINT}"
```

### 3. Run the Docker Container

After pulling the image, you can run the application using the following command:

```bash
docker run -it -p 5100:5100 \
  -e HOST="0.0.0.0" \
  -e PORT="5100" \
  -e DEBUG="True" \
  -e BROKER_URL="https://orion.tema.digital-enabler.eng.it" \
  -e BROKER_ENTITY_Maps4Flood_ID="urn:ngsi-ld:USE:PDM-05:Maps4Flood:" \
  -e BROKER_ENTITY_Maps4Fire_ID="urn:ngsi-ld:USE:PDM-05:Maps4Fire:" \
  -e BROKER_ENTITY_Maps4Object_ID="urn:ngsi-ld:USE:PDM-05:Maps4Object:" \
  -e BROKER_TYPE_ID="GeoTIFF" \
  -e CALLBACK_NGROK="https://f2955dbbf912.ngrok-free.app" \
  -e MINIO_ENDPOINT="storage.tema.digital-enabler.eng.it:443" \
  -e MINIO_ACCESS_KEY="AUMFK4CGDFORW7PC9URA" \
  -e MINIO_SECRET_KEY="v9L6zs+G8Qu0UKgfMi8FNIncXtZ+ASMJrAXQwpTB" \
  -e OBJECT_NAME="estimated_ogm_ND.tif" \
  -e BUCKET_NAME="use" \
  -e PROCESSING_UNIT="cpu" \
  -e PUBLIC_IP_ADDRESS="tema-project.ddns.net" \
  -e BASE_PATH="/pdm05" \
  -e API_ENDPOINT="notify" \
  -e OpenTopography_api_key="56da0f69ae202d4d9414278b0f6537bd" \
  -e OGM_OBJ_RESOLUTION=5 \
  -e OGM_ND_RESOLUTION=5 \
  -e SCALING_FACTOR=1 \
  inf_fusion:latest
```

```bash
docker run -it --rm \
  --name inf-fusion \
  -p 5100:5100 \
  -e HOST="0.0.0.0" \
  -e PORT="5100" \
  -e DEBUG="True" \
  -e BROKER_URL="https://orion.tema.digital-enabler.eng.it" \
  -e BROKER_ENTITY_Maps4Flood_ID="urn:ngsi-ld:USE:PDM-05:Maps4Flood:" \
  -e BROKER_ENTITY_Maps4Fire_ID="urn:ngsi-ld:USE:PDM-05:Maps4Fire:" \
  -e BROKER_ENTITY_Maps4Object_ID="urn:ngsi-ld:USE:PDM-05:Maps4Object:" \
  -e BROKER_TYPE_ID="GeoTIFF" \
  -e CALLBACK_NGROK="https://f2955dbbf912.ngrok-free.app" \
  -e BROKER_SUBSCRIPTION_ID="subscription123" \
  -e MINIO_ENDPOINT="storage.tema.digital-enabler.eng.it:443" \
  -e MINIO_ACCESS_KEY="AUMFK4CGDFORW7PC9URA" \
  -e MINIO_SECRET_KEY="v9L6zs+G8Qu0UKgfMi8FNIncXtZ+ASMJrAXQwpTB" \
  -e OBJECT_NAME="estimated_ogm_ND.tif" \
  -e BUCKET_NAME="use" \
  -e PROCESSING_UNIT="cpus" \
  -e PUBLIC_IP_ADDRESS="tema-project.ddns.net" \
  -e BASE_PATH="/pdm05/" \
  -e API_ENDPOINT="/notify/" \
  -e OpenTopography_api_key="56da0f69ae202d4d9414278b0f6537bd" \
  -e OGM_OBJ_RESOLUTION="20" \
  -e OGM_ND_RESOLUTION="20" \
  -e SCALING_FACTOR="1" \
  inf_fusion:latest
```
