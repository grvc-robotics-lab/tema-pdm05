# Flask Web Application - PDM-tech-05 (Information Fusion)

This repository contains a Flask web application for the **PDM‑tech‑05 (Information Fusion)** use case. It integrates with an **Orion Context Broker** and **MinIO** object storage to process GeoTIFF inputs (e.g., Occupancy Grid Maps) and exposes a callback endpoint for notifications.

---

## How to use

### 1) Pull the image

```bash
# Use :latest or pin to a specific version (e.g., :6.03)
docker pull ghcr.io/he-tema/inf_fusion_v02:latest
```
### 2. Environment Variables

The application uses environment variables that can be passed when running the Docker container.
The most relevant variables are listed below.

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
| `CALLBACK_URL` | Optional full callback URL override. If set, it is used directly. | — |
| `PUBLIC_IP_ADDRESS` | Public FQDN/IP used to construct `CALLBACK_URL` when no override is provided. | — |
| `BASE_PATH` | Base path without trailing slash. Example: `/pdm05`. | `/pdm05` |
| `API_ENDPOINT` | Final callback path segment without leading slash. Example: `notify`. | `notify` |
| `MINIO_ENDPOINT` | MinIO host:port. | `storage.tema.digital-enabler.eng.it:443` |
| `MINIO_ACCESS_KEY` | MinIO access key. | — |
| `MINIO_SECRET_KEY` | MinIO secret key. | — |
| `BUCKET_NAME` | MinIO bucket used by the application. | `use` |
| `OBJECT_NAME` | Object to process. | `estimated_ogm_ND.tif` |
| `PROCESSING_UNIT` | Processing backend. Use `cpu`. | `cpu` |
| `OpenTopography_api_key` | API key for OpenTopography. | — |
| `OGM_OBJ_RESOLUTION` | Object map resolution in meters/pixel. | `5` |
| `OGM_ND_RESOLUTION` | Natural-disaster OGM resolution in meters/pixel. | `5` |
| `OGM_UPLOAD_INTERVAL_SEC` | Minimum time between Maps4Fire/Maps4Flood uploads and entity updates. Fusion still continues locally between uploads. | `600` |
| `SCALING_FACTOR` | Scaling factor for processing. | `1` |
| `USE_TAKEOFF_ALTITUDE_FOR_GEOREF` | If `True`, UAV georeferencing uses takeoff/base ground altitude plus DJI `RelativeAltitude`. If `False`, it uses image metadata `GPSAltitude`. | `True` |
| `TAKEOFF_GROUND_ALTITUDE_M` | Fallback takeoff/base ground altitude in meters above sea level. Used only when `USE_TAKEOFF_ALTITUDE_FOR_GEOREF=True` and no DEM-derived value is available. | `150.3` |

> **Important:** `PROCESSING_UNIT` must be `cpu` (not `cpus`).

> **Historical trials:** For pure historical trials where the UAV images were not captured by USE and the real takeoff/base location is unknown or unreliable, set `USE_TAKEOFF_ALTITUDE_FOR_GEOREF=False` so the georeferencing uses `GPSAltitude` from the image metadata.

---

## UAV Takeoff Altitude for Georeferencing

When `USE_TAKEOFF_ALTITUDE_FOR_GEOREF=True`, the application computes the camera altitude as:

```text
camera altitude = takeoff/base ground altitude + DJI RelativeAltitude
```

For new Alert notifications, if the Alert contains the UAV common base:

```json
"uav": {
  "type": "Property",
  "value": {
    "common_base": {
      "type": "Point",
      "coordinates": ["<longitude>", "<latitude>"]
    },
    "multi_uav": {
      "enabled": "<boolean>",
      "number_of_uavs": "<integer-or-null>"
    },
    "sweep_m": "<number>"
  }
}
```

the application samples the downloaded COP30 DEM at `uav.value.common_base.coordinates` and uses that DEM-derived elevation as the takeoff/base ground altitude.

If the `uav.common_base` field is missing, invalid, outside the DEM, or DEM sampling fails, the application falls back to `TAKEOFF_GROUND_ALTITUDE_M`.

---

## Callback URL Configuration

You can either:

1. **Provide `CALLBACK_URL` directly**, or
2. **Let the app construct it** from `PUBLIC_IP_ADDRESS`, `BASE_PATH`, and `API_ENDPOINT`:

```python
CALLBACK_URL = f"https://{PUBLIC_IP_ADDRESS}{BASE_PATH}/{API_ENDPOINT}"
```

Use `BASE_PATH` without a trailing slash and `API_ENDPOINT` without a leading slash.

### 3. Run the Docker Container

After pulling the image, you can run the application using the following command:


```bash
docker run -it --rm \
  --name inf-fusion \
  -p 5505:5505 \
  -e HOST="0.0.0.0" \
  -e PORT="5505" \
  -e DEBUG="True" \
  -e BROKER_URL="https://orion.tema.digital-enabler.eng.it" \
  -e BROKER_ENTITY_Maps4Flood_ID="urn:ngsi-ld:USE:PDM-05:Maps4Flood:" \
  -e BROKER_ENTITY_Maps4Fire_ID="urn:ngsi-ld:USE:PDM-05:Maps4Fire:" \
  -e BROKER_ENTITY_Maps4Object_ID="urn:ngsi-ld:USE:PDM-05:Maps4Object:" \
  -e BROKER_TYPE_ID="GeoTIFF" \
  -e BROKER_SUBSCRIPTION_ID="subscription123" \
  -e MINIO_ENDPOINT="storage.tema.digital-enabler.eng.it:443" \
  -e MINIO_ACCESS_KEY= \
  -e MINIO_SECRET_KEY= \
  -e OBJECT_NAME="estimated_ogm_ND.tif" \
  -e BUCKET_NAME="use" \
  -e PROCESSING_UNIT="cpu" \
  -e PUBLIC_IP_ADDRESS="tema-project.ddns.net" \
  -e BASE_PATH="/pdm05" \
  -e API_ENDPOINT="notify" \
  -e OpenTopography_api_key= \
  -e OGM_OBJ_RESOLUTION="5" \
  -e OGM_ND_RESOLUTION="5" \
  -e OGM_UPLOAD_INTERVAL_SEC="600" \
  -e SCALING_FACTOR="1" \
  -e USE_TAKEOFF_ALTITUDE_FOR_GEOREF="True" \
  -e TAKEOFF_GROUND_ALTITUDE_M="150.3" \
  ghcr.io/he-tema/inf_fusion_v02:latest
```
