# TEMA PDM05 Information Fusion

This repository implements the PDM05 information-fusion service for the TEMA platform. It is an event-driven multimodal fusion framework that receives timestamped, spatially and temporally heterogeneous disaster-related observations through NGSI-LD notifications, downloads the associated products from object storage, georeferences them when needed, and recursively updates geospatial probability maps.

The framework supports Fire and Flood scenarios and publishes three operational map products:

- `Maps4Fire`: probabilistic active-fire occupancy grid maps.
- `Maps4Flood`: probabilistic flood occupancy grid maps.
- `Maps4Object`: georeferenced person/vehicle object-presence maps.

## Purpose

Disaster observations arrive asynchronously and differ in spatial resolution, footprint, timing, uncertainty, and physical meaning. This service converts those heterogeneous inputs into a common geodetic grid and updates the current disaster state using source-specific fusion logic. The result is a continuously maintained probabilistic map that can be uploaded and exposed through NGSI-LD entities.

## Input Sources

The service subscribes to NGSI-LD entities and processes the following data streams:

- `Alert`: initializes a new scenario, AOI, disaster type, BM identifier, ignition point, and optional UAV mission metadata.
- `FireSegmentation`: UAV active-fire segmentation masks.
- `BurntSegmentation`: UAV burnt-area segmentation masks.
- `FloodSegmentation`: UAV flood segmentation masks.
- `PersonVehicleDetection`: UAV person/vehicle detections for object tracking.
- `StandardArrivalTime`: FireSim arrival-time predictions.
- `FloodCalculationResult`: flood simulation/depth outputs.
- `HotspotResult` and `SinglePostResult`: geosocial analytics.
- `EOBurntArea`: satellite burnt-area products.
- `EOFloodExtent`: satellite flood-extent products.

## Fusion Outputs

The service writes GeoTIFF and JSON products under `estimated_OGM/` and uploads timestamped snapshots to MinIO. The corresponding NGSI-LD entities are updated with metadata including filename, bucket, MinIO URL, AOI coordinates, creation time, BM identifier, and scenario-specific fields.

For Flood maps, cells outside the alert AOI are written as `0.0` and marked as GeoTIFF NoData so clients can render the map as cropped/transparent outside the AOI.

## Fusion Logic

All probability maps are initialized with neutral probability `0.5` inside the AOI. Measurements update only the cells where they provide evidence; cells outside the measurement support keep their previous probability. If a cell has never been observed, it remains at `0.5`.

Flood fusion:

- UAV georeferenced flood segmentation uses per-pixel `georef_data` and updates only positive flood-support cells with recursive log-odds.
- Satellite flood masks are reprojected to the OGM grid and update only positive flood-support cells.
- Satellite zero/background cells are not treated as hard no-flood evidence, preventing satellite masks from erasing UAV-supported measurements.
- Flood simulation rasters are converted from depth to probability and assimilated into the current OGM.

Fire fusion:

- UAV active-fire segmentations increase active-fire probability locally.
- UAV burnt-area evidence decreases active-fire probability where burnt support is observed.
- FireSim predictions are fused using arrival-time perimeter logic.
- Geosocial hotspots provide weak, spatially gated support rather than direct high-confidence fire evidence.

Object fusion:

- Person and vehicle detections are georeferenced and tracked using Kalman filtering and data association.
- Tracks are published as Maps4Object JSON products.

## Main Components

- `app.py`: application entry point; starts Flask/SocketIO, manages subscriptions, and runs the service.
- `routes_modified_v9.py`: main notification handling, download, fusion, publication, and entity-update logic.
- `georeferencing_module.py`: georeferences UAV segmentations and detections using image metadata, DEM/takeoff altitude, and camera geometry.
- `Kalman_filter_estimating_Objects.py`: object tracking logic.
- `minio_client.py`: object-storage upload/download wrapper.
- `config.py`: environment-driven runtime configuration.
- `download_dem.py`: DEM download helper.
- `alert.py`: helper script for creating/testing alert notifications.

## Runtime Configuration

The service is configured through environment variables. Required variables include:

- `BROKER_URL`: NGSI-LD context broker base URL.
- `MINIO_ENDPOINT`: object-storage endpoint.
- `MINIO_ACCESS_KEY`: object-storage access key.
- `MINIO_SECRET_KEY`: object-storage secret key.
- `PUBLIC_IP_ADDRESS`: public host used to build callback URLs.
- `BASE_PATH`: service base path, for example `pdm05`.
- `API_ENDPOINT`: notification endpoint, for example `notify`.
- `BROKER_ENTITY_Maps4Flood_ID`: Maps4Flood entity ID prefix.
- `BROKER_ENTITY_Maps4Fire_ID`: Maps4Fire entity ID prefix.
- `BROKER_ENTITY_Maps4Object_ID`: Maps4Object entity ID prefix.

Common optional variables:

- `BUCKET_NAME`: MinIO bucket name. Default: `use`.
- `OGM_ND_RESOLUTION`: natural-disaster OGM grid resolution in meters. Default: `5`.
- `OGM_OBJ_RESOLUTION`: object-map grid resolution in meters. Default: `5`.
- `OGM_UPLOAD_INTERVAL_SEC`: minimum interval between published OGM uploads. Default: `600`.
- `USE_TAKEOFF_ALTITUDE_FOR_GEOREF`: whether UAV georeferencing uses takeoff-ground altitude plus relative altitude. Default: `True`.
- `TAKEOFF_GROUND_ALTITUDE_M`: fallback takeoff-ground altitude in meters.
- `OpenTopography_api_key`: API key used for DEM downloads.

Do not commit real credentials, API keys, or deployment secrets to the repository.

## Running Locally

Create and activate the Conda environment:

```bash
conda env create -f environment.yml
conda activate geo_env
```

Set the required environment variables, then run:

```bash
python app.py
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

### UAV Takeoff Altitude for Georeferencing

When `USE_TAKEOFF_ALTITUDE_FOR_GEOREF=True`, the application computes the camera altitude as:

```text
camera altitude = takeoff/base ground altitude + DJI RelativeAltitude
```

For new Alert notifications, if the Alert contains the UAV common base, the service can sample the DEM at that location and use the sampled ground elevation as the takeoff/base ground altitude. If no valid DEM-derived value is available, it falls back to `TAKEOFF_GROUND_ALTITUDE_M`.

## Running Locally

Create and activate the Conda environment:

```bash
conda env create -f environment.yml
conda activate geo_env
```

Set the required environment variables, then run:

```bash
python app.py
```

The application exposes:

- `/<BASE_PATH>`: basic health endpoint.
- `/<BASE_PATH>/notify`: NGSI-LD notification callback.
- `/<BASE_PATH>/logs`: live log view.

## Docker

Build the image:

```bash
docker build -t inf_fusion:latest .
```

### Callback URL Configuration

You can either provide `CALLBACK_URL` directly, or let the app construct it from `PUBLIC_IP_ADDRESS`, `BASE_PATH`, and `API_ENDPOINT`:

```python
CALLBACK_URL = f"https://{PUBLIC_IP_ADDRESS}{BASE_PATH}/{API_ENDPOINT}"
```

Use `BASE_PATH` without a trailing slash and `API_ENDPOINT` without a leading slash. For example:

```text
PUBLIC_IP_ADDRESS=tema-project.ddns.net
BASE_PATH=/pdm05
API_ENDPOINT=notify
```

This produces:

```text
https://tema-project.ddns.net/pdm05/notify
```

### Run the Docker Container

After pulling the image, run the application with the required environment variables:

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
  -e MINIO_ACCESS_KEY="" \
  -e MINIO_SECRET_KEY="" \
  -e OBJECT_NAME="estimated_ogm_ND.tif" \
  -e BUCKET_NAME="use" \
  -e PROCESSING_UNIT="cpu" \
  -e PUBLIC_IP_ADDRESS="tema-project.ddns.net" \
  -e BASE_PATH="/pdm05" \
  -e API_ENDPOINT="notify" \
  -e OpenTopography_api_key="" \
  -e OGM_OBJ_RESOLUTION="5" \
  -e OGM_ND_RESOLUTION="5" \
  -e OGM_UPLOAD_INTERVAL_SEC="600" \
  -e SCALING_FACTOR="1" \
  -e USE_TAKEOFF_ALTITUDE_FOR_GEOREF="True" \
  -e TAKEOFF_GROUND_ALTITUDE_M="150.3" \
  ghcr.io/he-tema/inf_fusion_v02:latest
```

Keep secret values outside the repository and inject them through the deployment environment.

## Repository Hygiene

Large generated products, runtime folders, and local datasets should be handled intentionally. The `.gitignore` keeps common runtime artifacts out of Git while allowing selected geospatial/data formats to be tracked when they are part of a reproducible case study or test dataset.

Before committing, inspect:

```bash
git status
git diff --stat
```

Avoid committing temporary downloads, local virtual environments, logs, or secrets.
