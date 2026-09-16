import os
from logging_config import logger

# Orion Context Broker
BROKER_URL = os.getenv('BROKER_URL')
if not BROKER_URL:
    logger.error("BROKER_URL environment variable is not set.")
    raise ValueError("BROKER_URL environment variable is not set.")
ORION_LD_ENDPOINT = f'{BROKER_URL}/ngsi-ld/v1/entities'

# MinIO Configuration
MINIO_ENDPOINT = os.getenv('MINIO_ENDPOINT')
if not MINIO_ENDPOINT:
    logger.error("MINIO_ENDPOINT environment variable is not set.")
    raise ValueError("MINIO_ENDPOINT environment variable is not set.")

MINIO_ACCESS_KEY = os.getenv('MINIO_ACCESS_KEY')
MINIO_SECRET_KEY = os.getenv('MINIO_SECRET_KEY')
BUCKET_NAME = os.getenv("BUCKET_NAME", "use")

# Additional Configurations
PUBLIC_IP_ADDRESS = os.getenv("PUBLIC_IP_ADDRESS")  # Public IP or domain
if not PUBLIC_IP_ADDRESS:
    logger.error("PUBLIC_IP_ADDRESS environment variable is not set.")
    raise ValueError("PUBLIC_IP_ADDRESS environment variable is not set.")
BASE_PATH = os.getenv("BASE_PATH", "/").rstrip('/').lstrip('/')
# BASE_PATH = os.getenv("BASE_PATH", "/").strip('/')
API_ENDPOINT = os.getenv("API_ENDPOINT", "notify").lstrip('/').rstrip('/')  # Ensure no leading slash
# API_ENDPOINT = os.getenv("API_ENDPOINT", "notify").strip('/')

# TEMP_CALLBACK for testing (e.g., using ngrok)
TEMP_CALLBACK = os.getenv("CALLBACK_NGROK")
# Dynamically construct CALLBACK_URL for reverse proxy scenario
if TEMP_CALLBACK:
    CALLBACK_URL = f"{TEMP_CALLBACK}/{BASE_PATH.strip('/')}/{API_ENDPOINT.strip('/')}"
else:
    CALLBACK_URL = f"https://{PUBLIC_IP_ADDRESS}/{BASE_PATH.strip('/')}/{API_ENDPOINT.strip('/')}"


logger.info(f"Constructed CALLBACK_URL: {CALLBACK_URL}")

# Application settings
HOST = os.getenv('HOST', '0.0.0.0')  # Default to all interfaces
DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'  # Convert to boolean
PORT = os.getenv('PORT')  # Default to port 80 if not set

# Other configurations
OBJECT_NAME = os.getenv('OBJECT_NAME', 'default_object.tif')
BROKER_TYPE_ID = os.getenv("BROKER_TYPE_ID", "GeoTIFF")
ENTITY_Maps4Flood_ID = os.getenv("BROKER_ENTITY_Maps4Flood_ID")
ENTITY_Maps4Fire_ID = os.getenv("BROKER_ENTITY_Maps4Fire_ID")
ENTITY_Maps4Object_ID = os.getenv("BROKER_ENTITY_Maps4Object_ID")
OpenTopography_api_key = os.getenv("OpenTopography_api_key", '56da0f69ae202d4d9414278b0f6537bd')
OGM_ND_RESOLUTION = os.getenv("OGM_ND_RESOLUTION", 5)
OGM_OBJ_RESOLUTION = os.getenv("OGM_OBJ_RESOLUTION", 5)
OGM_UPLOAD_INTERVAL_SEC = float(os.getenv("OGM_UPLOAD_INTERVAL_SEC", 10 * 60.0))
FLOODSIM_PLAYBACK_MODE = os.getenv("FLOODSIM_PLAYBACK_MODE", "immediate").strip().lower()
SCALING_FACTOR = os.getenv("SCALING_FACTOR", 1)
TRACK_CONFIRM_UPDATES = os.getenv("TRACK_CONFIRM_UPDATES", 1)
USE_TAKEOFF_ALTITUDE_FOR_GEOREF = os.getenv("USE_TAKEOFF_ALTITUDE_FOR_GEOREF", "True").lower() == "true"
TAKEOFF_GROUND_ALTITUDE_M = float(os.getenv("TAKEOFF_GROUND_ALTITUDE_M", 150.3))

if not OpenTopography_api_key:
    logger.error("No OpenTopography_api_key")

logger.debug("All configurations loaded successfully.")
