import os

# Orion Context Broker
BROKER_URL = os.getenv('BROKER_URL')
ORION_LD_ENDPOINT = f'{BROKER_URL}/ngsi-ld/v1/entities'

# MinIO Configuration
MINIO_ENDPOINT = os.getenv('MINIO_ENDPOINT')
MINIO_ACCESS_KEY = os.getenv('MINIO_ACCESS_KEY')
MINIO_SECRET_KEY = os.getenv('MINIO_SECRET_KEY')
BUCKET_NAME = os.getenv("BUCKET_NAME")

# Additional Configurations
PUBLIC_IP_ADDRESS = os.getenv("PUBLIC_IP_ADDRESS")  # Public IP or domain
BASE_PATH = os.getenv("BASE_PATH", "/")  # Default to "/"
API_ENDPOINT = os.getenv("API_ENDPOINT", "notify")  # Default to "notify"

# Application settings
HOST = os.getenv('HOST', '0.0.0.0')  # Default to all interfaces
DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'  # Convert to boolean
PORT = os.getenv('PORT', '80')  # Default to port 80 if not set

# TEMP_CALLBACK for testing (e.g., using ngrok)
TEMP_CALLBACK = os.getenv("CALLBACK_NGROK")
# Dynamically construct CALLBACK_URL for reverse proxy scenario
if TEMP_CALLBACK:  # If testing with ngrok
    CALLBACK_URL = TEMP_CALLBACK
else:
    # Ensure BASE_PATH formatting
    if not BASE_PATH.startswith('/'):
        BASE_PATH = '/' + BASE_PATH
    if BASE_PATH.endswith('/'):
        BASE_PATH = BASE_PATH[:-1]

    # Construct CALLBACK_URL without a port for reverse proxy
    CALLBACK_URL = f"https://{PUBLIC_IP_ADDRESS}{BASE_PATH}/{API_ENDPOINT}"

# Log the constructed CALLBACK_URL for debugging
print(f"Constructed CALLBACK_URL: {CALLBACK_URL}")


# Other configurations
OBJECT_NAME = os.getenv('OBJECT_NAME', 'default_object.tif')
BROKER_TYPE_ID = os.getenv("BROKER_TYPE_ID", "GeoTIFF")
ENTITY_Maps4Flood_ID = os.getenv("BROKER_ENTITY_Maps4Flood_ID")
ENTITY_Maps4Fire_ID = os.getenv("BROKER_ENTITY_Maps4Fire_ID")
ENTITY_Maps4Object_ID = os.getenv("BROKER_ENTITY_Maps4Object_ID")
