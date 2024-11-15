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
PUBLIC_IP_ADDRESS = os.getenv("PUBLIC_IP_ADDRESS")  # New
BASE_PATH = os.getenv("BASE_PATH")  # New
API_ENDPOINT = os.getenv("API_ENDPOINT")  # New

# Application settings
HOST = os.getenv('HOST')
DEBUG = os.getenv('DEBUG')
PORT = os.getenv('PORT')

# Dynamically construct CALLBACK_URL
CALLBACK_URL = f"https://{PUBLIC_IP_ADDRESS}:{PORT}{BASE_PATH}/{API_ENDPOINT}"

OBJECT_NAME = os.getenv('OBJECT_NAME')
BROKER_TYPE_ID = os.getenv("BROKER_TYPE_ID")
ENTITY_Maps4Flood_ID = os.getenv("BROKER_ENTITY_Maps4Flood_ID")
ENTITY_Maps4Fire_ID = os.getenv("BROKER_ENTITY_Maps4Fire_ID")
ENTITY_Maps4Object_ID = os.getenv("BROKER_ENTITY_Maps4Object_ID")
