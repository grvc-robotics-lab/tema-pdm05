# logging_config.py
import logging

# Set up logging configuration
logging.basicConfig(
    filename='app_routes.log',
    filemode='w',
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logger = logging.getLogger(__name__)
