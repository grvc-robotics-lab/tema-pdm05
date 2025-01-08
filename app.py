import config
from routes import app, socketio, subscribe_to_entities, list_subscriptions, \
    log_active_subscriptions
from logging_config import logger  # Import the logger from the logging config module


if __name__ == "__main__":
    logger.info("Starting the Flask application...")  # Log when the application starts

    subscribe_to_entities()
    # log_active_subscriptions()
    # list_subscriptions()

    host = config.HOST
    port = config.PORT
    logger.info(f"Flask app will run on host: {host}, port: {port}")
    print(f"Flask app will run on host: {host}, port: {port}")
    try:
        logger.info("Starting application...")
        socketio.run(app,
                     host=host,
                     port=port,
                     debug=config.DEBUG,
                     use_reloader=False,
                     log_output=True,
                     allow_unsafe_werkzeug=True)
    except Exception as e:
        logger.error(f"Failed to start the application: {e}", exc_info=True)  # Log any exceptions that occur
