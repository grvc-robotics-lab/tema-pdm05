# Use the Python 3.9 slim base image
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Install wget and other dependencies required to install Miniconda
RUN apt-get update && apt-get install -y wget && rm -rf /var/lib/apt/lists/*

# Download and install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /miniconda.sh && \
    bash /miniconda.sh -b -p /opt/conda && \
    rm /miniconda.sh

# Add Conda to the PATH
ENV PATH=/opt/conda/bin:$PATH

# Copy environment.yml to the container
COPY environment.yml .

# Create the Conda environment
RUN conda env create -f environment.yml

# Set Conda's default shell for subsequent RUN commands to the new environment
SHELL ["conda", "run", "-n", "geo_env", "/bin/bash", "-c"]

# Set PROJ_LIB environment variable
ENV PROJ_LIB=/opt/conda/envs/geo_env/share/proj

# Set environment variables for the application
ENV HOST=127.0.0.1
ENV PORT=5505
ENV DEBUG=True
ENV BROKER_URL=https://orion.tema.digital-enabler.eng.it
ENV BROKER_ENTITY_Maps4Flood_ID=urn:ngsi-ld:USE:PDM-05:Maps4Flood:01
ENV BROKER_ENTITY_Maps4Fire_ID=urn:ngsi-ld:USE:PDM-05:Maps4Fire:01
ENV BROKER_ENTITY_Maps4Object_ID=urn:ngsi-ld:USE:PDM-05:Maps4Object:01
ENV BROKER_TYPE_ID=GeoTIFF
ENV BROKER_SUBSCRIPTION_ID=subscription123
ENV CALLBACK_URL=https://informationfusion.pagekite.me/notify
ENV MINIO_ENDPOINT=storage.tema.digital-enabler.eng.it:443
ENV MINIO_ACCESS_KEY=D4xMAQylbJML0ppbLMtt
ENV MINIO_SECRET_KEY=rTV2pwa2PMApAzgV3tssGf7NKNVobM3MalAaSXpY
ENV OBJECT_NAME=estimated_ogm_ND.tif
ENV BUCKET_NAME=naples
ENV PROCESSING_UNIT=cpu

# Callback Configuration (New: environment variables without default values)
ENV PUBLIC_IP_ADDRESS=informationfusion.pagekite.me
ENV BASE_PATH=/pdm05
ENV API_ENDPOINT=notify


# Expose the necessary port
EXPOSE 5505

# Copy the rest of the application code
COPY . .

# Update PATH to use the Conda environment's executables by default
ENV PATH /opt/conda/envs/geo_env/bin:$PATH

# Run the Python application
CMD ["python", "app.py"]

