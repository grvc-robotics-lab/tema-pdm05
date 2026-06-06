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

# before conda env create
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# and make sure the prefix line is gone
RUN sed -i '/^prefix:/d' environment.yml
RUN conda env create -f environment.yml

# Set Conda's default shell for subsequent RUN commands to the new environment
SHELL ["conda", "run", "-n", "geo_env", "/bin/bash", "-c"]

# Set PROJ_LIB environment variable
ENV PROJ_LIB=/opt/conda/envs/geo_env/share/proj

# Set environment variables for the application
ENV HOST=
ENV PORT=
ENV DEBUG=True
ENV BROKER_URL=https://orion.tema.digital-enabler.eng.it
ENV BROKER_ENTITY_Maps4Flood_ID=urn:ngsi-ld:USE:PDM-05:Maps4Flood:01
ENV BROKER_ENTITY_Maps4Fire_ID=urn:ngsi-ld:USE:PDM-05:Maps4Fire:01
ENV BROKER_ENTITY_Maps4Object_ID=urn:ngsi-ld:USE:PDM-05:Maps4Object:01
ENV BROKER_TYPE_ID=GeoTIFF
ENV BROKER_SUBSCRIPTION_ID=
ENV CALLBACK_URL=
ENV MINIO_ENDPOINT=storage.tema.digital-enabler.eng.it:443
ENV MINIO_ACCESS_KEY=AUMFK4CGDFORW7PC9URA
ENV MINIO_SECRET_KEY=v9L6zs+G8Qu0UKgfMi8FNIncXtZ+ASMJrAXQwpTB
ENV OBJECT_NAME=estimated_ogm_ND.tif
ENV BUCKET_NAME=use
ENV PROCESSING_UNIT=cpu
ENV PUBLIC_IP_ADDRESS=tema-project.ddns.net
ENV BASE_PATH=/pdm05
ENV API_ENDPOINT=notify
ENV OpenTopography_api_key=56da0f69ae202d4d9414278b0f6537bd
ENV OGM_OBJ_RESOLUTION=5
ENV OGM_ND_RESOLUTION=5
ENV OGM_UPLOAD_INTERVAL_SEC=600
ENV SCALING_FACTOR=1
# Expose the necessary port
EXPOSE 5505

# Copy the rest of the application code
COPY . .

# Update PATH to use the Conda environment's executables by default
ENV PATH /opt/conda/envs/geo_env/bin:$PATH

# Run the Python application
CMD ["python", "app.py"]
