# AmyAI

## Classification of myocardial uptake in bone scintigraphy
Please cite: Halme, Hanna-Leena, et al. "Convolutional neural networks for detection of transthyretin amyloidosis in 2D scintigraphy images." EJNMMI research 12.1 (2022): 27.


# Setup

## Prerequisites

Docker engine https://docs.docker.com/get-started/get-docker/

## Download AmyAI repository
```bash
git clone https://github.com/hannahalme/AmyAI.git

cd AmyAI
```

## Get tfserving Docker image
```bash
sudo docker pull tensorflow/serving
```

## Build the Docker container
```bash
sudo docker-compose build --no-cache \
  --build-arg http_proxy \
  --build-arg https_proxy \
  --build-arg no_proxy \
```

## Move the whole-body bone scintigraphy images (DICOM files) to ./data folder
You can also add other image types or modalities, but the classifier will only process the WB images.

## Set up the container
```bash
sudo docker-compose up 
```
Optional arguments:
--input-dir : the directory where your DICOM data is (default: /data)
--output-dir : the directory where you want to save the output results (default: /out)
--pattern : Glob pattern to select input files (default: *.dcm)
--host : TF Serving host (default: tfserving) 
--port : TF Serving REST port (default: 8501)
--model : Model name (default: ResNet2)



