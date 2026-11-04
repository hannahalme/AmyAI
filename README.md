# AmyAI

## Classification of myocardial uptake in bone scintigraphy
AmyAI Docker container reads whole-body bone scintigraphy images (DICOM), preprocesses the images and detects myocardial uptake using a convolutional neural network. The results are saved in a csv file.

Please cite the following papers:

Halme, Hanna-Leena, et al. "Convolutional neural networks for detection of transthyretin amyloidosis in 2D scintigraphy images." EJNMMI research 12.1 (2022): 27.

Halme, Hanna-Leena, et al. "Early Diagnosis of ATTR Cardiomyopathy by AI-assisted Real-time Screening of Incidental Cardiac Uptake on Bone Scintigraphy". EANM Innovation (2026): 100287.


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
  --build-arg no_proxy
```

## Move the whole-body bone scintigraphy images (DICOM files) to ./data folder
You can also add other image types or modalities, but the classifier will only process the WB images.

## Set up the container
```bash
sudo docker-compose run
```
If your DICOM files are in some other directory, use:
```bash
sudo docker-compose run --rm   -v /path/to/data/dir:/data:ro   client
```

## Check output files
The results appear in directory "predictions". There should be two files:
- test_data_classification_results.csv
- test_data_classification_results.json

The files include the following fields:
AccessionNumber
SOPInstanceUID
StudyDate
StudyTime
StationName
PatientSex
PatientAge
prediction
probability

Prediction is the predicted class (0 for no myocardial uptake or 1 for myocardial uptake) and probability the associated probability for prediction.


## Using alternative CNN models
You can save your Tensorflow model to ./models and change the last line of docker-compose.yml:
```bash
command: ["--input-dir", "/data", "--output-dir", "/out", "--model", "yourmodel"]
```
