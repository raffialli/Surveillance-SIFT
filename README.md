# Human Detection in Video Surveillance

## Overview
This repository contains a PyTorch-based project designed for detecting human presence in video surveillance footage. The project leverages a custom neural network model to classify images as 'human' or 'non-human'. It includes scripts for preprocessing video frames, training the model on a labeled dataset, and applying the model to new video data to detect human presence.

## Project Structure
- `model.py`: Script for training the neural network model using PyTorch.
- `mail.py`: Script for processing video files and detecting human presence.
- `model_definition.py`: Contains the neural network architecture.
- `utils/`: Directory containing utility scripts for data preprocessing and other helper functions.


## Model Training
The model is trained on a dataset of surveillance images labeled as 'human' and 'non-human'. It includes the following key steps:
- Preprocessing the images (resizing, normalization).
- Handling class imbalance in the dataset.
- Defining a neural network architecture.
- Training the model and implementing early stopping for efficient training.

## Human Detection in Videos
The video processing script processes each frame of the input video files to detect human presence:
- Extracting frames from video files.
- Preprocessing frames to match the input requirements of the model.
- Classifying each frame using the trained model.
- Flagging videos with detected human presence.

## Requirements
- Python 3.x
- PyTorch
- OpenCV
- NumPy
- PIL

## Usage
To train the model, run:
python model_training.py

To process videos for human detection, run:
python video_processing.py


## Contributing
Contributions to this project are welcome. Please feel free to fork the repository, make changes, and submit pull requests.

## License
[MIT License](LICENSE)
