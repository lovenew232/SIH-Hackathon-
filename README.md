# Prototype for SIH-Hackathon

# Deepfake Detection System

This project presents a deepfake detection system developed during a hackathon. It utilizes a hybrid CNN-LSTM architecture to classify videos as either real or fake after extracting facial regions.
Overview

This system aims to identify deepfake videos by analyzing temporal and spatial features. It processes video data, focuses on facial expressions, and then applies a deep learning model to determine authenticity.
Features

   1. Google Drive Integration: Easily connects to Google Drive for data access.
   2. Automated Face Extraction: Detects and crops faces from video frames for targeted analysis.
   3.  CNN-LSTM Model: Combines ResNeXt (CNN) for spatial feature extraction with LSTM for temporal understanding.
   4.  Training & Evaluation Pipeline: Includes scripts for training the model and evaluating its performance with accuracy and confusion matrices.

# Setup and Installation

This project is designed to run in a Google Colab environment, leveraging its GPU capabilities.
# Prerequisites

    1. A Google Account.
    2. Your datasets (raw videos and Gobal_metadata.csv) organized in specific folders within your Google Drive (as referenced in the code, e.g., /content/drive/My Drive/Celeb_fake_face_only/).

# Colab Runtime

Important: Before running any code, change the Colab runtime type to GPU:
Runtime -> Change runtime type -> Hardware accelerator -> GPU

# Dependencies
The necessary libraries will be installed automatically by the notebooks. Key libraries include:

    1. face_recognition
    2. torch, torchvision
    3. opencv-python (cv2)
    4. numpy, pandas, scikit-learn, seaborn

# Data Preparation

The second attached file (or relevant sections for preprocessing) is used to prepare the dataset:

    1. Download (Optional): If your raw data is zipped on Google Drive, update the url in code2 to download and unzip it.
    2. Face Extraction: Run the create_face_videos function in code2 to process raw videos, extract faces, and save these "face-only" videos to your Google Drive. These processed videos are crucial for the main training script.

# Usage

1. Prepare Your Data:

   -> Open code2 in Colab.
   -> Mount Google Drive.
   -> Adjust video paths and run the face extraction process. This will save processed videos to your specified Google Drive folders.

2. Train and Evaluate the Model:

   -> Open file1 in Colab.
   -> Mount Google Drive.
   -> The script will automatically load the prepared face-only videos and the Gobal_metadata.csv from your Drive.
   -> Run all cells to train the model, evaluate its performance, and visualize results (loss/accuracy plots, confusion matrix).

# Model Architecture

The model uses a ResNeXt50_32x4d CNN for extracting spatial features from individual video frames. These frame-level features are then fed into an LSTM network to learn temporal dependencies across the video sequence, culminating in a final classification layer.
Files

    1. file1 (e.g., train_deepfake_detector.ipynb): Contains the main training and evaluation pipeline, including model definition, data loading, training loop, and performance visualization.
    2. code2 (e.g., preprocess_video_data.ipynb): Handles the data preparation steps, such as downloading raw videos, extracting frames, and detecting/cropping faces.
