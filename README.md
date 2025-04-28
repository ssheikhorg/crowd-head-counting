# Crowd Counting System for Bradford 2025

This project implements a **crowd counting system** using the **YOLOv8** object detection model for the **Bradford 2025 UK City of Culture** initiative. The system is designed to count the number of people in crowd images taken during public events, helping event organizers ensure public safety and efficient crowd management.

## Project Overview

The main goal of this project is to develop a real-time crowd-counting application that can be deployed for events, ensuring safety and assisting with crowd management. The system uses **YOLOv8** to detect and count people in images, offering a user-friendly web interface built with **FastAPI**.

## Features

- **Crowd Counting**: Accurately counts the number of individuals in images of public events.
- **Real-Time Processing**: Provides real-time crowd analysis with YOLOv8 object detection.
- **FastAPI Interface**: A simple web interface for uploading images and viewing results.
- **Model Export**: Trained model is exported to ONNX format for deployment flexibility.
- **Privacy Measures**: Implements privacy-preserving features such as data deletion after processing.

## Dataset

The system uses three main datasets for training the YOLOv8 model:

1. **ShanghaiTech Dataset**: A large-scale dataset containing images with varying crowd densities.
2. **UCF_CC_50 Dataset**: A dataset containing images of extremely dense crowds.
3. **UoB Graduation Ceremony Dataset**: A custom dataset with images of crowd gatherings at University of Bradford graduation ceremonies.

### Dataset Preprocessing

The images were preprocessed and resized to **640x640 pixels** to match the YOLOv8 model’s input size. Annotations were converted to **YOLO format** (normalized bounding boxes) for compatibility with the model.

## Installation

### Requirements

- Python 3.12+
- PyTorch
- FastAPI
- Ultralytics YOLOv8
- OpenCV
- Other dependencies listed in `requirements.txt`

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/ssheikhorg/crowd-head-counting.git
   cd crowd-head-counting
    ```

2. Download models and place them in the `models` directory from this [link](https://drive.google.com/drive/folders/116VfRuoNsNbmiWLyR978QCxVuyROCtcK?usp=drive_link).
3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the FastAPI server:
   ```bash
   python main.py
   ```