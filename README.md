# Emotion Detector using Deep Learning

## Overview

Emotion Detector is a deep learning-based application that recognizes human facial emotions from images. The project uses a Convolutional Neural Network (CNN) trained on the FER-2013 dataset to classify facial expressions into multiple emotion categories. A desktop GUI built with Tkinter provides an easy-to-use interface for loading images and viewing predictions.

This project demonstrates the complete machine learning workflow, including data preprocessing, model training, evaluation, and deployment through a graphical application.

## Features

* Facial emotion recognition using a CNN model
* Image-based emotion prediction
* Simple desktop GUI built with Tkinter
* Data preprocessing and model training pipeline
* Uses the FER-2013 facial expression dataset
* Modular Python code for training and inference

## Tech Stack

* Python
* TensorFlow / Keras
* OpenCV
* NumPy
* Pandas
* Matplotlib
* Tkinter

## Dataset

This project uses the **FER-2013 (Facial Expression Recognition 2013)** dataset, which contains grayscale facial images labeled with different emotions.
The model is trained to recognize the following emotions:
* Angry
* Disgust
* Fear
* Happy
* Sad
* Surprise
* Neutral

## Project Structure
Emotion-Detector/
│── emoji-creator-project-code/
│   ├── gui.py
│   ├── train.py
│   ├── ...
│
├── dataset/
├── models/
├── requirements.txt
├── README.md
└── .gitignore


## Installation

### Create a virtual environment

```bash
python -m venv venv
```

### Activate the virtual environment

**Windows**

bash
venv\Scripts\activate

**Linux/macOS**

bash
source venv/bin/activate

### Install dependencies

bash
pip install -r requirements.txt

## Running the Project

Train the model:

bash
python train.py

Launch the application:

bash
python gui.py

## Machine Learning Workflow

1. Load the FER-2013 dataset.
2. Preprocess and normalize facial images.
3. Train a Convolutional Neural Network.
4. Save the trained model.
5. Load the model for inference.
6. Predict the emotion displayed in the input image.

## Future Improvements

* Real-time webcam emotion detection
* Improved CNN architecture for higher accuracy
* Support for video emotion recognition
* Model optimization for faster inference
* Web-based deployment using Flask or FastAPI

## Learning Outcomes

Through this project, I gained practical experience with:

* Deep Learning using TensorFlow/Keras
* Convolutional Neural Networks (CNNs)
* Image preprocessing using OpenCV
* Model training and evaluation
* Building desktop applications with Tkinter
* End-to-end machine learning project development
