# Handwritten Digit Recognizer

A CNN-based Handwritten Digit Recognizer that uses the MNIST dataset to identify handwritten digits. The application also supports recognition of multiple handwritten digits drawn on the canvas.

## Features

- Handwritten digit recognition
- CNN-based image classification
- MNIST dataset
- Real-time prediction
- Confidence score
- Multi-digit recognition
- Interactive drawing canvas
- Image preprocessing using OpenCV
- Simple desktop GUI using Tkinter

## Technologies Used

- Python
- TensorFlow / Keras
- CNN
- MNIST
- NumPy
- OpenCV
- Pillow
- Tkinter

## Project Workflow

```text
User draws digit(s)
        ↓
Image acquisition
        ↓
Image preprocessing
        ↓
Digit segmentation
        ↓
Resize to 28 × 28
        ↓
CNN model
        ↓
Digit prediction
        ↓
Prediction + confidence