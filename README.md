Pest Detection using QCNN (Quantum Convolutional Neural Network)
This project implements pest detection using a hybrid classical-quantum model, focusing on Quantum Convolutional Neural Networks (QCNN) for image classification. The QCNN is used to detect pests from agricultural images and classify them into predefined categories.

Project Overview
The model uses classical preprocessing techniques and a quantum neural network component for improved performance. This approach aims to identify pests in crops and assist in early-stage detection for farmers.

Features
Image Preprocessing: Grayscale conversion, resizing, and normalization.
Classical Model: MLP (Multi-layer Perceptron) for comparison and baseline evaluation.
Quantum Model: Quantum Convolutional Neural Network (QCNN) using PennyLane.
Dataset: Pest images classified into categories for training, validation, and testing.
Metrics: Accuracy, Precision, Recall, and F1 score metrics to evaluate the model's performance.
Dataset
The dataset consists of images of various pests in crops, labeled into different classes. The images are processed, split into training, validation, and test datasets, and then passed through a machine learning pipeline for classification.

Training Split: Used for training the model.
Validation Split: Used for model evaluation during training.
Test Split: Used for final model evaluation.
Installation & Requirements
To run the project locally, clone this repository and install the required dependencies.
