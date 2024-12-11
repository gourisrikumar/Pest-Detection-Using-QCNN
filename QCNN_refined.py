import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pennylane as qml

# Constants for file paths
BASE_PATH = 'D:/VIT- STUDY MATERIAL/QCNN IN PEST DETECTION- PROJECT/pest analysis code/Classification/ip102_v1.1/'
CLASSES_FILE = 'D:\VIT- STUDY MATERIAL\QCNN IN PEST DETECTION- PROJECT\pest analysis code\Classification\classes.txt'
TRAIN_FILE = os.path.join(BASE_PATH, 'train.txt')
TEST_FILE = os.path.join(BASE_PATH, 'test.txt')
VAL_FILE = os.path.join(BASE_PATH, 'val.txt')
IMAGES_FOLDER = os.path.join(BASE_PATH, 'images')

# Load classes from classes.txt
def load_classes(classes_file):
    """Load class labels from a text file."""
    label_map = {}
    with open(classes_file, 'r') as f:
        for idx, line in enumerate(f):
            label_map[idx] = line.strip()
    return label_map

# Load dataset splits from text files
def load_dataset_split(split_file, images_folder):
    """Load dataset split from a text file."""
    image_paths = []
    labels = []
    with open(split_file, 'r') as f:
        for line in f:
            image_name, label = line.strip().split()
            image_path = os.path.join(images_folder, image_name)
            image_paths.append(image_path)
            labels.append(int(label))
    return image_paths, np.array(labels)

# Preprocess images: Convert to grayscale, resize, and normalize
def preprocess_images(image_paths, image_size=(32, 32)):
    """Preprocess images by converting to grayscale and normalizing."""
    images = []
    for img_path in image_paths:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Warning: Unable to read image {img_path}")
            continue
        img = cv2.resize(img, image_size)
        img_normalized = img / 255.0
        images.append(img_normalized.flatten())
    return np.array(images)

# Load label mapping from classes file
label_map = load_classes(CLASSES_FILE)

# Load dataset splits
train_image_paths, y_train = load_dataset_split(TRAIN_FILE, IMAGES_FOLDER)
test_image_paths, y_test = load_dataset_split(TEST_FILE, IMAGES_FOLDER)
val_image_paths, y_val = load_dataset_split(VAL_FILE, IMAGES_FOLDER)

# Preprocess the images
X_train = preprocess_images(train_image_paths)
X_test = preprocess_images(test_image_paths)
X_val = preprocess_images(val_image_paths)

# Define a PennyLane quantum node for the variational circuit (if needed later)
num_qubits = 4
dev = qml.device("default.qubit", wires=num_qubits)

@qml.qnode(dev)
def variational_circuit(params, x):
    """Define a variational circuit using PennyLane."""
    for i in range(num_qubits):
        qml.RY(x[i], wires=i)
    qml.templates.StronglyEntanglingLayers(params, wires=range(num_qubits))
    return [qml.expval(qml.PauliZ(i)) for i in range(num_qubits)]

# Define and train a classical MLP model as a substitute for quantum model
clf = MLPClassifier(hidden_layer_sizes=(64, 64), max_iter=500, random_state=42)
clf.fit(X_train, y_train)

# Function to evaluate the model on a given dataset
def evaluate_model(images, labels, dataset_name="Dataset"):
    """Evaluate the model on the provided dataset."""
    predictions = clf.predict(images)
    
    metrics = {
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision_score(labels, predictions, average='weighted'),
        "recall": recall_score(labels, predictions, average='weighted'),
        "f1": f1_score(labels, predictions, average='weighted')
    }

    # Calculate percentages
    metrics_pct = {k: v * 100 for k, v in metrics.items()}

    print(f"\n{dataset_name} Evaluation Metrics:")
    for key in metrics:
        print(f"{key.capitalize()}: {metrics[key]:.4f} ({metrics_pct[key]:.2f}%)")

    return metrics

# Calculate metrics for each dataset
train_metrics = evaluate_model(X_train, y_train, "Training")
val_metrics = evaluate_model(X_val, y_val, "Validation")
test_metrics = evaluate_model(X_test, y_test, "Test")

# Prepare data for plotting
datasets = ['Training', 'Validation', 'Test']
metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
metrics_values = [train_metrics[k] for k in ['accuracy', 'precision', 'recall', 'f1']]
val_values = [val_metrics[k] for k in ['accuracy', 'precision', 'recall', 'f1']]
test_values = [test_metrics[k] for k in ['accuracy', 'precision', 'recall', 'f1']]

# Function to plot each metric with raw values and percentages
def plot_metrics(metric_values_list):
    """Plot metrics across different datasets."""
    plt.figure(figsize=(12, 6))
    
    bar_width = 0.25
    index = np.arange(len(datasets))

    # Create bars for each metric
    for i in range(len(metrics_names)):
        plt.bar(index + i * bar_width,
                [metric_values_list[j][i] * 100 for j in range(len(datasets))],
                bar_width,
                label=metrics_names[i])

    plt.title("Metrics across Datasets")
    plt.xlabel("Datasets")
    plt.ylabel("Percentage (%)")
    plt.xticks(index + bar_width / 2 * (len(metrics_names) - 1), datasets)
    
    plt.legend()
    plt.tight_layout()
    plt.show()

# Display graphs for each metric
plot_metrics([metrics_values, val_values, test_values])

# Process uploaded image for model input
def process_image(image_path):
    """Process an uploaded image for model input."""
    img = cv2.imread(image_path)
    
    if img is None:
        raise ValueError(f"Image not found or cannot be read: {image_path}")
    
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale if needed
    img_resized = cv2.resize(img_gray, (32, 32))
    
    return img_resized.flatten() / 255.0

# Classify uploaded image using the trained model
def classify_pest(image_path):
    """Classify the pest based on the uploaded image."""
    img_data = process_image(image_path).reshape(1, -1)
    
    prediction = clf.predict(img_data)
    
    pest_name = label_map[int(prediction[0])]
    
    print(f"Detected pest: {pest_name}")
    
    return pest_name

# Display result with image and pest info
def display_result(image_path):
    """Display the classification result along with the image."""
    pest_name = classify_pest(image_path)
    
    img = cv2.imread(image_path)
    
    if img is None:
        raise ValueError(f"Image not found or cannot be read: {image_path}")
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    plt.imshow(img_rgb)
    plt.title(f"Detected Pest: {pest_name}")
    plt.axis('off')
    
    plt.show()

# Example usage of displaying result (make sure the path is correct)
try:
   display_result(os.path.join(IMAGES_FOLDER,'00005.jpg'))
except Exception as e:
   print(e)