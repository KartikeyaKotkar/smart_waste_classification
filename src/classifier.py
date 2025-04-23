import os
import cv2
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils import resample
from imblearn.ensemble import BalancedRandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from skimage.feature import hog

# Function to extract HOG features
def extract_hog_features(image):
    features = hog(image, 
                  orientations=9, 
                  pixels_per_cell=(8, 8), 
                  cells_per_block=(2, 2), 
                  visualize=False)
    return features

def load_dataset(dataset_path):
    categories = [folder for folder in os.listdir(dataset_path) 
                 if os.path.isdir(os.path.join(dataset_path, folder))]
    
    data = []
    labels = []
    
    for category in categories:
        category_path = os.path.join(dataset_path, category)
        for img_name in os.listdir(category_path):
            img_path = os.path.join(category_path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, (64, 64))  # Resize to consistent dimensions
                hog_features = extract_hog_features(img)
                data.append(hog_features)
                labels.append(category)
    
    return np.array(data), np.array(labels)

def balance_dataset(data, labels, target_samples=500):
    unique_classes = np.unique(labels)
    balanced_data = []
    balanced_labels = []
    
    for cls in unique_classes:
        cls_indices = np.where(labels == cls)[0]
        cls_data = data[cls_indices]
        cls_labels = labels[cls_indices]
        
        if len(cls_data) < target_samples:
            cls_data, cls_labels = resample(cls_data, cls_labels,
                                          replace=True,
                                          n_samples=target_samples,
                                          random_state=42)
        
        balanced_data.extend(cls_data)
        balanced_labels.extend(cls_labels)
    
    return np.array(balanced_data), np.array(balanced_labels)

def train_model(X_train, y_train):
    model = BalancedRandomForestClassifier(
        n_estimators=100,
        random_state=42,
        bootstrap=False,
        class_weight='balanced'
    )
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    
    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.2f}")
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=model.classes_))
    
    # Confusion matrix
    plt.figure(figsize=(10, 8))
    conf_matrix = confusion_matrix(y_test, y_pred, labels=model.classes_)
    sns.heatmap(conf_matrix, annot=True, fmt='d',
                xticklabels=model.classes_, yticklabels=model.classes_)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    plt.close()

def save_model(model, output_dir, model_name):
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, model_name)
    joblib.dump(model, model_path)
    print(f"[✔] Model saved to {model_path}")

def main():
    # Configuration
    dataset_path = r"C:\Users\karti\PycharmProjects\yearendMLproject\trashnet_dataset"
    output_dir = "../models"
    model_name = "waste_classifier_rf.pkl"
    
    # Load and prepare data
    print("Loading dataset...")
    data, labels = load_dataset(dataset_path)
    
    # Balance the dataset
    print("Balancing dataset...")
    balanced_data, balanced_labels = balance_dataset(data, labels)
    
    # Split into train/test sets
    print("Splitting dataset...")
    X_train, X_test, y_train, y_test = train_test_split(
        balanced_data, balanced_labels, test_size=0.2, random_state=42
    )
    
    # Train model
    print("Training model...")
    model = train_model(X_train, y_train)
    
    # Evaluate model
    print("Evaluating model...")
    evaluate_model(model, X_test, y_test)
    
    # Save model
    save_model(model, output_dir, model_name)

if __name__ == "__main__":
    main()