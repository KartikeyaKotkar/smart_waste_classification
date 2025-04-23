# Smart Waste Classifier 🗑️🤖

A Machine Learning project using Scikit-learn to classify images of waste materials into categories such as cardboard, glass, metal, paper, plastic, and trash.

## 📌 Objective

Build a lightweight ML model that can distinguish types of waste from images. This can be later used for smart bin sorting systems.

## ✅ Dataset

- **Source:** [TrashNet Dataset](https://github.com/garythung/trashnet)
- **Categories:** cardboard, glass, metal, paper, plastic, trash
- **Input Format:** Raw images resized to 64x64 and converted to grayscale

## 🛠️ Tech Stack

- **Language:** Python
- **Libraries:** Scikit-learn, OpenCV, NumPy, Matplotlib, Seaborn

## 🔍 Methodology

1. Preprocess images (resize, grayscale, flatten)
2. Train/Test split
3. Train SVM classifier
4. Evaluate using:
   - Confusion matrix
   - Accuracy
   - Precision/Recall/F1-Score

## 📊 Results

- Achieved good classification accuracy with SVM
- Confusion matrix and classification report are saved in `outputs/`

## 💡 Future Work

- Add HOG or color histogram features
- Try ensemble models like Random Forest
- Deploy as a Flask API or integrate with Android App

## 🙋‍♂️ Author

**Kartikeya Yatin Kotkar**  
BCA Final Year Project, 2025  
MIT-WPU

