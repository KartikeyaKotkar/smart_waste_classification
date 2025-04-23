# 🧠 Smart Waste Classifier

A machine learning-powered web application that classifies waste images into categories like paper, plastic, metal, etc., using **Balanced Random Forest** to handle imbalanced data. Built with **Flask** and designed for usability with a **dark mode UI** and **image preview** feature.

## 📸 Features

- ✅ Waste image classification (e.g., cardboard, glass, metal, paper, plastic, trash)
- 🎯 Uses **Balanced Random Forest Classifier** for accurate predictions even on imbalanced datasets
- 🖼️ Upload and preview image before classification
- 🌒 Clean, modern **dark mode UI**
- 📁 Uploads are stored in `static/uploads/`

## 🛠️ Tech Stack

- **Frontend**: HTML5, CSS3 (dark mode styling), JavaScript
- **Backend**: Python, Flask
- **Machine Learning**: `imbalanced-learn`, `scikit-learn`, `pandas`, `joblib`
- **Visualization**: `matplotlib`, `seaborn`

## 🧪 Dataset

- Dataset contains pre-extracted features (e.g., HOG or CNN embeddings)
- Classes: `cardboard`, `glass`, `metal`, `paper`, `plastic`, `trash`
- Format: `features_dataset.csv` with columns for features + a `label` column

## 📦 Installation

```bash
git clone https://github.com/your-username/smart-waste-classifier.git
cd smart-waste-classifier
pip install -r requirements.txt
```

## 🚀 Run the App

```bash
python flask_app/app.py
```

Access it at: [http://127.0.0.1:5000](http://127.0.0.1:5000)

## 📤 Image Upload & Preview

- Upload a waste image from your system
- Preview the image in the browser before submission
- The model predicts the waste category and displays it

## 🧠 Model Training

```python
python classifier.py
```

- Uses Balanced Random Forest to overcome class imbalance
- Automatically saves model as `balanced_random_forest_model.pkl`

## 📊 Example Output

```bash
Accuracy: 92.13%

Classification Report:
              precision    recall  f1-score   support

     cardboard       0.93      0.94      0.93        50
         glass       0.90      0.88      0.89        50
         metal       0.91      0.90      0.91        50
         paper       0.95      0.96      0.95        50
       plastic       0.89      0.90      0.89        50
         trash       0.93      0.92      0.92        50
```

## 🎨 UI Screenshot

> Include a screenshot here showing the dark mode UI and image preview

## 📁 Folder Structure

```
smart-waste-classifier/
├── flask_app/
│   ├── app.py
│   ├── templates/
│   ├── static/
│   │   └── uploads/
├── models/
│   └── balanced_random_forest_model.pkl
├── train_model.py
├── features_dataset.csv
└── README.md
```

## 🧠 Future Scope

- Use CNN for automatic feature extraction
- Export predictions and analytics as CSV/JSON
- Deploy to cloud (Heroku, Railway, etc.)

---

