import os
from flask import Flask, request, render_template
import joblib
import numpy as np
import cv2
from werkzeug.utils import secure_filename
from skimage.feature import hog
import logging

# Initialize Flask app
app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'models')
MODEL_NAME = 'waste_classifier_rf.pkl'
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_NAME)
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Create necessary directories
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Model loading with enhanced error handling
model = None
try:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

    # Verify model file is not empty
    if os.path.getsize(MODEL_PATH) == 0:
        raise ValueError("Model file is empty")

    model = joblib.load(MODEL_PATH)
    logger.info(f"Model loaded successfully from {MODEL_PATH}")
    logger.info(f"Model classes: {model.classes_}")

    # Verify model can make predictions
    test_features = np.zeros((1, 1764))  # Dummy features matching HOG output
    try:
        test_pred = model.predict(test_features)
        logger.info(f"Model test prediction: {test_pred}")
    except Exception as e:
        raise RuntimeError(f"Model prediction test failed: {str(e)}")

except Exception as e:
    logger.error(f"Failed to load model: {str(e)}")
    model = None


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def preprocess_image(image_path):
    """Process image to extract HOG features matching the training pipeline"""
    try:
        # Read image as grayscale using OpenCV
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError("Could not read image file")

        # Resize to 64x64
        img = cv2.resize(img, (64, 64))

        # Extract HOG features
        features = hog(img,
                       orientations=9,
                       pixels_per_cell=(8, 8),
                       cells_per_block=(2, 2),
                       visualize=False)
        return features
    except Exception as e:
        logger.error(f"Image processing error: {e}")
        raise


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if model is None:
            return render_template('index.html',
                                   error="System error: Model not available. Please try again later.",
                                   model_loaded=False)

        if 'file' not in request.files:
            return render_template('index.html',
                                   error="No file selected",
                                   model_loaded=True)

        file = request.files['file']

        if file.filename == '':
            return render_template('index.html',
                                   error="No file selected",
                                   model_loaded=True)

        if file and allowed_file(file.filename):
            try:
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                logger.info(f"Image saved to {filepath}")

                features = preprocess_image(filepath)

                # Verify features match model expectations
                if len(features) != 1764:  # Expected HOG features size
                    raise ValueError("Extracted features don't match model requirements")

                pred = model.predict([features])[0]
                confidence = round(model.predict_proba([features]).max() * 100, 2)

                return render_template('index.html',
                                       pred=pred,
                                       confidence=confidence,
                                       image_file=filename,
                                       model_loaded=True)

            except Exception as e:
                logger.error(f"Prediction error: {e}")
                return render_template('index.html',
                                       error=f"Processing error: {str(e)}",
                                       model_loaded=True)

    return render_template('index.html', model_loaded=(model is not None))


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')