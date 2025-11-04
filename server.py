# You can name this file app.py or server.py
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import webbrowser
from threading import Timer
from flask import Flask, render_template, request, jsonify

# --- 1. Setup and Configuration ---
app = Flask(__name__)

# --- CONFIGURATION (Modify these) ---

# TODO: Update this list to match your model's classes, in the correct order.
CLASS_NAMES = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary'] 

# TODO: Update this to the image size your model was trained on (e.g., 224, 224).
MODEL_IMAGE_SIZE = (457, 457) 

# --- THIS LINE IS UPDATED ---
MODEL_FILE_NAME = 'model.keras'

# --- 2. Load Your .keras Model ---
try:
    # Load the new .keras format
    model = tf.keras.models.load_model(MODEL_FILE_NAME)
    print(f"--- Model '{MODEL_FILE_NAME}' loaded successfully! ---")
    
except Exception as e:
    print(f"--- ERROR loading model '{MODEL_FILE_NAME}': {e} ---")
    print("--- Please make sure the model file is in the same directory as this script. ---")
    
# --- 3. Preprocessing and Postprocessing ---

def preprocess_image(image_file):
    """
    Reads an image file, resizes it, and normalizes it
    for the model.
    """
    try:
        # Read the image file from memory
        img = Image.open(io.BytesIO(image_file.read()))

        # Convert to RGB (if model expects 3 channels)
        if img.mode != 'RGB':
            img = img.convert('RGB')
            
        # Resize to the model's expected input size
        img = img.resize(MODEL_IMAGE_SIZE)
        
        # Convert to a numpy array
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        
        # Normalize the image (COMMON: divide by 255.0)
        # *IMPORTANT: Use the *exact same normalization as your training!
        img_array = img_array / 255.0
        
        # Add a batch dimension (model expects shape 1, H, W, C)
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
        
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None

def postprocess_prediction(prediction_array):
    """
    Converts the model's raw output array into a class name
    and confidence score.
    """
    # Use argmax to find the index of the highest probability
    predicted_index = np.argmax(prediction_array)
    
    # Get the corresponding class name
    predicted_class = CLASS_NAMES[predicted_index]
    
    # Get the confidence score
    confidence = float(np.max(prediction_array))
    
    return predicted_class, confidence

# --- 4. Define Server Routes ---

@app.route('/')
def home():
    """Serves the index.html page."""
    # This looks for 'index.html' in a folder named 'templates'
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handles the image upload and prediction."""
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
        
    file = request.files['image']
    
    if file.filename == '':
        return jsonify({'error': 'No image file selected'}), 400

    if file:
        # 1. Preprocess the image
        processed_image = preprocess_image(file)
        if processed_image is None:
            return jsonify({'error': 'Error processing image'}), 500
        
        # 2. Get prediction from model
        try:
            prediction_array = model.predict(processed_image)
            
            # 3. Postprocess the prediction
            class_name, confidence = postprocess_prediction(prediction_array)
            
            # 4. Send the result back as JSON
            return jsonify({
                'prediction': class_name,
                'confidence': f"{confidence * 100:.2f}%"
            })
            
        except Exception as e:
            return jsonify({'error': f'Model prediction error: {e}'}), 500

# --- 5. Auto-Launch and Run Server ---

def open_browser():
    """Waits 1 second and then opens the browser."""
    webbrowser.open_new('http://127.0.0.1:5000/')

if __name__ == '_main_':
    # Start a timer to open the browser 1 second after the script runs
    Timer(1, open_browser).start()
    
    # Run the Flask server
    # 'debug=False' is important for production and for the browser-opening timer
    app.run(port=5000, debug=False)