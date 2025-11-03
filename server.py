import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dense
from tensorflow.keras.applications import EfficientNetB5
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
import cv2
import io
import base64
from PIL import Image

# --- 1. Global Variables & Model Definition ---
app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing

# Define model constants
IMAGE_SIZE = (456, 456)
CLASSES = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']
MODEL_WEIGHTS_PATH = 'mode1.h5'

def build_model(num_classes):
    """
    Builds and returns the same EfficientNetB5 model structure used for training.
    """
    base_model = EfficientNetB5(include_top=False, weights=None, input_shape=(*IMAGE_SIZE, 3))
    base_model.trainable = False  # Not strictly necessary for inference, but good practice
    
    inputs = Input(shape=(*IMAGE_SIZE, 3))
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs, outputs)
    return model

# --- 2. Load Model ---
# We build the model structure and then load the trained weights into it.
try:
    model = build_model(num_classes=len(CLASSES))
    model.load_weights(MODEL_WEIGHTS_PATH)
    print(f"\n--- Model '{MODEL_WEIGHTS_PATH}' loaded successfully. ---")
except Exception as e:
    print(f"\n---!!! ERROR: Model weights '{MODEL_WEIGHTS_PATH}' not found. ---")
    print("Please make sure you have downloaded the 'model.h5' file from Kaggle,")
    print("renamed it, and placed it in the same directory as 'server.py'.")
    print(f"Details: {e}")
    model = None

# --- 3. Preprocessing Function (Must be identical to training) ---
def preprocess_image(image_pil):
    """
    Preprocesses a PIL image to be ready for the model.
    This includes resizing, noise reduction (like in training), and normalization.
    """
    try:
        # 1. Resize the image
        image_resized = image_pil.resize(IMAGE_SIZE, Image.LANCZOS)
        
        # 2. Convert to numpy array (float32, 0-255)
        image = np.array(image_resized).astype('float32')
        
        # 3. Handle grayscale (convert to 3-channel)
        if image.ndim == 2:
            image = np.stack((image,) * 3, axis=-1)
        
        # 4. Drop alpha channel if present
        if image.shape[2] == 4:
            image = image[:, :, :3]
            
        # --- Replicate the exact preprocessing from training ---
        
        # 5. Cast to uint8 for OpenCV
        image_uint8 = image.astype('uint8')
        
        # 6. Convert RGB -> BGR for OpenCV
        image_bgr = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2BGR)
        
        # 7. Apply Denoising
        denoised_image_bgr = cv2.fastNlMeansDenoisingColored(image_bgr, None, 10, 10, 7, 21)
        
        # 8. Convert BGR -> RGB
        denoised_image_rgb = cv2.cvtColor(denoised_image_bgr, cv2.COLOR_BGR2RGB)
        
        # 9. Apply standard EfficientNet preprocessing
        preprocessed_image = efficientnet_preprocess(denoised_image_rgb)
        
        # 10. Add batch dimension
        return np.expand_dims(preprocessed_image, axis=0)
        
    except Exception as e:
        print(f"Error in preprocessing: {e}")
        return None

# --- 4. Grad-CAM (Heatmap) Generation ---
def get_grad_cam(model, image_array, last_conv_layer_name="top_conv"):
    """
    Generates a Grad-CAM heatmap for the given model and image.
    """
    try:
        # Find the last convolutional layer
        conv_layer = None
        for layer in reversed(model.layers):
            if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.DepthwiseConv2D)):
                conv_layer = model.get_layer(layer.name)
                break
        
        if conv_layer is None:
            # Fallback for EfficientNet (which uses 'top_conv' in its top block)
            conv_layer = model.get_layer(last_conv_layer_name)

        # Create the Grad-CAM model
        grad_model = Model(
            [model.inputs], [conv_layer.output, model.output]
        )

        with tf.GradientTape() as tape:
            conv_layer_output, predictions = grad_model(image_array)
            predicted_class_index = np.argmax(predictions[0])
            predicted_class_output = predictions[:, predicted_class_index]

        # Get gradients
        grads = tape.gradient(predicted_class_output, conv_layer_output)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        # Generate the heatmap
        conv_layer_output = conv_layer_output[0]
        heatmap = conv_layer_output @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap) # Normalize
        heatmap = heatmap.numpy()

        return heatmap, predicted_class_index
        
    except Exception as e:
        print(f"Error in Grad-CAM: {e}")
        # Try finding layer by name if default fails
        if "top_conv" not in str(e):
            try:
                print("Grad-CAM failed, trying to find layer by name...")
                # Find the last conv layer in the base model
                last_conv_name = model.get_layer('efficientnetb5').get_layer('top_conv').name
                full_layer_name = f'efficientnetb5/{last_conv_name}'
                return get_grad_cam(model, image_array, last_conv_layer_name=full_layer_name)
            except Exception as e2:
                print(f"Grad-CAM failed on fallback: {e2}")
                return None, np.argmax(model.predict(image_array)[0])
        return None, np.argmax(model.predict(image_array)[0])


def overlay_heatmap(original_image_pil, heatmap):
    """
    Overlays the Grad-CAM heatmap onto the original image.
    """
    # Resize original image to heatmap size for base64 encoding
    img_resized = original_image_pil.resize((heatmap.shape[1], heatmap.shape[0]), Image.LANCZOS)
    img_array = np.array(img_resized)

    # Convert heatmap to 8-bit, apply colormap
    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_jet = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    
    # Superimpose the heatmap
    superimposed_img = (heatmap_jet * 0.4 + img_array).astype(np.uint8)
    
    # Convert to PIL image and then to base64
    final_img_pil = Image.fromarray(superimposed_img)
    buffered = io.BytesIO()
    final_img_pil.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

# --- 5. API Endpoints ---
@app.route('/')
def index():
    # This can be used to serve the HTML file in the future,
    # but for now, it just confirms the server is running.
    return "Brain Tumor AI Server is running. Open brain_tumor_app.html in your browser."

@app.route('/analyze_2d', methods=['POST'])
def analyze_2d():
    if model is None:
        return jsonify({"error": "Model is not loaded. Check server logs."}), 500
        
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    try:
        # 1. Read and preprocess image
        image_pil = Image.open(file.stream).convert('RGB')
        preprocessed_img = preprocess_image(image_pil)
        
        if preprocessed_img is None:
             return jsonify({"error": "Image preprocessing failed."}), 500

        # 2. Generate Grad-CAM heatmap and prediction
        heatmap, predicted_class_index = get_grad_cam(model, preprocessed_img)
        
        # 3. Get prediction details
        prediction_label = CLASSES[predicted_class_index]
        # Get raw prediction (confidence)
        all_predictions = model.predict(preprocessed_img)[0]
        confidence = float(all_predictions[predicted_class_index])
        
        # 4. Prepare heatmap for sending
        heatmap_base64 = None
        if heatmap is not None:
            # Resize heatmap to match model input size for overlay
            heatmap_resized = cv2.resize(heatmap, IMAGE_SIZE)
            heatmap_base64 = overlay_heatmap(image_pil.resize(IMAGE_SIZE), heatmap_resized)
        
        # 5. Send results
        return jsonify({
            "prediction": prediction_label,
            "confidence": f"{confidence * 100:.2f}%",
            "heatmapBase64": heatmap_base64
        })

    except Exception as e:
        print(f"Error during analysis: {e}")
        return jsonify({"error": f"An error occurred: {e}"}), 500

@app.route('/analyze_3d', methods=['POST'])
def analyze_3d():
    # This is the placeholder for your 3D volumetric analysis
    # (Future work: Use MONAI, NiBabel, and a 3D CNN)
    
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
        
    file = request.files['file']
    
    return jsonify({
        "error": f"3D analysis for '{file.filename}' is not yet implemented. This endpoint is a placeholder for your 3D model (e.g., using MONAI)."
    }), 501  # 501 Not Implemented

# --- 6. Run the Server ---
if __name__ == '__main__':
    print("Starting Flask server... Access the app by opening 'brain_tumor_app.html'.")
    app.run(debug=True, port=5000)

