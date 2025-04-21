from flask import Flask, request
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import pickle
import numpy as np
import time
import os
import tensorflow as tf
from collections import deque

# Define the EXACT ASLRecognizer class that was used to create the pickle file
class ASLRecognizer:
    def __init__(self, mean, std, label_encoder, feature_names):
        self.mean = mean
        self.std = std
        self.label_encoder = label_encoder
        self.feature_names = feature_names
    
    def predict(self, features, model):
        """
        Make a prediction using normalized features
        
        Args:
            features: numpy array of features
            model: loaded TensorFlow model
        """
        # Normalize the features
        features_scaled = (features - self.mean) / self.std
        
        # Make prediction
        prediction = model.predict(features_scaled.reshape(1, -1), verbose=0)
        predicted_class = np.argmax(prediction, axis=1)[0]
        confidence = float(prediction[0, predicted_class])
        
        # Convert back to label
        predicted_label = self.label_encoder.inverse_transform([predicted_class])[0]
        
        return predicted_label, confidence

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
socketio = SocketIO(app, cors_allowed_origins="*")

# Global variables
model = None
recognizer = None

# Load model and recognizer
MODEL_PATH = 'asl_recognition_model.keras'
RECOGNIZER_PATH = 'asl_recognizer.pkl'

def load_model_and_recognizer():
    global model, recognizer
    
    model_loaded = False
    recognizer_loaded = False
    
    # Load TensorFlow model
    if os.path.exists(MODEL_PATH):
        try:
            model = tf.keras.models.load_model(MODEL_PATH)
            print(f"TensorFlow model loaded successfully from {MODEL_PATH}")
            model_loaded = True
        except Exception as e:
            print(f"Error loading TensorFlow model: {e}")
            model = None
    else:
        print(f"Model file not found at {MODEL_PATH}")
    
    # Load recognizer with more detailed error reporting
    if os.path.exists(RECOGNIZER_PATH):
        try:
            with open(RECOGNIZER_PATH, 'rb') as f:
                recognizer = pickle.load(f)
            print(f"Recognizer loaded successfully from {RECOGNIZER_PATH}")
            recognizer_loaded = True
        except Exception as e:
            print(f"Error loading recognizer: {e}")
            import traceback
            traceback.print_exc()  # Print the full stack trace
            recognizer = None
    else:
        print(f"Recognizer file not found at {RECOGNIZER_PATH}")
    
    return model_loaded and recognizer_loaded

# Load models on startup
is_model_ready = load_model_and_recognizer()

# Track connected clients
clients = 0

# Track request frequency
request_times = deque(maxlen=100)  # Store last 100 request timestamps
request_freq = 0  # Requests per second

def update_request_frequency():
    global request_freq, request_times
    
    # Calculate frequency based on the time window
    now = time.time()
    
    # Remove timestamps older than 1 second
    while request_times and now - request_times[0] > 1.0:
        request_times.popleft()
    
    # Current frequency is the number of requests in the last second
    request_freq = len(request_times)
    
    return request_freq

def extract_features_from_landmarks(landmarks_flat):
    """
    Extract the hand features from landmarks that your model expects
    """
    # Reshape into 21×3 array
    landmarks = np.array(landmarks_flat).reshape(21, 3)
    
    # Get reference points
    wrist = landmarks[0]
    palm_indices = [0, 5, 9, 13, 17]  # Wrist and base of fingers
    palm_center = np.mean(landmarks[palm_indices], axis=0)
    
    # Initialize features dictionary
    features_dict = {}
    
    # Extract all features
    # 1. Joint angles
    fingers = [
        [(1, 2, 3), (2, 3, 4)],             # Thumb
        [(5, 6, 7), (6, 7, 8)],             # Index
        [(9, 10, 11), (10, 11, 12)],        # Middle
        [(13, 14, 15), (14, 15, 16)],       # Ring
        [(17, 18, 19), (18, 19, 20)]        # Pinky
    ]
    finger_names = ["thumb", "index", "middle", "ring", "pinky"]
    
    for i, finger in enumerate(fingers):
        for j, (p1, p2, p3) in enumerate(finger):
            joint_type = "knuckle" if j == 0 else "middle_joint"
            
            v1 = landmarks[p1] - landmarks[p2]
            v2 = landmarks[p3] - landmarks[p2]
            
            v1_norm = np.linalg.norm(v1)
            v2_norm = np.linalg.norm(v2)
            
            if v1_norm > 0 and v2_norm > 0:
                v1 = v1 / v1_norm
                v2 = v2 / v2_norm
                dot_product = np.clip(np.dot(v1, v2), -1.0, 1.0)
                angle = np.arccos(dot_product)
                features_dict[f"{finger_names[i]}_{joint_type}_angle"] = float(angle)
            else:
                features_dict[f"{finger_names[i]}_{joint_type}_angle"] = 0.0
    
    # 2. Fingertip distances from palm center
    fingertips = [4, 8, 12, 16, 20]
    
    for i, tip in enumerate(fingertips):
        dist = np.linalg.norm(landmarks[tip] - palm_center)
        features_dict[f"{finger_names[i]}_tip_to_palm_dist"] = float(dist)
    
    # 3. Fingertip heights relative to wrist
    for i, tip in enumerate(fingertips):
        height = landmarks[tip, 1] - wrist[1]
        features_dict[f"{finger_names[i]}_height"] = float(height)
    
    # 4. Key finger-to-finger distances
    thumb_index_dist = np.linalg.norm(landmarks[4] - landmarks[8])
    features_dict["thumb_to_index_dist"] = float(thumb_index_dist)
    
    thumb_pinky_dist = np.linalg.norm(landmarks[4] - landmarks[20])
    features_dict["thumb_to_pinky_dist"] = float(thumb_pinky_dist)
    
    # 5. Hand shape features
    avg_fingertip_dist = np.mean([np.linalg.norm(landmarks[tip] - palm_center) for tip in fingertips])
    features_dict["hand_curvature"] = float(avg_fingertip_dist)
    
    spread_distances = []
    for i in range(len(fingertips) - 1):
        dist = np.linalg.norm(landmarks[fingertips[i]] - landmarks[fingertips[i+1]])
        spread_distances.append(dist)
    
    features_dict["finger_spread"] = float(np.mean(spread_distances))
    
    # 6. Thumb opposition
    thumb_pinky_opposition = np.linalg.norm(landmarks[4] - landmarks[17])
    features_dict["thumb_pinky_opposition"] = float(thumb_pinky_opposition)
    
    # 7. Fingertip to palm plane
    palm_normal = np.cross(
        landmarks[5] - landmarks[0],
        landmarks[17] - landmarks[0]
    )
    if np.linalg.norm(palm_normal) > 0:
        palm_normal = palm_normal / np.linalg.norm(palm_normal)
        
        for i, tip in enumerate(fingertips):
            vec_to_tip = landmarks[tip] - landmarks[0]
            dist_to_plane = abs(np.dot(vec_to_tip, palm_normal))
            features_dict[f"{finger_names[i]}_dist_to_palm_plane"] = float(dist_to_plane)
    else:
        for i in range(len(fingertips)):
            features_dict[f"{finger_names[i]}_dist_to_palm_plane"] = 0.0
    
    return features_dict

@socketio.on('connect')
def handle_connect():
    global clients
    clients += 1
    print(f"Client connected! Total: {clients}")

@socketio.on('disconnect')
def handle_disconnect():
    global clients
    clients -= 1
    print(f"Client disconnected. Total: {clients}")

@socketio.on('hand_data')
def handle_hand_data(data):
    # Track request time
    request_times.append(time.time())
    freq = update_request_frequency()
    
    if model is None or recognizer is None:
        print(f"Data received (requests/sec: {freq})")
        emit('prediction_result', {
            'error': 'Model or recognizer not loaded. Please check server logs.',
            'request_frequency': freq
        })
        return
        
    try:
        start_time = time.time()
        
        # Get landmarks from client
        landmarks_flat = np.array(data)
        
        # Extract features
        features_dict = extract_features_from_landmarks(landmarks_flat)
        
        # Convert to array in the correct order
        features_array = np.array([features_dict[name] for name in recognizer.feature_names])
        
        # Use the recognizer's predict method
        predicted_label, confidence = recognizer.predict(features_array, model)
        
        # Prepare result
        result = {
            'prediction': predicted_label,
            'confidence': confidence,
            'request_frequency': freq
        }
        
        processing_time = (time.time() - start_time) * 1000  # ms
        result['processing_time'] = round(processing_time, 2)
        
        emit('prediction_result', result)
        print(f"Prediction: {predicted_label}, Confidence: {confidence:.4f}, " 
              f"Processing time: {result['processing_time']} ms, Frequency: {freq} req/sec")
    except Exception as e:
        emit('prediction_error', {
            'error': str(e),
            'request_frequency': freq
        })
        print(f"Prediction error: {str(e)}, Frequency: {freq} req/sec")

# Add a basic route for health check
@app.route('/')
def index():
    if model is None or recognizer is None:
        status = "WARNING: Model or recognizer not loaded!"
    else:
        status = "Model and recognizer loaded successfully."
    
    freq = update_request_frequency()
    return f"ASL recognition server running. Connected clients: {clients}. {status} Current request frequency: {freq} requests/sec"

# Add endpoint to get stats
@app.route('/stats')
def stats():
    freq = update_request_frequency()
    return {
        'clients': clients,
        'model_loaded': model is not None,
        'recognizer_loaded': recognizer is not None,
        'request_frequency': freq
    }

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    socketio.run(app, host='0.0.0.0', port=port, debug=False)