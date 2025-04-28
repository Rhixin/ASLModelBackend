from flask import Flask, request
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import numpy as np
import time
from collections import deque

from recognizer import ASLRecognizer
from input_handler import extract_features_from_landmarks
from model_loader import load_model_and_recognizer

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
socketio = SocketIO(app, cors_allowed_origins="*")

model, recognizer = load_model_and_recognizer()

clients = 0
request_times = deque(maxlen=100)
request_freq = 0

def update_request_frequency():
    global request_freq, request_times
    now = time.time()
    while request_times and now - request_times[0] > 1.0:
        request_times.popleft()
    request_freq = len(request_times)
    return request_freq

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
    request_times.append(time.time())
    freq = update_request_frequency()
    
    if model is None or recognizer is None:
        emit('prediction_result', {'error': 'Model or recognizer not loaded.', 'request_frequency': freq})
        return

    try:
        start_time = time.time()
        landmarks_flat = np.array(data)
        features_dict = extract_features_from_landmarks(landmarks_flat)
        features_array = np.array([features_dict[name] for name in recognizer.feature_names])
        
        predicted_label, confidence = recognizer.predict(features_array, model)
        
        processing_time = (time.time() - start_time) * 1000  # in ms
        
        emit('prediction_result', {
            'prediction': predicted_label,
            'confidence': confidence,
            'processing_time': round(processing_time, 2),
            'request_frequency': freq
        })
    except Exception as e:
        emit('prediction_error', {'error': str(e), 'request_frequency': freq})

@app.route('/')
def index():
    freq = update_request_frequency()
    status = "Model and recognizer loaded successfully.\n" if model and recognizer else "Model or recognizer not loaded.\n"
    return f"ASL recognition server running. \nClients: {clients}\n Status: {status} \nFrequency: {freq} requests/sec."

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
    socketio.run(app, host='0.0.0.0', port=10000, debug=False)
