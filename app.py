from flask import Flask, request
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import pickle
import numpy as np
import time
import os
from collections import deque

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
socketio = SocketIO(app, cors_allowed_origins="*")

# Load model
MODEL_PATH = 'gesture_clf.pkl'
model = None

if os.path.exists(MODEL_PATH):
    # Modify your model loading section
    try:
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        print(f"Model loaded successfully from {MODEL_PATH}")
    except Exception as e:
        print(f"Error loading model: {e}")
        model = None
        print("Running in test mode without model")
else:
    print(f"Model file not found at {MODEL_PATH}")

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
    
    if model is None:
        print(f"Data received (requests/sec: {freq})")
        emit('prediction_error', {
            'error': 'Model not loaded. Please check server logs.',
            'request_frequency': freq
        })
        return
        
    try:
        start_time = time.time()
        features = np.array(data['features']).reshape(1, -1)
        prediction = model.predict(features)[0]
        
        result = {
            'prediction': prediction.tolist(),
            'request_frequency': freq
        }
        
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(features)[0].tolist()
            result['probabilities'] = probabilities
        
        processing_time = (time.time() - start_time) * 1000  # ms
        result['processing_time'] = round(processing_time, 2)
        
        emit('prediction_result', result)
        print(f"Prediction: {prediction.tolist()}, Processing time: {result['processing_time']} ms, Frequency: {freq} req/sec")
    except Exception as e:
        emit('prediction_error', {
            'error': str(e),
            'request_frequency': freq
        })
        print(f"Prediction error: {str(e)}, Frequency: {freq} req/sec")

# Add a basic route for health check
@app.route('/')
def index():
    if model is None:
        status = "WARNING: Model not loaded!"
    else:
        status = "Model loaded successfully."
    
    freq = update_request_frequency()
    return f"Hand tracking server running. Connected clients: {clients}. {status} Current request frequency: {freq} requests/sec"

# Add endpoint to get stats
@app.route('/stats')
def stats():
    freq = update_request_frequency()
    return {
        'clients': clients,
        'model_loaded': model is not None,
        'request_frequency': freq
    }

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    # Remove the allow_unsafe_werkzeug parameter
    socketio.run(app, host='0.0.0.0', port=port, debug=False)