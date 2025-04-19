from flask import Flask, request
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import pickle
import numpy as np
import time
import os

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
socketio = SocketIO(app, cors_allowed_origins="*")

# Load model
MODEL_PATH = 'gesture_clf.pkl'
model = None

if os.path.exists(MODEL_PATH):
    try:
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        print(f"Model loaded successfully from {MODEL_PATH}")
    except Exception as e:
        print(f"Error loading model: {e}")
else:
    print(f"Model file not found at {MODEL_PATH}")

# Track connected clients
clients = 0

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
    if model is None:
        emit('prediction_error', {'error': 'Model not loaded. Please check server logs.'})
        return
        
    try:
        start_time = time.time()
        features = np.array(data['features']).reshape(1, -1)
        prediction = model.predict(features)[0]
        
        result = {'prediction': prediction.tolist()}
        
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(features)[0].tolist()
            result['probabilities'] = probabilities
        
        processing_time = (time.time() - start_time) * 1000  # ms
        result['processing_time'] = round(processing_time, 2)
        
        emit('prediction_result', result)
        print(f"Prediction: {prediction.tolist()}, Processing time: {result['processing_time']} ms")
    except Exception as e:
        emit('prediction_error', {'error': str(e)})
        print(f"Prediction error: {str(e)}")

# Add a basic route for health check
@app.route('/')
def index():
    if model is None:
        status = "WARNING: Model not loaded!"
    else:
        status = "Model loaded successfully."
    return f"Hand tracking server running. Connected clients: {clients}. {status}"

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    socketio.run(app, debug=False, host='0.0.0.0', port=port)