import tensorflow as tf
import pickle
import os

MODEL_PATH = 'model/asl_recognition_model.keras'
RECOGNIZER_PATH = 'model/asl_recognizer.pkl'

def load_model_and_recognizer():
    model = None
    recognizer = None

    if os.path.exists(MODEL_PATH):
        try:
            model = tf.keras.models.load_model(MODEL_PATH)
            print(f"TensorFlow model loaded successfully from {MODEL_PATH}")
        except Exception as e:
            print(f"Error loading TensorFlow model: {e}")
    
    if os.path.exists(RECOGNIZER_PATH):
        try:
            with open(RECOGNIZER_PATH, 'rb') as f:
                recognizer = pickle.load(f)
            print(f"Recognizer loaded successfully from {RECOGNIZER_PATH}")
        except Exception as e:
            print(f"Error loading recognizer: {e}")

    return model, recognizer
