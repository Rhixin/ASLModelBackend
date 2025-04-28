import numpy as np

class ASLRecognizer:
    def __init__(self, mean, std, label_encoder, feature_names):
        self.mean = mean
        self.std = std
        self.label_encoder = label_encoder
        self.feature_names = feature_names
    
    def predict(self, features, model):
        """
        Make a prediction using normalized features.
        """
        features_scaled = (features - self.mean) / self.std
        prediction = model.predict(features_scaled.reshape(1, -1), verbose=0)
        predicted_class = np.argmax(prediction, axis=1)[0]
        confidence = float(prediction[0, predicted_class])
        predicted_label = self.label_encoder.inverse_transform([predicted_class])[0]
        return predicted_label, confidence
