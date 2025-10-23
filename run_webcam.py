import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import sys

# Get model path from command line argument, default to 'best_model.keras'
if len(sys.argv) > 1:
    model_path = sys.argv[1]
else:
    model_path = 'best_model.keras'

# Load the trained model
print(f"Loading model from {model_path}...")
try:
    model = keras.models.load_model(model_path, safe_mode=False)
except:
    model = keras.models.load_model(model_path)
print("Model loaded successfully!")

# Get class names (should match your data folder structure)
class_names = ['null', 'paper', 'rock', 'scissors']  # Alphabetical order

# Initialize webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()

print("Webcam opened. Press 'q' to quit.")

while True:
    # Capture frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame")
        break
    
    # Prepare image for prediction
    # Resize to match model input size
    img = cv2.resize(frame, (224, 224))
    
    # Convert to RGB (OpenCV uses BGR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Expand dimensions to match model input shape (batch_size, height, width, channels)
    img_array = np.expand_dims(img_rgb, axis=0)
    
    # Make prediction
    predictions = model.predict(img_array, verbose=0)
    predicted_class = class_names[np.argmax(predictions)]
    confidence = np.max(predictions) * 100
    
    # Display prediction on frame
    text = f"{predicted_class}: {confidence:.1f}%"
    cv2.putText(frame, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                1.5, (0, 255, 0), 3, cv2.LINE_AA)
    
    # Show all predictions
    y_offset = 100
    for i, class_name in enumerate(class_names):
        prob_text = f"{class_name}: {predictions[0][i]*100:.1f}%"
        cv2.putText(frame, prob_text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, (255, 255, 255), 2, cv2.LINE_AA)
        y_offset += 35
    
    # Display frame
    cv2.imshow('Rock Paper Scissors - Press q to quit', frame)
    
    # Break loop on 'q' press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
print("Webcam closed.")

