import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import sys
import serial
import time

# Get model path from command line argument, default to 'best_model.keras'
if len(sys.argv) > 1:
    model_path = sys.argv[1]
else:
    model_path = 'best_model_transfer.keras'

# Serial port configuration
SERIAL_PORT = '/dev/cu.usbserial-10'  # Change this to your Arduino port
BAUD_RATE = 9600

# Initialize serial connection
print("Connecting to Arduino...")
try:
    arduino = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    time.sleep(2)  # Wait for Arduino to reset
    print("Connected to Arduino!")
except Exception as e:
    print(f"Error connecting to Arduino: {e}")
    print("Please check the port and try again.")
    exit()

# Load the trained model
print(f"Loading model from {model_path}...")
try:
    model = keras.models.load_model(model_path, safe_mode=False)
except:
    model = keras.models.load_model(model_path)
print("Model loaded successfully!")

# Get class names (should match your data folder structure)
class_names = ['null', 'paper', 'rock', 'scissors']  # Alphabetical order

# Mapping from predictions to Arduino commands
command_map = {
    'rock': b'r',
    'paper': b'p',
    'scissors': b's',
    'null': b'd'  # default position
}

# Initialize webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam")
    arduino.close()
    exit()

print("Webcam opened. Press 'q' to quit.")
print("Robot hand will mirror your gestures!")

last_command = None
command_cooldown = 0
COOLDOWN_FRAMES = 30  # Only send command every 30 frames (~1 second)

while True:
    # Capture frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame")
        break
    
    # Prepare image for prediction
    img = cv2.resize(frame, (224, 224))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_array = np.expand_dims(img_rgb, axis=0)
    
    # Make prediction
    predictions = model.predict(img_array, verbose=0)
    predicted_class = class_names[np.argmax(predictions)]
    confidence = np.max(predictions) * 100
    
    # Send command to Arduino if prediction is confident and changed
    if confidence > 45 and command_cooldown == 0:  # Only if >70% confident
        command = command_map[predicted_class]
        if command != last_command:
            arduino.write(command)
            last_command = command
            command_cooldown = COOLDOWN_FRAMES
            print(f"Sent command: {predicted_class}")
    
    # Decrement cooldown
    if command_cooldown > 0:
        command_cooldown -= 1
    
    # Display prediction on frame
    text = f"{predicted_class}: {confidence:.1f}%"
    color = (0, 255, 0) if confidence > 45 else (0, 165, 255)  # Green if confident, orange otherwise
    cv2.putText(frame, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                1.5, color, 3, cv2.LINE_AA)
    
    # Show robot status
    robot_text = f"Robot: {predicted_class if confidence > 45 else 'waiting...'}"
    cv2.putText(frame, robot_text, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 
                1.0, (255, 255, 0), 2, cv2.LINE_AA)
    
    # Show all predictions
    y_offset = 150
    for i, class_name in enumerate(class_names):
        prob_text = f"{class_name}: {predictions[0][i]*100:.1f}%"
        cv2.putText(frame, prob_text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, (255, 255, 255), 2, cv2.LINE_AA)
        y_offset += 35
    
    # Display frame
    cv2.imshow('Rock Paper Scissors Robot - Press q to quit', frame)
    
    # Break loop on 'q' press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
arduino.close()
cap.release()
cv2.destroyAllWindows()
print("Webcam and Arduino connection closed.")

