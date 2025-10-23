import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.applications import MobileNetV2
import plot_training

# Load datasets from the data folder
train_dataset = tf.keras.utils.image_dataset_from_directory(
    "./data", 
    image_size=(224, 224),
    label_mode="categorical",
    batch_size=8,
    shuffle=True,
    seed=1,
    subset="training",
    validation_split=0.3
)

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    "./data", 
    image_size=(224, 224),
    label_mode="categorical",
    batch_size=8,
    shuffle=True,
    seed=1,
    subset="validation",
    validation_split=0.3
)


# Print class names
class_names = train_dataset.class_names
print(f"Classes found: {class_names}")

# Load pre-trained MobileNetV2 model
base = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base.trainable = False  # Freeze the base model

# Build the model with transfer learning
model = keras.Sequential([
    # Data augmentation layers
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomTranslation(0.1, 0.1),
    
    # Preprocessing for MobileNetV2 (scale [0, 255] to [-1, 1])
    layers.Rescaling(scale=1./127.5, offset=-1),
    
    # Pre-trained base model
    base,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.3),
    layers.Dense(len(class_names), activation="softmax")
])

# Set up callback to save best model
callbacks = [
    ModelCheckpoint(monitor='val_loss', filepath='./best_model_transfer.keras', save_best_only=True)
]

# Compile the model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=tf.keras.losses.CategoricalCrossentropy(),
    metrics=["accuracy"]
)

# Print model summary
print("\nModel Summary:")
model.summary()

# Train the model
print("\nStarting training...")
history = model.fit(
    train_dataset, 
    epochs=20, 
    callbacks=callbacks, 
    validation_data=validation_dataset
)

# Evaluate the model
print("\nEvaluating model on validation data...")
val_loss, val_accuracy = model.evaluate(validation_dataset)
print(f"\nValidation accuracy: {val_accuracy * 100:.2f}%")

# Plot training history
plot_training.plot_training_history(history)

