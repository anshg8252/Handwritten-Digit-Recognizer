import tensorflow as tf
import numpy as np
from PIL import Image

# Load trained model
model = tf.keras.models.load_model("digit_model.keras")

# Load handwritten digit image
image = Image.open("digit.png").convert("L")

# Resize to MNIST size
image = image.resize((28, 28))

# Convert image to numpy array
image = np.array(image)

# Normalize pixel values
image = image / 255.0

# Reshape for CNN
image = image.reshape(1, 28, 28, 1)

# Make prediction
prediction = model.predict(image, verbose=0)

# Get predicted digit
digit = np.argmax(prediction)

# Get confidence
confidence = np.max(prediction) * 100

print("Predicted Digit:", digit)
print(f"Confidence: {confidence:.2f}%")