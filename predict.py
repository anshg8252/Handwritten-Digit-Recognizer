import tensorflow as tf
import numpy as np
from PIL import Image

# Load trained model
model = tf.keras.models.load_model("digit_model.keras")

# Load image
image = Image.open("digit.png").convert("L")

# Convert to numpy array
image_array = np.array(image)

# Create binary image
# Dark pixels = digit, light pixels = background
image_array = np.where(image_array < 130, 255, 0).astype(np.uint8)

# Convert back to PIL image
image = Image.fromarray(image_array)

# Find digit boundaries
bbox = image.getbbox()

if bbox:
    image = image.crop(bbox)

# Resize digit while maintaining proportions
image.thumbnail((20, 20))

# Create 28x28 black canvas
canvas = Image.new("L", (28, 28), 0)

# Center digit
x = (28 - image.width) // 2
y = (28 - image.height) // 2

canvas.paste(image, (x, y))

# Convert to numpy
image_array = np.array(canvas)

# Normalize
image_array = image_array / 255.0

# Reshape for CNN
image_array = image_array.reshape(1, 28, 28, 1)

# Predict
prediction = model.predict(image_array, verbose=0)

digit = np.argmax(prediction)
confidence = np.max(prediction) * 100

print("Predicted Digit:", digit)
print(f"Confidence: {confidence:.2f}%")