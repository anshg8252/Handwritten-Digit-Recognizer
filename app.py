import tkinter as tk
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw

# Load trained model
model = tf.keras.models.load_model("digit_model.keras")

# Window
root = tk.Tk()
root.title("Handwritten Digit Recognizer")
root.geometry("500x600")

# Canvas
canvas_size = 400

canvas = tk.Canvas(
    root,
    width=canvas_size,
    height=canvas_size,
    bg="black"
)

canvas.pack(pady=20)

# PIL image
image = Image.new(
    "L",
    (canvas_size, canvas_size),
    0
)

draw = ImageDraw.Draw(image)


# Draw digit
def paint(event):
    brush_size = 5

    # Keep drawing inside a safe margin
    margin = 40

    x = max(margin, min(event.x, canvas_size - margin))
    y = max(margin, min(event.y, canvas_size - margin))

    canvas.create_oval(
        x - brush_size,
        y - brush_size,
        x + brush_size,
        y + brush_size,
        fill="white",
        outline="white"
    )

    draw.ellipse(
        [
            x - brush_size,
            y - brush_size,
            x + brush_size,
            y + brush_size
        ],
        fill=255
    )
    x = event.x
    y = event.y

    brush_size = 12

    canvas.create_oval(
        x - brush_size,
        y - brush_size,
        x + brush_size,
        y + brush_size,
        fill="white",
        outline="white"
    )

    draw.ellipse(
        [
            x - brush_size,
            y - brush_size,
            x + brush_size,
            y + brush_size
        ],
        fill=255
    )


# Clear canvas
def clear_canvas():
    canvas.delete("all")

    draw.rectangle(
        [0, 0, canvas_size, canvas_size],
        fill=0
    )

    result_label.config(
        text="Prediction: -\nConfidence: -"
    )


# Predict digit
def predict_digit():

    # Convert PIL image to numpy
    img_array = np.array(image)

    # Find the handwritten digit
    bbox = Image.fromarray(img_array).getbbox()

    if bbox is None:
        result_label.config(
            text="Please draw a digit first!"
        )
        return

    # Crop unnecessary black space
    img = image.crop(bbox)

    # Resize while maintaining aspect ratio
    img.thumbnail((20, 20))

    # Create 28x28 black canvas
    canvas_image = Image.new("L", (28, 28), 0)

    # Center the digit
    x = (28 - img.width) // 2
    y = (28 - img.height) // 2

    canvas_image.paste(img, (x, y))

    # Convert to numpy
    img_array = np.array(canvas_image)

    # Normalize
    img_array = img_array / 255.0

    # Reshape for CNN
    img_array = img_array.reshape(
        1, 28, 28, 1
    )

    # Prediction
    prediction = model.predict(
        img_array,
        verbose=0
    )

    digit = np.argmax(prediction)

    confidence = np.max(prediction) * 100

    result_label.config(
        text=f"Prediction: {digit}\n"
             f"Confidence: {confidence:.2f}%"
    )

    # Resize to 28x28
    img = image.resize((28, 28))

    # Convert to numpy
    img_array = np.array(img)

    # Normalize
    img_array = img_array / 255.0

    # Reshape for CNN
    img_array = img_array.reshape(
        1, 28, 28, 1
    )

    # Prediction
    prediction = model.predict(
        img_array,
        verbose=0
    )

    digit = np.argmax(prediction)

    confidence = np.max(prediction) * 100

    result_label.config(
        text=f"Prediction: {digit}\n"
             f"Confidence: {confidence:.2f}%"
    )


# Mouse drawing
canvas.bind(
    "<B1-Motion>",
    paint
)

# Predict button
predict_button = tk.Button(
    root,
    text="Predict",
    command=predict_digit,
    width=15,
    height=2
)

predict_button.pack(pady=5)


# Clear button
clear_button = tk.Button(
    root,
    text="Clear",
    command=clear_canvas,
    width=15,
    height=2
)

clear_button.pack(pady=5)


# Result label
result_label = tk.Label(
    root,
    text="Prediction: -\nConfidence: -",
    font=("Arial", 18)
)

result_label.pack(pady=20)


# Start application
root.mainloop()