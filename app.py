import tkinter as tk
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw
import cv2


# Load trained model
model = tf.keras.models.load_model("digit_model.keras")


# -----------------------------
# Window
# -----------------------------

root = tk.Tk()
root.title("Handwritten Digit Recognizer")
root.geometry("520x680")
root.resizable(False, False)


# -----------------------------
# Title
# -----------------------------

title = tk.Label(
    root,
    text="Handwritten Digit Recognizer",
    font=("Arial", 24, "bold")
)

title.pack(pady=(20, 5))


subtitle = tk.Label(
    root,
    text="Draw one or more digits",
    font=("Arial", 12)
)

subtitle.pack(pady=(0, 15))


# -----------------------------
# Canvas
# -----------------------------

canvas_size = 400

canvas = tk.Canvas(
    root,
    width=canvas_size,
    height=canvas_size,
    bg="black",
    highlightthickness=2
)

canvas.pack()


# PIL image
image = Image.new(
    "L",
    (canvas_size, canvas_size),
    0
)

draw = ImageDraw.Draw(image)


# -----------------------------
# Draw
# -----------------------------

def paint(event):

    brush_size = 5

    margin = 40

    x = max(
        margin,
        min(event.x, canvas_size - margin)
    )

    y = max(
        margin,
        min(event.y, canvas_size - margin)
    )

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


# -----------------------------
# Clear
# -----------------------------

def clear_canvas():

    canvas.delete("all")

    draw.rectangle(
        [0, 0, canvas_size, canvas_size],
        fill=0
    )

    result_label.config(
        text="Prediction: -"
    )

    confidence_label.config(
        text="Confidence: -"
    )


# -----------------------------
# Prepare one digit
# -----------------------------

def prepare_digit(digit_image):

    # Find bounding box
    bbox = digit_image.getbbox()

    if bbox is None:
        return None

    # Crop
    digit_image = digit_image.crop(bbox)

    # Resize
    digit_image.thumbnail((20, 20))

    # Create 28x28 image
    processed = Image.new(
        "L",
        (28, 28),
        0
    )

    # Center
    x = (28 - digit_image.width) // 2
    y = (28 - digit_image.height) // 2

    processed.paste(
        digit_image,
        (x, y)
    )

    # Convert to numpy
    arr = np.array(processed)

    # Normalize
    arr = arr / 255.0

    # CNN input shape
    arr = arr.reshape(
        1,
        28,
        28,
        1
    )

    return arr


# -----------------------------
# Predict multiple digits
# -----------------------------

def predict_digit():

    # Convert drawing to numpy
    img = np.array(image)

    # Threshold
    _, binary = cv2.threshold(
        img,
        50,
        255,
        cv2.THRESH_BINARY
    )

    # Find connected components
    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    digit_regions = []

    for contour in contours:

        x, y, w, h = cv2.boundingRect(contour)

        # Ignore very small objects
        if w < 10 or h < 15:
            continue

        # Ignore extremely large regions
        if w > 150 or h > 300:
            continue

        digit_regions.append(
            (x, y, w, h)
        )

    # No digits
    if not digit_regions:

        result_label.config(
            text="Please draw a digit!"
        )

        confidence_label.config(
            text=""
        )

        return

    # Sort digits from left to right
    digit_regions.sort(
        key=lambda region: region[0]
    )

    result = ""
    confidences = []

    # Process every detected digit
    for x, y, w, h in digit_regions:

        # Add padding
        padding = 10

        x1 = max(0, x - padding)
        y1 = max(0, y - padding)

        x2 = min(
            canvas_size,
            x + w + padding
        )

        y2 = min(
            canvas_size,
            y + h + padding
        )

        digit_image = image.crop(
            (x1, y1, x2, y2)
        )

        # Prepare for CNN
        input_image = prepare_digit(
            digit_image
        )

        if input_image is None:
            continue

        # Prediction
        prediction = model.predict(
            input_image,
            verbose=0
        )

        digit = np.argmax(prediction)

        confidence = (
            np.max(prediction) * 100
        )

        result += str(digit)

        confidences.append(
            confidence
        )

    # Average confidence
    if confidences:

        avg_confidence = (
            sum(confidences)
            / len(confidences)
        )

        result_label.config(
            text=f"Prediction: {result}"
        )

        confidence_label.config(
            text=f"Confidence: {avg_confidence:.2f}%"
        )


# -----------------------------
# Buttons
# -----------------------------

button_frame = tk.Frame(root)

button_frame.pack(pady=15)


predict_button = tk.Button(
    button_frame,
    text="Predict",
    command=predict_digit,
    width=15,
    height=2,
    font=("Arial", 12, "bold")
)

predict_button.grid(
    row=0,
    column=0,
    padx=10
)


clear_button = tk.Button(
    button_frame,
    text="Clear",
    command=clear_canvas,
    width=15,
    height=2,
    font=("Arial", 12, "bold")
)

clear_button.grid(
    row=0,
    column=1,
    padx=10
)


# -----------------------------
# Results
# -----------------------------

result_label = tk.Label(
    root,
    text="Prediction: -",
    font=("Arial", 22, "bold")
)

result_label.pack(pady=(5, 5))


confidence_label = tk.Label(
    root,
    text="Confidence: -",
    font=("Arial", 15)
)

confidence_label.pack()


# -----------------------------
# Mouse
# -----------------------------

canvas.bind(
    "<B1-Motion>",
    paint
)


# -----------------------------
# Start
# -----------------------------

root.mainloop()