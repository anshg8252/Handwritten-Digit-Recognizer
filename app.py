import tkinter as tk
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw
import cv2


# Load trained model
model = tf.keras.models.load_model("digit_model.keras")


# =============================
# Window
# =============================

root = tk.Tk()
root.title("Handwritten Digit Recognizer")
root.geometry("600x750")
root.resizable(False, False)


# =============================
# Colors
# =============================

BG_COLOR = "#f4f6f8"
CARD_COLOR = "#ffffff"
TEXT_COLOR = "#1f2937"


root.configure(bg=BG_COLOR)


# =============================
# Header
# =============================

header = tk.Frame(
    root,
    bg=BG_COLOR
)

header.pack(
    pady=(25, 10)
)


title = tk.Label(
    header,
    text="Handwritten Digit Recognizer",
    font=("Arial", 25, "bold"),
    bg=BG_COLOR,
    fg=TEXT_COLOR
)

title.pack()


subtitle = tk.Label(
    header,
    text="Draw one or more digits and let the CNN recognize them",
    font=("Arial", 11),
    bg=BG_COLOR,
    fg="#6b7280"
)

subtitle.pack(pady=5)


# =============================
# Drawing Area
# =============================

canvas_frame = tk.Frame(
    root,
    bg=CARD_COLOR,
    padx=10,
    pady=10
)

canvas_frame.pack()


canvas_size = 400


canvas = tk.Canvas(
    canvas_frame,
    width=canvas_size,
    height=canvas_size,
    bg="black",
    highlightthickness=0
)

canvas.pack()


# PIL image
image = Image.new(
    "L",
    (canvas_size, canvas_size),
    0
)

draw = ImageDraw.Draw(image)


# =============================
# Drawing Function
# =============================

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


# =============================
# Clear Function
# =============================

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


# =============================
# Prepare Digit
# =============================

def prepare_digit(digit_image):

    bbox = digit_image.getbbox()

    if bbox is None:
        return None

    digit_image = digit_image.crop(bbox)

    digit_image.thumbnail((20, 20))

    processed = Image.new(
        "L",
        (28, 28),
        0
    )

    x = (28 - digit_image.width) // 2
    y = (28 - digit_image.height) // 2

    processed.paste(
        digit_image,
        (x, y)
    )

    arr = np.array(processed)

    arr = arr / 255.0

    arr = arr.reshape(
        1,
        28,
        28,
        1
    )

    return arr


# =============================
# Prediction
# =============================

def predict_digit():

    img = np.array(image)

    _, binary = cv2.threshold(
        img,
        50,
        255,
        cv2.THRESH_BINARY
    )

    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    digit_regions = []

    for contour in contours:

        x, y, w, h = cv2.boundingRect(contour)

        if w < 10 or h < 15:
            continue

        if w > 150 or h > 300:
            continue

        digit_regions.append(
            (x, y, w, h)
        )

    if not digit_regions:

        result_label.config(
            text="Please draw a digit!"
        )

        confidence_label.config(
            text=""
        )

        return

    # Sort from left to right
    digit_regions.sort(
        key=lambda region: region[0]
    )

    result = ""
    confidences = []

    for x, y, w, h in digit_regions:

        padding = 10

        x1 = max(
            0,
            x - padding
        )

        y1 = max(
            0,
            y - padding
        )

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

        input_image = prepare_digit(
            digit_image
        )

        if input_image is None:
            continue

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


# =============================
# Buttons
# =============================

button_frame = tk.Frame(
    root,
    bg=BG_COLOR
)

button_frame.pack(
    pady=20
)


predict_button = tk.Button(
    button_frame,
    text="Predict",
    command=predict_digit,
    width=16,
    height=2,
    font=("Arial", 12, "bold"),
    relief="flat"
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
    width=16,
    height=2,
    font=("Arial", 12, "bold"),
    relief="flat"
)

clear_button.grid(
    row=0,
    column=1,
    padx=10
)


# =============================
# Results
# =============================

result_label = tk.Label(
    root,
    text="Prediction: -",
    font=("Arial", 24, "bold"),
    bg=BG_COLOR,
    fg=TEXT_COLOR
)

result_label.pack(
    pady=(5, 5)
)


confidence_label = tk.Label(
    root,
    text="Confidence: -",
    font=("Arial", 15),
    bg=BG_COLOR,
    fg="#6b7280"
)

confidence_label.pack()


# =============================
# Mouse Event
# =============================

canvas.bind(
    "<B1-Motion>",
    paint
)


# =============================
# Start Application
# =============================

root.mainloop()