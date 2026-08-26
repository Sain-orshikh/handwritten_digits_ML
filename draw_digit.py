"""Draw a digit and test it against the trained model.

Run this after training in main.ipynb and saving W1.txt/b1.txt/W2.txt/b2.txt.
"""

import numpy as np
import tkinter as tk
from PIL import Image, ImageDraw

CANVAS_SIZE = 280
BRUSH_RADIUS = 10


def relu(x):
    return np.maximum(0, x)


def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=0, keepdims=True))
    return exp_x / np.sum(exp_x, axis=0, keepdims=True)


def forward_prop(W1, b1, W2, b2, x):
    Z1 = W1.dot(x) + b1
    A1 = relu(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return A2


def load_weights():
    W1 = np.loadtxt('W1.txt')
    b1 = np.loadtxt('b1.txt').reshape(-1, 1)
    W2 = np.loadtxt('W2.txt')
    b2 = np.loadtxt('b2.txt').reshape(-1, 1)
    return W1, b1, W2, b2


class DigitApp:
    def __init__(self, root, weights):
        self.W1, self.b1, self.W2, self.b2 = weights

        self.canvas = tk.Canvas(root, width=CANVAS_SIZE, height=CANVAS_SIZE, bg='black')
        self.canvas.grid(row=0, column=0, columnspan=2, padx=10, pady=10)
        self.canvas.bind('<B1-Motion>', self.paint)

        self.image = Image.new('L', (CANVAS_SIZE, CANVAS_SIZE), color=0)
        self.draw = ImageDraw.Draw(self.image)

        self.result_label = tk.Label(root, text="Draw a digit, then click Predict", font=('Arial', 14))
        self.result_label.grid(row=1, column=0, columnspan=2, pady=5)

        tk.Button(root, text="Predict", command=self.predict, font=('Arial', 12)).grid(row=2, column=0, sticky='ew', padx=10, pady=10)
        tk.Button(root, text="Clear", command=self.clear, font=('Arial', 12)).grid(row=2, column=1, sticky='ew', padx=10, pady=10)

    def paint(self, event):
        x, y = event.x, event.y
        r = BRUSH_RADIUS
        self.canvas.create_oval(x - r, y - r, x + r, y + r, fill='white', outline='white')
        self.draw.ellipse([x - r, y - r, x + r, y + r], fill=255)

    def clear(self):
        self.canvas.delete('all')
        self.draw.rectangle([0, 0, CANVAS_SIZE, CANVAS_SIZE], fill=0)
        self.result_label.config(text="Draw a digit, then click Predict")

    def predict(self):
        small = self.image.resize((28, 28), Image.LANCZOS)
        pixels = np.array(small, dtype=np.float64) / 255.0
        x = pixels.reshape(784, 1)

        A2 = forward_prop(self.W1, self.b1, self.W2, self.b2, x)
        digit = int(np.argmax(A2))
        confidence = float(A2[digit, 0]) * 100
        self.result_label.config(text=f"Prediction: {digit}  ({confidence:.1f}% confidence)")


def main():
    weights = load_weights()
    root = tk.Tk()
    root.title("Digit Recognizer")
    DigitApp(root, weights)
    root.mainloop()


if __name__ == '__main__':
    main()
