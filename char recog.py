
import numpy as np
import tensorflow.compat.v1 as tf
from tkinter import *
from PIL import Image, ImageFilter
import pyperclip
import matplotlib.pyplot as plt

# Disable TensorFlow v2 behavior
tf.disable_v2_behavior()

# Image Preprocessing
def input_emnist(st):
    try:
        im = Image.open(st).convert("L")  # Convert to grayscale
    except Exception as e:
        print(f"Error opening image: {e}")
        return None

    width, height = im.size

    new_image = Image.new("L", (28, 28), (255))

    if width > height:
        nheight = int(round((28.0 / width * height), 0))
        nheight = max(1, nheight)
        img = im.resize((28, nheight), Image.Resampling.LANCZOS).filter(ImageFilter.SHARPEN)
        wtop = int(round(((28 - nheight) / 2), 0))
        new_image.paste(img, (0, wtop))
    else:
        nwidth = int(round((28.0 / height * width), 0))
        nwidth = max(1, nwidth)
        img = im.resize((nwidth, 28), Image.Resampling.LANCZOS).filter(ImageFilter.SHARPEN)
        wleft = int(round(((28 - nwidth) / 2), 0))
        new_image.paste(img, (wleft, 0))

    # Normalize
    tv = list(new_image.getdata())
    tva = [(255 - x) * 1.0 / 255.0 for x in tv]
    tva = [0.0 if x <= 0.45 else x for x in tva]

    n_image = np.array(tva).reshape(1, 784)

    # Show Images
    plt.imshow(im, cmap='gray')
    plt.title("Input Image")
    plt.show()

    plt.imshow(new_image, cmap='gray')
    plt.title("Rescaled Image")
    plt.show()

    plt.imshow(n_image.reshape(28, 28), cmap='gray')
    plt.title("Normalized Image")
    plt.show()

    return n_image


# Model Prediction
def model_predict(n_image):
    x = tf.placeholder(tf.float32, shape=[None, 784])
    hold_prob = tf.placeholder(tf.float32)

    # Replace with your trained model structure (e.g., ConvNet)
    y_pred = tf.layers.dense(x, 59)  # Mock prediction layer for now

    prediction = tf.argmax(y_pred, 1)

    saver = tf.train.Saver()

    with tf.Session() as sess:
        # Adjust path to your model checkpoint
        saver.restore(sess, "model/cnn_model.ckpt")

        result = sess.run(prediction, feed_dict={x: n_image, hold_prob: 1.0})

    # Sample label dictionary (A-Z, 0-9)
    labels_dict = {i: chr(65 + i) for i in range(26)}  # A-Z
    predicted_char = labels_dict.get(result[0], '?')
    return f"The predicted character is: {predicted_char}", predicted_char


# GUI for File Input
def run_gui():
    def getVal():
        nonlocal st
        st = entry.get()
        root.destroy()

    st = ""

    root = Tk()
    root.geometry("400x200")
    root.title("Character Recognition")

    label1 = Label(root, text="Enter the name of the image file:")
    label1.pack()

    entry = Entry(root, bd=5)
    entry.pack()

    submit = Button(root, text="Submit", command=getVal)
    submit.pack()

    root.mainloop()

    return st


# Main Execution
if __name__ == "__main__":
    # Step 1: Get Image File Path from GUI
    filename = r"W:\img.jpg" # Adjust to your file path

    # Step 2: Preprocess the Input Image
    n_image = input_emnist(filename)

    if n_image is None:
        print("Image processing failed.")
    else:
        # Step 3: Make Prediction using the trained model
        result_text, predicted_char = model_predict(n_image)

        # Step 4: Copy the Prediction to Clipboard
        pyperclip.copy(predicted_char)

        # Step 5: Display Prediction Result in a new GUI
        root2 = Tk()
        root2.geometry("400x200")
        root2.title("Prediction Result")

        label2 = Label(root2, text=result_text)
        label2.config(font=("Courier", 20))
        label2.pack()
        root2.mainloop()
