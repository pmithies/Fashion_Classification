import os
import numpy as np
from flask import Flask, request, render_template, send_from_directory
from tensorflow import keras
import tensorflow as tf
from keras.preprocessing import image  # Import the image module from keras.preprocessing

app = Flask(__name__)
model = keras.models.load_model("fashion_mnist_model.h5")

if not os.path.exists("uploads"):
    os.makedirs("uploads")

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        image = request.files["image"]
        if image:
            image_path = os.path.join("uploads", image.filename)
            image.save(image_path)

            img = image.load_img(image_path, target_size=(28, 28), color_mode="grayscale")  # Use load_img from the keras.preprocessing.image module

            img_array = image.img_to_array(img)
            img_array = tf.image.rgb_to_grayscale(img_array)
            img_array = tf.image.flip_left_right(img_array)
            img_array = tf.image.transpose(img_array)
            img_array = img_array / 255.0
            img_array = np.expand_dims(img_array, 0)

            predictions = model.predict(img_array)
            predicted_class = np.argmax(predictions)

            return render_template("index.html", prediction=class_names[predicted_class], image=image_path)

    return render_template("index.html", prediction=None, image=None)

if __name__ == "__main__":
    app.run(debug=True)
