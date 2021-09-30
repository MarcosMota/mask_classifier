from flask import Flask
from flask import request, jsonify

import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

app = Flask(__name__)

model = tf.keras.models.load_model('models/keras_model.h5')

@app.route('/api/predict/', methods=['POST'])
def predict():

    file = request.files['image']
    image = Image.open(file.stream)

    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    #image = Image.open('test/1/1.jpeg')
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)
    
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
    data[0] = normalized_image_array

    predictions = model.predict(data)
    print(predictions)

    predictions = tf.nn.sigmoid(predictions)
    print(predictions)

    predictions = tf.where(predictions < 0.55, 0, 1)
    print(predictions)
    prediction = predictions.numpy()[0][0]

    classes = {
        0:'sem máscara',
        1:'com máscara'
    }

    print(f"classe: {classes[prediction]}")
    return jsonify({
        "classe": classes[prediction]
    })


app.run()