from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np

# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

app = Flask(__name__)
MODEL_PATH = 'model/model_aug.h5'
model = load_model(MODEL_PATH)
model.make_predict_function()         
print('Model loaded. Check http://127.0.0.1:5000/')


def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(200, 200))

    x = image.img_to_array(img)
    x = np.array([x])

    preds = model.predict(x)
    classes = ['Mata Tertutup', 'Mata Terbuka', 'Tidak Menguap', 'Menguap']  
    predict = classes[np.argmax(preds, axis=1)[0]] 
    return predict


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']

        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        preds = model_predict(file_path, model)

        result = str(preds)               
        return result
    return None


if __name__ == '__main__':
    app.run(debug=True)

