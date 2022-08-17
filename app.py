import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2
import os

from keras.applications.vgg16 import VGG16
from flask import Flask, request

app = Flask(__name__)

UPLOAD_FOLDER = ''
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/model/study', methods=['POST'])
def studypredict():
    data =  [11001, 11002, 11003, 12001, 12002, 12003, 21001, 21003, 21003, 21004, 21005, 21006, 31001, 31002, 31003, 31004, 31005]
    file = request.files['file']
    wname = request.form['wname']
    file.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))

    imagearr = x = x_features = pred = []
    index = 0
    answer = False
    y = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224,224,3))
    model = tf.keras.models.load_model('./cnnlstm_datasetver4.1.h5')

    #frame cut
    vidcap = cv2.VideoCapture(file.filename)
    success, image = vidcap.read()

    success = True

    while success:
        success, image = vidcap.read()
        if success:
            if (int(vidcap.get(1)) % 5 == 0): # n프레임당 한장씩 캡쳐
                image = cv2.resize(image, (224,224))
                imagearr.append(image)

    #test
    x = np.array(imagearr)

    x_features = base_model.predict(x)
    x_features = x_features.reshape(x_features.shape[0],
                x_features.shape[1] * x_features.shape[2], x_features.shape[3])

    pred = model.predict(x_features)
    print(pred)
    

    for i in range(len(pred)):
        for _ in range(11):
            index = np.argmax(pred[i])
            y[index]+=1

    index = np.argmax(y)
    
    if (wname == str(data[index])):
        answer = True

    return str(answer)

@app.route('/model/test', methods=['POST'])
def testpredict():
    data =  [11001, 11002, 11003, 12001, 12002, 12003, 21001, 21003, 21003, 21004, 21005, 21006, 31001, 31002, 31003, 31004, 31005]
    file = request.files['file']
    file.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))

    imagearr = x = x_features = answer = []
    # y = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    y = [0, 0, 0, 0, 0, 0]
    # y = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    index = 0
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224,224,3))
    model = tf.keras.models.load_model('./cnnlstm_datasetver4.1.h5')

    #frame cut
    vidcap = cv2.VideoCapture(file.filename)
    success, image = vidcap.read()

    success = True

    while success:
        success, image = vidcap.read()
        if success:
            if (int(vidcap.get(1)) % 5 == 0): # n프레임당 한장씩 캡쳐
                image = cv2.resize(image, (224,224))
                imagearr.append(image)

    #test
    x = np.array(imagearr)

    x_features = base_model.predict(x)
    x_features = x_features.reshape(x_features.shape[0],
                x_features.shape[1] * x_features.shape[2], x_features.shape[3])

    answer = model.predict(x_features)
    print(answer)
    

    for i in range(len(answer)):
        for _ in range(11):
            index = np.argmax(answer[i])
            y[index]+=1

    index = np.argmax(y)

    return str(index)

if __name__ == '__main__':
    app.run(debug=True)
    
