import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2
import os
import requests

from keras.applications.vgg16 import VGG16
from flask import Flask, request

app = Flask(__name__)

UPLOAD_FOLDER = ''
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/model/study', methods=['POST'])
def studypredict():
    file = request.files['file']
    word_id = request.form['word_id']
    file.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))
    # data = [11001, 11002, 11003, 12001, 12002, 12003, 21001, 21003, 21003, 21004, 21005, 21006, 31001, 31002, 31003, 31004, 31005]
    
    imagearr = x = x_features = pred = []
    index = 0
    answer = False
    y = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    base_model = VGG16(weights='imagenet', include_top=False,
                       input_shape=(224, 224, 3))
    model = tf.keras.models.load_model('./model/cnnlstm_datasetver7.0.h5')

    # frame cut
    vidcap = cv2.VideoCapture(file.filename)
    success, image = vidcap.read()

    success = True

    while success:
        success, image = vidcap.read()
        if success:
            if (int(vidcap.get(1)) % 5 == 0):  # n프레임당 한장씩 캡쳐
                image = cv2.resize(image, (224, 224))
                # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                imagearr.append(image)

    # test
    x = np.array(imagearr)

    x_features = base_model.predict(x)
    x_features = x_features.reshape(x_features.shape[0],
                                    x_features.shape[1] * x_features.shape[2], x_features.shape[3])

    pred = model.predict(x_features)
    np.set_printoptions(precision=3, suppress=True)
    print(pred)
    # print(format(pred,'.2f'))

    # pred의 길이만큼 최댓값을 찾아서 y에 넣어준다 (100 곱해줘서 %로 만들어줌)
    for i in range(len(pred)):
        index = np.argmax(pred[i])
        y[index] +=pred[i][index] * 100

    # y를 내림차순으로 정렬하여 상위 3개의 퍼센트를 구한다.
    sort_predict=sorted(y,reverse = True)
    rank_result=sort_predict[:3]
    rank_word=[]

    print(y)
    print(rank_result)

    # y에서 상위 3개 퍼센트의 인덱스를 찾아 어떤 단어인지 파악한다.
    for i in range(len(rank_result)):
        rank_word.append(y.index(rank_result[i]))
    
    for i in range(3):
        rank_result[i]=rank_result[i]/len(pred)
        
    print("-------------------------")
    print(rank_word, rank_result)

    print(word_id)
    print(rank_word[0]+1)
    if (word_id == str(rank_word[0]+1)):
        answer = True
    

    print(answer)
    result_json={
        "result" : answer,
        "rank" : rank_result,
        "rank_word" : rank_word
    }

    return result_json


@app.route('/model/test', methods=['POST'])
def testpredict():
    data = [11001, 11002, 11003, 12001, 12002, 12003, 21001, 21003,
            21003, 21004, 21005, 21006, 31001, 31002, 31003, 31004, 31005]
    file = request.files['file']
    wname = request.form['wname']
    testindex = request.form['testindex']
    # 나중에 DB에 저장하는 것으로 변경 될 가능성 있음
    file.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))

    imagearr = x = x_features = pred = []
    index = 0
    answer = False
    y = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    base_model = VGG16(weights='imagenet', include_top=False,
                       input_shape=(224, 224, 3))
    model = tf.keras.models.load_model('./model/cnnlstm_datasetver7.0.h5')

    # frame cut
    vidcap = cv2.VideoCapture(file.filename)
    success, image = vidcap.read()

    success = True

    while success:
        success, image = vidcap.read()
        if success:
            if (int(vidcap.get(1)) % 5 == 0):  # n프레임당 한장씩 캡쳐
                image = cv2.resize(image, (224, 224))
                imagearr.append(image)

    # test
    x = np.array(imagearr)

    x_features = base_model.predict(x)
    x_features = x_features.reshape(x_features.shape[0],
                                    x_features.shape[1] * x_features.shape[2], x_features.shape[3])

    pred = model.predict(x_features)
    print(pred)
    
    for i in range(len(pred)):        
        index = np.argmax(pred[i])
        y[index] += 1

    index = np.argmax(y)

    if (wname == str(data[index])):
        answer = True

    # json = {
    #     "index": testindex,
    #     "result": answer
    # }

    testindex = 18
    answer =  0

    json = {
        "testListIdx" : testindex,
        "result" : answer
    }

    res = requests.patch("http://127.0.0.1:8080/test", json=json)

    # res = requests.put("http://127.0.0.1:8080", json=json)

    return str(res)  # str은 미정


if __name__ == '__main__':
    app.run(debug=True)
