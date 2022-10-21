import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2
import os
import requests
import mediapipe as mp
from keras.applications.vgg16 import VGG16
from flask import Flask, request

app = Flask(__name__)

UPLOAD_FOLDER = ''
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

imagearr = []
x = []
pred = []
rank_result=[]
rank_word=[]
index = 0
data = [11001, 11002, 11003, 12001, 12002, 12003, 21001, 21003,
        21003, 21004, 21005, 21006, 31001, 31002, 31003, 31004, 31005]

# mediapipe 이미지 디텍팅 실습
seq_length = 30

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

@app.route('/model/study', methods=['POST'])
def studypredict():
    file = request.files['file']
    word_id = request.form['word_id']
    modelfilename = request.form['modelfilename']
    level = request.form['level']
    file.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))
    answer = False
    seq = []
    y = [0, 0, 0, 0, 0, 0]
    cnt=0
    y_pred=[]

    model = tf.keras.models.load_model('./model/' + modelfilename)

    # frame cut
    cap = cv2.VideoCapture(file.filename)

    while cap.isOpened():
        ret, img = cap.read()
        # img0 = img.copy()
        if(ret==False):
            break

        img = cv2.flip(img, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = hands.process(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        if result.multi_hand_landmarks is not None:
            for res in result.multi_hand_landmarks:
                joint = np.zeros((21, 4))
                for j, lm in enumerate(res.landmark):
                    joint[j] = [lm.x, lm.y, lm.z, lm.visibility]

                # Compute angles between joints
                v1 = joint[[0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0,
                            13, 14, 15, 0, 17, 18, 19], :3]  # Parent joint
                v2 = joint[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
                            14, 15, 16, 17, 18, 19, 20], :3]  # Child joint
                v = v2 - v1  # [20, 3]
                # Normalize v
                v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

                # Get angle using arcos of dot product
                angle = np.arccos(np.einsum('nt,nt->n',
                                            v[[0, 1, 2, 4, 5, 6, 8, 9, 10,
                                                12, 13, 14, 16, 17, 18], :],
                                            v[[1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19], :]))  # [15,]

                angle = np.degrees(angle)  # Convert radian to degree

                d = np.concatenate([joint.flatten(), angle])

                seq.append(d)

                mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

                if len(seq) < seq_length:
                    continue

                input_data = np.expand_dims(
                    np.array(seq[-seq_length:], dtype=np.float32), axis=0)

                #predict한 퍼센트 배열
                y_pred = model.predict(input_data).squeeze()
                # predict한 영상 장 수 count위함
                cnt+=1
                print('y_pred',y_pred) # ex )[9.9989879e-01 1.0122546e-04 2.3264384e-09]

                np.set_printoptions(precision=3, suppress=True)

                for i in range(len(y_pred)):
                    y[i]+=y_pred[i]*100
                print('result',y)


                # i_pred = 최댓값 위치
                i_pred = int(np.argmax(y_pred))
                print('i_pred',i_pred)

                conf = y_pred[i_pred]

                print("conf",conf)

                if conf < 0.9:
                    continue  

    cap.release()

    for i in range(len(y_pred)):
        y[i]=y[i]/cnt

    # 내림차순으로 정렬하여 상위 3개의 퍼센트를 구한다.
    sort_predict=sorted(y,reverse = True)
    rank_result=[]
    rank_result=sort_predict[:3]
    rank_word=[]

    for i in range(len(rank_result)):
        rank_word.append(y.index(rank_result[i]))

    print(level)

    if(level == "2"):
        print(rank_word)
        for i in range(len(rank_word)):
            rank_word[i] += 6
    elif(level == "3"):
        for i in range(len(rank_word)):
            rank_word[i] += 12
        
        
    print("-------------------------")
    print(rank_word, rank_result)
    print(word_id)
    print(rank_word[0]+1)

    if (word_id == str(rank_word[0]+1)):
        answer = True

    if(rank_word[0] == 0):
        answer = False      

    print(answer)
    result_json={
        "result" : answer,
        "rank" : rank_result,
        "rank_word" : rank_word
    }

    return result_json


@app.route('/model/test', methods=['POST'])
def testpredict():
    file = request.files['file']
    wname = request.form['wname']
    testListIndex = request.form['testListIndex']
    modelfilename = request.form['modelfilename']
    file.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))
    answer = 0
    model = tf.keras.models.load_model('./model/' + modelfilename)
    seq = []
    y = [0, 0, 0, 0, 0, 0]
    cnt=0
    y_pred=[]

    # frame cut
    cap = cv2.VideoCapture(file.filename)

    while cap.isOpened():
        ret, img = cap.read()
        # img0 = img.copy()
        if(ret==False):
            break

        img = cv2.flip(img, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = hands.process(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        if result.multi_hand_landmarks is not None:
            for res in result.multi_hand_landmarks:
                joint = np.zeros((21, 4))
                for j, lm in enumerate(res.landmark):
                    joint[j] = [lm.x, lm.y, lm.z, lm.visibility]

                # Compute angles between joints
                v1 = joint[[0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0,
                            13, 14, 15, 0, 17, 18, 19], :3]  # Parent joint
                v2 = joint[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
                            14, 15, 16, 17, 18, 19, 20], :3]  # Child joint
                v = v2 - v1  # [20, 3]
                # Normalize v
                v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

                # Get angle using arcos of dot product
                angle = np.arccos(np.einsum('nt,nt->n',
                                            v[[0, 1, 2, 4, 5, 6, 8, 9, 10,
                                                12, 13, 14, 16, 17, 18], :],
                                            v[[1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19], :]))  # [15,]

                angle = np.degrees(angle)  # Convert radian to degree

                d = np.concatenate([joint.flatten(), angle])

                seq.append(d)

                mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

                if len(seq) < seq_length:
                    continue

                input_data = np.expand_dims(
                    np.array(seq[-seq_length:], dtype=np.float32), axis=0)

                #predict한 퍼센트 배열
                y_pred = model.predict(input_data).squeeze()
                # predict한 영상 장 수 count위함
                cnt+=1
                print('y_pred',y_pred) # ex )[9.9989879e-01 1.0122546e-04 2.3264384e-09]

                np.set_printoptions(precision=3, suppress=True)

                for i in range(len(y_pred)):
                    y[i]+=y_pred[i]*100
                print('result',y)

                # i_pred = 최댓값 위치
                i_pred = int(np.argmax(y_pred))
                print('i_pred',i_pred)

                conf = y_pred[i_pred]

                print("conf",conf)

                if conf < 0.9:
                    continue

    cap.release()
    
    for i in range(len(pred)):        
        index = np.argmax(pred[i])
        y[index] += 1

    index = np.argmax(y)

    if (wname == str(data[index])):
        answer = 1

    if(pred == []):
        answer = 0  

    json = {
        "testListIdx" : testListIndex,
        "result" : answer
    }

    res = requests.patch("http://127.0.0.1:8080/test", json=json)

    print(cap)
    print(modelfilename)
    print(file)
    print(file.filename)

    return str(res)  # str은 미정

if __name__ == '__main__':
    app.run(debug=True)
