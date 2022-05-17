# import pickle
# import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

def training_ML():
    tf.random.set_seed(0)

    # 데이터셋 load
    xy = pd.read_csv('final_dataset.csv', delimiter=',', header=None)
    X = xy.iloc[:, 0:-1]
    Y = xy.iloc[:, [-1]]

    # Y는 one-hot encoding
    enc = LabelEncoder()
    y1 = enc.fit_transform(Y)  # 1) 문자열label을 정수로 변환
    Y = pd.get_dummies(y1).values  # 2) 정수 -> one-hot encoding

    # 의논 필요) 메소드 자체에서 랜덤 test data를 설정하는데, 이를 인덱싱으로 정확하게 카테고리별 test data를 설정해야할까?? 
    # 1) train/test (8:2)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 1)

    # 2) train/test/validation (7:2:1)
    # X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state = 1)
    # X_test, X_val, Y_test, Y_val = train_test_split(X_test, Y_test, test_size = 0.33, random_state = 1)  # 이 때의 test_size는 validation 크기

    # define model
    model = Sequential()
    model.add(Dense(64, input_shape=(39,), activation='relu'))
    # model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(6, activation='softmax'))
    model.compile(loss = 'categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
    # model.summary()  # 모델 구성을 간략히 보여줌

    # trainning
    # 1) validation data 적용 X
    model.fit(X_train, Y_train, epochs=10, batch_size=32)  # 옵션 끝에 validation_Data=(x_val, y_val) 추가

    # 2) validation data 적용 O
    # model.fit(X_train, Y_train, epochs=10, batch_size=32, validation_data=(X_val, Y_val))  # 옵션 끝에 validation_Data=(x_val, y_val) 추가

    # 예측 정확도 측정 model.predict: 주어진 input(X)의 예측값을 return
    Y_pred = model.predict(X, verbose=0)  # 각 y마다의 softmax 값인 듯...?(이를 이용해 상위 몇개 결과를 노출하는 식으로?) verbose 값이 뭘 의미하는지 알아낼 것
    predicted = Y_pred.argmax(axis=-1)  # softmax에서 제일 큰 값의 인덱스(0~5)로 변환
    print("Miss: ", (y1 != predicted).sum())  # 예측 틀린 개수

    # 모델 성능 평가 model.evaluate: 주어진 input(X)의 예측값을 실제 값(Y)과 비교하여 정확도 측정
    results = model.evaluate(X_test, Y_test)
    print("test loss, test acc:", results)

    print("==========머신 러닝 훈련 완료==========")

    return model
    # 특정 데이터셋 한 개를 넣어보기
    # testX = X.iloc[[4500],:]
    # testY = y1[4500]

    # Y_pred_2 = model.predict(testX)
    # predicted_2 = Y_pred_2.argmax(axis=-1)
    # print(f"예측: {predicted_2}, 실제: {testY}")

    # with open('C:/Users/user/save_model/saved_model.pkl', 'wb') as fw:
    #     pickle.dump(model, fw)
    # joblib.dump(model, 'saved_modle.pkl')

# <성능 조율 및 분석>
# optimizer별 accuracy (실행 시마다 약간의 오차 있음)
# SGD: 0.8884528875350952
# RMSprop: 0.9215246438980103
# Adam: 0.9232062697410583
# Adamax: 9097533822059631
# Nadam: 0.9181614518165588

# validation 적용 O -> acc: 0.9151785969734192 (Adam 기준)
# validation 적용 X -> acc: 0.9232062697410583 (Adam 기준)

# 히든 레이어 셀 개수 63->32 또는 128로 설정. 딱히 성능이 더 우세한 게 없음

# dropout(0.2) 실행: cell 개수에서 20%를 0으로 설정. 오히려 성능이 0.01 정도 더 낮음