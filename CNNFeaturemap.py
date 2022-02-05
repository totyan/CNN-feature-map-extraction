import keras as ks
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.utils import np_utils
from sklearn.datasets import fetch_openml
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image


#MNISTデータのインポート
mnist = fetch_openml('mnist_784', version=1,)

#説明変数と目的変数に分ける
X, y = mnist['data'], mnist['target']

#トレーニングデータとテストデータに分ける
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
y_test_backup = y_test



#平らに並んでいる画像データを28×28×1へリサイズ（28行28列行列に変換）
X_train = X_train.reshape(X_train.shape[0], 28, 28,1)
X_test = X_test.reshape(X_test.shape[0], 28, 28,1)


#Kerasで扱えるようにデータタイプをfloat32に変換
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

'''
#画像の出力
def img_show(img):
    img = np.squeeze(img)
    pil_img = Image.fromarray(np.uint8(img))
    cv2.imwrite("Image.jpg",Img)
    pil_img.show()

img_show(X_test[0].reshape(28, 28))
'''

#正規化する（ピクセルの最大値255で各データを割りxの値が全て0-1の範囲になるようにする）
X_train /= 255
X_test /= 255

#yの値を0,1で表されるバイナリクラスに変換
y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)

#モデルの宣言
model = Sequential()

#宣言したモデルにレイヤーを追加
model.add(Conv2D(8, kernel_size=(3, 3),activation='relu',input_shape=(28,28,1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(16, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))

#学習プロセスの設定（訓練データをどのように学習させるか）
model.compile(loss='categorical_crossentropy',optimizer='sgd',metrics=['accuracy'])

#モデルの訓練（学習）
fit = model.fit(X_train, y_train, epochs=10,batch_size=128, validation_data=(X_test, y_test))

#出力する中間層を選択
# 畳み込み層のみを抽出
conv_layers = [l.output for l in model.layers[:5]]
conv_model = Model(inputs=model.inputs, outputs=conv_layers)

# 畳み込み層の出力を取得
conv_outputs = conv_model.predict(X_test)

for i in range(len(conv_outputs)):
    print(f'layer {i}:{conv_outputs[i].shape}')

def plot_conv_outputs(outputs):
    filters = outputs.shape[2]
    fig = plt.figure()
    for i in range(filters):
        plt.subplots_adjust(wspace=0.1)
        plt.subplots_adjust(hspace=0)
        plt.subplot(filters/6 + 1, 6, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.xlabel(f'filter {i}')
        plt.imshow(outputs[:,:,i])
    p = "your_path"
    fig.savefig(p+str(i)+"img.png")
#1層目(Conv2D)
plot_conv_outputs(conv_outputs[0][0])
#2層目(Conv2D)
plot_conv_outputs(conv_outputs[2][0])
#2層目(Conv2D)
plot_conv_outputs(conv_outputs[4][0])
'''
#テストデータを使いモデルの精度の評価
loss_and_metrics = model.evaluate(X_test, y_test, batch_size=128,)
print(loss_and_metrics)

# X_testを使って予測を行う
predictions = model.predict_classes(X_test)
x = list(predictions)
y = list(y_test_backup)
results = pd.DataFrame({'Actual': y, 'Predictions': x})
print(results[1:10])
print(fit.history.keys())


plt.plot(range(1, 11), fit.history['loss'], label="loss for training")
plt.plot(range(1, 11), fit.history['val_loss'], label="loss for validation")
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
'''