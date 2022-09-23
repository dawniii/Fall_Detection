#2个adl+1个fall
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from scipy.stats import stats
from sklearn.model_selection import train_test_split
from tcn import TCN, tcn_full_summary
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import glob,os
gpu = tf.config.experimental.list_physical_devices(device_type='GPU')
tf.config.experimental.set_memory_growth(gpu[0], True)
from sklearn.model_selection import KFold
import send_warning as W
import csv
ParentDir="E:/Fall_Detection/UMAFall/data/"
file = glob.glob(os.path.join(ParentDir, "*.csv"))
df=pd.DataFrame()
TIME_STEPS=1
step=1
file_count=0
for f in file:
    file_count +=1
    dk = pd.read_csv(f, header=32, sep=';')
    dk = dk.loc[dk[' Sensor Type'] == 0]  # Accelerometer = 0
    dk = dk.loc[dk[' Sensor ID'] == 4]  # 4; ANKLE; SensorTag
    dk=dk[[' X-Axis', ' Y-Axis', ' Z-Axis']]
    if (file_count%2 == 0):  # 跌倒
        dk.insert(3, 'label_1', [1] * len(dk))
    else:  # 非跌倒
        dk.insert(3, 'label_1', [0] * len(dk))
    df = pd.concat([df, dk])

df.reset_index(drop=True, inplace=True)
scale_columns =[' X-Axis', ' Y-Axis', ' Z-Axis']
scaler = MinMaxScaler(feature_range=(-1, 1))
scaler = scaler.fit(df[scale_columns])
df.loc[:, scale_columns] = scaler.transform(df[scale_columns].to_numpy())
count=0
segments=[]
labels=[]
FEATURES =3
RANDOM_SEED = 42
TIME_STEPS=1
step=1
for i in range(0, len(df) - TIME_STEPS, step):
    x = df[' X-Axis'].values[i:i + TIME_STEPS].reshape(-1, 1)
    y = df[' Y-Axis'].values[i:i + TIME_STEPS].reshape(-1, 1)
    z = df[' Z-Axis'].values[i:i + TIME_STEPS].reshape(-1, 1)
    segment=np.hstack((x, y, z))
    segments.append(segment)
    label = stats.mode(df['label_1'][i:i + TIME_STEPS])[0][0]  # 出现最多的类别
    labels.append(label)
# labels=list(df['label_1'])

reshaped_segments =np.asarray(segments,dtype=np.float32).reshape(-1,TIME_STEPS,FEATURES)
labels=np.asarray(pd.get_dummies(labels),dtype=np.float32)#是利用pandas实现one hot encode的方式。
floder = KFold(n_splits=3, shuffle=False)
LOSS=[]
ACC=[]
TN=[]
FP=[]
FN=[]
TP=[]
Accuracy=[]
Precision=[]
Recall=[]
k_fold_num=0

for train_index, test_index in floder .split(reshaped_segments):
    k_fold_num += 1
    X_train, X_test = reshaped_segments[train_index], reshaped_segments[test_index]
    y_train, y_test = labels[train_index], labels[test_index]
    model_TCN = tf.keras.models.Sequential([
        TCN(input_shape=(X_train.shape[1], X_train.shape[2]), dilations=(1, 2, 4, 8, 16, 32)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(y_train.shape[1], activation='softmax')
    ])
    model_TCN.compile(loss='binary_crossentropy', metrics=['acc'], optimizer='adam')
    if (k_fold_num == 1):
        checkpoint_save_path = "E:/Fall_Detection/UMAFall/checkpoint/TCN_UMAFall1.ckpt"
        if os.path.exists(checkpoint_save_path + '.index'):
            print('-------------load the model-----------------')
            model_TCN.load_weights(checkpoint_save_path)
    if (k_fold_num == 2):
        checkpoint_save_path = "E:/Fall_Detection/UMAFall/checkpoint/TCN_UMAFall2.ckpt"
        if os.path.exists(checkpoint_save_path + '.index'):
            print('-------------load the model-----------------')
            model_TCN.load_weights(checkpoint_save_path)
    if (k_fold_num == 3):
        checkpoint_save_path = "E:/Fall_Detection/UMAFall/checkpoint/TCN_UMAFall3.ckpt"
        if os.path.exists(checkpoint_save_path + '.index'):
            print('-------------load the model-----------------')
            model_TCN.load_weights(checkpoint_save_path)

    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_save_path,
        save_weights_only=True,
        monitor='acc',
        mode='max',
        save_best_only=True)
  #  history_TCN = model_TCN.fit(X_train, y_train, epochs=32, batch_size=64, verbose=1, shuffle=True,
                               # callbacks=[model_checkpoint_callback])
    model_TCN.summary()
    loss1, acc1 = model_TCN.evaluate(X_test, y_test, verbose=1)
    LOSS.append(loss1)
    ACC.append(acc1)
    y_pred_label = np.argmax(model_TCN.predict(X_test), axis=1)
    y_test_label = np.argmax(y_test, axis=1)
    # 混淆矩阵
    # C=confusion_matrix(y_test_label, y_pred_label)
    # sns.heatmap(C, annot=True)
    confusion_mat = confusion_matrix(y_test_label, y_pred_label)
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion_mat, display_labels=['ADL', 'Fall'])
    disp.plot(
        include_values=True,  # 混淆矩阵每个单元格上显示具体数值
        cmap=plt.cm.Blues,
        ax=None,  # 同上
        xticks_rotation="horizontal",  # 同上
        values_format="d"  # 显示的数值格式
    )
    title = "Confusion matrix"
    disp.ax_.set_title(title)
    if (k_fold_num == 1):
        plt.savefig('E:/Fall_Detection/UMAFall/figure/k1-TCN-C.png', bbox_inches='tight')
        plt.close()
        warnings = []
        warnings = W.is_Fall(y_pred_label)
        plt.subplot(2, 1, 1)
        plt.plot(test_index, y_pred_label, label='y_pred_label', c='orange')
        plt.legend()
        plt.subplot(2, 1, 2)
        plt.plot(test_index, y_test_label, label='y_test_label', c='orange')
        plt.scatter(test_index[warnings], np.array(len(warnings) * [1]), label="warning", c='red', s=8)
        # plt.scatter(test_index[reset_index], np.array(len(reset_index) * [1]), label="reset", c='green', s=8)
        plt.legend()
        plt.savefig('E:/Fall_Detection/UMAFall/figure/w1.svg', bbox_inches='tight')
        plt.close()
        with open("test1.csv", "w", newline='') as csvfile:
            writer = csv.writer(csvfile, dialect='excel')
            writer.writerows(zip(y_test_label, y_pred_label,test_index))
    if (k_fold_num == 2):
        plt.savefig('E:/Fall_Detection/UMAFall/figure/k2-TCN-C.png', bbox_inches='tight')
        plt.close()
        warnings = []
        warnings = W.is_Fall(y_pred_label)
        plt.subplot(2, 1, 1)
        plt.plot(test_index, y_pred_label, label='y_pred_label', c='orange')
        plt.legend()
        plt.subplot(2, 1, 2)
        plt.plot(test_index, y_test_label, label='y_test_label', c='orange')
        plt.scatter(test_index[warnings], np.array(len(warnings) * [1]), label="warning", c='red', s=8)
        # plt.scatter(test_index[reset_index], np.array(len(reset_index) * [1]), label="reset", c='green', s=8)
        plt.legend()
        plt.savefig('E:/Fall_Detection/UMAFall/figure/w2.svg', bbox_inches='tight')
        plt.close()
        with open("test2.csv", "w", newline='') as csvfile:
            writer = csv.writer(csvfile, dialect='excel')
            writer.writerows(zip(y_test_label, y_pred_label,test_index))
    if (k_fold_num == 3):
        plt.savefig('E:/Fall_Detection/UMAFall/figure/k3-TCN-C.png', bbox_inches='tight')
        plt.close()
        warnings = []
        warnings = W.is_Fall(y_pred_label)
        plt.subplot(2, 1, 1)
        plt.plot(test_index, y_pred_label, label='y_pred_label', c='orange')
        plt.legend()
        plt.subplot(2, 1, 2)
        plt.plot(test_index, y_test_label, label='y_test_label', c='orange')
        plt.scatter(test_index[warnings], np.array(len(warnings) * [1]), label="warning", c='red', s=8)
        # plt.scatter(test_index[reset_index], np.array(len(reset_index) * [1]), label="reset", c='green', s=8)
        plt.legend()
        plt.savefig('E:/Fall_Detection/UMAFall/figure/w3.svg', bbox_inches='tight')
        plt.close()
        with open("test3.csv", "w", newline='') as csvfile:
            writer = csv.writer(csvfile, dialect='excel')
            writer.writerows(zip(y_test_label, y_pred_label,test_index))

    tn, fp, fn, tp = confusion_matrix(y_test_label, y_pred_label).ravel()
    TN.append(tn)
    FP.append(fp)
    FN.append(fn)
    TP.append(tp)
    Accuracy.append((tp + tn) / (tp + tn + fp + fn))
    Precision.append(tp / (tp + fp))
    Recall.append(tp / (tp + fn))

for i in range(3):
    print('-------------' + str(i + 1) + '-------------')

    print("K fold  TCN acc:{:4.4f}%".format(100 * ACC[i]))
    print("K fold  TCN loss: {:4.4f}%".format(100 * LOSS[i]))
    print("将ADL预测为ADL:", TN[i])
    print("将跌倒预测为跌倒:", FP[i])
    print("将ADL预测为跌倒:", FN[i])
    print("将跌倒预测为ADL:", TP[i])
    print("K fold Accuracy:", Accuracy[i])
    print("K fold precision:", Precision[i])  # 查准率
    print("K fold recall:", Recall[i])  # 查全率

print('--------------------------------------------------')
print("K fold  TCN acc:{:4.4f}%".format(100 * ACC[i]))
print("K fold  TCN loss: {:4.4f}%".format(100 * LOSS[i]))
print("K fold average Accuracy:", np.array(Accuracy).mean())
print("K fold average precision:", np.array(Precision).mean())  # 查准率
print("K fold average recall:",np.array(Recall).mean())  # 查全率


