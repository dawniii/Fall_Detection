
"""
Activity ID Description Duration(s)
1 Falling forward using hands 10
2 Falling forward using knees 10
3 Falling backwards 10(文件里没有)
4 Falling sidewards 10
5 Falling sitting in empty chair 10
6 Walking 60
7 Standing 60
8 Sitting 60
9 Picking up an object 10
10 Jumping 30
11 Laying 60
活动 11 中受试者 8 的试验 2 和 3 不可用
"""
# 对subject1的数据进行拼接
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
"""
S01_A06_T01->S01_A01_T01->S01_A07_T01->S01_A02_T01->S01_A09_T01->S01_A10_T01->S01_A04_T01
S01_A06_T02->S01_A01_T02->S01_A07_T02->S01_A02_T02->S01_A09_T02->S01_A10_T02->S01_A04_T02
S01_A06_T03->S01_A01_T03->S01_A07_T01->S01_A02_T03->S01_A09_T03->S01_A10_T03->S01_A04_T03
"""
df=pd.DataFrame()
activity=[6,1,7,2,9,10,4]
trial=[1,2,3]
is_fall=[0,1,0,1,0,0,1]
parent_dir='E:/Fall_Detection/UPFall/data/'
for t in range(3):
    for a in range(7):
        child_dir='S01_'+'A'+'%02d'%activity[a]+'_T0'+str(t+1)
        f=parent_dir+child_dir+'.csv'
        dk=pd.read_csv(f,error_bad_lines=False,header=None,skiprows=2)
        #'BELT_ACC_X', 'BELT_ACC_Y', 'BELT_ACC_Z', 'BELT_ANG_X', 'BELT_ANG_Y', 'BELT_ANG_Z'
        dk=dk[[2,3,4,5,6,7]]
        if(is_fall[a]==1):#跌倒
            dk.insert(6, 'label_1', [1] * len(dk))
        else:#非跌倒
            dk.insert(6, 'label_1', [0] * len(dk))
        df=pd.concat([df,dk])
df = pd.DataFrame(df.values, columns=['BELT_ACC_X', 'BELT_ACC_Y', 'BELT_ACC_Z', 'BELT_ANG_X', 'BELT_ANG_Y', 'BELT_ANG_Z','label_1'])
df.reset_index(drop=True, inplace=True)
scale_columns =['BELT_ACC_X', 'BELT_ACC_Y', 'BELT_ACC_Z', 'BELT_ANG_X', 'BELT_ANG_Y', 'BELT_ANG_Z','label_1']
scaler = MinMaxScaler(feature_range=(-1, 1))
scaler = scaler.fit(df[scale_columns])
df.loc[:, scale_columns] = scaler.transform(df[scale_columns].to_numpy())
count=0
segments=[]
labels=[]
FEATURES =6
RANDOM_SEED = 42
TIME_STEPS=1
step=1
for i in range(0, len(df) - TIME_STEPS, step):
    x = df['BELT_ACC_X'].values[i:i + TIME_STEPS].reshape(-1, 1)
    y = df['BELT_ACC_Y'].values[i:i + TIME_STEPS].reshape(-1, 1)
    z = df['BELT_ACC_Z'].values[i:i + TIME_STEPS].reshape(-1, 1)
    xs = df['BELT_ANG_X'].values[i:i + TIME_STEPS].reshape(-1, 1)
    ys = df['BELT_ANG_Y'].values[i:i + TIME_STEPS].reshape(-1, 1)
    zs = df['BELT_ANG_Z'].values[i:i + TIME_STEPS].reshape(-1, 1)
    segment=np.hstack((x, y, z,xs,ys,zs))
    segments.append(segment)
labels=list(df['label_1'])

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
        checkpoint_save_path = "E:/Fall_Detection/UPFall/checkpoint/TCN_UPFall1.ckpt"
        if os.path.exists(checkpoint_save_path + '.index'):
            print('-------------load the model-----------------')
            model_TCN.load_weights(checkpoint_save_path)
    if (k_fold_num == 2):
        checkpoint_save_path = "E:/Fall_Detection/UPFall/checkpoint/TCN_UPfall2.ckpt"
        if os.path.exists(checkpoint_save_path + '.index'):
            print('-------------load the model-----------------')
            model_TCN.load_weights(checkpoint_save_path)
    if (k_fold_num == 3):
        checkpoint_save_path = "E:/Fall_Detection/UPFall/checkpoint/TCN_UPfall3.ckpt"
        if os.path.exists(checkpoint_save_path + '.index'):
            print('-------------load the model-----------------')
            model_TCN.load_weights(checkpoint_save_path)

    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_save_path,
        save_weights_only=True,
        monitor='acc',
        mode='max',
        save_best_only=True)
   # history_TCN = model_TCN.fit(X_train, y_train, epochs=32, batch_size=64, verbose=1, shuffle=True,
                                #callbacks=[model_checkpoint_callback])
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
    # 使用sklearn工具包中的ConfusionMatrixDisplay可视化混淆矩阵，参考plot_confusion_matrix
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
        plt.savefig('E:/Fall_Detection/UPFall/figure/k1-TCN-C.png', bbox_inches='tight')
        plt.close()
        warnings = []
        reset_index = []
        # test=y_pred_label[5420:11600]
        warnings = W.is_Fall(y_pred_label)
        print(warnings)
        print(reset_index)
        plt.subplot(2, 1, 1)
        plt.plot(test_index, y_pred_label, label='y_pred_label', c='orange')
        plt.legend()
        plt.subplot(2, 1, 2)
        plt.plot(test_index, y_test_label, label='y_test_label', c='orange')
        plt.scatter(test_index[warnings], np.array(len(warnings) * [1]), label="warning", c='red', s=8)
        # plt.scatter(test_index[reset_index], np.array(len(reset_index) * [1]), label="reset", c='green', s=8)
        plt.legend()
        plt.savefig('E:/Fall_Detection/UPFall/figure/w1.svg', bbox_inches='tight')
        plt.close()
        with open("test1.csv", "w", newline='') as csvfile:
            writer = csv.writer(csvfile, dialect='excel')
            writer.writerows(zip(y_test_label, y_pred_label, test_index))
    if (k_fold_num == 2):
        plt.savefig('E:/Fall_Detection/UPFall/figure/k2-TCN-C.png', bbox_inches='tight')
        plt.close()
        warnings = []
        reset_index = []
        # test=y_pred_label[5420:11600]
        warnings = W.is_Fall(y_pred_label)
        print(warnings)
        print(reset_index)
        plt.subplot(2, 1, 1)
        plt.plot(test_index, y_pred_label, label='y_pred_label', c='orange')
        plt.legend()
        plt.subplot(2, 1, 2)
        plt.plot(test_index, y_test_label, label='y_test_label', c='orange')
        plt.scatter(test_index[warnings], np.array(len(warnings) * [1]), label="warning", c='red', s=8)
        # plt.scatter(test_index[reset_index], np.array(len(reset_index) * [1]), label="reset", c='green', s=8)
        plt.legend()
        plt.savefig('E:/Fall_Detection/UPFall/figure/w2.svg', bbox_inches='tight')
        plt.close()
        with open("test2.csv", "w", newline='') as csvfile:
            writer = csv.writer(csvfile, dialect='excel')
            writer.writerows(zip(y_test_label, y_pred_label, test_index))
    if (k_fold_num == 3):
        plt.savefig('E:/Fall_Detection/UPFall/figure/k3-TCN-C.png', bbox_inches='tight')
        plt.close()
        warnings = []
        reset_index = []
        # test=y_pred_label[5420:11600]
        warnings = W.is_Fall(y_pred_label)
        print(warnings)
        print(reset_index)
        plt.subplot(2, 1, 1)
        plt.plot(test_index, y_pred_label, label='y_pred_label', c='orange')
        plt.legend()
        plt.subplot(2, 1, 2)
        plt.plot(test_index, y_test_label, label='y_test_label', c='orange')
        plt.scatter(test_index[warnings], np.array(len(warnings) * [1]), label="warning", c='red', s=8)
        # plt.scatter(test_index[reset_index], np.array(len(reset_index) * [1]), label="reset", c='green', s=8)
        plt.legend()
        plt.savefig('E:/Fall_Detection/UPFall/figure/w3.svg', bbox_inches='tight')
        plt.close()
        with open("test3.csv", "w", newline='') as csvfile:
            writer = csv.writer(csvfile, dialect='excel')
            writer.writerows(zip(y_test_label, y_pred_label, test_index))
    plt.close()
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


