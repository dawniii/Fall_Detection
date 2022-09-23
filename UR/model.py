import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from scipy.stats import stats
from sklearn.model_selection import KFold
from tcn import TCN, tcn_full_summary
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import MinMaxScaler
import glob,os
gpu = tf.config.experimental.list_physical_devices(device_type='GPU')
tf.config.experimental.set_memory_growth(gpu[0], True)
import send_warning as W
import Butterworth
import csv
"""
        数据处理：
        共30条非跌倒序列和30条跌倒序列，1条非跌倒序列＋1条跌倒序列依次进行拼接。
        对于跌倒序列，只取x轴加速度峰值左右50个采样点作为输入。
"""
Parent_Fall="E:/Fall_Detection/UR/data/fall/"
Parent_Adl="E:/Fall_Detection/UR/data/adl/"

labels=[]
df=pd.DataFrame()
for i in range(30):
    adl_dir1='adl-'+str('%02d' % (i+1))+"-acc.csv"
    dk = pd.read_csv(Parent_Adl+adl_dir1,header=None)
    dk.insert(5, 'label_1', [0] * len(dk))
    df = pd.concat([df, dk])
    # adl_dir2='adl-'+str('%02d' % (i*2+2))+"-acc.csv"
    # dk = pd.read_csv(Parent_Adl+adl_dir2,header=None)
    # dk.insert(5, 'label_1', [0] * len(dk))
    # df = pd.concat([df, dk])
    fall_dir='fall-'+str('%02d' % (i+1))+"-acc.csv"
    dk = pd.read_csv(Parent_Fall+fall_dir,header=None)
    max_id = dk[2].idxmax()
    dk=dk[max_id-50:max_id+50]
    dk.insert(5, 'label_1', [1] * len(dk))
    df = pd.concat([df, dk])
df = pd.DataFrame(df.values, columns=['timestep','acc_all','acc_x','acc_y','acc_z','label_1'])
df.reset_index(drop=True, inplace=True)
# df['acc_all']=Butterworth.BW_func(df['acc_all'])
# df['acc_x']=Butterworth.BW_func(df['acc_all'])
# df['acc_y']=Butterworth.BW_func(df['acc_all'])
# df['acc_z']=Butterworth.BW_func(df['acc_all'])
"""
        数据归一化和格式调整
"""
FEATURES =3
#RANDOM_SEED = 42
TIME_STEPS=1
step=1
segments=[]
labels=[]
# 归一化，映射到-1到1之间
scale_columns = ['acc_all','acc_x','acc_y','acc_z']
# 归一化，映射到-1到1之间
scaler = MinMaxScaler(feature_range=(-1, 1))
scaler = scaler.fit(df[scale_columns])
df.loc[:, scale_columns] = scaler.transform(df[scale_columns].to_numpy())
for i in range (0,len(df)-TIME_STEPS,step):
    # all= df['acc_all'].values[i:i + TIME_STEPS].reshape(-1, 1)
    x = df['acc_x'].values[i:i + TIME_STEPS].reshape(-1, 1)
    y = df['acc_y'].values[i:i + TIME_STEPS].reshape(-1, 1)
    z = df['acc_z'].values[i:i + TIME_STEPS].reshape(-1, 1)
    label = stats.mode(df['label_1'][i:i + TIME_STEPS])[0][0]  # 出现最多的类别
    segments.append(np.hstack((x,y,z)))
    labels.append(label)

"""
        3折交叉验证，前15条跌倒序列进行训练，后5条跌倒序列进行测试
"""
#设置shuffle=False，每次运行结果都相同
floder = KFold(n_splits=3,  shuffle=False)
reshaped_segments =np.asarray(segments,dtype=np.float32).reshape(-1,TIME_STEPS,FEATURES)
labels=np.asarray(pd.get_dummies(labels),dtype=np.float32)#是利用pandas实现one hot encode的方式。
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

for train_index, test_index in floder.split(reshaped_segments):
    k_fold_num +=1
    X_train, X_test = reshaped_segments[train_index], reshaped_segments[test_index]
    y_train, y_test = labels[train_index], labels[test_index]
    #------------------------TCN--------------------------
    model_TCN = tf.keras.models.Sequential([
        TCN(input_shape=(X_train.shape[1],X_train.shape[2]),dilations=(1, 2,4,8,16,32)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(y_train.shape[1], activation='softmax')
    ])
    #---------------------model.compile--------------------------------------------------
    model_TCN.compile(loss='binary_crossentropy',metrics=['acc'],optimizer='adam')
    if(k_fold_num==1):
        checkpoint_save_path ="E:/Fall_Detection/UR/checkpoint/TCN_UR1.ckpt"
        if os.path.exists(checkpoint_save_path + '.index'):
            print('-------------load the model-----------------')
            model_TCN.load_weights(checkpoint_save_path)
    if(k_fold_num==2):
        checkpoint_save_path ="E:/Fall_Detection/UR/checkpoint/TCN_UR2.ckpt"
        if os.path.exists(checkpoint_save_path + '.index'):
            print('-------------load the model-----------------')
            model_TCN.load_weights(checkpoint_save_path)
    if(k_fold_num==3):
        checkpoint_save_path = "E:/Fall_Detection/UR/checkpoint/TCN_UR3.ckpt"
        if os.path.exists(checkpoint_save_path + '.index'):
            print('-------------load the model-----------------')
            model_TCN.load_weights(checkpoint_save_path)
    # if(k_fold_num==4):
    #     checkpoint_save_path = "E:/Fall_Detection/UR/checkpoint/TCN_UR4.ckpt"
    #     if os.path.exists(checkpoint_save_path + '.index'):
    #         print('-------------load the model-----------------')
    #         model_TCN.load_weights(checkpoint_save_path)

    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_save_path,
        save_weights_only=True,
        monitor='acc',
        mode='max',
        save_best_only=True)
    #model_TCN.fit(X_train, y_train, epochs=32, batch_size=64, verbose=1,shuffle=True,callbacks=[model_checkpoint_callback])
    model_TCN.summary()
    loss1, acc1 = model_TCN.evaluate(X_test, y_test, verbose=1)
    LOSS.append(loss1)
    ACC.append(acc1)
    y_pred_label=np.argmax(model_TCN.predict(X_test), axis=1)
    y_test_label=np.argmax(y_test, axis=1)
    #混淆矩阵
    # C=confusion_matrix(y_test_label, y_pred_label)
    # sns.heatmap(C, annot=True)
    confusion_mat= confusion_matrix(y_test_label, y_pred_label)
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
        plt.savefig('E:/Fall_Detection/UR/figure/k1-TCN-C.png', bbox_inches='tight')
        plt.close()
        warnings = []
        reset_index = []
        # test=y_pred_label[5420:11600]
        warnings = W.UR_Fall(y_pred_label)
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
        plt.savefig('E:/Fall_Detection/UR/figure/w1.svg', bbox_inches='tight')
        plt.close()
        with open("test1.csv", "w", newline='') as csvfile:
            writer = csv.writer(csvfile, dialect='excel')
            writer.writerows(zip(y_test_label, y_pred_label, test_index))
    if (k_fold_num == 2):
        plt.savefig('E:/Fall_Detection/UR/figure/k2-TCN-C.png', bbox_inches='tight')
        plt.close()
        warnings = []
        reset_index = []
        # test=y_pred_label[5420:11600]
        warnings = W.UR_Fall(y_pred_label)
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
        plt.savefig('E:/Fall_Detection/UR/figure/w2.svg', bbox_inches='tight')
        plt.close()
        with open("test2.csv", "w", newline='') as csvfile:
            writer = csv.writer(csvfile, dialect='excel')
            writer.writerows(zip(y_test_label, y_pred_label, test_index))
    if (k_fold_num == 3):
        plt.savefig('E:/Fall_Detection/UR/figure/k3-TCN-C.png', bbox_inches='tight')
        plt.close()
        warnings = []
        reset_index = []
        # test=y_pred_label[5420:11600]
        warnings = W.UR_Fall(y_pred_label)
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
        plt.savefig('E:/Fall_Detection/UR/figure/w3.svg', bbox_inches='tight')
        plt.close()
        with open("test3.csv", "w", newline='') as csvfile:
            writer = csv.writer(csvfile, dialect='excel')
            writer.writerows(zip(y_test_label, y_pred_label, test_index))

    # tn:将ADL预测为ADL  tp:将跌倒预测为跌倒  fp:将ADL预测为跌倒 1 fn：将跌倒预测为ADL
    tn, fp, fn, tp = confusion_matrix(y_test_label, y_pred_label).ravel()
    TN.append(tn)
    FP.append(fp)
    FN.append(fn)
    TP.append(tp)
    Accuracy.append((tp + tn) / (tp + tn + fp + fn))
    Precision.append(tp / (tp + fp))
    Recall.append(tp / (tp + fn))
    plt.legend()
for i in range(3):
    print('-------------'+str(i+1)+'-------------')

    print("K fold  TCN acc:{:4.4f}%" .format(100*ACC[i]))
    print("K fold average TCN loss: {:4.4f}%" .format(100*LOSS[i]))
    print("K fold tn:", TN[i])  # 查准率
    print("K fold fp:",  FP[i])  # 查全率
    print("K fold fn:", FN[i])  # 查准率
    print("K fold tp:",  TP[i])  # 查全率
    print("K fold Accuracy:", Accuracy[i])
    print("K fold precision:", Precision[i])  # 查准率
    print("K fold recall:",  Recall[i])  # 查全率
print('--------------------------------------------------')
print("K fold  TCN acc:{:4.4f}%".format(100 * ACC[i]))
print("K fold  TCN loss: {:4.4f}%".format(100 * LOSS[i]))
print("K fold average Accuracy:", np.array(Accuracy).mean())
print("K fold average precision:", np.array(Precision).mean())  # 查准率
print("K fold average recall:",np.array(Recall).mean())  # 查全率

