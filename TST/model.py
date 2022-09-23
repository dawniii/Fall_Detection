import matplotlib.pyplot as plt
import pandas as pd
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
from timeit import default_timer as timer
import send_warning as W
import csv
# step1 import 相关模块
# ES08 male 55
# ADL->Fall ->ADL 不管时间顺序
# step2:指定输入网络的训练集和测试集，如指定训练集的输入 x_train 和标签y_train，测试集的输入 x_test 和标签 y_test。
FEATURES =18
RANDOM_SEED = 42
TIME_STEPS=1
step=1
files=['E:/Fall_Detection/TST/data/fb4fd/ES08/Acquisitions/ADLs/APBE/APBEST/2017-02-21_17.35.43.txt',
       'E:/Fall_Detection/TST/data/fb4fd/ES08/Acquisitions/Falls/FBELFR/FBELFRST/2017-02-21_16.40.47.txt',
       'E:/Fall_Detection/TST/data/fb4fd/ES08/Acquisitions/ADLs/APBE/APBEWK/2017-02-21_17.34.49.txt',
       'E:/Fall_Detection/TST/data/fb4fd/ES08/Acquisitions/Falls/FBELFR/FBELFRSTRC/2017-02-21_16.41.50.txt',
       'E:/Fall_Detection/TST/data/fb4fd/ES08/Acquisitions/ADLs/APSQ/APSQST/2017-02-21_17.37.33.txt',
       'E:/Fall_Detection/TST/data/fb4fd/ES08/Acquisitions/Falls/FBELFR/FBELFRWK/2017-02-21_16.37.47.txt',
       'E:/Fall_Detection/TST/data/fb4fd/ES08/Acquisitions/ADLs/APSQ/APSQWK/2017-02-21_17.36.42.txt',
       'E:/Fall_Detection/TST/data/fb4fd/ES08/Acquisitions/Falls/FBELFR/FBELFRWK/2017-02-21_16.37.47.txt',
       'E:/Fall_Detection/TST/data/fb4fd/ES08/Acquisitions/ADLs/ASCH/ASCHST/2017-02-21_17.41.06.txt',
       'E:/Fall_Detection/TST/data/fb4fd/ES08/Acquisitions/Falls/FBELFR/FBELFRWKRC/2017-02-21_16.38.47.txt',
       'E:/Fall_Detection/TST/data/fb4fd/ES08/Acquisitions/ADLs/ASCH/ASCHWK/2017-02-21_17.39.34.txt',
       'E:/Fall_Detection/TST/data/fb4fd/ES08/Acquisitions/Falls/FBESFR/FBESFRST/2017-02-21_16.48.33.txt',
       'E:/Fall_Detection/TST/data/fb4fd/ES08/Acquisitions/ADLs/ASSO/ASSOST/2017-02-21_17.43.06.txt',
       'E:/Fall_Detection/TST/data/fb4fd/ES08/Acquisitions/Falls/FBESFR/FBESFRSTRC/2017-02-21_16.49.22.txt',
       'E:/Fall_Detection/TST/data/fb4fd/ES08/Acquisitions/ADLs/ASSO/ASSOWK/2017-02-21_17.42.13.txt',
       ]
#files[0]、files[2]、files[4]、files[6]、files[8]ADL
#files[1]、files[3]、files[5]、files[7] Fall
df=pd.DataFrame()
for i in range(len(files)):
       dk=pd.read_csv(files[i],sep=',',header=None)
       dk=dk[[0,1,2,3,4,5,6,7,8,12,13,14,15,16,17,21,22,23]]
       if(i%2==0):
              dk.insert(18, 'label_1', [0] * len(dk))
       else:
              dk.insert(18, 'label_1', [1] * len(dk))
       df = pd.concat([df, dk])
df = pd.DataFrame(df.values, columns=['fsr1_l', 'fsr2_l', 'fsr3_l', 'fsr1_r', 'fsr2_r', 'fsr3_r','accel_x_r','accel_y_r','accel_z_r','gyro_x_r','gyro_y_r','gyro_z_r','accel_x_l','accel_y_l','accel_z_l','gyro_x_l','gyro_y_l','gyro_z_l','label_1'])
df.reset_index(drop=True, inplace=True)


scale_columns =['fsr1_l', 'fsr2_l', 'fsr3_l', 'fsr1_r', 'fsr2_r', 'fsr3_r','accel_x_r','accel_y_r','accel_z_r','gyro_x_r','gyro_y_r','gyro_z_r','accel_x_l','accel_y_l','accel_z_l','gyro_x_l','gyro_y_l','gyro_z_l']
scaler = MinMaxScaler(feature_range=(-1, 1))
scaler = scaler.fit(df[scale_columns])
df.loc[:, scale_columns] = scaler.transform(df[scale_columns].to_numpy())
count=0
segments=[]
labels=[]
for i in range(0, len(df) - TIME_STEPS, step):
    fl1 = df['fsr1_l'].values[i:i + TIME_STEPS].reshape(-1, 1)
    fl2 = df['fsr2_l'].values[i:i + TIME_STEPS].reshape(-1, 1)
    fl3 = df['fsr3_l'].values[i:i + TIME_STEPS].reshape(-1, 1)
    fr1 = df['fsr1_r'].values[i:i + TIME_STEPS].reshape(-1, 1)
    fr2= df['fsr2_r'].values[i:i + TIME_STEPS].reshape(-1, 1)
    fr3= df['fsr3_r'].values[i:i + TIME_STEPS].reshape(-1, 1)
    accxr=df['accel_x_r'].values[i:i + TIME_STEPS].reshape(-1, 1)
    accyr=df['accel_y_r'].values[i:i + TIME_STEPS].reshape(-1, 1)
    acczr=df['accel_z_r'].values[i:i + TIME_STEPS].reshape(-1, 1)
    groxr=df['gyro_x_r'].values[i:i + TIME_STEPS].reshape(-1, 1)
    groyr=df['gyro_y_r'].values[i:i + TIME_STEPS].reshape(-1, 1)
    grozr=df['gyro_z_r'].values[i:i + TIME_STEPS].reshape(-1, 1)
    accxl = df['accel_x_l'].values[i:i + TIME_STEPS].reshape(-1, 1)
    accyl = df['accel_y_l'].values[i:i + TIME_STEPS].reshape(-1, 1)
    acczl = df['accel_z_l'].values[i:i + TIME_STEPS].reshape(-1, 1)
    groxl = df['gyro_x_l'].values[i:i + TIME_STEPS].reshape(-1, 1)
    groyl = df['gyro_y_l'].values[i:i + TIME_STEPS].reshape(-1, 1)
    grozl = df['gyro_z_l'].values[i:i + TIME_STEPS].reshape(-1, 1)
    segment=np.hstack((fl1,fl2,fl3,fr1,fr2,fr3,accxr,accyr,acczr,groxr,groyr,grozr,accxl,accyl,acczl,groxl,groyl,grozl))
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
    # step3 逐层搭建网络结构，model = tf.keras.models.Sequential()。
    # nb_filters：在卷积层中使用的过滤器数量
    # kernel_size： 整在每个卷积层中使用的内核大小。
    # dilations： 一个膨胀列表
    # nb_stacks： 要使用的残差块的堆栈数。
    #  return_sequences=False,是返回输出序列中的最后一个输出，还是返回完整序列。
    # 感受野= 1+nb_stacks*(kernel_size*dilations[i]+..)
    model_TCN = tf.keras.models.Sequential([
        TCN(input_shape=(X_train.shape[1], X_train.shape[2]), dilations=(1, 2, 4, 8, 16, 32)),
        tf.keras.layers.Dropout(0.2),# 提高模型泛化能力的目的在神经网络的训练过程中，将一部分神经元按照一定概率从神经网络中暂时舍弃，使用时被舍弃的神经元恢复链接，
        # softmax为每个输出分类的结果都赋予一个概率值，表示属于每个类别的可能性。
        tf.keras.layers.Dense(y_train.shape[1], activation='softmax')
    ])
    # step4 在 model.compile()中配置训练方法，选择训练时使用的优化器、损失函数和最终评价指标。
    model_TCN.compile(loss='binary_crossentropy', metrics=['acc'], optimizer='adam')
    if (k_fold_num == 1):
        checkpoint_save_path = "E:/Fall_Detection/TST/checkpoint/TCN_TST1.ckpt"
        if os.path.exists(checkpoint_save_path + '.index'):
            print('-------------load the model-----------------')
            model_TCN.load_weights(checkpoint_save_path)

    if (k_fold_num == 2):
        checkpoint_save_path = "E:/Fall_Detection/TST/checkpoint/TCN_TST2.ckpt"
        if os.path.exists(checkpoint_save_path + '.index'):
            print('-------------load the model-----------------')
            model_TCN.load_weights(checkpoint_save_path)

    if (k_fold_num == 3):
        checkpoint_save_path = "E:/Fall_Detection/TST/checkpoint/TCN_TST3.ckpt"
        if os.path.exists(checkpoint_save_path + '.index'):
            print('-------------load the model-----------------')
            model_TCN.load_weights(checkpoint_save_path)


    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_save_path,
        save_weights_only=True,
        monitor='acc',
        mode='max',
        save_best_only=True)
   # step5 在 model.fit()中执行训练过程，告知训练集和测试集的输入值和标签、每个 batch 的大小（batchsize）和数据集的迭代次数（epoch）。
    #history_TCN = model_TCN.fit(X_train, y_train, epochs=32, batch_size=64, verbose=1, shuffle=True,
                               #callbacks=[model_checkpoint_callback])
    # step6 使用 model.summary()打印网络结构，统计参数数目。
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
    title = "TCN_TST"
    disp.ax_.set_title(title)
    if (k_fold_num == 1):
        plt.savefig('E:/Fall_Detection/TST/figure/k1-TCN-C.png', bbox_inches='tight')
        plt.close()
        warnings = []
        reset_index = []
        # test=y_pred_label[5420:11600]
        warnings,reset_index = W.TST_Fall(y_pred_label)
        print(warnings)
        print(reset_index)
        plt.subplot(2, 1, 1)
        plt.plot(test_index, y_pred_label, label='y_pred_label', c='orange')
        plt.legend()
        plt.subplot(2, 1, 2)
        plt.plot(test_index, y_test_label, label='y_test_label', c='orange')
        plt.scatter(test_index[warnings], np.array(len(warnings) * [1]), label="warning", c='red', s=8)
        #plt.scatter(test_index[reset_index], np.array(len(reset_index) * [1]), label="reset", c='green', s=8)
        plt.legend()
        plt.savefig('E:/Fall_Detection/TST/figure/w1.svg', bbox_inches='tight')
        plt.close()
        with open("test1.csv", "w", newline='') as csvfile:
            writer = csv.writer(csvfile, dialect='excel')
            writer.writerows(zip(y_test_label, y_pred_label, test_index))
    if (k_fold_num == 2):
        plt.savefig('E:/Fall_Detection/TST/figure/k2-TCN-C.png', bbox_inches='tight')
        plt.close()
        warnings = []
        reset_index = []
        # test=y_pred_label[5420:11600]
        warnings,reset_index = W.TST_Fall(y_pred_label)
        print(warnings)
        print(reset_index)
        plt.subplot(2, 1, 1)
        plt.plot(test_index, y_pred_label, label='y_pred_label', c='orange')
        plt.legend()
        plt.subplot(2, 1, 2)
        plt.plot(test_index, y_test_label, label='y_test_label', c='orange')
        plt.scatter(test_index[warnings], np.array(len(warnings) * [1]), label="warning", c='red', s=8)
        #plt.scatter(test_index[reset_index], np.array(len(reset_index) * [1]), label="reset", c='green', s=8)
        plt.legend()
        plt.savefig('E:/Fall_Detection/TST/figure/w2.svg', bbox_inches='tight')
        plt.close()
        with open("test2.csv", "w", newline='') as csvfile:
            writer = csv.writer(csvfile, dialect='excel')
            writer.writerows(zip(y_test_label, y_pred_label, test_index))
    if (k_fold_num == 3):
        plt.savefig('E:/Fall_Detection/TST/figure/k3-TCN-C.png', bbox_inches='tight')
        plt.close()
        warnings = []
        reset_index = []
        # test=y_pred_label[5420:11600]
        warnings,reset_index = W.TST_Fall(y_pred_label)
        print(warnings)
        print(reset_index)
        plt.subplot(2, 1, 1)
        plt.plot(test_index, y_pred_label, label='y_pred_label', c='orange')
        plt.legend()
        plt.subplot(2, 1, 2)
        plt.plot(test_index, y_test_label, label='y_test_label', c='orange')
        plt.scatter(test_index[warnings], np.array(len(warnings) * [1]), label="warning", c='red', s=8)
        #plt.scatter(test_index[reset_index], np.array(len(reset_index) * [1]), label="reset", c='green', s=8)
        plt.legend()
        plt.savefig('E:/Fall_Detection/TST/figure/w3.svg', bbox_inches='tight')
        plt.close()
        with open("test3.csv", "w", newline='') as csvfile:
            writer = csv.writer(csvfile, dialect='excel')
            writer.writerows(zip(y_test_label, y_pred_label, test_index))
    tn, fp, fn, tp = confusion_matrix(y_test_label, y_pred_label).ravel()
    TN.append(tn)
    FP.append(fp)
    FN.append(fn)
    TP.append(tp)
    Accuracy.append((tp + tn) / (tp + tn + fp + fn))
    Precision.append(tp / (tp + fp))
    Recall.append(tp / (tp + fn))

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

