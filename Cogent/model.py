import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tcn import TCN, tcn_full_summary
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from scipy.stats import stats
from sklearn.preprocessing import MinMaxScaler
import glob,os
gpu = tf.config.experimental.list_physical_devices(device_type='GPU')
tf.config.experimental.set_memory_growth(gpu[0], True)
from sklearn.model_selection import KFold
import csv
TIME_STEPS=1
FEATURES =6
RANDOM_SEED = 42
step=1
segments=[]

labels=[]
labels_number=[]

filepath="E:/Fall_Detection/Cogent/data/falls/subject_1"
df = pd.read_csv(filepath)
dk=pd.read_csv(filepath)
#df['annotation_1'] = df['annotation_1'].replace(2, 1)
# 删除annotation_1列包含数字2的行
df = df[~df['annotation_1'].isin([2])]
dk = df[~df['annotation_1'].isin([2])]
#调整df的标号
df.reset_index(drop=True, inplace=True)
dk.reset_index(drop=True, inplace=True)
i = 0
start_list=[]#统计所有跌倒的开始点
end_list=[]#统计所有跌倒的结束
while (i < len(df)):
    if (df['annotation_1'][i] == 1.0):  # 一次跌倒发生的开始
        start_fall_index = i  # 一次跌倒发生的开始点
        end = i + 1
        while ((0.0 in list(df['annotation_1'][start_fall_index:end])) == False):
            i = i + 1
            end = i
        end_fall_index = end - 2
        start_list.append(start_fall_index)
        end_list.append(end_fall_index)
        # 找到峰值所在的坐标
        max_id = df['ch_accel_x'][start_fall_index:end_fall_index + 1].idxmax()
        print(start_fall_index, end_fall_index, df['ch_accel_x'][start_fall_index:end_fall_index + 1].max(),
              df['ch_accel_x'][start_fall_index:end_fall_index + 1].idxmax())  # end_fall_index 一次跌倒发生的结束点
        # 选取峰值附近的数据，假设跌倒发生在2s内，就让max_id向左向右的100个数据注释不变。
        j = start_fall_index
        while (j < max_id - 50):
            df['annotation_1'][j] = 0.0
            j = j + 1
        k = end_fall_index
        while (k > max_id + 50):
            df['annotation_1'][k] = 0.0
            k = k - 1
    else:
        i = i + 1

scale_columns = ['ch_accel_x','ch_accel_y','ch_accel_z','ch_gyro_x','ch_gyro_y','ch_gyro_z']
# 归一化，映射到-1到1之间
scaler = MinMaxScaler(feature_range=(-1, 1))
scaler = scaler.fit(df[scale_columns])
df.loc[:, scale_columns] = scaler.transform(df[scale_columns].to_numpy())
for i in range (0,len(df)-TIME_STEPS,step):
    x = df['ch_accel_x'].values[i:i + TIME_STEPS].reshape(-1, 1)
    y = df['ch_accel_y'].values[i:i + TIME_STEPS].reshape(-1, 1)
    z = df['ch_accel_z'].values[i:i + TIME_STEPS].reshape(-1, 1)
    xs = df['ch_gyro_x'].values[i:i + TIME_STEPS].reshape(-1, 1)
    ys = df['ch_gyro_y'].values[i:i + TIME_STEPS].reshape(-1, 1)
    zs = df['ch_gyro_z'].values[i:i + TIME_STEPS].reshape(-1, 1)
    label = stats.mode(df['annotation_1'][i:i + TIME_STEPS])[0][0]  # 出现最多的类别
    label_number = stats.mode(df['annotation_1'][i:i + TIME_STEPS])[1][0]  # 出现最多的类别的个数
    segments.append(np.hstack((x,y,z,xs, ys, zs)))
    labels.append(label)
    labels_number.append(label_number)
# n_splits 表示划分为几块（至少是2）
# shuffle 表示洗牌操作，也就是是否打乱，默认False，即不打乱
# random_state 表示是否固定随机起点，一般在 shuffle == True时使用

#------------------3折交叉验证-------------------------
floder = KFold(n_splits=3,shuffle=False)
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
        checkpoint_save_path1 ="E:/Fall_Detection/Cogent/checkpoint/TCN_cogent1.ckpt"
        if os.path.exists(checkpoint_save_path1 + '.index'):
            print('-------------load the model-----------------')
            model_TCN.load_weights(checkpoint_save_path1)
    if(k_fold_num==2):
        checkpoint_save_path1 ="E:/Fall_Detection/Cogent/checkpoint/TCN_cogent2.ckpt"
        if os.path.exists(checkpoint_save_path1 + '.index'):
            print('-------------load the model-----------------')
            model_TCN.load_weights(checkpoint_save_path1)
    if(k_fold_num==3):
        checkpoint_save_path1 = "E:/Fall_Detection/Cogent/checkpoint/TCN_cogent3.ckpt"
        if os.path.exists(checkpoint_save_path1 + '.index'):
            print('-------------load the model-----------------')
            model_TCN.load_weights(checkpoint_save_path1)

    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_save_path1,
        save_weights_only=True,
        monitor='acc',
        mode='max',
        save_best_only=True)
    history_TCN=model_TCN.fit(X_train, y_train, epochs=32, batch_size=64, verbose=1,shuffle=True,callbacks=[model_checkpoint_callback])
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
    if(k_fold_num==1):
        plt.savefig('E:/Fall_Detection/Cogent/figure/k1-TCN-C.png',bbox_inches='tight')
        plt.close()
        # warnings=[]
        # reset_index=[]
        # warnings=W.is_Fall(y_pred_label)
        # plt.subplot(2,1,1)
        # plt.plot(test_index, y_pred_label, label='y_pred_label', c='orange')
        # plt.legend()
        # plt.subplot(2,1,2)
        # plt.plot(test_index, y_test_label, label='y_test_label', c='orange')
        # plt.scatter(test_index[warnings], np.array(len(warnings) * [1]), label="warning", c='red', s=8)
        # # plt.scatter(test_index[reset_index], np.array(len(reset_index) * [1]), label="reset", c='green', s=8)
        # plt.legend()
        # plt.savefig('E:/Fall_Detection/Cogent/figure/w1.svg', bbox_inches='tight')
        # plt.close()
        # check_type.Check_Type(y_pred_label, y_test_label, df, test_index,'count1.txt',dk)
        with open("test1.csv", "w", newline='') as csvfile:
            writer = csv.writer(csvfile, dialect='excel')
            writer.writerows(zip(y_test_label, y_pred_label,test_index))
    if(k_fold_num==2):
        plt.savefig('E:/Fall_Detection/Cogent/figure/k2-TCN-C.png',bbox_inches='tight')
        plt.close()
        # reset_index=[]
        # warnings=W.is_Fall(y_pred_label)
        # plt.subplot(2,1,1)
        # plt.plot(test_index, y_pred_label, label='y_pred_label', c='orange')
        # plt.legend()
        # plt.subplot(2,1,2)
        # plt.plot(test_index, y_test_label, label='y_test_label', c='orange')
        # plt.scatter(test_index[warnings], np.array(len(warnings) * [1]), label="warning", c='red', s=8)
        # # plt.scatter(test_index[reset_index], np.array(len(reset_index) * [1]), label="reset", c='green', s=8)
        # plt.legend()
        # plt.savefig('E:/Fall_Detection/Cogent/figure/w2.svg', bbox_inches='tight')
        # plt.close()
        # check_type.Check_Type(y_pred_label, y_test_label, df, test_index,'count2.txt',dk)
        with open("test2.csv", "w", newline='') as csvfile:
            writer = csv.writer(csvfile, dialect='excel')
            writer.writerows(zip(y_test_label, y_pred_label,test_index))
    if(k_fold_num==3):
        plt.savefig('E:/Fall_Detection/Cogent/figure/k3-TCN-C.png',bbox_inches='tight')
        plt.close()
        # reset_index=[]
        # warnings=W.is_Fall(y_pred_label)
        # plt.subplot(2,1,1)
        # plt.plot(test_index, y_pred_label, label='y_pred_label', c='orange')
        # plt.legend()
        # plt.subplot(2,1,2)
        # plt.plot(test_index, y_test_label, label='y_test_label', c='orange')
        # plt.scatter(test_index[warnings], np.array(len(warnings) * [1]), label="warning", c='red', s=8)
        # plt.legend()
        # plt.savefig('E:/Fall_Detection/Cogent/figure/w3.svg', bbox_inches='tight')
        # plt.close()
        # check_type.Check_Type(y_pred_label, y_test_label, df, test_index,'count3.txt',dk)
        with open("test3.csv", "w", newline='') as csvfile:
            writer = csv.writer(csvfile, dialect='excel')
            writer.writerows(zip(y_test_label, y_pred_label,test_index))

    # tn:将ADL预测为ADL  tp:将跌倒预测为跌倒  fp:将ADL预测为跌倒 1 fn：将跌倒预测为ADL
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

