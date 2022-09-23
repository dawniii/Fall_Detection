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
import send_warning as W
TIME_STEPS=1
FEATURES =6
RANDOM_SEED = 42
step=1
segments=[]

labels=[]
labels_number=[]

filepath="E:/Fall_Detection/Cogent/data/falls/subject_1"
df = pd.read_csv(filepath)
dk = pd.read_csv(filepath)
#df['annotation_1'] = df['annotation_1'].replace(2, 1)
# 删除annotation_1列包含数字2的行
df = df[~df['annotation_1'].isin([2])]
dk = dk[~dk['annotation_1'].isin([2])]
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
plt.subplot(3,1,1)
plt.plot(df['ch_accel_x'])
# plt.scatter(np.arange(len(df)),df['annotation_1'],s=3,c='red')
plt.scatter(np.arange(len(df)),df['annotation_2'],s=3,c='green',label='annotation_2')
plt.legend()
# plt.subplot(2,1,2)
# plt.plot(df['ch_accel_y'])
# # plt.scatter(np.arange(len(df)),df['annotation_1'],s=3,c='red')
# plt.scatter(np.arange(len(df)),df['annotation_2'],s=3,c='green')
plt.subplot(3,1,2)
plt.plot(df['ch_accel_x'])
plt.scatter(np.arange(len(df)),df['annotation_1'],s=3,c='red')
plt.legend()
# plt.scatter(np.arange(len(dk)),df['annotation_1'],s=3,c='blue')
plt.subplot(3,1,3)
plt.plot(df['ch_accel_x'])
# plt.scatter(np.arange(len(df)),df['annotation_1'],s=3,c='red')
plt.scatter(np.arange(len(dk)),dk['annotation_1'],s=3,c='blue')
#plt.scatter(np.arange(len(df)),df['annotation_2'],s=3,c='green')
# plt.subplot(3,2,3)
# plt.plot(df['ch_accel_z'])
# plt.scatter(np.arange(len(df)),df['annotation_1'],s=5,c='red')
# plt.scatter(np.arange(len(df)),df['annotation_2'],s=5,c='green')
# plt.subplot(3,2,4)
# plt.plot(df['ch_gyro_x'])
# plt.scatter(np.arange(len(df)),df['annotation_1'],s=5,c='red')
# plt.subplot(3,2,5)
# plt.plot(df['ch_gyro_y'])
# plt.scatter(np.arange(len(df)),df['annotation_1'],s=5,c='red')
# plt.subplot(3,2,6)
# plt.plot(df['ch_gyro_z'])
# plt.scatter(np.arange(len(df)),df['annotation_1'],s=5,c='red')
plt.show()