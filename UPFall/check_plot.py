import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

test1=pd.read_csv('test1.csv',header=None,names=["y_test_label", "y_pred_label","test_index"])
test2=pd.read_csv('test2.csv',header=None,names=["y_test_label", "y_pred_label","test_index"])
test3=pd.read_csv('test3.csv',header=None,names=["y_test_label", "y_pred_label","test_index"])


def plot(y_test_label,y_pred_label,test_index,warnings,reset):
    plt.subplot(2,1,1)
    plt.scatter(test_index,y_test_label,label='true',s=3,c='green')
    plt.scatter(test_index[warnings], np.array(len(warnings) * [1]), label="warning", c='red', s=8, marker='^')
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.scatter(test_index,y_pred_label,label='pred',s=3,c='orange')
    plt.scatter(test_index[warnings], np.array(len(warnings) * [1]), label="warning", c='red', s=8,marker='^')
    plt.scatter(test_index[reset], np.array(len(reset) * [1]), label="reset", c='blue', s=8,marker='o')
    plt.legend()
    plt.show()
def is_Fall(y_pred_label):
    fall_num=0 #统计跌倒样本点
    adl_num=0 #统计非跌倒样本点
    p=0 #指针，从第一个采样点开始移动
    FALL_MIN=5 #发生一次跌倒时要求的最小跌倒采样点个数15 10(40*0.25) 5(15*0.3) 12(40*0.3)
    warnings=[]#记录发送警报的时刻
    reset_index=[]#记录清零的坐标
    is_warning = False  # 某次跌倒是否发送警报的状态，默认为False
    while(p<len(y_pred_label)):
        if(y_pred_label[p]==0 and fall_num==0):
            p +=1
            continue
        elif(y_pred_label[p]==1):
            fall_num +=1
            p +=1
        elif(y_pred_label[p]==0 and fall_num !=0):
            adl_num +=1
            p +=1
        index = fall_num/(fall_num+adl_num)
        if(fall_num>=FALL_MIN and index>=0.6):#如果跌倒个数大于FALL_MIN,并且比值大于0.6
            if(is_warning==False):#如果还未发送警报
                warnings.append(p)#在此处发送警报
                is_warning=True
        elif(is_warning==True):
            if(index<0.6):
                is_warning = False
                fall_num = 0
                adl_num = 0
                reset_index.append(p)
            else:
                continue
        if(fall_num<FALL_MIN and index<0.5):#如果跌倒个数小于FALL_MIN，且占比非常小，重新计数。
            is_warning = False
            fall_num = 0
            adl_num = 0
            reset_index.append(p)

    return warnings,reset_index
#y=[1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0]
#is_Fall(y)
# def result(test):
#     warnings=[]
#     reset=[]
#     warnings,reset=is_Fall(test['y_pred_label'])
#     plot(test['y_test_label'],test['y_pred_label'],test['test_index'],warnings,reset)
# result(test1)
# result(test2)
# result(test3)

def is_Fall2(y_pred_label,win_size,step):
    """
    遇到第一个跌倒采样点，开始统计win_size窗口内的跌倒样本点的个数。窗口大小尽可能的和发生一次跌倒的采样点个数一致。(比如采样频率为100HZ，跌倒持续时间为2s，窗口大小就为200)
    如何移动滑窗？ 有无重叠？重叠比例为多少？
    清零条件？
    """
    p=0
    MIN_INDEX=0.1
    MAX_INDEX=0.5
    warnings=[]
    while(p<len(y_pred_label)-win_size):
        if(y_pred_label[p]==0):
            p +=1
        elif(y_pred_label[p]==1):
            fall_num=sum(y_pred_label[p:p+win_size])
            index=fall_num/win_size
            if(index<MIN_INDEX):
                p=p+win_size
               # continue
            elif(index>MAX_INDEX):
                warnings.append(p+win_size)
                p=p+win_size
               # continue
            else:
                p=p+step
    return warnings

def result2(test):
    warnings=[]
    reset=[]
    warnings=is_Fall2(test['y_pred_label'],30,15)
    plot(test['y_test_label'],test['y_pred_label'],test['test_index'],warnings,reset)

result2(test1)
result2(test2)
result2(test3)