import numpy as np
import matplotlib.pyplot as plt

#y_pred=[1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]



"""
1.指针p从第一个采样点开始移动，当遇到0时，直接跳过。
2.当p指向第一个1时，用adl_num和fall_num开始分别统计0的个数和1的个数，并计算跌倒样本占比index=fall_num/(adl_num+fall_num)
3.一连串0中出现极少数的1：如果fall_num<FALL_MIN并且index<0.1，则adl_num和fall_num清0。
4.发生跌倒的情况：
    如果跌倒个数>FALL_MIN并且index>0.6，则记录发送警报的时刻，将状态is_warning=True。
    当is_warning=True时，如果index>0.6，则继续移动指针，如果index<0.6，说明跌倒已结束，adl_num和fall_num清0。
记录清零的时刻
Cogent  
#TST
#UMAFall  
#UPFall  
UR      
"""

def is_Fall(y_pred_label):
    fall_num=0 #统计跌倒样本点
    adl_num=0 #统计非跌倒样本点
    p=0 #指针，从第一个采样点开始移动
    FALL_MIN=35 #发生一次跌倒时要求的最小跌倒采样点个数
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

    return warnings
y_pred_label=[1,1,1,1,1,1,0,0]
w=is_Fall(y_pred_label)
print(w)
def TST_Fall(y_pred_label):
    fall_num=0 #统计跌倒样本点
    adl_num=0 #统计非跌倒样本点
    p=0 #指针，从第一个采样点开始移动
    FALL_MIN=15 #发生一次跌倒时要求的最小跌倒采样点个数
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
        if(fall_num>=FALL_MIN and index>=0.75):#如果跌倒个数大于FALL_MIN,并且比值大于0.6
            if(is_warning==False):#如果还未发送警报
                warnings.append(p)#在此处发送警报
                is_warning=True
        elif(is_warning==True):
            if(index<0.8):
                is_warning = False
                fall_num = 0
                adl_num = 0
                reset_index.append(p)
            else:
                continue
        if(fall_num<FALL_MIN and index<0.75):#如果跌倒个数小于FALL_MIN，且占比非常小，重新计数。
            is_warning = False
            fall_num = 0
            adl_num = 0
            reset_index.append(p)

    return warnings,reset_index
def UR_Fall(y_pred_label):
    fall_num=0 #统计跌倒样本点
    adl_num=0 #统计非跌倒样本点
    p=0 #指针，从第一个采样点开始移动
    FALL_MIN=5 #发生一次跌倒时要求的最小跌倒采样点个数
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
        if(fall_num<FALL_MIN and index<0.6):#如果跌倒个数小于FALL_MIN，且占比非常小，重新计数。
            is_warning = False
            fall_num = 0
            adl_num = 0
            reset_index.append(p)

    return warnings

# w = is_Fall(y_pred)
# plt.scatter(np.arange(len(y_pred)),y_pred,s=5)
# plt.scatter(w, [1] * len(w),s=5)
# plt.show()

def cogent_Fall(y_pred_label):
    fall_num=0 #统计跌倒样本点
    adl_num=0 #统计非跌倒样本点
    p=0 #指针，从第一个采样点开始移动
    FALL_MIN=5 #发生一次跌倒时要求的最小跌倒采样点个数
    warnings=[]#记录发送警报的时刻
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
            if(index<0.8):
                is_warning = False
                fall_num = 0
                adl_num = 0
            continue
        if(fall_num<FALL_MIN and index<0.1):#如果跌倒个数小于FALL_MIN，且占比非常小，重新计数。
            is_warning = False
            fall_num = 0
            adl_num = 0

    return warnings
# w = cogent_Fall(y_pred)
# plt.scatter(np.arange(len(y_pred)),y_pred,s=5)
# plt.scatter(w, [1] * len(w),s=5)
# plt.show()



"""
 统计：fall_num,adl_num
 第一次跌倒样本/非跌倒样本>=1.5时，就发送警报，后面可能还有跌倒样本，继续计算,当没有非跌倒样本出现时，该比值是递增的。
 如果非跌倒样本增多，该比值会递减，当递减到比值为1时，fall_num和adl_num置0。
 因为非跌倒样本和跌倒样本个数差异较大，所以只有遇到跌倒样本时，才进行计数。
 
"""
