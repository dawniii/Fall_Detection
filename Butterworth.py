"""
巴特沃斯滤波器
低通
巴特沃斯滤波器的特点是通频带内的频率响应曲线最大限度平坦，没有起伏，而在阻频带则逐渐下降为零。 在振幅的对数对角频率的波特图上，从某一边界角频率开始，振幅随着角频率的增加而逐步减少，趋向负无穷大。巴特沃斯滤波器的频率特性曲线，无论在通带内还是阻带内都是频率的单调函数。因此，当通带的边界处满足指标要求时，通带内肯定会有裕量。所以，更有效的设计方法应该是将精确度均匀的分布在整个通带或阻带内，或者同时分布在两者之内。这样就可用较低阶数的系统满足要求。这可通过选择具有等波纹特性的逼近函数来达到。
原文链接：https://blog.csdn.net/weixin_42762173/article/details/121530932
"""
from scipy import signal
"""
1、低通滤波

这里假设采样频率为1000hz,信号本身最大的频率为500hz，要滤除400hz以上频率成分，即截至频率为400hz,则wn=2*400/1000=0.8。Wn=0.8

scipy.signal.butter(N, Wn, btype='low', analog=False, output='ba')

输入参数：
N:滤波器的阶数
Wn：归一化截止频率。计算公式Wn=2*截止频率/采样频率。（注意：根据采样定理，采样频率要大于两倍的信号本身最大的频率，才能还原信号。截止频率一定小于信号本身最大的频率，所以Wn一定在0和1之间）。当构造带通滤波器或者带阻滤波器时，Wn为长度为2的列表。
btype : 滤波器类型{‘lowpass’, ‘highpass’, ‘bandpass’, ‘bandstop’},
output : 输出类型{‘ba’, ‘zpk’, ‘sos’}

输出参数：
b，a: IIR滤波器的分子（b）和分母（a）多项式系数向量。output='ba'
z,p,k: IIR滤波器传递函数的零点、极点和系统增益. output= 'zpk'
sos: IIR滤波器的二阶截面表示。output= 'sos'

CogentLab数据集的采样频率为100HZ 100>=2w  w<=50   截至频率为5HZ  Wn=2*5/100=0.1

"""
def BW_func(data):
    b, a = signal.butter(4, 0.5, 'lowpass')   #配置滤波器 8 表示滤波器的阶数 4阶0.3效果不太好
    filtedData = signal.filtfilt(b, a, data,padlen=5)  #data为要过滤的信号
    return filtedData
