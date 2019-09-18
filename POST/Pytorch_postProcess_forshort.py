# -*- coding: utf-8 -*-
__author__ = 'Atlas'
'''
Author: Atlas Kim
Description: 只读取，绘图比较，输出excel、给matlab
Modification: 因为还要输net参数，直接在主函数里调用比较方便
不知道为什么debug可以运行，换一台电脑也可以
但是直接运行就不行
假死原因：simplot(y, predDat)
重新使用小段数据反演，推测模型，还是一样，感觉没有学到里头的东西

'''

import torch
import torch.nn as nn
from torch.autograd import *
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import xlrd
import xlwt
import os
import sys


def uigetpath(fileend = '.pkl'):
    import tkinter as tk
    import tkinter.filedialog
    root = tk.Tk()
    root.withdraw()
    filepath = tkinter.filedialog.askopenfilename(filetypes=[('NET_files', fileend)])
    return filepath



def postPlot(x, y):
    # 单次生成
    filepath = uigetpath(fileend = '.pkl')
    checkpoint = torch.load(filepath)
    # 重新构造
    # 可能有点问题
    NEUNUM = len(checkpoint['lstm.weight_hh_l0'][0])
    tmodel = LSTMpred(1, NEUNUM).to('cuda')
    tmodel.load_state_dict(checkpoint)
    # plot 都在cpu空间
    testx = ToVariable(x).to('cuda')
    predDat = tmodel(testx).data.to('cpu').numpy()
    simplot(y, predDat)
    print('SimplotEnd2')
    return y, predDat

def NNdataCreate(pklpath, x, NL, BATCH):
    """

    :param pklpath: pkl文件路径
    :param x: 高低不平顺幅值短序
    :return: 轨向不平衡孙
    """
    checkpoint = torch.load(pklpath)
    # 可能有点问题
    NEUNUM = len(checkpoint['lstm.weight_hh_l0'][0])
    # 读取模型
    tmodel = LSTMpred2(1, NEUNUM, batchsize=BATCH, num_layer=NL).to('cuda')
    tmodel.load_state_dict(checkpoint)
    # plot 都在cpu空间
    testx = ToVariable(x).to('cuda')
    predDat = tmodel(testx).data.to('cpu').numpy()

    return predDat

def simplot(trueDat, predDat):

    plt.figure()
    # plt.plot(y.numpy())
    plt.plot(trueDat, label='Truedata')
    plt.plot(predDat, label='Predict', alpha=0.7)
    plt.legend()

    # plt.show()
    plt.pause(5)
    plt.draw()


    print('SimplotEnd1')


def save2excel(data, xlname='Pred_Truth.xls'):
    """
    data: [array(predict),array(Truth)]
    """

    xlsfilename = xlname
    workbook = xlwt.Workbook(encoding='utf-8')
    wsheet = workbook.add_sheet('Test', cell_overwrite_ok=True)
    for j in range(len(data)):
        for i in range(len(data[j])):
            wsheet.write(i, j, label=float(data[j][i]))
    workbook.save(xlsfilename)
    print("Excel out finished.")
    print(os.path.abspath(xlname))


def loaddata(xlpath, length=-1, start=1):
    # 打开文件

    workbook = xlrd.open_workbook(xlpath)
    # 获取所有sheet
    print(workbook.sheet_names())  # [u'sheet1', u'sheet2']
    sheet2_name = workbook.sheet_names()[0]
    sheet2 = workbook.sheet_by_index(0)
    # sheet的名称，行数，列数
    print(sheet2.name, sheet2.nrows, sheet2.ncols)
    # 获取整行和整列的值（数组）
    rows = sheet2.row_values(0)  # 获取第四行内容
    # 9 左高低，11 左轨向   10右高低   12右轨向
    if length == -1:
        inputs = sheet2.col_values(9, start_rowx=start)
        targets = sheet2.col_values(11, start_rowx=start)
    else:
        inputs = sheet2.col_values(9, start_rowx=start, end_rowx=start + length)
        targets = sheet2.col_values(11, start_rowx=start, end_rowx=start + length)
    print('Datasetlens: ', len(inputs))
    return inputs, targets


def SeriesGen(N):
    x = torch.arange(0, N, 0.01)
    return x, torch.sin(x)


def trainDataGen(x, seq, k, step=10 * 4):
    """

    :param x: input
    :param seq: output
    :param k: 每个batch的大小
    :return: 列表 数组
    数据长度-k-1 batch个数
    """
    dat = list()
    L = len(seq)
    # k 其实就是训练集的长度单元，
    # 多个训练集，[[x1,x2....],[y1y2]]
    num = 0
    for i in range(0, L - k - 1, step):
        indat = x[i:i + k]
        outdat = seq[i:i + k]
        dat.append((indat, outdat))
        #        print('TrainData: length ', len(dat))
        #        print(num)
        num += 1
    print('Batch Number:', len(dat))
    print('Batch size:', len(dat[0][0]))
    return dat


def ToVariable(x):
    tmp = torch.FloatTensor(x)
    return Variable(tmp)

# class LSTMpred1(nn.Module):
#
#     def __init__(self, input_size, hidden_dim):
#         super(LSTMpred1, self).__init__()
#         self.input_dim = input_size
#         self.hidden_dim = hidden_dim
#         self.lstm = nn.LSTM(input_size, hidden_dim)
#         self.hidden2out = nn.Linear(hidden_dim, 1)
#         self.hidden = self.init_hidden()
#
#     def init_hidden(self):
#         return (Variable(torch.zeros(1, 1, self.hidden_dim)).cuda(),
#                 Variable(torch.zeros(1, 1, self.hidden_dim)).cuda())
#
#     def forward(self, seq):
#         lstm_out, self.hidden = self.lstm(
#             seq.view(len(seq), 1, -1), self.hidden)
#         outdat = self.hidden2out(lstm_out.view(len(seq), -1))
#         return outdat
# 0917 更改作废
class LSTMpred2_ori0917(nn.Module):
    def __init__(self, input_size, hidden_dim, num_layer=1):
        super(LSTMpred2, self).__init__()
        self.input_dim = input_size
        self.hidden_dim = hidden_dim
        self.num_layer = num_layer
        # 数据归一化操作
        # self.bn1 = nn.BatchNorm1d(num_features=320)
        # 增加DROPout 避免过拟合
        self.lstm = nn.LSTM(input_size, hidden_dim, num_layer, dropout=0.5)
        self.hidden2out = nn.Linear(hidden_dim, 1)
        self.hidden = self.init_hidden()

    # 第一个求导应该不用的吧
    def init_hidden(self):
        return (Variable(torch.zeros(self.num_layer, 1, self.hidden_dim)).cuda(),
                Variable(torch.zeros(self.num_layer, 1, self.hidden_dim)).cuda())

    def forward(self, seq):
        lstm_out, self.hidden = self.lstm(
            seq.view(len(seq), 1, -1), self.hidden)
        outdat = self.hidden2out(lstm_out.view(len(seq), -1))
        return outdat



# 0917 COLAB 批量计算

class LSTMpred2(nn.Module):

    def __init__(self, input_size, hidden_dim, batchsize, num_layer=1):
        super(LSTMpred2, self).__init__()
        self.input_dim = input_size
        self.hidden_dim = hidden_dim
        self.num_layer = num_layer
        self.batchsize = batchsize
        # self.hidden = self.init_hidden()
        # 数据归一化操作
        # self.bn1 = nn.BatchNorm1d(num_features=320)
        # 增加DROPout 避免过拟合
        self.lstm = nn.LSTM(input_size, hidden_dim, num_layer, dropout=0.5)
        # outfeature = 1
        self.hidden2out = nn.Linear(self.hidden_dim, 1)

    # 第一个求导应该不用的吧
    def init_hidden(self):
        return (
        Variable(torch.zeros(self.num_layer, self.batchsize, self.hidden_dim)).cuda(),
        Variable(torch.zeros(self.num_layer, self.batchsize, self.hidden_dim)).cuda())

    def forward(self, seq):
        # 三个句子，10个单词，1000

        # hc维度应该是 [层数，batch, hiddensize]
        # out 维度应该是[单词, batch, hiddensize]
        # lstm_out, self.hidden = self.lstm(seq.view(len(seq), 1, -1), self.hidden)
        # seq =1  batch 1 vec 200

        # vecinput 行数据的个数
        # input >>> [seq_len, batchsize, input_size]
        # out >>> [seq_len, bathchsize, hiddenlayernum]
        # h,c >>> [层数，batchsize, hiddensize]
        lstm_out, self.hidden = self.lstm(
            seq.view(int(len(seq) / self.batchsize), self.batchsize, 1), self.hidden)
        # 是不是多对一的话留下最后结果
        # outdat = self.hidden2out(lstm_out[-1].view(self.batchsize, -1))
        # return outdat.view(-1)
        outdat = self.hidden2out(lstm_out.view(len(seq), -1))
        return outdat

def main():
    # 配置输入batch
    xlpath = uigetpath(fileend = '.xls')
    pklpath = uigetpath(fileend = '.pkl')
    # 12500~ 15000 有坏值
    # xy[]
    x, y = loaddata(xlpath, length=-1, start=1)
    tlen = len(x)
    tstart = 0
    x = x[tstart:tstart+tlen]
    trueDat = y[tstart:tstart+tlen]

    # 生成短序序列 n 短序列长度
    n = 200
    # [[],[],[短序列]]
    xs = [x[i:i + n] for i in range(0, len(x), n)]
    predDat = []
    predDat2 = [predDat.extend(NNdataCreate(pklpath, li)) for li in xs]
    save2excel([trueDat, predDat, x], xlname=xlpath+'_PyPost_short.xls')
    print('True','|Predict')

if __name__ == '__main__':
    main()