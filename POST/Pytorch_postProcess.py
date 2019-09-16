# -*- coding: utf-8 -*-
__author__ = 'Atlas'
'''
Author: Atlas Kim
Description: 只读取，绘图比较，输出excel、给matlab
Modification: 因为还要输net参数，直接在主函数里调用比较方便
不知道为什么debug可以运行，换一台电脑也可以
但是直接运行就不行
假死原因：simplot(y, predDat)

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

    filepath = uigetpath(fileend = '.pkl')
    checkpoint = torch.load(filepath)
    # 重新构造
    # 可能有点问题
    NEUNUM = len(checkpoint['lstm.weight_hh_l0'][0])
    # 两层
    tmodel = LSTMpred2(1, NEUNUM,2).to('cuda')
    tmodel.load_state_dict(checkpoint)
    # plot 都在cpu空间
    testx = ToVariable(x).to('cuda')
    predDat = tmodel(testx).data.to('cpu').numpy()
    simplot(y, predDat)
    print('SimplotEnd2')
    return y, predDat


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


class LSTMpred(nn.Module):

    def __init__(self, input_size, hidden_dim):
        super(LSTMpred, self).__init__()
        self.input_dim = input_size
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_size, hidden_dim)
        self.hidden2out = nn.Linear(hidden_dim, 1)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (Variable(torch.zeros(1, 1, self.hidden_dim)).cuda(),
                Variable(torch.zeros(1, 1, self.hidden_dim)).cuda())

    def forward(self, seq):
        lstm_out, self.hidden = self.lstm(
            seq.view(len(seq), 1, -1), self.hidden)
        outdat = self.hidden2out(lstm_out.view(len(seq), -1))
        return outdat





def main():
    # 配置输入batch
    # xlpath = r'excelTest37000.xlsx'
    # xlpath = r'Irr_1000_AM6.xls'
    # xlpath = r'Irr_1000_GML'
    xlpath = uigetpath(fileend = '.xls')

    # 12500~ 15000 有坏值
    x, y = loaddata(xlpath, length=-1, start=1)

    tlen = len(x)
    tstart = 0
    x = x[tstart:tstart+tlen]
    y = y[tstart:tstart+tlen]
    trueDat, predDat = postPlot(x, y)
    print('SimplotEnd3')

    save2excel([trueDat, predDat], xlname=xlpath+'_PyPost.xls')
    print('True','|Predict')

if __name__ == '__main__':
    main()