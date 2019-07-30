# -*- coding: utf-8 -*-
__author__ = 'Atlas'
'''
Author: Atlas Kim
Description: 
Modification: 因为还要输net参数，直接在主函数里调用比较方便
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

import tkinter as tk
import tkinter.filedialog



def postPlot(model, x, y, num):

    root = tk.Tk()
    root.withdraw()
    filepath = tkinter.filedialog.askopenfilename(filetypes=[('NET_files', '.pkl')])
    # ask目录
    tkinter.filedialog.askdirectory(initialdir=os.getcwd(),
                                    title='Please select a directory')
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model'])  # 推荐
    optimizer.load_state_dict(checkpoint['optimizer'])
    start_epoch = checkpoint['epoch'] + 1
    # plot 都在cpu空间
    testx = ToVariable(x)
    predDat = model(x).data.numpy()
    simplot(y, predDat)
    root.destroy()


def main():
    postPlot()

if __name__ == '__main__':
    main()