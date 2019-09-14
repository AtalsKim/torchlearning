clear;close all;
% 用于pyexcel的处理
% 用于对数据集的预处理
% 输入是标准的轨检数据 10,12 是高低
% 输出是xls文件,基本信息不变

[Filename, Foldname]=uigetfile({'*.xls'},"选择XLS");       
% 第一列： 轨检车轨向数据/真值，IFFT 轨向数据
% 第二列： LSTM轨向数据
% 第三列： IFFT高低数据

excelpath = [Foldname,Filename];
data_ori = xlsread(excelpath); %nx2
% JKLM
preprocessErrWlet(data_ori(1:end,10),'J',Filename)
preprocessErrWlet(data_ori(1:end,11),'K',Filename)
preprocessErrWlet(data_ori(1:end,12),'L',Filename)
preprocessErrWlet(data_ori(1:end,13),'M',Filename)





function  preprocessErrWlet(dataori,COLNAME,Filename)

    %% 异常值消除
    % 遍历错误项消除0.003
    n = 1;
    err_ori = dataori;
    [value_ti] =errowipe(err_ori,n, 0.003);

    figure;
    plot(err_ori);
    hold on;
    plot(value_ti);
    legend('原始信号','异常值消除');
    ylabel('幅值 / mm')
    xlabel('序列编号')

    %% 趋势项消除

    qs_ori = dataori;
    % 后去小波基本参数
    [C,L] = wavedec(qs_ori,6,'bior4.4');
    % 获取低频系数
    ca6 = appcoef(C,L,'bior4.4',6);

    % 高频信号
    cd6 = detcoef(C,L,6);
    cd5 = detcoef(C,L,5);
    cd4 = detcoef(C,L,4);
    cd3 = detcoef(C,L,3);
    cd2 = detcoef(C,L,2);
    cd1 = detcoef(C,L,1);


    % 消除超低频信号
    ca6 = zeros(length(ca6),1);


    % 重建系数矩阵
    c1 = [ca6;cd6;cd5;cd4;cd3;cd2;cd1];
    % 根据系数重构
    s2 = waverec(c1,L,'bior4.4');

    figure;
    plot(qs_ori);
    hold on;
    plot(s2);
    legend('原始信号','趋势项消除');
    ylabel('幅值 / mm')
    xlabel('序列编号')

    % 计算平均值

    figure;
    bar([mean(qs_ori),mean(s2)]);
    set(gca,'xtick',1:2,'xticklabel',{'原始信号','趋势项消除'});
    disp(['原始信号, ','趋势项消除']);
    disp([mean(qs_ori),mean(s2)]);

    %% 输出结果至xls
    % JKLM
%     COLNAME = 'A';
    RANGE = [COLNAME,'2:',COLNAME,num2str(length(s2)+1)];
    xlswrite(['Prep_',Filename],s2,RANGE);
end






