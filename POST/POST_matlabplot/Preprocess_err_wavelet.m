clear;close all;
% 用于pyexcel的处理
% 用于对数据集的预处理
% 输入是标准的轨检数据 10,12 是高低
% 输出是xls文件,基本信息不变

[Filename, Foldname]=uigetfile({'*.xls';'*.xlsx'},"选择XLS");       
% 第一列： 轨检车轨向数据/真值，IFFT 轨向数据
% 第二列： LSTM轨向数据
% 第三列： IFFT高低数据

excelpath = [Foldname,Filename];
data_ori = xlsread(excelpath,'J:M'); %nx2
% 删除零行
dataori = deleteZline(data_ori);
% JKLM
preprocessErrWlet(data_ori(1:end,1),'J',Filename);
preprocessErrWlet(data_ori(1:end,2),'K',Filename);
preprocessErrWlet(data_ori(1:end,3),'L',Filename);
preprocessErrWlet(data_ori(1:end,4),'M',Filename);






function dataori = deleteZline(dataori)

% 去除零行
% dataori = [1,2,3,4,0,0,0,0,0,2,0,0,0,1];

% 删除三个以上连续数值
k = find(diff([0,dataori(:,2)'])).';
g = find(k(2:2:end)- k(1:2:end)>=3);
% 需要删除的首尾
dellist = [k(2*g-1),k(2*g)-1];
delnum = [];
for i = 1:1:length(dellist(:,1))
    delnum = [delnum, dellist(i,1):1:dellist(i,2)];
end
% 删除行
dataori(delnum) = [];
end




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
    
    % 最高波长为0.25m 
    % 滤掉的长波为 0.25*2^n   1-0.5 2-1     9-128 8-64 
    % 高铁350 的控制波长为120m
    % 后去小波基本参数
    
    level = 8;
    [C,L] = wavedec(qs_ori,level,'bior4.4');
    % 获取低频系数
    ca8 = appcoef(C,L,'bior4.4',level);

    % 高频信号
    
    cd8 = detcoef(C,L,8);
    cd7 = detcoef(C,L,7);
    cd6 = detcoef(C,L,6);
    cd5 = detcoef(C,L,5);
    cd4 = detcoef(C,L,4);
    cd3 = detcoef(C,L,3);
    cd2 = detcoef(C,L,2);
    cd1 = detcoef(C,L,1);


    % 消除超低频信号
    ca8 = zeros(length(ca8),1);


    % 重建系数矩阵
    c1 = [ca8;cd8;cd7;cd6;cd5;cd4;cd3;cd2;cd1];
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






