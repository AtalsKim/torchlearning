clear;close all;
% 用于pyexcel的处理


[Filename, Foldname]=uigetfile({'*.xls'},"选择XLS");       
% 第一列： 轨检车轨向数据/真值，IFFT 轨向数据
% 第二列： LSTM轨向数据
% 第三列： IFFT高低数据


excelpath = [Foldname,Filename];
data_ori = xlsread(excelpath); %nx2
% 真 假
%IFFT 轨向
tru = data_ori(1:end,1);
% 预测轨向
predt = data_ori(1:end,2);
% 原始值
longi = data_ori(1:end,2);


figure;
plot(tru);
hold on;
plot(predt);
hold on;
plot(longi);
legend('IFFT轨向','LSTM', '原始高低');
ylabel('幅值 / mm')



%% 专门用于excel文件的谱分析
% 窗长
sampN = 1000;
fs = 4;
window = hann(sampN);
[Pxx, f] = pwelch(predt, window, sampN/2,sampN...
    ,fs, 'onesided');
[Pxx2, f2] = pwelch(tru, window, sampN/2,sampN...
    ,fs, 'onesided');

figure;
loglog(f, Pxx);
hold on;
loglog(f2, Pxx2);
title('轨道不平顺谱')
legend('原始信号','LSTM');
ylabel('PSD / mm^2/(1/m)')
xlim([0.01, 1]);

%% 均值、方差、中位数
d_min = min(data_ori);
d_max = max(data_ori);
d_mean = mean(data_ori);
d_std = std(data_ori);
d_var = var(data_ori);
d_median = median(data_ori);

%%  CDF累计分布图绘制

figure;
cdfplot(tru);
hold on;
cdfplot(predt);
legend('真值','LSTM');


%% 平稳性测试 
% adftest( )
% [h_adf1,pValue1,stat1,cValue1,reg1]
adfresults = zeros(3, 4);
[adfresults(2,1),adfresults(2,2),adfresults(2,3),...
    adfresults(2,4)] = adftest(data_ori(:,1));
[adfresults(3,1),adfresults(3,2),adfresults(3,3),...
    adfresults(3,4)] = adftest(data_ori(:,2));


%% 单纯绘制bar-curve

figure;
h = histfit(1000*data_ori(:,1),30,'normal');
title('IFFT results')
figure;
histfit(1000*data_ori(:,2),30,'normal');
title('LSTM results')


%% MIC测试计算
mine_results1 = mine(tru', longi');
mine_results2 = mine(predt', longi');

%% pearson 相关系数
pearsonValue1 = corr(tru, longi);
pearsonValue2 = corr(predt, longi);
plot(longi);hold on;plot(predt)







% 'XTickLabel'
% set(get(h(2),'ylabel'),'string', '频数')
% set(h(1), 'XTickLabel', '频数'); 
% ylabel('频数')
% xlabel('幅值 / mm')


% %% 绘制正态分布图
% data = predt; %随机产生一个1000行1列的服从标准正态分布的随机数random normal distribution
% [counts,centers] = hist(data, 15);%返回指定数据，特定间隔（如7）的各个分割区间的数量和中间值
% % 画频率分布直方图
% figure;
% bar(centers, counts / sum(counts));%画直方图，x是各个区间中间值，y是对应的概率（区间数/总数量）
% % 分布参数拟合
% [mu,sigma]=normfit(data); %计算数据data的正态分布参数特征值，均值μ和标准方差δ
% % 画已知分布的概率密度曲线
% x1 = min(data):0.0001:max(data);%获取密集的x间隔数据
% y1 = pdf('Normal', x1, mu,sigma, 'LineWidth', 10);%probability density functions概率密度函数，
% y2 = normpdf(x1, mu,sigma);
% hold on;
















