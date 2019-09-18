clear;close all;
% ����pyexcel�Ĵ���


[Filename, Foldname]=uigetfile({'*.xls'},"ѡ��XLS");       
% ��һ�У� ��쳵��������/��ֵ��IFFT ��������
% �ڶ��У� LSTM��������
% �����У� IFFT�ߵ�����


excelpath = [Foldname,Filename];
data_ori = xlsread(excelpath); %nx2
% �� ��
%IFFT ����
tru = data_ori(1:end,1);
% Ԥ�����
predt = data_ori(1:end,2);
% ԭʼֵ
longi = data_ori(1:end,2);


figure;
plot(tru);
hold on;
plot(predt);
hold on;
plot(longi);
legend('IFFT����','LSTM', 'ԭʼ�ߵ�');
ylabel('��ֵ / mm')



%% ר������excel�ļ����׷���
% ����
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
title('�����ƽ˳��')
legend('ԭʼ�ź�','LSTM');
ylabel('PSD / mm^2/(1/m)')
xlim([0.01, 1]);

%% ��ֵ�������λ��
d_min = min(data_ori);
d_max = max(data_ori);
d_mean = mean(data_ori);
d_std = std(data_ori);
d_var = var(data_ori);
d_median = median(data_ori);

%%  CDF�ۼƷֲ�ͼ����

figure;
cdfplot(tru);
hold on;
cdfplot(predt);
legend('��ֵ','LSTM');


%% ƽ���Բ��� 
% adftest( )
% [h_adf1,pValue1,stat1,cValue1,reg1]
adfresults = zeros(3, 4);
[adfresults(2,1),adfresults(2,2),adfresults(2,3),...
    adfresults(2,4)] = adftest(data_ori(:,1));
[adfresults(3,1),adfresults(3,2),adfresults(3,3),...
    adfresults(3,4)] = adftest(data_ori(:,2));


%% ��������bar-curve

figure;
h = histfit(1000*data_ori(:,1),30,'normal');
title('IFFT results')
figure;
histfit(1000*data_ori(:,2),30,'normal');
title('LSTM results')


%% MIC���Լ���
mine_results1 = mine(tru', longi');
mine_results2 = mine(predt', longi');

%% pearson ���ϵ��
pearsonValue1 = corr(tru, longi);
pearsonValue2 = corr(predt, longi);
plot(longi);hold on;plot(predt)







% 'XTickLabel'
% set(get(h(2),'ylabel'),'string', 'Ƶ��')
% set(h(1), 'XTickLabel', 'Ƶ��'); 
% ylabel('Ƶ��')
% xlabel('��ֵ / mm')


% %% ������̬�ֲ�ͼ
% data = predt; %�������һ��1000��1�еķ��ӱ�׼��̬�ֲ��������random normal distribution
% [counts,centers] = hist(data, 15);%����ָ�����ݣ��ض��������7���ĸ����ָ�������������м�ֵ
% % ��Ƶ�ʷֲ�ֱ��ͼ
% figure;
% bar(centers, counts / sum(counts));%��ֱ��ͼ��x�Ǹ��������м�ֵ��y�Ƕ�Ӧ�ĸ��ʣ�������/��������
% % �ֲ��������
% [mu,sigma]=normfit(data); %��������data����̬�ֲ���������ֵ����ֵ�̺ͱ�׼�����
% % ����֪�ֲ��ĸ����ܶ�����
% x1 = min(data):0.0001:max(data);%��ȡ�ܼ���x�������
% y1 = pdf('Normal', x1, mu,sigma, 'LineWidth', 10);%probability density functions�����ܶȺ�����
% y2 = normpdf(x1, mu,sigma);
% hold on;
















