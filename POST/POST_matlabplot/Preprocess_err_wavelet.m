clear;close all;
% ����pyexcel�Ĵ���
% ���ڶ����ݼ���Ԥ����
% �����Ǳ�׼�Ĺ������ 10,12 �Ǹߵ�
% �����xls�ļ�,������Ϣ����

[Filename, Foldname]=uigetfile({'*.xls';'*.xlsx'},"ѡ��XLS");       
% ��һ�У� ��쳵��������/��ֵ��IFFT ��������
% �ڶ��У� LSTM��������
% �����У� IFFT�ߵ�����

excelpath = [Foldname,Filename];
data_ori = xlsread(excelpath,'J:M'); %nx2
% ɾ������
dataori = deleteZline(data_ori);
% JKLM
preprocessErrWlet(data_ori(1:end,1),'J',Filename);
preprocessErrWlet(data_ori(1:end,2),'K',Filename);
preprocessErrWlet(data_ori(1:end,3),'L',Filename);
preprocessErrWlet(data_ori(1:end,4),'M',Filename);






function dataori = deleteZline(dataori)

% ȥ������
% dataori = [1,2,3,4,0,0,0,0,0,2,0,0,0,1];

% ɾ����������������ֵ
k = find(diff([0,dataori(:,2)'])).';
g = find(k(2:2:end)- k(1:2:end)>=3);
% ��Ҫɾ������β
dellist = [k(2*g-1),k(2*g)-1];
delnum = [];
for i = 1:1:length(dellist(:,1))
    delnum = [delnum, dellist(i,1):1:dellist(i,2)];
end
% ɾ����
dataori(delnum) = [];
end




function  preprocessErrWlet(dataori,COLNAME,Filename)

    %% �쳣ֵ����
    % ��������������0.003
    n = 1;
    err_ori = dataori;
    [value_ti] =errowipe(err_ori,n, 0.003);

    figure;
    plot(err_ori);
    hold on;
    plot(value_ti);
    legend('ԭʼ�ź�','�쳣ֵ����');
    ylabel('��ֵ / mm')
    xlabel('���б��')

    %% ����������

    qs_ori = dataori;
    
    % ��߲���Ϊ0.25m 
    % �˵��ĳ���Ϊ 0.25*2^n   1-0.5 2-1     9-128 8-64 
    % ����350 �Ŀ��Ʋ���Ϊ120m
    % ��ȥС����������
    
    level = 8;
    [C,L] = wavedec(qs_ori,level,'bior4.4');
    % ��ȡ��Ƶϵ��
    ca8 = appcoef(C,L,'bior4.4',level);

    % ��Ƶ�ź�
    
    cd8 = detcoef(C,L,8);
    cd7 = detcoef(C,L,7);
    cd6 = detcoef(C,L,6);
    cd5 = detcoef(C,L,5);
    cd4 = detcoef(C,L,4);
    cd3 = detcoef(C,L,3);
    cd2 = detcoef(C,L,2);
    cd1 = detcoef(C,L,1);


    % ��������Ƶ�ź�
    ca8 = zeros(length(ca8),1);


    % �ؽ�ϵ������
    c1 = [ca8;cd8;cd7;cd6;cd5;cd4;cd3;cd2;cd1];
    % ����ϵ���ع�
    s2 = waverec(c1,L,'bior4.4');

    figure;
    plot(qs_ori);
    hold on;
    plot(s2);
    legend('ԭʼ�ź�','����������');
    ylabel('��ֵ / mm')
    xlabel('���б��')

    % ����ƽ��ֵ

    figure;
    bar([mean(qs_ori),mean(s2)]);
    set(gca,'xtick',1:2,'xticklabel',{'ԭʼ�ź�','����������'});
    disp(['ԭʼ�ź�, ','����������']);
    disp([mean(qs_ori),mean(s2)]);

    %% ��������xls
    % JKLM
%     COLNAME = 'A';
    RANGE = [COLNAME,'2:',COLNAME,num2str(length(s2)+1)];
    xlswrite(['Prep_',Filename],s2,RANGE);
end






