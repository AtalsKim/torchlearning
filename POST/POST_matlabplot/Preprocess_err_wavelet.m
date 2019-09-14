clear;close all;
% ����pyexcel�Ĵ���
% ���ڶ����ݼ���Ԥ����
% �����Ǳ�׼�Ĺ������ 10,12 �Ǹߵ�
% �����xls�ļ�,������Ϣ����

[Filename, Foldname]=uigetfile({'*.xls'},"ѡ��XLS");       
% ��һ�У� ��쳵��������/��ֵ��IFFT ��������
% �ڶ��У� LSTM��������
% �����У� IFFT�ߵ�����

excelpath = [Foldname,Filename];
data_ori = xlsread(excelpath); %nx2
% JKLM
preprocessErrWlet(data_ori(1:end,10),'J',Filename)
preprocessErrWlet(data_ori(1:end,11),'K',Filename)
preprocessErrWlet(data_ori(1:end,12),'L',Filename)
preprocessErrWlet(data_ori(1:end,13),'M',Filename)





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
    % ��ȥС����������
    [C,L] = wavedec(qs_ori,6,'bior4.4');
    % ��ȡ��Ƶϵ��
    ca6 = appcoef(C,L,'bior4.4',6);

    % ��Ƶ�ź�
    cd6 = detcoef(C,L,6);
    cd5 = detcoef(C,L,5);
    cd4 = detcoef(C,L,4);
    cd3 = detcoef(C,L,3);
    cd2 = detcoef(C,L,2);
    cd1 = detcoef(C,L,1);


    % ��������Ƶ�ź�
    ca6 = zeros(length(ca6),1);


    % �ؽ�ϵ������
    c1 = [ca6;cd6;cd5;cd4;cd3;cd2;cd1];
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






