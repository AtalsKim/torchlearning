function [C,L] = xiaofilter(signal,wname,level)

% С����������
% modelΪС����������
% level ΪС�������Ĳ㼶
% signal Ϊ��������ź�
% C = dec
% [com , L] = xiaofilter(value_ti(:,1),'bior4.4',6);


[C,L] = wavedec(signal,level,wname);


