function [C,L] = xiaofilter(signal,wname,level)

% 小波分析函数
% model为小波基的名称
% level 为小波分析的层级
% signal 为待处理的信号
% C = dec
% [com , L] = xiaofilter(value_ti(:,1),'bior4.4',6);


[C,L] = wavedec(signal,level,wname);


