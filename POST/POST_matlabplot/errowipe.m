function [value_ti] =errowipe(six_origin,n,cor)

%进行逐点遍历的异常值剔除 
%1：原始数据
%2：数据维度
%轨检车0.25m一个数据点


for i1 = 1:1:n
    
    %备份数据
    x2 = six_origin(:,i1);
    x2_u = six_origin(:,i1);

    %进行剔除算法的次数
    cishu = 1;
    %精度参数cor = 0.003 默认
    for i = 1:1:length(x2_u)

        for j = i+2:2:length(x2_u)

            cha = abs(x2_u((i+j)/2)-((x2_u(i)+x2_u(j))/2));

            if cha>abs(i-j)*250*cor*0.5

                x2_u((i+j)/2)=(x2_u(i)+x2_u(j))/2;
                cishu = cishu + 1;
                %进行剔除算法点的坐标
                % dot(cishu)=(i+j)*0.5;
            end
        end
    end
    %绘图区
    value_ti (:,i1) = x2_u;
end