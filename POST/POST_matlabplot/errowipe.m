function [value_ti] =errowipe(six_origin,n,cor)

%�������������쳣ֵ�޳� 
%1��ԭʼ����
%2������ά��
%��쳵0.25mһ�����ݵ�


for i1 = 1:1:n
    
    %��������
    x2 = six_origin(:,i1);
    x2_u = six_origin(:,i1);

    %�����޳��㷨�Ĵ���
    cishu = 1;
    %���Ȳ���cor = 0.003 Ĭ��
    for i = 1:1:length(x2_u)

        for j = i+2:2:length(x2_u)

            cha = abs(x2_u((i+j)/2)-((x2_u(i)+x2_u(j))/2));

            if cha>abs(i-j)*250*cor*0.5

                x2_u((i+j)/2)=(x2_u(i)+x2_u(j))/2;
                cishu = cishu + 1;
                %�����޳��㷨�������
                % dot(cishu)=(i+j)*0.5;
            end
        end
    end
    %��ͼ��
    value_ti (:,i1) = x2_u;
end