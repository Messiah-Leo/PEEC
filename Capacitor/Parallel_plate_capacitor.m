clear
clc
% 参数设定
L = 10e-3; % 极板单位边长，单位：米
Lx = L;
Ly = L;
Epsilon = 8.854e-12; % 真空介电常数，单位：法/米
d_values = 4e-3; % 极板间的单位距离，单位：米
division = 20; % 将极板划分为 division x division 个小单元
delta = 2*Lx/division; % 每个单元的边长

% V矩阵，代表上下两个极板的电位分布，1为上极板，-1为下极板
V = vertcat(ones([division^2,1]), (-1)*ones([division^2,1]));

% 初始化P矩阵
P = zeros([2*division^2,2*division^2]);

% 计算小单元之间的距离并填充P矩阵
for p_index = 1:2*division^2
    z = floor((p_index-1) / (division^2)); % 判断在上极板(0)或下极板(1)
    y = floor((p_index-1-z*division^2) / division)+1; % 计算y坐标
    x = p_index-z*division^2-(y-1)*division; % 计算x坐标
    for r_z = 0:1
        for r_y = 1:division
            for r_x = 1:division
                % 计算两个小单元的距离R
                R = sqrt((delta*abs(x-r_x)).^2 + (delta*abs(y-r_y)).^2 + (d_values*2*abs(r_z-z)).^2);
                r_index = r_z*division^2 + (r_y-1)*division + r_x; % 计算目标单元的索引
                if (R == 0)
                    % 如果R为0，处理自相互作用项
                    P(p_index,r_index) = 2*delta*0.8814/(pi*Epsilon);
                else
                    % 否则直接用中心点电荷作近似处理
                    P(p_index,r_index) = delta^2 / (4*pi*Epsilon*R);
                end
            end
        end
    end
end

% 求解线性方程组，得到各个单元的电荷分布
sol = linsolve(P,V);

% 设置绘图网格
[x, y] = meshgrid(1:division, 1:division);
x = delta.*(x-0.5)-Lx; % 坐标转换
y = delta.*(y-0.5)-Ly;
z_upper = d_values*ones(division,division); % 上极板的z坐标
charge_upper = reshape(sol(1:division^2)./delta^2,division,division); % 上极板的电荷分布

% 绘制上极板的电荷分布
figure;
surf(x, y, z_upper,charge_upper); % 上极板电荷的三维表面图
shading interp; % 使表面平滑
hold on;

% 绘制下极板的电荷分布（取负号）
surf(x, y, -z_upper,-charge_upper); % 下极板电荷的三维表面图
shading interp; % 使表面平滑
hold off;

% 设置坐标轴标签
xlabel('x'),ylabel('y'),zlabel('z')
colormap jet; % 设置颜色图为jet模式
colorbar; % 显示颜色条

% 定义x和z方向的范围，用于计算空间中的电场
division2 = 40; % 空间点划分
x_r = linspace(-3*L, 3*L, division2); % x方向的范围
z_r = linspace(-2*d_values, 2*d_values, division2); % z方向的范围

% 生成用于电场计算的网格点
[X, Z] = meshgrid(x_r, z_r);
P2 = zeros(length(x_r)*length(z_r),2*division^2); % 初始化P2矩阵，用于存储空间点与极板上单元的距离

% 计算空间点与上极板的距离并填充P2矩阵
for p_index = 1:division2^2
    for r_index = 1:division^2
        % 计算空间点与上极板单元的距离R
        R = sqrt(abs(X(p_index)-x(r_index)).^2 + abs(y(r_index)).^2 + abs(Z(p_index)-z_upper(r_index)).^2);
        if (R == 0)
            % 处理自相互作用
            P2(p_index,r_index) = 2*delta*0.8814/(pi*Epsilon);
        else
            % 根据库仑定律计算影响系数
            P2(p_index,r_index) = delta^2 / (4*pi*Epsilon*R);
        end
    end
    
    % 计算空间点与下极板的距离并填充P2矩阵
    for r_index = division^2+1:2*division^2
        % 计算空间点与下极板单元的距离R
        R = sqrt(abs(X(p_index)-x(r_index-division^2)).^2 + abs(y(r_index-division^2)).^2 + abs(Z(p_index)+z_upper(r_index-division^2)).^2);
        if (R == 0)
            P2(p_index,r_index) = 2*delta*0.8814/(pi*Epsilon);
        else
            P2(p_index,r_index) = delta^2 / (4*pi*Epsilon*R);
        end
    end
end

% 计算空间中电势分布
P_p = reshape(P2*sol,division2,division2);

% 计算电势的梯度，即电场
[px,py] = gradient(P_p);

% 绘制电势的等高线图和电场的矢量场
figure
contour(X,Z,P_p,20) % 绘制电势的等高线
hold on
quiver(x_r,z_r,px,py) % 绘制电场的矢量场
hold off
