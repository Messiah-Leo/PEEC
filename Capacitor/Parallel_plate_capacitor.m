clear
clc
% 参数设定
L = 10e-3; % 单位：米
Lx = L;
Ly = L;
Epsilon = 8.854e-12; % 真空中的介电常数，单位：法/米
d_values = 4e-3; % d的取值范围，单位：米
division = 20; % 将极板划分
delta = 2*Lx/division; % 单元格的边长

V = vertcat(ones([division^2,1]), (-1)*ones([division^2,1]));
P = zeros([2*division^2,2*division^2]);
for p_index = 1:2*division^2
    z = floor((p_index-1) / (division^2));
    y = floor((p_index-1-z*division^2) / division)+1;
    x = p_index-z*division^2-(y-1)*division;
    for r_z = 0:1
        for r_y= 1:division
            for r_x = 1:division
                R = sqrt((delta*abs(x-r_x)).^2 + (delta*abs(y-r_y)).^2 + (d_values*2*abs(r_z-z)).^2);
                r_index = r_z*division^2 + (r_y-1)*division + r_x;
                if (R == 0)
                    P(p_index,r_index) = 2*delta*0.8814/(pi*Epsilon);
                else
                    P(p_index,r_index) = delta^2 / (4*pi*Epsilon*R);
                end
            end
        end
    end
end

sol = linsolve(P,V);
[x, y] = meshgrid(1:division, 1:division);
x = delta.*(x-0.5)-Lx;
y = delta.*(y-0.5)-Ly;
z_upper = d_values*ones(division,division);
charge_upper = reshape(sol(1:division^2)./delta^2,division,division);
figure;
surf(x, y, z_upper,charge_upper);
shading interp;
hold on;
surf(x, y, -z_upper,-charge_upper);
shading interp;
hold off;
xlabel('x'),ylabel('y'),zlabel('z')
colormap jet;
colorbar;


% Define the range for x and z
division2 = 40;
x_r = linspace(-3*L, 3*L, division2);
z_r = linspace(-2*d_values, 2*d_values, division2);

% Create a grid of points
[X, Z] = meshgrid(x_r, z_r);
P2 = zeros(length(x_r)*length(z_r),2*division^2);

for p_index = 1:division2^2
    for r_index = 1:division^2
                R = sqrt(abs(X(p_index)-x(r_index)).^2 + abs(y(r_index)).^2 + abs(Z(p_index)-z_upper(r_index)).^2);
                if (R == 0)
                    P2(p_index,r_index) =2*delta*0.8814/(pi*Epsilon);
                else
                    P2(p_index,r_index) = delta^2 / (4*pi*Epsilon*R);
                end
    end
    for r_index = division^2+1:2*division^2
                R = sqrt(abs(X(p_index)-x(r_index-division^2)).^2 + abs(y(r_index-division^2)).^2 + abs(Z(p_index)+z_upper(r_index-division^2)).^2);
                if (R == 0)
                    P2(p_index,r_index) =2*delta*0.8814/(pi*Epsilon);
                else
                    P2(p_index,r_index) = delta^2 / (4*pi*Epsilon*R);
                end
    end
end

P_p = reshape(P2*sol,division2,division2);
[px,py] = gradient(P_p);
% Plot the vector field
figure
contour(X,Z,P_p,20)
hold on
quiver(x_r,z_r,px,py)
hold off