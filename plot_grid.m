L = 20;
load(sprintf('grid_L%d.mat', L));

x = sin(thetas) .* cos(phis);
y = sin(thetas) .* sin(phis);
z = cos(thetas);

scatter3(x, y, z, 20, 'b', 'filled');
hold on;

% Underlying sphere
vw = [70 25];
res = 201;
phi = linspace(0, 2 * pi, res);
theta = linspace(0, pi, ceil(res/2));
[Phi, Theta] = meshgrid(phi, theta);

r = 0.98;
X = r .* sin(Theta) .* cos(Phi);
Y = r .* sin(Theta) .* sin(Phi);
Z = r .* cos(Theta);

surf(X, Y, Z, 0.8 * ones(size(X)));
colormap(gray);
caxis([0, 1]);
shading interp;
daspect([1 1 1]);
axis tight;
view(vw);
