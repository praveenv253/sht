function plot_grid(f, thetas, phis)

	x = sin(thetas) .* cos(phis);
	y = sin(thetas) .* sin(phis);
	z = cos(thetas);

	if nargin == 0
		scatter3(x, y, z, 20, 'b', 'filled');
	else
		scatter3(x, y, z, 20, real(f), 'filled');
	end

	%hold on;

	% Underlying sphere
	%vw = [70 25];
	%res = 201;
	%phi = linspace(0, 2 * pi, res);
	%theta = linspace(0, pi, ceil(res/2));
	%[Phi, Theta] = meshgrid(phi, theta);

	%r = 0.98;
	%X = r .* sin(Theta) .* cos(Phi);
	%Y = r .* sin(Theta) .* sin(Phi);
	%Z = r .* cos(Theta);

	%surf(X, Y, Z, 'g');
	%colormap(gray);
	%caxis([0, 1]);
	%shading interp;
	%daspect([1 1 1]);
	%axis tight;
	%view(vw);
end
