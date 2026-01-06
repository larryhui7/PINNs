L = 1; H = 1;
E = 5; % Young's modulus
nu = 0.3; % Poisson's ratio

model = createpde("structural","static-planestress");
rect = [3,4,0,L,L,0,0,0,H,H]';
g = decsg(rect);
geometryFromEdges(model, g);
generateMesh(model, "Hmax", 0.05);

figure
pdegplot(model, "VertexLabels", "on")
hold on
pdegplot(model, "EdgeLabels", "on")
hold on
pdemesh(model)

structuralProperties(model, "YoungsModulus", E, "PoissonsRatio", nu);

% Edge 1: Allow movement in x, fix in y
structuralBC(model, "Edge", 1, "YDisplacement", 0);

% Edge 4: Allow movement in y, fix in x
structuralBC(model, "Edge", 4, "XDisplacement", 0);

% Apply only in the x direction
structuralBoundaryLoad(model, "Edge", 2, ...
    "SurfaceTraction", @(location, state) [cos(pi * location.y/(2)); zeros(size(location.x))]);

% Solve and plot
result = solve(model);
figure
pdeplot(model, "XYData", result.Displacement.ux, "Deformation", result.Displacement, ColorMap="jet");
title('Displacement in x-direction')
colorbar
