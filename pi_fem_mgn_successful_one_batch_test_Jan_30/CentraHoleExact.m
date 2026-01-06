run("gmsh_CentralHolePlate.m")
[x, y, LM, IX, ID, srlist, edgesNodesMatrix, numnp, ndf, numel, nen, numed] = processGmsh(msh);
plotMesh(x, y, srlist)

model = createpde(2);  % 2D Problem so u,v components
xyglobe = [x; y];  
conn = IX'; 
msh_to_pde = geometryFromMesh(model, xyglobe, conn);

% Material properties (example: steel)
E = 200e9;      % Young's modulus (Pa)
nu = 0.3;       % Poisson's ratio
rho = 7850;     % Density (kg/m³)

structuralProperties(model, 'YoungsModulus', E, ...
                            'PoissonsRatio', nu, ...
                            'MassDensity', rho);

% Identify boundaries (edges) from your Gmsh mesh + Fix left edge (edge 1)
leftEdge = srlist{1}; 
structuralBC(model, 'Edge', leftEdge, 'Constraint', 'fixed');

% Apply traction/pressure on the right edge (edge 2)
rightEdge = srlist{2};  
traction = 1e6;      
structuralBoundaryLoad(model, 'Edge', rightEdge, 'SurfaceTraction', [traction; 0]);

% Generate mesh
generateMesh(model, 'Hmax', 0.1);  % Adjust `Hmax` for refinement

% Solve the static structural problem
result = solve(model);

% Extract displacements
u = result.NodalSolution(:,1);  % x-displacements
v = result.NodalSolution(:,2);  % y-displacements

% Combine into FEM solution vector (U_FEM)
U_FEM = [u; v];

figure;
pdeplot(model, 'XYData', result.NodalSolution(:,1), 'ColorMap', 'jet');
title('FEM Solution: x-Displacement');
axis equal;

figure;
pdeplot(model, 'XYData', result.NodalSolution(:,2), 'ColorMap', 'jet');
title('FEM Solution: y-Displacement');
axis equal;
