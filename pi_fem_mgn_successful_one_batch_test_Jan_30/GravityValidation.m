run("gmsh_nen_25_numel_16.m")
[x, y, LM, IX, ID, srlist, edgesNodesMatrix, numnp, ndf, numel, nen, numed] = processGmsh(msh);

plotMesh(x, y, srlist)

E = 7; % Youngs modulus, Pa
nu = 0.25; % Poissons ratio, no units
g = -0.1; % Gravitational constant
rho = 1; % Material density
D = D_mat(E, nu); % material matrix  - plane stress

nodesLeftBdry = find(x == 0);
fixed_nodes = [nodesLeftBdry; nodesLeftBdry]; % Fixed left edge nodes
f_x = [0; g*rho]; % global body load

name = "XLeft_YLeft.mat";
load(name, "femResults")
intDisp = interpolateDisplacement(femResults, x, y);
u_exact = intDisp.ux';
v_exact = intDisp.uy';

[K, F_uu, K_uu] = globalAssembly(numnp, ndf, numel, x, y, IX, LM, f_x, D, fixed_nodes, ID);

[u_fem, v_fem, x_def_fem, y_def_fem] = solveFEM(ID, fixed_nodes, numnp, ndf, K, F_uu, x, y);

error = 100*sqrt((u_fem - u_exact).^2 +  (v_fem - v_exact).^2)/max(intDisp.Magnitude);
errorMax = max(error);

plotMesh(x_def_fem, y_def_fem, srlist);

sqR = errorRelative(u_exact, v_exact, u_fem, v_fem);

% Simulate residual calculation
UV_exact = reshape([u_exact; v_exact], [], 1);
UV_fem = reshape([u_fem; v_fem], [], 1);

R = K_uu*UV_fem - F_uu; % checks out now!