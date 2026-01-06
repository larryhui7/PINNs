clc
clear 
close all

%% Load ".m" mesh file exported from "gmsh.exe" software
% run("gmsh_nen_25_numel_16.m")
% run("gmsh_nen_337_numel_304.m")
 run("gmsh_tri_nen_61_numed_216.m")
% run("gmsh_nen_337_numel_304.m")
padNds = 350; 
padEds = 1500;
%% Process Gmsh to get node positions and connectivity data
[x, y, LM, IX, ID, srlist, edgesNodesMatrix, numnp, ndf, numel, nen, numed] = processGmsh(msh);
edgesToPlot = srlist(:, 1:size(srlist, 2)/2);

%% Assign physical properties to the mesh
E = 7; % Youngs modulus, Pa
nu = 0.25; % Poissons ratio, no units
g = -0.1; % Gravitational constant
rho = 1; % Material density
D = D_mat(E, nu); % material matrix

%% Assign boundary conditions and body load: x and y directions separately
nodesLeftBdry = find(x == 0);
fixed_nodes = [nodesLeftBdry; nodesLeftBdry]; % Fixed left edge nodes
f_x = [0; g*rho]; % global body load

%% Get global stiffness matrix and force vector
fixed_x_dof = ID(1, fixed_nodes(1, :)); fixed_y_dof = ID(2, fixed_nodes(2, :)); fixed_dof = [fixed_x_dof, fixed_y_dof];
free_dof = sort(setdiff(1:numnp*ndf, fixed_dof));

[K, F_uu, K_uu, M, M_uu, M_uu_inv] = globalAssembly(numnp, ndf, numel, x, y, IX, LM, f_x, D, fixed_dof, rho);

%% Get transient FEM results 
ntsteps = 300;
dt = 0.001;
tlist_train = ntsteps*dt;
tlist_eval = 0:dt:(3*ntsteps*dt);
solnTr = solveFEMTransient(free_dof, numnp, ndf, K, F_uu, M, tlist_eval);

%% Format inputs 
% Masks
dirMaskX = ismember(1:numnp, fixed_nodes(1,:)); dirMaskY = ismember(1:numnp, fixed_nodes(2,:));
dirMaskRev = paddata([double(~dirMaskX); double(~dirMaskY)], padNds, Dimension = 2);

% Node features
nodeAttr0 = gpuArray(dlarray(paddata([x; y; zeros(1, numnp); zeros(1, numnp); zeros(1, numnp); zeros(1, numnp)], padNds, Dimension = 2), "CUB"));
nodeAttr = nodeAttr0;

% Edge features: direction and magnitude
XY = [x; y];
sender_nodes = XY(:, srlist(1,:)); receiver_nodes = XY(:, srlist(2,:)); direc = sender_nodes - receiver_nodes;
direc_norm = vecnorm(direc); edgeAttr = gpuArray(dlarray(paddata([direc; direc_norm], padEds, Dimension = 2), "CUB"));

% Other inputs to the MGN: connectivity and masks
edgeMask = gpuArray(dlarray(paddata(ones(1, size(srlist,2)), padEds, Dimension = 2), "CUB"));
srlist = gpuArray(dlarray(paddata(srlist, padEds, Dimension = 2, FillValue = 1), "CUB"));
edgesNodesMatrix = gpuArray(dlarray(paddata(edgesNodesMatrix, [padEds, padNds]), "CUB"));
nodeMask = gpuArray(dlarray(paddata(ones(1, numnp), padNds, Dimension = 2), "CUB"));

%% Perform evaluation
load("netTransient.mat")
M_inv_in = dlarray(paddata(M_uu_inv, 2*[padNds, padNds]));
F_in = dlarray(paddata(F_uu, 2*padNds, Dimension = 1));
K_in = dlarray(paddata(K_uu, 2*[padNds, padNds]));
ux_eval_mat = zeros(padNds, length(tlist_eval)); uy_eval_mat = zeros(padNds, length(tlist_eval));
vx_eval_mat = zeros(padNds, length(tlist_eval)); vy_eval_mat = zeros(padNds, length(tlist_eval));
A_prev = reshape(M_inv_in*F_in, 2, []);

for it = 2:length(tlist_eval)
    % make prediction
    output = permute(stripdims(net.forward(nodeAttr,...
        edgeAttr, srlist, edgeMask, edgesNodesMatrix, nodeMask)), [1, 3, 2]);
    A_pred = output(1:2, :);
    U_prev = nodeAttr(3:4, :);
    V_prev = nodeAttr(5:6, :);
    % convert to displacement
    V_pred = (V_prev + 1/2*(A_prev + A_pred)*dt).*dirMaskRev;
    U_pred = (U_prev + 1/2*(V_prev + V_pred)*dt).*dirMaskRev;
    % store the displacement and velocity
    % adjust the node feature inputs
    ux_eval_mat(:, it) = extractdata(gather(U_pred(1, :)))';
    uy_eval_mat(:, it) = extractdata(gather(U_pred(2, :)))';
    vx_eval_mat(:, it) = extractdata(gather(V_pred(1, :)))';
    vy_eval_mat(:, it) = extractdata(gather(V_pred(2, :)))';
    
    % Adjust node inputs
    nodeAttr(3:4, :) = U_pred;
    nodeAttr(5:6, :) = V_pred;

    % Reset previous acceleration
    uv_prev = reshape(U_pred, [], 1);
    A_prev = reshape(M_inv_in*(F_in - K_in*uv_prev), 2, []);
end

%% Evaluate error
error_ux = solnTr.ux - ux_eval_mat(1:numnp, :);
error_uy = solnTr.uy - uy_eval_mat(1:numnp, :);

R_ux = 1 - sum(error_ux.^2, 1)./sum(solnTr.ux.^2, 1);
R_uy = 1 - sum(error_uy.^2, 1)./sum(solnTr.uy.^2, 1);

%% Plot correlation coefficients
figure 
hold on
xline(tlist_train,"--", "DisplayName", "Training cut-off", "LineWidth", 1.5)
plot(tlist_eval, R_ux, "DisplayName", "R_{ux}", "LineWidth", 3)
plot(tlist_eval, R_uy, "DisplayName", "R_{uy}", "LineWidth", 3)
ylim([0, 1])
xlabel("time [s]")
ylabel("R^2")
grid on
box on
legend("Location", "southeast")
title("Testing on training model: dt = " + num2str(dt))


%% Plot trajectories
formatSpec = '%.3f';
scaleFactor = 10;
x_def_fem = x' + scaleFactor*solnTr.ux; 
x_def_pred = x' + scaleFactor*ux_eval_mat(1:numnp, :); 
x_max = max(x_def_fem, [], "all"); x_min = min(x_def_fem, [], "all");
y_def_fem = y' + scaleFactor*solnTr.uy;
y_def_pred = y' + scaleFactor*uy_eval_mat(1:numnp, :); 
y_max = max(y_def_fem, [], "all"); y_min = min(y_def_fem, [], "all");

femFig = figure;
for n = 1:length(tlist_eval)
    clf(femFig)
    xPlain = [x(edgesToPlot(1,:)); x(edgesToPlot(2,:))];
    yPlain = [y(edgesToPlot(1,:)); y(edgesToPlot(2,:))];

    xPred = [x_def_pred(edgesToPlot(1,:), n), x_def_pred(edgesToPlot(2,:), n)]';
    yPred = [y_def_pred(edgesToPlot(1,:), n), y_def_pred(edgesToPlot(2,:), n)]';

    xFEM = [x_def_fem(edgesToPlot(1,:), n), x_def_fem(edgesToPlot(2,:), n)]';
    yFEM = [y_def_fem(edgesToPlot(1,:), n), y_def_fem(edgesToPlot(2,:), n)]';
    hold on
    plot(xPlain, yPlain, "Color", [0.75, 0.75, 0.75])
    plot(xFEM, yFEM, "k", "LineWidth", 2);
    plot(xPred, yPred, "r --", "LineWidth", 2);
    box on
    grid on
    hold off
    axis equal
    xlim(1.1*[x_min x_max]);
    ylim(1.1*[y_min y_max]);
    title("Timestep: " + num2str(tlist_eval(n), formatSpec) + " s, Scale: " + num2str(scaleFactor))
    subtitle("R_{u_x}^2: " + num2str(R_ux(n), formatSpec) + ", R_{u_y}^2: " + num2str(R_uy(n), formatSpec))
    %drawnow 
    exportgraphics(femFig,'testAnimated.gif','Append',true)
end

