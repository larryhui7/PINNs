clc; clear; close all

%% Load ".m" mesh file exported from "gmsh.exe" software
run("gmsh_nen_25_numel_16.m")
%run("gmsh_nen_337_numel_304.m")
%run("gmsh_centralHolePlate.m")
%% Process Gmsh to get node positions and connectivity data
[x, y, LM, IX, ID, srlist, edgesNodesMatrix, numnp, ndf, numel, nen, numed] = processGmsh(msh);

%% Plot mesh
plotMesh(x, y, srlist)

%% Assign physical properties to the mesh - see if can you more realistic values like E = 200e8 Pa
E = 7; % Youngs modulus, Pa
nu = 0.25; % Poissons ratio, no units
g = -0.1; % Gravitational constant
rho = 1; % Material density
D = D_mat(E, nu); % material matrix  - plane stress

%% Assign boundary conditions and body load: x and y directions separately
nodesLeftBdry = find(x == 0);
fixed_nodes = [nodesLeftBdry; nodesLeftBdry]; % Fixed left edge nodes
f_x = [0; g*rho]; % global body load

%% Pull more precise solution for the given mesh: interpolation from more fine triangular FEM mesh
% GAval = -0.1;
% W = 1;
% L = 1;
% YoungsModulus = 7;
% nu = 0.25;
% rho = 1;
name = "XLeft_YLeft.mat";
load(name, "femResults")
intDisp = interpolateDisplacement(femResults, x, y);
u_exact = intDisp.ux';
v_exact = intDisp.uy';

%% Get global stiffness matrix and force vector
[K, F_uu, K_uu] = globalAssembly(numnp, ndf, numel, x, y, IX, LM, f_x, D, fixed_nodes, ID);

%% Solve the system in FEM: validation step
[u_fem, v_fem, x_def_fem, y_def_fem] = solveFEM(ID, fixed_nodes, numnp, ndf, K, F_uu, x, y);

error = 100*sqrt((u_fem - u_exact).^2 +  (v_fem - v_exact).^2)/max(intDisp.Magnitude);
errorMax = max(error);

plotMesh(x_def_fem, y_def_fem, srlist);

sqR = errorRelative(u_exact, v_exact, u_fem, v_fem);

% Simulate residual calculation
UV_exact = reshape([u_exact; v_exact], [], 1);
UV_fem = reshape([u_fem; v_fem], [], 1);

R = K_uu*UV_fem - F_uu; % checks out now!

%% Format inputs into the training code
% Node features information 
dirMaskX = ismember(1:numnp, fixed_nodes(1,:));
dirMaskY = ismember(1:numnp, fixed_nodes(2,:));

%maskULoss = double(ismember(1:numnp, [3, 10 , 9, 8, 2]));
%maskULoss = gpuArray(dlarray(reshape([maskULoss; maskULoss], [], 1)));

dirMaskRev = reshape([double(~dirMaskX); double(~dirMaskY)], [], 1);

% Initial conditions 
Ux0 = zeros(1, numnp); % initial displacement x direction
Uy0 = zeros(1, numnp); % initial displacement y direction 
Vx0 = zeros(1, numnp); % initial velocity x direction
Vy0 = zeros(1, numnp); % initial velocity y direction

%nodeAttr = gpuArray(dlarray([x; y; dirMaskX; dirMaskY], "CUB")); % this node features - play around best inputs 
nodeAttr = (dlarray([x; y; Ux0; Uy0; Vx0; Vy0], "CUB"));

% Edge features calculation
loc_i = nodeAttr(1:2, srlist(1,:));
loc_j = nodeAttr(1:2, srlist(2,:));
direc = (loc_i-loc_j);
direc_norm = vecnorm(direc);

% could make all the below gpu array
edgeAttr = (dlarray([direc; direc_norm], "CUB"));
edgeMask = (dlarray(ones(1, size(srlist,2)), "CUB"));
srlist = (dlarray(srlist, "CUB"));
edgesNodesMatrix = (dlarray(edgesNodesMatrix, "CUB"));
nodeMask = (dlarray(ones(1, numnp), "CUB"));

dirVecU = (dlarray(zeros(numnp, 1)));
dirVecV = (dlarray(zeros(numnp, 1)));
K_in = (dlarray(K_uu));
F_in = (dlarray(F_uu));
UV_fem_in = (dlarray(UV_fem));

u_std = std(u_exact);
v_std = std(v_exact);
u_mean = mean(u_exact);
v_mean = mean(v_exact);

%% Training parameters - hyperparameter training - see what makes things faster
initLearnRate = 0.001; % initial learning rate
learnRateDecay = 0.001; % learning rate decay
epochs = 1e5; % max epochs
gradDecay = 0.9; % weights gradient decay (MATLAB default is 0.9)
sqGradDecay = 0.9; % weights square gradient decay (MATLAB default is 0.999)

%% Generate training monitor
monitor_single = trainingProgressMonitor; % initate trainition monitor

% Set metrics
monitor_single.Metrics = ["LossTotal", "sqR_U", "sqR_V"];
groupSubPlot(monitor_single, "Loss","LossTotal");
groupSubPlot(monitor_single, "sqR", ["sqR_U", "sqR_V"]);

% Set axis
yscale(monitor_single,"Loss","log")
monitor_single.XLabel = "Iteration";

% Set progress
monitor_single.Status = "Configuring";
monitor_single.Progress = 0;

% Initiate monitor information
monitor_single.Info = ["Epoch", "LearnRate", "sqR_U", "sqR_V"];

%% Initiate training
iteration = 0;
epoch = 0;
averageGrad = [];
averageSqGrad = [];
learnRate = initLearnRate;
accFun = dlaccelerate(@modelloss);
clearCache(accFun)
epochsToPlot = 1:10:epochs;
epochsToSave = 1:100:epochs;

numNodeFeatures = 6;
numEdgeFeatures = 3;
batchSize = 1;
outputSize = 2;
net = meshGraphNetwork(numNodeFeatures, numEdgeFeatures, numnp, numed, batchSize, outputSize);
numnpGpu = (dlarray(numnp));

%% Perform the training
monitor_single.Status = "Running ADAM Optimizer";
figure
while epoch < epochs && ~monitor_single.Stop
    epoch = epoch + 1;

    iteration = iteration + 1;

    % Evaluate the model gradients and loss
    [loss, gradients, UV_pred] = ...
        dlfeval(accFun,net, nodeAttr, edgeAttr, srlist, edgeMask, edgesNodesMatrix, nodeMask,...
        dirMaskRev, K_in, F_in, numnpGpu);

    % Calculate training error
    u_pred = extractdata(gather(UV_pred(1:2:end)))';
    v_pred = extractdata(gather(UV_pred(2:2:end)))';
    magn = 5;
    x_def = x + magn*u_pred;
    y_def = y + magn*v_pred;
    
    if any(epochsToPlot == epoch)
        curFig = gcf;
        clf(curFig)
        hold on
        scatter(x + magn*u_exact, y + magn*v_exact, 50, "red", "filled")
        scatter(x_def, y_def, 50, "blue", "filled")
        axis equal
    end

    u_error = u_exact - u_pred;
    v_error = v_exact - v_pred;

    sqU = 1 - sum(u_error.^2, "all")/sum(u_exact.^2, "all");
    sqV = 1 - sum(v_error.^2, "all")/sum(v_exact.^2, "all");

    % Update training parameters
    gradients = dlupdate(@gather, gradients); % keep params on CPU

    % Update the network parameters using the ADAM optimizer.
    [net, averageGrad, averageSqGrad] =...
        adamupdate(net, gradients, averageGrad, averageSqGrad, iteration, learnRate, gradDecay, sqGradDecay);

    % Record data in the training monitor
    updateInfo(monitor_single,...
        "Epoch", epoch,...
        "LearnRate", learnRate, ...
        "sqR_U", sqU, ...
        "sqR_V", sqV);
    if any(abs([sqU, sqR]) > 10)
        sqU = 0;
        sqV = 0;
    end 
    recordMetrics(monitor_single, iteration, ...
        "LossTotal", loss, ...
        "sqR_U", sqU,...
        "sqR_V", sqV);
    

    % Update learning rate.
    learnRate = initLearnRate/(1+learnRateDecay*epoch);

    % Update progress percentage.
    monitor_single.Progress = 100*epoch/epochs;
end

function [loss, gradients, UV_pred] =...
    modelloss(net, nodeAttr, edgeAttr, srlist, edgeMask, edgesNodesMatrix, nodeMask,...
    dirMaskRev, K, F, numnp)

% Make prediction
UV_pred = permute(stripdims(net.forward(nodeAttr, edgeAttr, srlist, edgeMask, edgesNodesMatrix, nodeMask)), [1, 3, 2]);

% Apply Dirichlet BC directly: set displacements of fixed coordinates to
% zero
UV_pred = reshape(UV_pred, [], 1).*dirMaskRev;

% Calculate the residual and loss: [K_uu]*{u} - {F_uu} = R
loss = sum((K*UV_pred - F).^2, "all")/numnp;

% Evaluate gradients
gradients = dlgradient(loss, net.Learnables);
end