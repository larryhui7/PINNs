clc
close all
clear

%% Reference for the dataset and its 29 cells: this database is generated in the gen_data_simple.m 
% nodeFeatures_pad (1) - "CUB" - node attributes - x-position, y-position, one-hot nodes
% edgeAttr_pad (2) -  "CUB" - edge attributes vector - 3 x numEdges size 
% srlist_pad (3) -  "CUB" - sender-receiver list - 2 x numEdges size
% edgeAttrMask_pad (4) - "CUB"- edgesAttributes mask
% edgesNodesMatrix_pad (5)- "CUB" - edgeNodesMatrix mask
% nodeMask_pad (6), - "CUB" - nodeFeatures mask
% dir_maskRev_X_pad (7) - reverse dirichlet mask 1 when not dirichlet in x direction
% dir_maskRev_Y_pad (8) - reverse dirichlet mask 1 when not dirichlet in y direction
% dir_mask_X_pad (9) - dirichlet mask 1 when dirichlet in x direction
% dir_mask_Y_pad (10) - dirichlet mask 1 when dirichlet in y direction
% pde_X_ux_pad (11) - equilib. mat. in X direction that multiplies x - displacement 
% pde_X_uy_pad (12) - equilib. mat. in X direction that multiplies y - displacement
% pde_X_pad (13) - padding for equilib. mat in X direction
% pde_Y_ux_pad (14) - equilib. mat. in Y direction that multiplies x - displacement
% pde_Y_uy_pad (15) - equilib. mat. in Y direction that multiplies y - displacement 
% pde_Y_pad (16) - padding for equilib. mat in Y direction
% neu_X_ux_pad (17) - neumann bc/traction. mat. in X direction that multiplies x - displacement 
% neu_X_uy_pad (18) - neumann bc/traction. mat. in X direction that multiplies y - displacement 
% neu_X_pad (19) - padding for neumann bc/traction. mat. in X direction
% neu_Y_ux_pad (20) - neumann bc/traction. mat. in Y direction that multiplies x - displacement 
% neu_Y_uy_pad (21) - neumann bc/traction. mat. in Y direction that multiplies Y - displacement 
% neu_Y_pad (22) - padding for neumann boundary condition/traction. mat. in Y direction
% numPdeX (23) - scalar number of equlib nodes in x direction
% numPdeY (24) - scalar number of equlib nodes in y direction
% numNeuX (25) - scalar number of traction bc nodes in x direction
% numNeuY (26) - scalar number of traction bc nodes in y direction
% U_pad(27) - ground truth data x-displacement
% V_pad (28) - ground truth data y-displacement
% nds_pad (29) - nodes x and y positions

%% Input mesh data
load("multiInput.mat") % loads dataset, averages and standard deviation for U and V
ds_read_inputs = read(ds_inputs); % pull 31 cells from the dataset (see above reference)
[numEdges, numNodes] = size(ds_read_inputs{1, 5}); % measure edgesNodesMatrix which is a massive binary matrix which stores all edges and nodes connectivity information
numBatches = size(ds_read_inputs, 1);
numNodeFeatures = size(ds_read_inputs{1, 1}, 1); % should be four: x-position, y-position, x-dirichlet, y-dirichlet
edgeFeatures = size(ds_read_inputs{1, 2}, 1); % should be three: x-direction, y-direction, length
outputSize = 2; % x-displcament, y-displacement
numDsOutputs = size(ds_read_inputs, 2);
numCUB = 6; % number of CUB type inputs into training
magn = 1; % reset deflection multiplier for the validation dataset

%% Define train and validation datasets, pull validation data to plot during the training
val_idx = 9; % choose some batch to be validation
tr_idx = setdiff(1:numBatches, val_idx); % remove validation batch from the rest and redefine as training
ds_val = read(subset(ds_inputs, val_idx)); % split database pull validation database only
U_val_exact = ds_val{27};
V_val_exact = ds_val{28};
U_val_dir = U_val_exact.*ds_val{9}; % only dirichlet exact x displacement
V_val_dir = V_val_exact.*ds_val{10}; % only dirichlet exact y displacement
X_val = ds_val{29}(1,:); % x-position nodes
Y_val = ds_val{29}(2,:); % y-position nodes
SS_val_U = sum(U_val_exact.^2, "all"); % precalculate part of the R squared statistics value 
SS_val_V = sum(V_val_exact.^2, "all"); % precalculate part of the R squared statistics value 
% Convert plain vectors into deep learning arrays -  "dlarray" - used
% within Matlab's "dlnetwork" - these have either no special properties or
% alternatively some x-y-z-etc dimensions can be assigned labels like "C"-
% channel, "U" - unassigned, or "B" batch  - these below are all inputs
% into our custom meshgraphnetwork 
nodeFeatures_val = dlarray(ds_val{1}, "CUB"); 
edgeAttr_val = dlarray(ds_val{2}, "CUB");
srlist_val = dlarray(ds_val{3}, "CUB");
edgeAttrMask_val = dlarray(ds_val{4}, "CUB");
edgesNodesMatrix_val = dlarray(ds_val{5}, "CUB");
nodeMask_val = dlarray(ds_val{6}, "CUB");

%% Define network: see folder dlnetworkMGN for details and additional arguments to overwrite default ones
net = meshGraphNetwork(numNodeFeatures, edgeFeatures, numNodes, numEdges, numBatches, outputSize);

%% Define minibatchque: MATLABs way to keep track of inputs into our network, its useful because we can reset it or shuffle it between separate epochs (complete cycles through all data)
mbqFormat = horzcat(repmat("CUB", 1, numCUB), strings(1, numDsOutputs - numCUB));
mbqTrain = minibatchqueue(subset(ds_inputs, tr_idx), numDsOutputs, MiniBatchSize = 1, MiniBatchFormat = mbqFormat, OutputEnvironment = "gpu");

%% Training parameters
initLearnRate = 1e-4; % initial learning rate
learnRateDecay = initLearnRate/2; % learning rate decay
epochs = 1e6; % max epochs 
gradDecay = 0.9; % weights gradient decay (MATLAB default is 0.9)
sqGradDecay = 0.999; % weights square gradient decay (MATLAB default is 0.999)

%% Generate training monitor
monitor_single = trainingProgressMonitor; % initate trainition monitor

% Set metrics
monitor_single.Metrics = ["LossTotal", "LossEquilibriumX","LossEquilibriumY", "LossTractionX", "LossTractionY", "TrainingSqR_U", "TrainingSqR_V", "ValidationSqR_U", "ValidationSqR_V"];
groupSubPlot(monitor_single, "Loss",["LossTotal", "LossEquilibriumX","LossEquilibriumY", "LossTractionX", "LossTractionY"]);
groupSubPlot(monitor_single, "TrainingSqR", ["TrainingSqR_U", "TrainingSqR_V"]);
groupSubPlot(monitor_single, "ValidationSqR", ["ValidationSqR_U", "ValidationSqR_V"]);

% Set axis
yscale(monitor_single,"Loss","log")
monitor_single.XLabel = "Iteration";

% Set progress
monitor_single.Status = "Configuring";
monitor_single.Progress = 0;

% Initiate monitor information
monitor_single.Info = ["Epoch", "LearnRate", "sqR_U_Train", "sqR_V_Train", "sqR_U_Val", "sqR_V_Val"];

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

%% Perform the training
monitor_single.Status = "Running ADAM Optimizer";
while epoch < epochs && ~monitor_single.Stop
    epoch = epoch + 1;
    % Reset minibatch
    shuffle(mbqTrain)
    for itr = 1:length(tr_idx)
        iteration = iteration + 1;
        % Read mini-batch of data.
        [nodeFeatures, edgeAttr, srlist,...
            edgeAttrMask, edgesNodesMatrix, nodeMask,...
            dir_maskRev_X, dir_maskRev_Y, dir_mask_X, dir_mask_Y, ...
            pde_X_ux, pde_X_uy, pde_X, pde_Y_ux, pde_Y_uy, pde_Y,...
            neu_X_ux, neu_X_uy, neu_X, neu_Y_ux, neu_Y_uy, neu_Y,...
            trPdeX, trPdeY, trNeuX, trNeuY, U_tr_exact, V_tr_exact] = next(mbqTrain);

        % Get Dirichlet data
        nodeMaskPermuted = permute(stripdims(nodeMask), [3,2,1]);
        U_tr_n = (U_tr_exact - U_mean)/U_std.*nodeMaskPermuted;
        V_tr_n = (V_tr_exact - V_mean)/V_std.*nodeMaskPermuted;
        U_tr_dir = U_tr_n.*dir_mask_X;
        V_tr_dir = V_tr_n.*dir_mask_Y;

        % Evaluate the model gradients and loss
        [loss, gradients, residuals, U_pred, V_pred] = ...
            dlfeval(accFun, net, nodeFeatures, edgeAttr, srlist,...
            edgeAttrMask, edgesNodesMatrix, nodeMask,...
            dir_maskRev_X, dir_maskRev_Y, U_tr_dir, V_tr_dir, ...
            pde_X_ux, pde_X_uy, pde_X, pde_Y_ux, pde_Y_uy, pde_Y,...
            neu_X_ux, neu_X_uy, neu_X, neu_Y_ux, neu_Y_uy, neu_Y,...
            trPdeX, trPdeY, trNeuX, trNeuY, U_tr_n, V_tr_n);

        % Calculate training error
        U_tr_pred = (U_pred*U_std + U_mean).*nodeMaskPermuted;
        V_tr_pred = (V_pred*V_std + V_mean).*nodeMaskPermuted;

        U_tr_error = U_tr_exact - U_tr_pred;
        V_tr_error = V_tr_exact - V_tr_pred;

        tr_U_error = 1 - sum(U_tr_error.^2, "all")/sum(U_tr_exact.^2, "all");
        tr_V_error = 1 - sum(V_tr_error.^2, "all")/sum(V_tr_exact.^2, "all");

        % Update training parameters
        gradients = dlupdate(@gather, gradients); % keep params on CPU

        % Update the network parameters using the ADAM optimizer.
        [net, averageGrad, averageSqGrad] =...
            adamupdate(net, gradients, averageGrad, averageSqGrad, iteration, learnRate, gradDecay, sqGradDecay);
        
        % Record data in the training monitor
    recordMetrics(monitor_single, iteration, ...
        "LossTotal", loss, ...
        "LossEquilibriumX", residuals(1),...
        "LossEquilibriumY", residuals(2), ...
        "LossTractionX", residuals(3), ...
        "LossTractionY", residuals(4),...
        "TrainingSqR_U", tr_U_error, ...
        "TrainingSqR_V", tr_V_error);
    updateInfo(monitor_single,...
        "sqR_U_Train", tr_U_error, ...
        "sqR_V_Train", tr_V_error);
    end

    %% Validation
    UV_val = permute(stripdims(net.forward(nodeFeatures_val, edgeAttr_val, srlist_val,...
        edgeAttrMask_val, edgesNodesMatrix_val, nodeMask_val)), [3, 1, 2]);

    U_val_pred = ds_val{7}.*(UV_val(:,1)*U_std + U_mean) + U_val_dir;
    V_val_pred = ds_val{8}.*(UV_val(:,2)*V_std + V_mean) + V_val_dir;

    % Calculate validation error
    U_val_error = U_val_exact - U_val_pred;
    V_val_error = V_val_exact - V_val_pred;

    val_U_error = 1 - sum(U_val_error.^2, "all")/SS_val_U;
    val_V_error = 1 - sum(V_val_error.^2, "all")/SS_val_V;

    % Plot nodes distribution
    if any(epochsToPlot == epoch)
        curFig = gcf;
        clf(curFig)
        hold on
        scatter(X_val' + magn*U_val_exact, ...
                Y_val' + magn*V_val_exact, ...
                50, "k", "filled", "DisplayName", "Exact")
        scatter(X_val' + magn*gather(extractdata(U_val_pred)), ...
                Y_val' + magn*gather(extractdata(V_val_pred)), ...
                50, "r", "filled", "DisplayName", "Validation")
        title("Epoch: " + string(epoch) + " of " + string(epochs), ...
            "R_U^2: " +  string(val_U_error) + " and R_V^2: " +  string(val_V_error))
        axis equal
        drawnow
    end

    if any(epochsToSave == epoch)
        save("netCurrent.mat", "monitor_single", "residuals", "net")
    end 

    % Record data in the training monitor
    recordMetrics(monitor_single, iteration, ...
        "ValidationSqR_U", val_U_error, ...
        "ValidationSqR_V", val_V_error);

    updateInfo(monitor_single, "Epoch", string(epoch) + " of " + string(epochs),...
        "LearnRate", learnRate,...
        "sqR_U_Val", val_U_error, ...
        "sqR_V_Val", val_V_error);

    % Update learning rate.
    learnRate = initLearnRate/(1+learnRateDecay*epoch);

    % Update progress percentage.
    monitor_single.Progress = 100*epoch/epochs;

end

%% Update progress monitor
monitor_single.Status = "Finished ADAM Optimizer";

% % Plot validation dataset
% U_val = gather(extractdata(U_val*U_std + U_mean))*uScale(end);
% V_val = gather(extractdata(V_val*V_std + V_mean))*vScale(end);
% 
% figure
% hold on
% scatter(X_val' + magn*U_val_exact, Y_val' + magn*V_val_exact,50, "k", "filled", "DisplayName", "Exact")
% scatter(X_val' + magn*U_val, Y_val' + magn*V_val,50, "r", "filled", "DisplayName", "Validation")
% axis equal

function [loss, gradients, residuals, U_pred, V_pred] =...
    modelloss(net, nodeAttr, edgeAttr, srlist, edgeMask, edgesNodesMatrix, nodeMask,...
    dirMaskRevX, dirMaskRevY, dirVecU, dirVecV, K, F, U_exact, V_exact)

% Make prediction
UV_pred = permute(stripdims(net.forward(nodeAttr, edgeAttr, srlist, edgeMask, edgesNodesMatrix, nodeMask)), [3, 1, 2]);
U_pred = UV_pred(:, 1, :);
V_pred = UV_pred(:, 2, :);

% Apply Dirichlet BC directly
U_pred = dirMaskRevX.*U_pred + dirVecU;
V_pred = dirMaskRevY.*V_pred + dirVecV;

% Apply PDE
equibX = pagemtimes(pdeX_ux, U_pred) + pagemtimes(pdeX_uy, V_pred) - pdeX;
equibY = pagemtimes(pdeY_ux, U_pred) + pagemtimes(pdeY_uy, V_pred) - pdeY;
lossPdeX = sum(dirMaskRevX.*equibX.^2, "all")/numPdeX;
lossPdeY = sum(dirMaskRevY.*equibY.^2, "all")/numPdeY;

% % Sanity check
% equibX = pagemtimes(pdeX_ux, U_exact) + pagemtimes(pdeX_uy, V_exact) - pdeX;
% equibY = pagemtimes(pdeY_ux, U_exact) + pagemtimes(pdeY_uy, V_exact) - pdeY;
% lossPdeX = sum(dirMaskRevX.*equibX.^2, "all")/numPdeX;
% lossPdeY = sum(dirMaskRevY.*equibY.^2, "all")/numPdeY;

% Apply Neumann
lossNeumannX = sum((pagemtimes(neuX_ux, U_pred) + pagemtimes(neuX_uy, V_pred) - neuX).^2, "all")/numNeuX;
lossNeumannY = sum((pagemtimes(neuY_ux, U_pred) + pagemtimes(neuY_uy, V_pred) - neuY).^2, "all")/numNeuY;

% % Sanity check
% lossNeumannX = sum((pagemtimes(neuX_ux, U_exact) + pagemtimes(neuX_uy, V_exact) - neuX).^2, "all")/numNeuX;
% lossNeumannY = sum((pagemtimes(neuY_ux, U_exact) + pagemtimes(neuY_uy, V_exact) - neuY).^2, "all")/numNeuY;

% Calculate loss
loss = sum([lossPdeX, lossPdeY, lossNeumannX, lossNeumannY]);
gradients = dlgradient(loss, net.Learnables);
residuals = [lossPdeX, lossPdeY, lossNeumannX, lossNeumannY];
end