clc; clear; close all

plot_msh = 0;
plot_fem = 0;

if canUseGPU
    env = @(x) gpuArray(x);
else 
    env = @(x) x ;
end

set(groot, 'defaultTextInterpreter', 'latex');
set(groot, 'defaultLegendInterpreter', 'latex');

%% Initialize structure and base solution
% Geometry
L = 1; H = 1;
% Material properties
E = 5; % Young's modulus
nu = 0.3; % Poisson's ratio
la = E*nu/(1-nu^2);
mu = 0.5*E/(1+nu);
% Applied force
F = @(y) cos(pi*y/(2*H)); 
plot_fem = 1; 
% Solve the model: beam is structural pde or femodel, and results is solved results
[beam, results] = structure(L,H,E,nu,10,F);
% Plot the exact solution: plot_fem is binary to indicate if plot should be
% created or not
plotR(plot_fem, beam, results)

%% Generate training data: uniform grid 
% Alternatively can sample points from a fine finite element mesh (or even just random sampling of some form) instead of creating a grid

numTrainPointsSide = 40;
[xy_dom, b_xy, u_xy, l_xy, r_xy, B_bc, U_bc, L_bc, R_bc, xy_total] = trainData(numTrainPointsSide, L, H, F, plot_msh); 
% b: bottom, u: upper, l: left, r: right
% _bc: indicates boundary condition (traction or dirichlet) of that edge

% Convert boundary arrays into deep learning arrays and gpuArrays if gpu is available
U_XY = env(dlarray(u_xy,"BC"));  B_XY = env(dlarray(b_xy,"BC")); R_XY = env(dlarray(r_xy,"BC")); L_XY = env(dlarray(l_xy,"BC"));
B_BCX = env(dlarray(B_bc(:,1),"BC"));  B_BCY = env(dlarray(B_bc(:,2),"BC"));
R_BCX = env(dlarray(R_bc(:,1),"BC"));  R_BCY = env(dlarray(R_bc(:,2),"BC"));
U_BCX = env(dlarray(U_bc(:,1),"BC"));  U_BCY = env(dlarray(U_bc(:,2),"BC"));
L_BCX = env(dlarray(L_bc(:,1),"BC"));  L_BCY = env(dlarray(L_bc(:,2),"BC"));

% Convert domain data into the arraydatastore, converted into minibatchqueue later
ds = arrayDatastore(xy_dom);

%% Generate testing data, used to track shape during training and calculate error
% Define the body
beam = femodel(Geometry = geometryFromEdges(decsg([3,4, 0, L, L, 0, 0, 0 H H]')));

% Generate Mesh
beam = generateMesh(beam, "Hedge", {1:4, 0.5});
xy_test = beam.Geometry.Mesh.Nodes;
XY_test = dlarray(xy_test, "CB");
uv_test = interpolateDisplacement(results, xy_test(1,:), xy_test(2,:));
magn = 10;
x_test_deformed = xy_test(1,:)' + magn*uv_test.ux;
y_test_deformed = xy_test(2,:)' + magn*uv_test.uy;

figure
 hold on 
 scatter(xy_test(1,:), xy_test(2, :), 50, 'b', "filled", "DisplayName", "Undeformed")
 scatter(x_test_deformed, y_test_deformed, 50, 'g', "filled", "DisplayName", "Undeformed")
 title("Test mesh at " + num2str(magn) + " scale")
 axis equal

%% Define Deep Learning Model
numLayers = 5;
numNeurons = 40; 
nIn = 2; nOut = 2;
net = createNN(numLayers,numNeurons, nIn, nOut);

%% Initialize modelloss function acceleration
accfun = dlaccelerate(@modelLoss); 
clearCache(accfun) % important to clear if making changes to modelloss

%% Specify Training Options
epochs = 3000;
miniBatchSize = 50;

mbq = minibatchqueue(ds, ...
    MiniBatchSize = miniBatchSize, ...
    MiniBatchFormat = "BC");

%% Generate training monitor
monitor = trainingProgressMonitor; % initate trainition monitor

% Set progress
monitor.Status = "Configuring";
monitor.Progress = 0;

% Initiate monitor information
monitor.Info = ["Epoch", "Loss", "LearnRate", "R2_u", "R2_v", "MiniBatchSize", "EdgeSize", "DomainNodesCount", "NumHiddenLayers", "NumHiddenNeurons"];
updateInfo(monitor,...
        "DomainNodesCount", size(xy_dom, 1),...
        "MiniBatchSize", miniBatchSize,...
        "EdgeSize", numTrainPointsSide, ...
        "NumHiddenLayers", numLayers-1, ...
        "NumHiddenNeurons", numNeurons); % fill in some useful static data on the model

% Set metrics:
monitor.Metrics = ["Loss", "LossPDE", "LossNeu", "LossDir", "R2_u", "R2_v"];
groupSubPlot(monitor, "Loss", ["Loss", "LossPDE", "LossNeu", "LossDir"]);
groupSubPlot(monitor, "Data", ["R2_u", "R2_v"]);

% Set axes
yscale(monitor,"Loss","log")
monitor.XLabel = "Epoch";

%% Initialize training
epochsToPlot = 1:10:epochs; % frequency to update deformation figure

% ADAM optimization parameters
initialLearnRate = 0.01;
decayRate = 0.005;

% Initialize the parameters for the Adam solver
averageGrad = [];
averageSqGrad = [];
monitor.Status = "Running ADAM Optimizer";

% Initialize iterables
learningRate = initialLearnRate;
epoch = 0;
iteration = 0;
figure

%% Perform training
while epoch < epochs && ~monitor.Stop
   epoch = epoch +1;
   shuffle(mbq);
   while hasdata(mbq)
        iteration = iteration + 1;
        XY_dom_batch = next(mbq);
        [loss, gradients, mseG, mseT, mseUV] = dlfeval(accfun, net, XY_dom_batch, ...
            U_XY, B_XY, R_XY, L_XY, U_BCX, U_BCY, B_BCX, B_BCY, R_BCX, R_BCY, L_BCX, L_BCY, la, mu);

        % Update training parameters
        gradients = dlupdate(@gather, gradients); % keep params on CPU
        [net, averageGrad, averageSqGrad] = adamupdate(net, gradients, averageGrad, ...
            averageSqGrad, iteration, learningRate);
   end

   % Get error
   uv_pred = extractdata(forward(net, XY_test));
   u_R = 1 - sum((uv_test.ux - uv_pred(1, :)').^2, "all")/ sum(uv_test.ux.^2, "all");
   v_R = 1 - sum((uv_test.uy - uv_pred(2, :)').^2, "all")/ sum(uv_test.uy.^2, "all");

   % Update the deformation plot 
    if any(epochsToPlot == epoch)
        x_def = xy_test(1,:) + magn*uv_pred(1, :);
        y_def = xy_test(2,:) + magn*uv_pred(2, :);
        curFig = gcf;
        clf(curFig)
        hold on
        scatter(x_test_deformed, y_test_deformed, 50, "green", "filled")
        scatter(x_def, y_def, 50, "red", "filled")
        title("Epoch: " + num2str(epoch))
        subtitle("$R^2_{ux}$: " + num2str(u_R) + " $R^2_{uy}$: " + num2str(v_R))
        axis equal
    end

   % Update training monitor
    updateInfo(monitor,...
        "Epoch", epoch,...
        "Loss", loss,...
        "LearnRate", learningRate, ...
        "R2_u", u_R, ...
        "R2_v", v_R);
    
    recordMetrics(monitor, epoch, ...
        "Loss", loss,...
        "LossPDE", mseG,...
        "LossNeu", mseT,...
        "LossDir", mseUV,...
        "R2_u", u_R, ...
        "R2_v", v_R);

    monitor.Progress = 100*epoch/epochs;

    % Update learning rate.
        learningRate = initialLearnRate / (1+decayRate*iteration);
end

function parameters = createNN(numLayers,numNeurons, nIn, nOut)
    layers = featureInputLayer(nIn);
    
    for i = 1:numLayers-1
        layers = [
            layers
            fullyConnectedLayer(numNeurons)
            tanhLayer];
    end
    
    layers = [
        layers
        fullyConnectedLayer(nOut)];
    parameters = dlnetwork(layers);
end

function plotR(plot_soln, beam, R)
if plot_soln == 1
    figure
    subplot(3,2,1)
    pdeplot(beam.Mesh, XYData=R.Displacement.ux,ColorMap="jet");

    title('$U$ (m)')
    axis equal
    subplot(3,2,3)
    pdeplot(beam.Mesh, XYData=R.Displacement.uy,ColorMap="jet");
    title('$V$ (m)')
    axis equal

    subplot(3,2,5)
    pdeplot(beam.Mesh, XYData=R.Displacement.Magnitude, Deformation = R.Displacement,ColorMap="jet");
    title('Def. Magn (m)')
    axis equal

    subplot(3,2,2)
    pdeplot(beam.Mesh, XYData=R.Stress.sxx,ColorMap="jet");
    title('$\sigma_{xx}$ (Pa)')
    axis equal

    subplot(3,2,4)
    pdeplot(beam.Mesh, XYData=R.Stress.syy,ColorMap="jet");
    title('$\sigma_{yy}$ (Pa)')
    axis equal

    subplot(3,2,6)
    pdeplot(beam.Mesh, XYData=R.Stress.sxy,ColorMap="jet");
    title('$\sigma_{xy}$ (Pa)')
    axis equal

    % fontsize(scale=3) %uncomment to adjust fontsize of the enture figure
end
end

%% LOSS FUNCTION
function [loss, gradients, L_D, L_N, L_Dir] = modelLoss(net, XY_dom_batch, ...
    U_XY, B_XY, R_XY, L_XY, ...
    U_BCX, U_BCY, B_BCX, B_BCY, R_BCX, R_BCY, L_BCX, L_BCY, ...
    la, mu)

% DOMAIN LOSS
% Predicted displacement field (U, V) for domain points
uv_dom = forward(net, XY_dom_batch); 
U_dom = uv_dom(1,:);
V_dom = uv_dom(2,:);

% Compute gradients of batched displacement then separate
U = dlgradient(sum(U_dom, "all"), XY_dom_batch, 'EnableHigherDerivatives', true);
V = dlgradient(sum(V_dom, "all"), XY_dom_batch, 'EnableHigherDerivatives', true);
U_x = U(1,:); U_y = U(2,:);
V_x = V(1,:); V_y = V(2,:);

% Small strain components
e_xx = U_x;
e_yy = V_y;
e_xy = 0.5 * (U_y + V_x);

% Stress components
sigma_xx = la * (e_xx + e_yy) + 2 * mu * e_xx;
sigma_yy = la * (e_xx + e_yy) + 2 * mu * e_yy;
sigma_xy = 2 * mu * e_xy;

% Derivatives of stress components
sigma_xx_dl = dlgradient(sum(sigma_xx, "all"), XY_dom_batch, 'EnableHigherDerivatives', true);
sigma_xy_dl = dlgradient(sum(sigma_xy, "all"), XY_dom_batch, 'EnableHigherDerivatives', true);
sigma_xx_x = sigma_xx_dl(1,:);
sigma_xy_y = sigma_xy_dl(2,:);
total_x = sigma_xx_x + sigma_xy_y;

sigma_xy_dl = dlgradient(sum(sigma_xy, "all"), XY_dom_batch, 'EnableHigherDerivatives', true);
sigma_yy_dl = dlgradient(sum(sigma_yy, "all"), XY_dom_batch, 'EnableHigherDerivatives', true);
sigma_xy_x = sigma_xy_dl(1,:);
sigma_yy_y = sigma_yy_dl(2,:);
total_y = sigma_xy_x + sigma_yy_y;

% Compute domain loss as the mean squared residuals of equilibrium equations
L_D = mean(total_x.^2 + total_y.^2);

% BOUNDARY LOSS IN X-DIRECTION
% Dirichlet left side
uv_left = forward(net, L_XY);
U_left = uv_left(1,:);
L_dir_x = l2loss(U_left, L_BCX);

% Neumann right side (traction)
uv_right = forward(net, R_XY);
U_right = uv_right(1,:);
V_right = uv_right(2,:);

% Compute gradients of U and V at the right boundary
grad_U = dlgradient(sum(U_right, "all"), R_XY, 'EnableHigherDerivatives', true);
grad_V = dlgradient(sum(V_right, "all"), R_XY, 'EnableHigherDerivatives', true);
U_x_r = grad_U(1,:); U_y_r = grad_U(2,:);
V_x_r = grad_V(1,:); V_y_r = grad_V(2,:);

% Traction components on right side
sigma_xx_r = la*(U_x_r + V_y_r) + 2 * mu * U_x_r;
N_x_right = mean((sigma_xx_r - R_BCX).^2);

e_xy_r = 0.5 * (U_y_r + V_x_r);
sigma_xy_r = 2 * mu * e_xy_r;
N_y_right = mean((sigma_xy_r - R_BCY).^2);

% Neumann bottom side 
uv_bottom = forward(net, B_XY);
U_bottom = uv_bottom(1,:);
V_bottom = uv_bottom(2,:);

grad_U = dlgradient(sum(U_bottom, "all"), B_XY, 'EnableHigherDerivatives', true);
U_y_b = grad_U(2,:);
grad_V = dlgradient(sum(V_bottom, "all"), B_XY, 'EnableHigherDerivatives', true);
V_x_b = grad_V(1,:);

e_xy_b = 0.5*(U_y_b + V_x_b);
sigma_xy_b = 2 * mu * e_xy_b;
N_x_bottom = mean((sigma_xy_b + B_BCX).^2);

% Neumann top side
uv_top = forward(net, U_XY);
U_top = uv_top(1,:);
V_top = uv_top(2,:);

grad_U = dlgradient(sum(U_top, "all"), U_XY, 'EnableHigherDerivatives', true);
U_y_t = grad_U(2,:);
grad_V = dlgradient(sum(V_top, "all"), U_XY, 'EnableHigherDerivatives', true);
V_x_t = grad_V(1,:);

e_xy_t = 0.5 * (U_y_t + V_x_t);
sigma_xy_t = 2 * mu * e_xy_t;
N_x_top = mean((sigma_xy_t - U_BCX).^2);

% Total Neumann loss in x-dir
L_N_x = N_x_right + N_x_bottom + N_x_top;

% BOUNDARY LOSS IN Y-DIRECTION
% Dirichlet bottom side
uv_bottom = forward(net, B_XY);
V_bottom = uv_bottom(2,:);
L_dir_y = l2loss(V_bottom, B_BCY);

% Neumann left (shear traction)
uv_left = forward(net, L_XY);
U_left = uv_left(1,:);
V_left = uv_left(2,:);

grad_U = dlgradient(sum(U_left, "all"), L_XY, 'EnableHigherDerivatives', true);
grad_V = dlgradient(sum(V_left, "all"), L_XY, 'EnableHigherDerivatives', true);
U_y_l = grad_U(2,:);
V_x_l = grad_V(1,:);

e_xy_l = 0.5 * (U_y_l + V_x_l);
sigma_xy_l = 2 * mu * e_xy_l;
N_y_left = mean((sigma_xy_l + L_BCY).^2);

% Neumann top (normal traction)
uv_top = forward(net, U_XY);
V_top = uv_top(2,:);

grad_U = dlgradient(sum(uv_top(1,:), "all"), U_XY, 'EnableHigherDerivatives', true);
grad_V = dlgradient(sum(V_top, "all"), U_XY, 'EnableHigherDerivatives', true);
U_x_t = grad_U(1,:);
V_y_t = grad_V(2,:);

sigma_yy_t = la * (U_x_t + V_y_t) + 2 * mu * V_y_t;
N_y_top = mean((sigma_yy_t - U_BCY).^2);

% Total Neumann in y-dir
L_N_y = N_y_right + N_y_left + N_y_top;

% TOTAL LOSS
L_Dir = L_dir_y + L_dir_x; 
L_N = L_N_x + L_N_y; 
loss = L_D + L_N + L_Dir; 

gradients = dlgradient(loss, net.Learnables);

end

%% STRUCTURE FUNCTION –– DO NOT MODIFY 
function [beam, results] = structure(L, H, E, nu, nel, F)
    beam = createpde("structural","static-planestress");
    rect = [3,4,0,L,L,0,0,0,H,H]';
    g = decsg(rect);
    geometryFromEdges(beam, g);
    generateMesh(beam, "Hmax", 0.05);
    
    elementSize = L / nel;  
    generateMesh(beam, "Hmax", elementSize);

    structuralProperties(beam, "YoungsModulus", E, "PoissonsRatio", nu);

    % Bottom edge allow movement in x not y
    structuralBC(beam, "Edge", 1, "YDisplacement", 0);

    % Left edge allow movement in y not in x
    structuralBC(beam, "Edge", 4, "XDisplacement", 0);

    % Apply only in the x direction
    structuralBoundaryLoad(beam, "Edge", 2, ...
        "SurfaceTraction", @(location, state) [F(location.y); zeros(size(location.x))]);

    results = solve(beam);
end

%% TRAINING DATA POINTS
function [xy_dom, b_xy, u_xy, l_xy, r_xy, ...
          B_bc, U_bc, L_bc, R_bc, xy_total] = trainData(numTrainPointsSide, L, H, F, plot_msh)

% Create a grid with +2 points in each direction so we can get the boundaries
x_all = linspace(0, L, numTrainPointsSide +2);
y_all = linspace(0, H, numTrainPointsSide +2);
[Xg, Yg] = meshgrid(x_all, y_all);

% Domain points + boundary points
xy_total = [Xg(:), Yg(:)];

% Domain points
X_int = Xg(2:end-1, 2:end-1);
Y_int = Yg(2:end-1, 2:end-1);
xy_dom = [X_int(:), Y_int(:)];

% Boundary points are the 1st and last rows/columns:
% Bottom y=0
b_xy = [Xg(1,:)', Yg(1,:)'];
% Top y=H
u_xy = [Xg(end,:)', Yg(end,:)'];
% Left x = 0
l_xy = [Xg(:,1), Yg(:,1)];
% Right x = L
r_xy = [Xg(:,end), Yg(:,end)];

% Boundary Conditions
B_bc = zeros(size(b_xy));  
L_bc = zeros(size(l_xy));
U_bc = zeros(size(u_xy));

% Traction Neumann 
R_bc = [F(r_xy(:,2)), zeros(size(r_xy,1), 1)];

% Plot the mesh
if plot_msh
    figure; hold on
    scatter(xy_dom(:,1), xy_dom(:,2), 20, 'b', 'filled', 'DisplayName', 'Interior')
    scatter(b_xy(:,1),   b_xy(:,2),   20, 'r', 'filled', 'DisplayName', 'Bottom')
    scatter(u_xy(:,1),   u_xy(:,2),   20, 'g', 'filled', 'DisplayName', 'Top')
    scatter(l_xy(:,1),   l_xy(:,2),   20, 'c', 'filled', 'DisplayName', 'Left')
    scatter(r_xy(:,1),   r_xy(:,2),   20, 'm', 'filled', 'DisplayName', 'Right')
    legend; axis equal
    title("Training Data (Domain + Boundary Points)")
end
end