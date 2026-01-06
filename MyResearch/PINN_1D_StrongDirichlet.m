clear; close all; clc;

%% Generate Training Data
n = 100;
x_all = linspace(0,1,n+2);   
x_dom = x_all(2:end-1);      
X = dlarray(x_dom, "CB");    

% Dirichlet at x = 0: u(0)=0
X0 = dlarray(0, "CB");
U0 = dlarray(0, "CB");

% Neumann at x = 1
Xn = dlarray(1, "CB");

%% Define Neural Network Architecture
numLayers = 9;
numNeurons = 20;

layers = featureInputLayer(1);
for i = 1:numLayers-1
    layers = [layers
              fullyConnectedLayer(numNeurons)
              tanhLayer];
end
layers = [layers
          fullyConnectedLayer(1)]

% Create dlnetwork
net = dlnetwork(layers);
net = dlupdate(@double, net);

%% Modified Loss Function (Strong Dirichlet Enforcement)
function [loss, gradients] = modelLoss(net, X, Xn, E, t_target)
    % Enforce u(x) = x * NN(x) construction
    %--------------------------------------------------------------
    % Domain outputs (automatically satisfies u(0)=0)
    NN_dom = forward(net, X);   % Network's raw output
    u_dom = X .* NN_dom;        % u(x) = x * NN(x)
    
    % Compute PDE residual (Force equilibrium)
    u_dom_x = dlgradient(sum(u_dom, "all"), X, 'EnableHigherDerivatives', true);
    u_dom_xx = dlgradient(sum(u_dom_x, "all"), X, 'EnableHigherDerivatives', true);
    L_D = mean((E * u_dom_xx).^2);
    
    % Neumann condition at x=1
    %--------------------------------------------------------------
    NN_N = forward(net, Xn);    % Network's raw output at x=1
    u_N = Xn .* NN_N;           % u(1) = 1 * NN(1)
    
    % Compute traction condition
    u_N_x = dlgradient(sum(u_N, "all"), Xn, 'EnableHigherDerivatives', true);
    sigma_N = E * u_N_x;
    L_N = mean((sigma_N - t_target).^2);

    % Total loss (No Dirichlet term needed)
    loss = L_D + L_N;
    gradients = dlgradient(loss, net.Learnables);
end

%% Updated Training Setup (Remove Dirichlet references)
% ... (keep previous data generation code)

% Update loss function handle (remove X0 and U0)
accfun = dlaccelerate(@modelLoss);
lossFcn = @(net) dlfeval(accfun, net, X, Xn, E, t_bar);

%% Training Options and Data Conversion
E = 10;                 % Young's modulus
t_bar = 1;              % Applied traction

maxIterations = 1000;
gradientTolerance = 1e-5;
stepTolerance = 1e-5;
solverState = lbfgsState;

accfun = dlaccelerate(@modelLoss);
lossFcn = @(net) dlfeval(accfun, net, X, Xn, E, t_bar);

monitor = trainingProgressMonitor( ...
    Metrics="TrainingLoss", ...
    Info=["Iteration" "GradientsNorm" "StepNorm"], ...
    XLabel="Iteration");

%% Training Loop
iteration = 0;
while iteration < maxIterations && ~monitor.Stop
    iteration = iteration + 1;
    
    [net, solverState] = lbfgsupdate(net, lossFcn, solverState);
    
    updateInfo(monitor, Iteration=iteration, ...
                        GradientsNorm=solverState.GradientsNorm, ...
                        StepNorm=solverState.StepNorm);
    recordMetrics(monitor, iteration, TrainingLoss=solverState.Loss);
    monitor.Progress = 100 * iteration / maxIterations;
    
    if solverState.GradientsNorm < gradientTolerance || ...
       solverState.StepNorm < stepTolerance || ...
       solverState.LineSearchStatus == "failed"
        break;
    end
end

%% Post Processing
numPredictions = 100;
xTest = linspace(0,1,numPredictions);
XTest = dlarray(xTest, "CB");

% PINN prediction
u_pred = forward(net, XTest);
u_pred = extractdata(u_pred);

% Analytical solution
u_exact = xTest / E;

% Compute error
relError = norm(u_pred - u_exact) / norm(u_exact);

% Plot of the solutions
figure
plot(xTest, u_pred, 'b-', LineWidth=1.5)
hold on
plot(xTest, u_exact, 'r--', LineWidth=1.5)
xlabel('$x$')
ylabel('$u(x)$')
legend('PINN Prediction','Analytical Solution', 'Interpreter','latex' )
title('PINN vs Analytical Solution', 'Interpreter','latex')
grid on