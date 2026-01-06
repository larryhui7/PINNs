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

function [loss, gradients] = modelLoss(net, X, X0, Xn, U0, E, t_target)
    % Domain Loss
    u_dom = forward(net, X);
    u_dom_x = dlgradient(sum(u_dom, "all"), X, 'EnableHigherDerivatives', true);
    u_dom_xx = dlgradient(sum(u_dom_x, "all"), X, 'EnableHigherDerivatives', true);
    L_D = mean((E * u_dom_xx).^2);
    
    % Neumann Loss
    u_N = forward(net, Xn);
    u_N_x = dlgradient(sum(u_N, "all"), Xn, 'EnableHigherDerivatives', true);
    sigma_N = E * u_N_x;
    n_N = numel(extractdata(Xn));
    L_N = (1/n_N) * (sigma_N - t_target).^2;

    % Dirichlet Loss
    u_Dir = forward(net, X0);
    n_Dir = numel(extractdata(X0));
    L_Dir = (1/n_Dir) * sum((u_Dir - U0).^2);

    % Total Loss
    loss = L_D + L_N + L_Dir;
    gradients = dlgradient(loss, net.Learnables);
end

%% Training Options and Data Conversion
E = 10;                 % Young's modulus
t_bar = 1;              % Applied traction

maxIterations = 1000;
gradientTolerance = 1e-5;
stepTolerance = 1e-5;
solverState = lbfgsState;

accfun = dlaccelerate(@modelLoss);
lossFcn = @(net) dlfeval(accfun, net, X, X0, Xn, U0, E, t_bar);

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