% Generate Training Data
% Training the model requires a data set of collocation points that enforce the boundary conditions, enforce the initial conditions, and fulfill the Burger's equation.
% Select 25 equally spaced time points to enforce each of the boundary conditions  and . 

numBoundaryConditionPoints = [25 25];

x0BC1 = -1*ones(1,numBoundaryConditionPoints(1));
x0BC2 = ones(1,numBoundaryConditionPoints(2));

t0BC1 = linspace(0,1,numBoundaryConditionPoints(1));
t0BC2 = linspace(0,1,numBoundaryConditionPoints(2));

u0BC1 = zeros(1,numBoundaryConditionPoints(1));
u0BC2 = zeros(1,numBoundaryConditionPoints(2));

% Select 50 equally spaced spatial points to enforce the initial condition u(x,0)=−sin(πx).
numInitialConditionPoints  = 50;

x0IC = linspace(-1,1,numInitialConditionPoints);
t0IC = zeros(1,numInitialConditionPoints);
u0IC = -sin(pi*x0IC);

% Group together the data for initial and boundary conditions.
X0 = [x0IC x0BC1 x0BC2];
T0 = [t0IC t0BC1 t0BC2];
U0 = [u0IC u0BC1 u0BC2];

% Uniformly sample 10,000 points (t,x)∈(0,1)×(−1,1) to enforce the output of the network to fulfill the Burger's equation.
numInternalCollocationPoints = 10000;

points = rand(numInternalCollocationPoints,2);

dataX = 2*points(:,1)-1;
dataT = points(:,2);

%------------------------------------------
% Define Neural Network Architecture
% Define a multilayer perceptron neural network architecture.
% Define a multilayer perceptron neural network architecture with 9 fully connect operations with 20 hidden neurons each. The first fully connect operation has two input channels corresponding to the inputs  and . The last fully connect operation has one output .
% Specify the network hyperparameters.
numLayers = 9;
numNeurons = 20;

% Create the neural network.
layers = featureInputLayer(2);

for i = 1:numLayers-1
    layers = [
        layers
        fullyConnectedLayer(numNeurons)
        tanhLayer];
end

layers = [
    layers
    fullyConnectedLayer(1)]

% Convert the layer array to a dlnetwork object.
net = dlnetwork(layers)

% Training a PINN can result in better accuracy when the learnable 
% parameters have data type double. Convert the network learnables to
% double using the dlupdate function. Note that not all neural networks 
% support learnables of type double, for example, networks that use GPU 
% optimizations that rely on learnables with type single.
net = dlupdate(@double,net);

%----------------------
% Define Model Loss Function

function [loss,gradients] = modelLoss(net,X,T,X0,T0,U0)

% Make predictions with the initial conditions.
XT = cat(1,X,T);
U = forward(net,XT);

% Calculate derivatives with respect to X and T.
gradientsU = dlgradient(sum(U,"all"),{X,T},EnableHigherDerivatives=true);
Ux = gradientsU{1};
Ut = gradientsU{2};

% Calculate second-order derivatives with respect to X.
Uxx = dlgradient(sum(Ux,"all"),X,EnableHigherDerivatives=true);

% Calculate mseF. Enforce Burger's equation.
f = Ut + U.*Ux - (0.01./pi).*Uxx;
zeroTarget = zeros(size(f),"like",f);
mseF = l2loss(f,zeroTarget);

% Calculate mseU. Enforce initial and boundary conditions.
XT0 = cat(1,X0,T0);
U0Pred = forward(net,XT0);
mseU = l2loss(U0Pred,U0);

% Calculated loss to be minimized by combining errors.
loss = mseF + mseU;

% Calculate gradients with respect to the learnable parameters.
gradients = dlgradient(loss,net.Learnables);

end

%-----------------------------
% Specify the training options:
%   Train for 1500 iterations
%   Stop training early when the norm of the gradients or steps are smaller than 0.00001.
%   Use the default options for the L-BFGS solver state.

maxIterations = 1500;
gradientTolerance = 1e-5;
stepTolerance = 1e-5;
solverState = lbfgsState;

% Convert the training data to dlarray objects. Specify that the inputs X
% and T have format "BC" (batch, channel) and that the initial conditions
% have format "CB" (channel, batch).
X = dlarray(dataX,"BC");
T = dlarray(dataT,"BC");
X0 = dlarray(X0,"CB");
T0 = dlarray(T0,"CB");
U0 = dlarray(U0,"CB");

% Accelerate the loss function using dlaccelerate.
accfun = dlaccelerate(@modelLoss);

% Create a function handle containing the loss function for the L-BFGS 
% update. In order to evaluate the dlgradient function inside the modelLoss
% function using automatic differentiation, use the dlfeval function.
lossFcn = @(net) dlfeval(accfun,net,X,T,X0,T0,U0);

% Initialize the TrainingProgressMonitor object. At each iteration, plot 
% the loss and monitor the norm of the gradients and steps. Because the 
% timer starts when you create the monitor object, make sure that you 
% create the object close to the training loop.
monitor = trainingProgressMonitor( ...
    Metrics="TrainingLoss", ...
    Info=["Iteration" "GradientsNorm" "StepNorm"], ...
    XLabel="Iteration");

% Train the network using a custom training loop using the full data set at each iteration. For each iteration:
%   Update the network learnable parameters and solver state using the lbfgsupdate function. 
%   Update the training progress monitor.
%   Stop training early if the norm of the gradients are less the specified tolerances or if the line search fails.
iteration = 0;
while iteration < maxIterations && ~monitor.Stop
    iteration = iteration + 1;
    [net, solverState] = lbfgsupdate(net,lossFcn,solverState);

    updateInfo(monitor, ...
        Iteration=iteration, ...
        GradientsNorm=solverState.GradientsNorm, ...
        StepNorm=solverState.StepNorm);

    recordMetrics(monitor,iteration,TrainingLoss=solverState.Loss);

    monitor.Progress = 100 * iteration/maxIterations;

    if solverState.GradientsNorm < gradientTolerance || ...
            solverState.StepNorm < stepTolerance || ...
            solverState.LineSearchStatus == "failed"
        break
    end

end

%-----------------------------------
% Evaluate Model Accuracy
% For values of t at 0.25, 0.5, 0.75, and 1, compare the predicted values 
% of the deep learning model with the true solutions of the Burger's equation.
% Set the target times to test the model at. For each time, calculate 
% the solution at 1001 equally spaced points in the range [-1,1].
tTest = [0.25 0.5 0.75 1];
numPredictions = 1001;
XTest = linspace(-1,1,numPredictions);
XTest = dlarray(XTest,"CB");

% Test the model.
figure
tiledlayout("flow")

for i=1:numel(tTest)
    t = tTest(i);
    TTest = t*ones(1,numPredictions);
    TTest = dlarray(TTest,"CB");

    % Make predictions.
    XTTest = cat(1,XTest,TTest);
    UPred = forward(net,XTTest);

    % Calculate target.
    UTest = solveBurgers(extractdata(XTest),t,0.01/pi);

    % Calculate error.
    UPred = extractdata(UPred);
    err = norm(UPred - UTest) / norm(UTest);

    % Plot prediction.
    nexttile
    plot(XTest,UPred,"-");
    ylim([-1.1, 1.1])

    % Plot target.
    hold on
    plot(XTest, UTest,"--")
    hold off

    title("t = " + t + ", Error = " + gather(err));
end

legend(["Prediction" "Target"])

%----------------------------------
% The solveBurgers function returns the true solution of Burger's equation at times t as outlined in [3].
% INSERT FEM SOLUTION
function U = solveBurgers(X,t,nu)

% Define functions.
f = @(y) exp(-cos(pi*y)/(2*pi*nu));
g = @(y) exp(-(y.^2)/(4*nu*t));

% Initialize solutions.
U = zeros(size(X));

% Loop over x values.
for i = 1:numel(X)
    x = X(i);

    % Calculate the solutions using the integral function. The boundary
    % conditions in x = -1 and x = 1 are known, so leave 0 as they are
    % given by initialization of U.
    if abs(x) ~= 1
        fun = @(eta) sin(pi*(x-eta)) .* f(x-eta) .* g(eta);
        uxt = -integral(fun,-inf,inf);
        fun = @(eta) f(x-eta) .* g(eta);
        U(i) = uxt / integral(fun,-inf,inf);
    end
end

end