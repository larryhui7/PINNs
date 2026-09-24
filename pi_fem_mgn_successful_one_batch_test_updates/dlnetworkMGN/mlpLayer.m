function mlp = mlpLayer(hiddenSize, outputSize, args)
%% 1. Set defaults
arguments
    hiddenSize (1,1) double {mustBeInteger, mustBePositive}
    outputSize (1,1) double {mustBeInteger, mustBePositive}
    args.Name string = ''
    args.unflatSize (1,1) double {mustBeInteger,mustBePositive} = 1
    args.ActivationLayer = reluLayer
    args.NumHiddenLayers (1,1) double {mustBeInteger, mustBePositive} = 1
    args.FinalActivationLayer (1,1) logical = false
    args.FinalNormalizationLayer (1,1) logical = false
    args.DropOutLayer (1,1) logical = false
    args.WeightsInitializer string = "glorot"
    args.NormalizationType string {mustBeMember(args.NormalizationType, ["layer", "batch"])} = "layer"
    args.flatLayerName string = ''
    args.unflatLayerName string = ''
end

%% 2. Initiate flattened structure and main deep learning layers
mlp = [
    % 1. Custom functionLayer: "CBU" -> "CB" (all nodes accross all graphs become separate batches)
    flattenIntoBatchLayer("Name", args.flatLayerName) 
    % 2. Repeat hidden layers: ...
    %  i. A fully connected layer multiplies the input by a weight matrix and ...
    %  then adds a bias vector.
    %  ii. Activation function layer
    repmat([fullyConnectedLayer(hiddenSize, "WeightsInitializer",args.WeightsInitializer);...
            args.ActivationLayer()],[args.NumHiddenLayers,1])
    % 3. Final linear layer (no activation function used by default)
    fullyConnectedLayer(outputSize)]; % 3. 

%% 3. Optional follow up layers
% i. if final layer also must be processed through an activation function
if args.FinalActivationLayer
    mlp = [mlp; args.ActivationLayer()];
end

% ii. if we want to randomly set some outputs (10%) to zero
if args.DropOutLayer
    mlp = [mlp; dropoutLayer(0.1)];
end

% iii. if we want to include layer normalization layer which normalizes 
% a mini-batch of data across all channels for each observation
% independently. Scale and offset are learnable parameters.
% iv. alternatively batch normalization, which normalizes each channel
% across the mini-batch (all nodes/edges of all graphs after flattening)
if args.FinalNormalizationLayer
    if args.NormalizationType == "batch"
        mlp = [mlp; batchNormalizationLayer];
    else
        mlp = [mlp; layerNormalizationLayer];
    end
end

%% 4. Unflatten the results and finalize mlp defintion 
mlp = [mlp; ...
       % Custom functionLayer: "CB" -> "CBU" (split nodes accross separate graphs)
       unflattenBatchLayer(USize = args.unflatSize, Name = args.unflatLayerName)];
mlp = dlnetwork(mlp, Initialize = false);
mlp = networkLayer(mlp, Name = args.Name);
end