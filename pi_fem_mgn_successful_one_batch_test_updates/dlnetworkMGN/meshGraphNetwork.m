function [net] = meshGraphNetwork(numNodeFeatures, numEdgeFeatures, numNodes, numEdges, batchSize, outputSize, args)
%% MESHGRAPHNETWORK
%{
MESHGRAPHNETWORK Constructs a mesh graph neural network (MGN)
  This function builds a deep learning model for processing graph-based 
  data based on features of the nodes and their edges. Predictions are 
  node-wise. It includes encoding, message-passing (processing), and
  decoding layers. 

  INITIALIZATION INPUTS AND OUTPUTS:
      Input dimensions:
        - numNodeFeatures  - Number of features per node
        - numEdgeFeatures  - Number of features per edge
        - numNodes         - Number of nodes in the graph (use padded value for batched training)
        - numEdges         - Number of edges in the graph (use padded value for batched training)
        - batchSize        - Number of graphs processed simultaneously
        - outputSize       - Number of output features per node
        - args             - Structure with additional hyperparameters

      Output:
        - net              - Constructed deep learning network (dlnetwork)
%}
%% ACTUAL DLNETWORK INPUTS AND OUTPUTS:
%{
      Input dlarrays:
          - nodeFeatures:        [numNodeFeatures, batchSize, numNodes] (Node feature matrix)
          - edgeFeatures:        [numEdgeFeatures, batchSize, numEdges] (Edge feature matrix)
          - srList:              [2, batchSize, numEdges] (Node-wise sender-receiver list defining graph connectivity or the respective edges - must input bidirectional data - meaning physical edges are doubled [s, r; r, s])
          - edgeMask:            [1, batchSize, numEdges] (Mask indicating valid edges)
          - edgesOfANode:        [numEdges, batchSize, numNodes] (Node-edge adjacency matrix)
          - nodeMask:            [1, batchSize, numNodes] (Mask indicating valid nodes)

      Output dlarray:
          - decoder:             [outputSize, batchSize, numNodes] (Final node embeddings)
%}
%% ADDITIONAL FUNCTIONS USED:
%{
      - 'mlpLayer': Constructs a dlnetworkLayer multi-layer perceptron (MLP) used for encoding and decoding.
          Inputs:  Feature size, hidden layer size, activation functions, normalization settings.
          Outputs: Transformed embeddings.
          Notes: 
            - Does not include an input layer
            - Internally flattens 3D batched data into 2D matrices via custom function 'flattenBatchLayer'
              and unflattens it via custom function 'unflattenBatchLayer' after passing through the MLP

      - 'graphProcessorLayer': Implements message passing between nodes and edges as dlnetworkLayer
          Inputs: Hidden neuron size, number of nodes, number of edges, batch size, activation functions.
          Outputs: Updated node and edge embeddings after each message-passing step.
          Notes:
            - Each i'th graph layer has two independent trainable mlp subnetworks shared accross all nodes and edges
            - Both mlps have residual connections (addition layers) between input unupdated embeddings and updated embeddings
                - i'th edge processor 
                    - Input:  concatenation of an edge embedding with two connecting (sender and receiver) node embeddings
                              using custom function 'AccumulateNodesToEdgesLayer', [edgeEmbeddingSize + 2*nodeEmbeddingSize, batchSize, numEdges]
                    - Output: updated edge embedding (including residual connection), [edgeEmbeddingSize, batchSize, numEdges]
                - i'th node processor
                    - Input:  concatenation of a node embedding with a sum of updated adjacent edge embeddings
                              using custom function 'AccumulateEdgesToNodeLayer', [nodeEmbeddingSize + edgeEmbeddingSize, batchSize, numNodes]
                    - Output: updated node embedding (including residual connection), [nodeEmbeddingSize, batchSize, numNodes]
%}

%% Default values
arguments 
    numNodeFeatures (1,1) double {mustBePositive, mustBeInteger}
    numEdgeFeatures (1,1) double {mustBePositive, mustBeInteger}
    numNodes (1,1) double {mustBePositive, mustBeInteger}
    numEdges (1,1) double {mustBePositive, mustBeInteger}
    batchSize (1,1) double {mustBePositive, mustBeInteger}
    outputSize (1,1) double {mustBePositive, mustBeInteger}
    args.numGraphLayers (1,1) double {mustBePositive, mustBeInteger} = 3 % Number of message passing steps
    args.numHiddenNeurons (1,1) double {mustBePositive, mustBeInteger} = 32 % Number of hidden neurons within encoders/processors/decoders
    args.encInnerLayers (1,1) double {mustBePositive, mustBeInteger} = 2 % Number of hidden layers in the encoder
    args.graphInnerLayers (1,1) double {mustBePositive, mustBeInteger} = 2 % Number of hidden layers in both edge and node processors
    args.decInnerLayers (1,1) double {mustBePositive, mustBeInteger} = 2 % Number of hidden layers in the decoder
    args.actLayer string = "swish"
    args.encFinActLayer (1,1) logical = true
    args.encFinNormLayer (1,1) logical = true 
    args.graphFinActLayer (1,1) logical = true
    args.graphFinNormLayer (1,1) logical = true
    args.decFinActLayer (1,1) logical = false
    args.decFinNormLayer (1,1) logical = false
    args.encWeightsInitializer string = "glorot"
    args.graphWeightsInitializer string = "glorot"
    args.decWeightsInitializer string = "glorot"
    args.normType string = "layer" % "layer" or "batch": normalization after the encoder MLPs (processors always use LayerNorm)
end

%% Select activation function based on user input
switch args.actLayer
    case "relu"
        actLayer = reluLayer;
    case "leaky-relu"
        actLayer = leakyReluLayer;
    case "elu"
        actLayer = eluLayer;
    case "swish"
        actLayer = swishLayer;
    case "tanh"
        actLayer = tanhLayer;
    case "sig"
        actLayer = sigmoidLayer;
    case "soft-plus"
        actLayer = softplusLayer;
    case "clipped-relu"
        actLayer = clippedReluLayer(2);
end

%% Encoders
% Node encoder
net = [    
    inputLayer([numNodeFeatures,batchSize,numNodes],"CBU",...
    Name="nodeFeatures")
    mlpLayer(args.numHiddenNeurons, args.numHiddenNeurons,...
    FinalActivationLayer = args.encFinActLayer,...
    FinalNormalizationLayer = args.encFinNormLayer, ...
    Name = "nodeEncoder",...
    unflatSize = numNodes,...
    NumHiddenLayers = args.encInnerLayers,...
    ActivationLayer = actLayer,...
    WeightsInitializer=args.encWeightsInitializer,...
    NormalizationType = args.normType)];
net = dlnetwork(net, Initialize = false);

% Edge encoder
edge_encoder = [
    inputLayer([numEdgeFeatures,batchSize,numEdges],"CBU", Name="edgeFeatures")
    mlpLayer(args.numHiddenNeurons, args.numHiddenNeurons, FinalActivationLayer = args.encFinActLayer, FinalNormalizationLayer = args.encFinNormLayer,...
    Name = "edgeEncoder", unflatSize = numEdges,...
    NumHiddenLayers = args.encInnerLayers,...
    ActivationLayer = actLayer,...
    WeightsInitializer = args.encWeightsInitializer,...
    NormalizationType = args.normType)];

net = net.addLayers(edge_encoder);

%% Graph Processing Layers
% Define additional graph structure input layers
net = net.addLayers(inputLayer([2,batchSize,numEdges], "CBU", Name = "srList"));
net = net.addLayers(inputLayer([1,batchSize,numEdges], "CBU", Name = "edgeMask"));
net = net.addLayers(inputLayer([numEdges,batchSize,numNodes],"CBU", Name = "edgesOfANode"));
net = net.addLayers(inputLayer([1,batchSize,numNodes], "CBU", Name = "nodeMask"));

% Add and (inter)connect message-passing steps
for i = 1:args.numGraphLayers
    graphLayer = graphProcessorLayer(args.numHiddenNeurons, args.numHiddenNeurons, numNodes, numEdges, batchSize,...
                                     Name = "graph_layer_" + i, ...
                                     FinalActivationLayer = args.graphFinActLayer,...
                                     FinalNormalizationLayer = args.graphFinNormLayer,...
                                     NumHiddenLayers = args.graphInnerLayers,...
                                     ActivationLayer = actLayer,...
                                     WeightsInitializer = args.graphWeightsInitializer,...
                                     NormalizationType = "layer"); % BatchNorm nested two networkLayers deep fails dlnetwork/initialize (R2025b), so processors keep LayerNorm
    net = net.addLayers(graphLayer);

    % Connect processors sequentially so that updated edge and node embeddings are passed along as inputs into the next set of graph layers
    if i > 1 % (*) Skip first graph layer, as it requires separate connection with encoders instead
        net = net.connectLayers("graph_layer_"+(i-1)+"/node_embeddings","graph_layer_"+i+"/node_embedding_input");
        net = net.connectLayers("graph_layer_"+(i-1)+"/edge_embeddings","graph_layer_"+i+"/edge_embedding_input");
    end

    % Include fixed connections which dont vary per message passing step (MPS), but are called upon during every MPS
    net = net.connectLayers("srList","graph_layer_"+i+"/node2edge/send_receive_list");
    net = net.connectLayers("edgesOfANode","graph_layer_"+i+"/edge2node/edges_of_node_list");
    net = net.connectLayers("edgeMask","graph_layer_"+i+"/node2edge/mask");
    net = net.connectLayers("nodeMask","graph_layer_"+i+"/edge2node/mask");
end

% (*) Connect encoder to the first graph layer
net = net.connectLayers("nodeEncoder","graph_layer_1/node_embedding_input");
net = net.connectLayers("edgeEncoder","graph_layer_1/edge_embedding_input");

%% Decoder
decoder = mlpLayer(args.numHiddenNeurons, outputSize,...
                   Name = "decoder",...
                   FinalActivationLayer = args.decFinActLayer, ...
                   FinalNormalizationLayer = args.decFinNormLayer,...
                   unflatSize = numNodes,...
                   NumHiddenLayers = args.decInnerLayers,...
                   ActivationLayer = actLayer,...
                   WeightsInitializer = args.decWeightsInitializer);

net = net.addLayers(decoder);
% (*) Connect decoder to the last graph layer
net = net.connectLayers("graph_layer_" + args.numGraphLayers + "/node_embeddings", "decoder");
net = net.initialize();
net.OutputNames = "decoder"; % get rid of edge embeddings output

end