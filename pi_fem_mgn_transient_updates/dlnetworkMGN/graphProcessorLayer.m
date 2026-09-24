function layer = graphProcessorLayer(edgeEmbeddingSize, nodeEmbeddingSize, numNodes, numEdges, batchSize, args)
% GRAPHPROCESSORLAYER   Creates a graph encoder layer which accumulates edge
% features onto nodes and node features onto edges, and returns both the
% updated node and edge embeddings.

arguments
    edgeEmbeddingSize (1,1) double {mustBeInteger, mustBePositive}
    nodeEmbeddingSize (1,1) double {mustBeInteger, mustBePositive}
    numNodes (1,1) double {mustBeInteger, mustBePositive}
    numEdges (1,1) double {mustBeInteger, mustBePositive}
    batchSize (1,1) double {mustBeInteger, mustBePositive}
    args.Name = ''
    args.FinalActivationLayer (1,1) logical = false
    args.FinalNormalizationLayer (1,1) logical = false
    args.NumHiddenLayers (1,1) double {mustBeInteger, mustBePositive} = 4
    args.ActivationLayer = reluLayer
    args.DropOutLayer (1,1) logical = false
    args.WeightsInitializer string = "glorot"
end

%% Define inputs into the total processor unit
net = dlnetwork(); % initiate network
edgeInputLayer = inputLayer([edgeEmbeddingSize, batchSize, numEdges], "CBU", Name = "edge_embedding_input");
nodeInputLayer = inputLayer([nodeEmbeddingSize, batchSize, numNodes], "CBU", Name = "node_embedding_input");

%% I. Edge processor   
% 0. Predefine local network that processes edges+connected nodes into updated edges
edge_processor = mlpLayer(edgeEmbeddingSize + 2*nodeEmbeddingSize, edgeEmbeddingSize,...
    FinalActivationLayer = args.FinalActivationLayer,...
    FinalNormalizationLayer = args.FinalNormalizationLayer,...
    Name = 'edge_processor', ...
    NumHiddenLayers = args.NumHiddenLayers, ...
    ActivationLayer = args.ActivationLayer,...
    DropOutLayer = args.DropOutLayer,...
    WeightsInitializer = args.WeightsInitializer,...
    flatLayerName = "flat_edge",...
    unflatLayerName = "unflat_edge",...
    unflatSize = numEdges);

% 1. input edge data (from edge encoder or previous message passing step - connection defined out of this function),
% 2. gather edges + connected nodes into input form for the edge processor 
% 3. process edge + connected nodes data through the trainable mlp
% 4. add residual connection

layers_edge_processor = [edgeInputLayer
    AccumulateNodesToEdgesLayer(Name = "node2edge") 
    edge_processor % automatically inserts into 'in1' of the "edge_embeddings" addition layer
    additionLayer(2, Name = "edge_embeddings") % note on the name: "edge_embeddings" = "unflat_edge" + "edge_embedding_input" 
    ];
net = addLayers(net, layers_edge_processor); % connect layers with graph network
net = connectLayers(net, "edge_embedding_input", "edge_embeddings/in2"); % fill in 'in2' of the addition layer representing residual layer

%% II. Node processor
% 0. Predefine local network that processes nodes + sum(adjacent updated edge embeddings) into updated nodes
node_processor = mlpLayer(edgeEmbeddingSize + nodeEmbeddingSize, nodeEmbeddingSize,...
    FinalActivationLayer = args.FinalActivationLayer,...
    FinalNormalizationLayer = args.FinalNormalizationLayer,...
    Name = "node_processor", ...
    NumHiddenLayers = args.NumHiddenLayers, ...
    ActivationLayer = args.ActivationLayer,...
    DropOutLayer = args.DropOutLayer,...
    WeightsInitializer = args.WeightsInitializer,...
    flatLayerName = "flat_node",...
    unflatLayerName = "unflat_node",...
    unflatSize = numNodes);

% 1. input node data (from node encoder or previous message passing step - connection defined out of this function),
% 2. gather nodes+averaged(connected residuals of edges) into input form for the node processor 
% 3. process nodes + averaged(connected residuals of edges)
% 4. add residual connection

layers_node_processor = [nodeInputLayer
    AccumulateEdgesToNodeLayer(Name = "edge2node")
    node_processor
    additionLayer(2, Name = "node_embeddings"),...
    ];
net = addLayers(net, layers_node_processor); % connect layers with network
net = connectLayers(net, "node_embedding_input", "node_embeddings/in2");


%% III. Connect edge and node processors within graph layer
% 'node to edge' aggregation step used in edge processor as well as node processor both input "unflat_node_embedding_input" 
net = connectLayers(net, "node_embedding_input","node2edge/node_embedding");
% 'edge to node' aggregation step used in node processor inputs processed edge residuals
net = connectLayers(net, "edge_embeddings","edge2node/edge_embedding");

%% Check connections between layers and residual calculations
% plot(net)
%% Prepare output message passing step network
layer = networkLayer(net, Name = args.Name, OutputNames=["node_embeddings","edge_embeddings"]);
end