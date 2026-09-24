function [x, y, LM, IX, ID, srlist, edgesNodesMatrix, numnp, ndf, numel, nen, numed] = processGmsh(msh)
%PROCESSGMSH Process output from the Gmsh into mesh data used for FEM code
% and MGN inputs
%   Detailed explanation goes here...

% edges = msh.LINES(:, 1:2)'; % only external edges; might be useful later!
%% ID matrix: map between global node number and global degree-of-freedom
numnp = msh.nbNod; % denotes the total number of nodes in the mesh
ndf = 2; % denotes the number of degrees of freedom per node before any boundary conditions are imposed
ID = [1:2:numnp*ndf; 2:2:numnp*ndf];

%% IX connectivity matrix: connects local and global node numbers, each
% column is each new element
numel = size(msh.QUADS, 1); % the total number of elements in the mesh
nen = 4; % number of nodes per element
IX = msh.QUADS(:, [2,3,4,1])'; % reorder nodes to match our FEM code (1 is bottom left and count up counter-clockwise) and remove last column, since it does not carry element information

%% LM array contains the list of globally numbered degrees of freedom in the corresponding
...order to that of the local degrees of freedom of the element
LM = zeros(ndf*nen, numel); 
for iel = 1:numel
    LM(:, iel) = reshape(ID(:, IX(:,iel)), [], 1);
end

%% Pull full edge connectivity data from the mesh
edgesEl = [1:nen, 1];
slist = [];
rlist = [];
for iL = 1:nen % cycle through all edges of each element, keep repeated ones as well
    slist = [slist, IX(edgesEl(iL), :)]; % sender list
    rlist = [rlist, IX(edgesEl(iL+1), :)]; % receiver list
end 
srlist = [slist; rlist]; % stack sender nodes and receiver nodes on top of each other
srlist = [max(srlist); min(srlist)]; % per each pair s-r defined arbitrarily, rename s as max node # of the pair and r as min node # of the pair, this ensures that repeated edges are all names the same such that [5 4; 4 5] -> [ 5 5; 4 4]
srlist = unique(srlist','rows')'; % remove repeated edges
srlist = [srlist, flip(srlist)];

numed = size(srlist, 2);
%% Get edges/nodes connectivity matrix
edgesNodesMatrix = createEdgeDataMatrix(numnp, srlist);

%% Pull x and y arrays
x = msh.POS(:, 1)';
y = msh.POS(:, 2)';
end

function edgesNodesMatrix = createEdgeDataMatrix(num_nodes, srlist)

% This createEdgeDataMatrix.m creates a matrix out of a cell list
% with neighboring edges per node

edgesOfANodeList = cell(num_nodes, 1);

for iNode = 1:num_nodes
    % Edges where iNode is a sender (top row)
    senders = find(srlist(1,:) == iNode);
    % Edges where iNode is a receiver (bottom row)
    receivers = find(srlist(2,:) == iNode);

    edgesOfANodeList{iNode} = [senders, receivers];
end

numEdges = max(size(srlist));
numNodes = numel(edgesOfANodeList);

edgesNodesMatrix = false(numEdges, numNodes);
for iNode = 1:numNodes
    edgesNodesMatrix(edgesOfANodeList{iNode}, iNode) = 1;
end

% Average num edges per node: optional - when aggregating edges to nodes in
% the node processor, default function inside dlnetwork is just sum- thus
% can choose here to either let all edges weight 1 or instead averge so
% that nodes with more edge connections are weighted equally as nodes with
% less edge connections
% Leave as default of all ones.
%sumEdges = 1./sum(edgesNodesMatrix);
%edgesNodesMatrix = sumEdges.*edgesNodesMatrix;

edgesNodesMatrix = double(edgesNodesMatrix);

end

