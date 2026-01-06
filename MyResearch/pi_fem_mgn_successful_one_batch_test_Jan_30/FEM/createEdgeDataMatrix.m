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

% Average num edges per node
sumEdges = 1./sum(edgesNodesMatrix);
edgesNodesMatrix = sumEdges.*edgesNodesMatrix;

end

