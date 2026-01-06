function [x, y, LM, IX, ID, srlist, edgesNodesMatrix, numnp, ndf, numel, nen, numed] = processGmsh(msh)
%PROCESSGMSH Process output from the Gmsh into mesh data used for FEM code
%   Detailed explanation goes here...

%edges = msh.LINES(:, 1:2)'; % only external edges; might be useful later!
%% ID matrix: map between global node number and global degree-of-freedom
numnp = msh.nbNod; % denotes the total number of nodes in the mesh
ndf = 2; % denotes the number of degrees of freedom per node before any boundary conditions are imposed
ID = [1:2:numnp*ndf; 2:2:numnp*ndf];

%% IX connectivity matrix: connects local and global node numbers, each

if isfield(msh, 'QUADS') && ~isempty(msh.QUADS)
    % For quadrilateral elements.
    numel = size(msh.QUADS, 1);
    nen = 4; 
    IX = msh.QUADS(:, [2,3,4,1])';
elseif isfield(msh, 'TRIANGLES') && ~isempty(msh.TRIANGLES)
    numel = size(msh.TRIANGLES, 1);
    nen = 3;
    IX = msh.TRIANGLES(:, 1:3)';
else
    error('No QUADS or TRIANGLES field found in the msh structure.');
end

% % column is each new element
% numel = size(msh.QUADS, 1); % the total number of elements in the mesh
% nen = 4; % number of nodes per element
% IX = msh.QUADS(:, [2,3,4,1])'; % reorder nodes to match our fem code and remove last column, since it does not carry element information

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
    slist = [slist, IX(edgesEl(iL), :)];
    rlist = [rlist, IX(edgesEl(iL+1), :)];
end 
srlist = [slist; rlist]; % stack sender nodes and receiver nodes on top of each other
srlist = [max(srlist); min(srlist)]; % sort them so that identical edges are all in the same form 
srlist = unique(srlist','rows')'; % remove repeated edges
srlist = [srlist, flip(srlist)];

numed = size(srlist, 2);
%% Get edges/nodes connectivity matrix
edgesNodesMatrix = createEdgeDataMatrix(numnp, srlist);

%% Pull x and y arrays
x = msh.POS(:, 1)';
y = msh.POS(:, 2)';
end

