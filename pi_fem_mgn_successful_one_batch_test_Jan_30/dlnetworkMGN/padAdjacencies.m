function A = padAdjacencies(Ac)
% Extract row=height (edge index) and width = column (node index) from the
% edge-node matrix: output is two 1xbatchSz vectors
[h,w] = cellfun(@(A) size(A), Ac);

% Determine max edges(h) and nodes(w) accross all batches
maxh = max(h);
maxw = max(w);

% Regenerate the edge-node adjacency matrix but now with respect to max
% dimensions and number of batches
A = zeros([maxh,maxw,numel(Ac)]); % initiate with all false (0)
for i = 1:numel(Ac)
    Ai = Ac{i}; % extract original edge-node matrix of batch i
    A(1:h(i),1:w(i),i) = Ai; % transfer original to padded using ranges...
    % of rows and columns calculated earlier
end
A = double(permute(A,[1,3,2])); % Reorder to match with the rest of data
end  