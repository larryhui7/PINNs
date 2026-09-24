function A = padMatrices(Ac)
% PADMATRICES Convert cell data (batch per cell) of matrices into padded 3D
% array

% Extract row = height and width = column from the matrix
[rows, columns] = cellfun(@(A) size(A), Ac);

% Determine max rows and columns accross all batches
maxh = max(rows);
maxw = max(columns);

% Regenerate the matrix but now with respect to max dimensions and total batches
A = zeros([maxh, maxw, numel(Ac)]); % initiate new matrix
for i = 1:numel(Ac)
    A(1:rows(i), 1:columns(i), i) = Ac{i}; % transfer original to padded using ranges...
    % of rows and columns calculated earlier
end

end   