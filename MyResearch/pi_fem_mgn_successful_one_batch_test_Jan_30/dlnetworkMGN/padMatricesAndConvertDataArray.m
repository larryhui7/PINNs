function A = padMatricesAndConvertDataArray(Ac,numBatches)
% PADMATRICESANDCONVERTDATAARRAY Convert cell data (batch per cell) of matrices into padded 3D
% array, and then converts it into array datastore reading in the batch
% direction
A = padMatrices(Ac);
A = arrayDatastore(A, "IterationDimension", 3, "ReadSize", numBatches);

end 
