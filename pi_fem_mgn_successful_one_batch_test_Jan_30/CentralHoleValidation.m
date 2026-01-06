%% Validation on a New Geometry

run("gmsh_CentralHolePlateFine.m")
[x, y, LM, IX, ID, srlist, edgesNodesMatrix, numnp, ndf, numel, nen, numed] = processGmsh(msh);
plotMesh(x, y, srlist)

%% NEURAL NET PREDICTION

% Combine the x and y coordinates (returned as row vectors) into a single 2-by-N array.
xy_val = [x; y];

% Convert coordinates to DL
XY_val = dlarray(xy_val, "CB");  % "C" = channel, "B" = batch

% Use trained Neural Net (row 1: disp. x, row 2: disp. y)
uv_val = extractdata(forward(net, XY_val));

% magnification if too small
magn = 2;
x_val_deformed = xy_val(1,:)' + magn * uv_val(1,:)';
y_val_deformed = xy_val(2,:)' + magn * uv_val(2,:)';

% Plot the new (validation) geometry: undeformed and deformed configurations.
figure;
hold on;
scatter(xy_val(1,:), xy_val(2,:), 50, 'b', "filled", "DisplayName", "Undeformed");
scatter(x_val_deformed, y_val_deformed, 50, 'r', "filled", "DisplayName", "Deformed");
title("Validation Geometry: Deformation Predicted by the Trained Model");
axis equal;
legend;
