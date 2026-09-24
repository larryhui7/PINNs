clc
clear
close all

%{
Edge and vertex data reference for initial geometry
1 Node: [0, num.W] top-left corner
2 Node: [num.L, num.W] top-right
3 Node: [num.L, 0] bottom-right
4 Node: [0, 0] bottom-left

1 Edge: Top (from node 1 to node 2)
2 Edge: Right (from node 2 to node 3)
3 Edge: Bottom (from node 3 to node 4)
4 Edge: Left (from node 4 to node 1)
%}

GAval = -0.1;
W = 1;
L = 1;
YoungsModulus = 7;
nu = 0.25;
rho = 1;

% Nodes position
N1 = [0, W]; % top-left corner
N2 = [L, W]; % top-right
N3 = [L, 0]; % bottom-right
N4 = [0, 0]; % bottom-left
model = createpde("structural", "static-planestress");

rect = [3,4,N1(1),N2(1),N3(1),N4(1),N1(2),N2(2),N3(2),N4(2)]';
g = decsg(rect);
geometryFromEdges(model, g);
generateMesh(model, "Hmax", 0.1);

figure
pdegplot(model, "VertexLabels","on")
hold on
pdegplot(model, "EdgeLabels","on")
hold on
pdemesh(model)

structuralProperties(model,"YoungsModulus", YoungsModulus, "PoissonsRatio", nu, "MassDensity", rho)

I = 1/12*W^3;
%% Cantilever plate
% 1 Edge: Top (from node 1 to node 2) 
% 2 Edge: Right (from node 2 to node 3) 
% 3 Edge: Bottom (from node 3 to node 4)
% 4 Edge: Left (from node 4 to node 1)

%structuralBC(model,"Edge", 2,"XDisplacement",0) % fix left/right
%structuralBC(model,"Edge", 3,"YDisplacement",0) % fix top/bottom
structuralBC(model,"Edge", 4,"Constraint","fixed") 
structuralBodyLoad(model,"GravitationalAcceleration", [0; GAval]) % Body load

% Solution
femResults = solve(model);

% Plot results
figure
scaleFactor = 1;
pdeplot(femResults.Mesh, XYData=femResults.Displacement.Magnitude, Deformation=femResults.Displacement, DeformationScaleFactor=scaleFactor, Mesh = "on")
colorbar