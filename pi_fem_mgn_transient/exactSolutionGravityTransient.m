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

gm = fegeometry(decsg([3,4,N1(1),N2(1),N3(1),N4(1),N1(2),N2(2),N3(2),N4(2)]'));
model = femodel(AnalysisType="structuralTransient", Geometry=gm);
model.PlanarType = "planeStress";
model.MaterialProperties = materialProperties("YoungsModulus", YoungsModulus, "PoissonsRatio", nu, "MassDensity", rho);
model = generateMesh(model, Hmax=0.05);


% figure
% pdegplot(model, "VertexLabels","on")
% hold on
% pdegplot(model, "EdgeLabels","on")
% hold on
% pdemesh(model)


I = 1/12*W^3;
%% Cantilever plate
% 1 Edge: Top (from node 1 to node 2) 
% 2 Edge: Right (from node 2 to node 3) 
% 3 Edge: Bottom (from node 3 to node 4)
% 4 Edge: Left (from node 4 to node 1)

model.EdgeBC(4) = edgeBC("Constraint","fixed"); 
model.FaceLoad = faceLoad(Gravity=[0 GAval]); 
model.FaceIC = faceIC(Displacement=[0;0],Velocity=[0;0]); %zero initial velocity and displacement

% Solution
ntsteps = 750;
tlist = linspace(0, 0.875, ntsteps);
dt = tlist(2);
femResults = solve(model, tlist);


%% Generate code using Live Scripts
% Data to visualize
meshData2 = femResults.Mesh;
nodalData2 = femResults.Displacement.Magnitude(:,1);
deformationData2 = [femResults.Displacement.ux(:,1) ...
    femResults.Displacement.uy(:,1)];

% Create PDE result visualization
resultViz2 = pdeviz(meshData2,abs(real(nodalData2)), ...
    "DeformationData",deformationData2, ...
    "DeformationScaleFactor",2, ...
    "MeshVisible",true, ...
    "ColorLimits",[0 0.09128]);

% Fix axes limits for animation
resultViz2.XLimits = [-0.027367 1.3684];
resultViz2.YLimits = [-0.93598 1.0353];

% Animate
for ii2 = 1:ntsteps
    resultViz2.NodalData = femResults.Displacement.Magnitude(:,ii2);
    resultViz2.DeformationData = [femResults.Displacement.ux(:,ii2) ...
        femResults.Displacement.uy(:,ii2)];
    pause(0.002)
end

% Clear temporary variables
clearvars meshData2 nodalData2 deformationData2 ii2
