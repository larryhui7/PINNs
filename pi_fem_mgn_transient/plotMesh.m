function plotMesh(x, y, Disp, srlist, numed,scaleFactor, intMesh)
x_def_sc = x' + scaleFactor*Disp.ux; 
x_max = max(x_def_sc, [], "all");
x_min = min(x_def_sc, [], "all");
y_def_sc = y' + scaleFactor*Disp.uy;
y_max = max(y_def_sc, [], "all");
y_min = min(y_def_sc, [], "all");
edgesToPlot = srlist(:, 1:numed/2);
numnp = length(x);

error = 100*sqrt((Disp.ux - intMesh.ux).^2 +  (Disp.uy - intMesh.uy).^2)/max(intMesh.Magnitude);
errorMax = max(error);

figure
hold on 

plot([x_def_sc(edgesToPlot(1,:)), x_def_sc(edgesToPlot(2,:))]', [y_def_sc(edgesToPlot(1,:)), y_def_sc(edgesToPlot(2,:))]', "black", "LineWidth",1)
scatter(x_def_sc, y_def_sc, 50, error, "filled")
colorbar
%text(x_def_sc+0.02, y_def_sc, string(1:numnp),AffectAutoLimits="on", Color= "red")
axis equal
xlim(1.1*[x_min x_max]);
ylim(1.1*[y_min y_max]);
title("Max relative error is " + num2str(errorMax) + "%")

end

