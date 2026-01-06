function plotTr(x,y, Disp,srlist,tlist, scaleFactor, intMesh)
x_def_sc = x' + scaleFactor*Disp.ux; 
x_max = max(x_def_sc, [], "all"); x_min = min(x_def_sc, [], "all");
y_def_sc = y' + scaleFactor*Disp.uy;
y_max = max(y_def_sc, [], "all"); y_min = min(y_def_sc, [], "all");
edgesToPlot = srlist(:, 1:size(srlist, 2)/2);

error = 100*sqrt((Disp.ux - intMesh.ux).^2 +  (Disp.uy - intMesh.uy).^2)/max(intMesh.Magnitude, [], "all");
errorVec = max(error);
errorMax = max(errorVec);

femFig = figure;
n = 1;
hold on
xData = [x_def_sc(edgesToPlot(1,:), n), x_def_sc(edgesToPlot(2,:), n)]';
yData = [y_def_sc(edgesToPlot(1,:), n), y_def_sc(edgesToPlot(2,:), n)]';
plot(xData, yData, "black", "LineWidth", 1);
scatter(x_def_sc(:, n), y_def_sc(:, n), 50, error(:,n), "filled")
colorbar
clim([0 errorMax])
hold off
axis equal
xlim(1.1*[x_min x_max]);
ylim(1.1*[y_min y_max]);
title("Timestep: " + num2str(0) + " s")

% Animate
for n = 2:length(tlist)
    clf(femFig)
    xData = [x_def_sc(edgesToPlot(1,:), n), x_def_sc(edgesToPlot(2,:), n)]';
    yData = [y_def_sc(edgesToPlot(1,:), n), y_def_sc(edgesToPlot(2,:), n)]';
    hold on
    plot(xData, yData, "black", "LineWidth", 1);
    scatter(x_def_sc(:, n), y_def_sc(:, n), 50, error(:,n), "filled")
    colorbar
    clim([0 errorMax])
    hold off
    axis equal
    xlim(1.1*[x_min x_max]);
    ylim(1.1*[y_min y_max]);
    title("Max relative error is " + num2str(errorMax) + "%")
    subtitle("Timestep: " + num2str(tlist(n)) + " s, max relative error: " + num2str(errorVec(n)) + "%")
    drawnow limitrate
end
end