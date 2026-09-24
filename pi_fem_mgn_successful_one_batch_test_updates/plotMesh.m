function plotMesh(x, y, srlist)

numnp = length(x);

figure
hold on 
plot([x(srlist(1,:)); x(srlist(2,:))], [y(srlist(1,:)); y(srlist(2,:))], "black", "LineWidth",1)
scatter(x, y, 50, "blue", "filled")
text(x+0.02, y, string(1:numnp),AffectAutoLimits="on", Color= "red")
axis equal


end

