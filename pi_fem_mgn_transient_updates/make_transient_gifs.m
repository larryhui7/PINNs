%% GIFs for the transient PI-MGN (netTransient.mat), CPU-only version of evaluationModel.m
%   gifs/transient_<mesh>.gif : autoregressive rollout to 0.9 s (3x the 0.3 s training window).
%       Left : MGN deformed mesh coloured by |u|, FEM (Newmark) mesh black dashed, x10 deflection.
%       Right: R^2(u_x), R^2(u_y) vs time and the tip deflection u_y(t), drawn up to the current time.
%   gifs/transient_training_curves.gif : training monitor (monitorTransient.mat) replayed.
addpath(genpath(pwd))
if ~exist("gifs", "dir"); mkdir gifs; end
meshes = ["gmsh_nen_25_numel_16.m", "coarse_square"; "gmsh_nen_337_numel_304.m", "fine_square"; ...
          "gmsh_tri_nen_61_numed_216.m", "coarse_tri"];
S = load("netTransient.mat"); net = S.net;
for k = 1:size(meshes, 1)
    r = rollout(net, meshes(k, 1));
    ref = load("R_ux_" + meshes(k, 2) + ".mat");
    fprintf("%s: final R2 ux=%.3f uy=%.3f (stored R_ux final %.3f)\n", meshes(k, 2), r.Rux(end), r.Ruy(end), ref.R_ux(end));
    rolloutGif(r, meshes(k, 2), fullfile("gifs", "transient_" + meshes(k, 2) + ".gif"));
end
trainingGif(fullfile("gifs", "transient_training_curves.gif"));

function r = rollout(net, meshFile)
run(meshFile)
padNds = 350; padEds = 1500;
[x, y, LM, IX, ID, srlist, edgesNodesMatrix, numnp, ndf, nElem] = processGmsh(msh);
E = 7; nu = 0.25; g = -0.1; rho = 1; D = D_mat(E, nu);
nodesLeft = find(x == 0); fixed_nodes = [nodesLeft; nodesLeft]; f_x = [0; g*rho];
fixed_dof = [ID(1, fixed_nodes(1, :)), ID(2, fixed_nodes(2, :))];
free_dof = sort(setdiff(1:numnp*ndf, fixed_dof));
[K, F_uu, K_uu, M, ~, M_uu_inv] = globalAssembly(numnp, ndf, nElem, x, y, IX, LM, f_x, D, fixed_dof, rho);
dt = 0.001; t = 0:dt:0.9;
fem = solveFEMTransient(free_dof, numnp, ndf, K, F_uu, M, t);
dirMaskRev = paddata([double(~ismember(1:numnp, fixed_nodes(1,:))); double(~ismember(1:numnp, fixed_nodes(2,:)))], padNds, Dimension = 2);
nodeAttr = dlarray(paddata([x; y; zeros(4, numnp)], padNds, Dimension = 2), "CUB");
d = [x(srlist(1,:)) - x(srlist(2,:)); y(srlist(1,:)) - y(srlist(2,:))];
edgeAttr = dlarray(paddata([d; vecnorm(d)], padEds, Dimension = 2), "CUB");
edgeMask = dlarray(paddata(ones(1, size(srlist, 2)), padEds, Dimension = 2), "CUB");
srl = dlarray(paddata(srlist, padEds, Dimension = 2, FillValue = 1), "CUB");
enm = dlarray(paddata(edgesNodesMatrix, [padEds, padNds]), "CUB");
nodeMask = dlarray(paddata(ones(1, numnp), padNds, Dimension = 2), "CUB");
Minv = dlarray(paddata(M_uu_inv, 2*[padNds, padNds])); Fin = dlarray(paddata(F_uu, 2*padNds, Dimension = 1));
Kin = dlarray(paddata(K_uu, 2*[padNds, padNds]));
ux = zeros(numnp, numel(t)); uy = ux;
A_prev = reshape(Minv*Fin, 2, []);
for it = 2:numel(t)       % same trapezoidal update as evaluationModel.m
    out = permute(stripdims(predict(net, nodeAttr, edgeAttr, srl, edgeMask, enm, nodeMask)), [1, 3, 2]);
    A = out(1:2, :); U0 = nodeAttr(3:4, :); V0 = nodeAttr(5:6, :);
    V = (V0 + 0.5*(A_prev + A)*dt).*dirMaskRev;
    U = (U0 + 0.5*(V0 + V)*dt).*dirMaskRev;
    Ue = extractdata(U); ux(:, it) = Ue(1, 1:numnp)'; uy(:, it) = Ue(2, 1:numnp)';
    nodeAttr(3:4, :) = U; nodeAttr(5:6, :) = V;
    A_prev = reshape(Minv*(Fin - Kin*reshape(U, [], 1)), 2, []);
end
r = struct("x", x(:), "y", y(:), "faces", IX', "t", t, "ux", ux, "uy", uy, "fx", fem.ux, "fy", fem.uy);
r.Rux = 1 - sum((fem.ux - ux).^2, 1)./sum(fem.ux.^2, 1);
r.Ruy = 1 - sum((fem.uy - uy).^2, 1)./sum(fem.uy.^2, 1);
[~, r.tip] = max(x + 1e-3*y);   % right-most node (top one on ties): plotted as the tip
end

function rolloutGif(r, name, outFile)
magn = 10; frames = [1:6:numel(r.t), repmat(numel(r.t), 1, 15)];
cmax = max(hypot(r.fx, r.fy), [], "all");
xs = r.x + magn*[r.fx, r.ux]; ys = r.y + magn*[r.fy, r.uy];
lims = [min(xs(:)) - 0.05, max(xs(:)) + 0.05, min(ys(:)) - 0.05, max(ys(:)) + 0.05];
fig = figure("Visible", "off", "Position", [100 100 1100 480], "Color", "w");
tl = tiledlayout(fig, 2, 2, "TileSpacing", "compact", "Padding", "compact");
for n = 1:numel(frames)
    k = frames(n); tk = r.t(k);
    ax = nexttile(tl, 1, [2 1]); cla(ax); hold(ax, "on")
    patch(ax, "Faces", r.faces, "Vertices", [r.x r.y], "FaceColor", "none", "EdgeColor", [.85 .85 .85]);
    patch(ax, "Faces", r.faces, "Vertices", [r.x + magn*r.ux(:, k), r.y + magn*r.uy(:, k)], ...
          "FaceVertexCData", hypot(r.ux(:, k), r.uy(:, k)), "FaceColor", "interp", "EdgeColor", [.3 .3 .3], "LineWidth", 0.5);
    patch(ax, "Faces", r.faces, "Vertices", [r.x + magn*r.fx(:, k), r.y + magn*r.fy(:, k)], ...
          "FaceColor", "none", "EdgeColor", "k", "LineStyle", "--", "LineWidth", 1.2);
    colormap(ax, turbo); clim(ax, [0 cmax]); cb = colorbar(ax); cb.Label.String = "|u| (MGN)";
    axis(ax, "equal"); axis(ax, lims); box(ax, "on"); grid(ax, "on")
    title(ax, sprintf("%s: t = %.3f s, deflection x%d", strrep(name, "_", " "), tk, magn))
    phase = "training window"; if tk > 0.3; phase = "extrapolation (unseen times)"; end
    subtitle(ax, phase + "   (black dashed: Newmark FEM)")

    ax2 = nexttile(tl, 2); cla(ax2); hold(ax2, "on")
    plot(ax2, r.t(2:k), r.Rux(2:k), "Color", [0.16 0.47 0.84], "LineWidth", 1.5, "DisplayName", "R^2 u_x");
    plot(ax2, r.t(2:k), r.Ruy(2:k), "Color", [0.92 0.41 0.20], "LineWidth", 1.5, "DisplayName", "R^2 u_y");
    xline(ax2, 0.3, "--", "training cut-off", "HandleVisibility", "off");
    xlim(ax2, [0 r.t(end)]); ylim(ax2, [0 1]); grid(ax2, "on"); box(ax2, "on")
    legend(ax2, "Location", "southeast"); ylabel(ax2, "R^2 vs FEM")

    ax3 = nexttile(tl, 4); cla(ax3); hold(ax3, "on")
    plot(ax3, r.t(1:k), r.fy(r.tip, 1:k), "k--", "LineWidth", 1.5, "DisplayName", "FEM");
    plot(ax3, r.t(1:k), r.uy(r.tip, 1:k), "Color", [0.92 0.41 0.20], "LineWidth", 1.5, "DisplayName", "MGN");
    xline(ax3, 0.3, "--", "HandleVisibility", "off");
    xlim(ax3, [0 r.t(end)]); ylim(ax3, [min(r.fy(r.tip, :))*1.15, max(0, max(r.fy(r.tip, :)))*1.15 + 1e-4]);
    grid(ax3, "on"); box(ax3, "on"); legend(ax3, "Location", "northeast")
    ylabel(ax3, "right-edge u_y"); xlabel(ax3, "time [s]")
    writeFrame(fig, outFile, n == 1, 0.06);
end
close(fig)
end

function trainingGif(outFile)
m = load("monitorTransient.mat"); md = m.monitor_single.MetricData; info = m.monitor_single.InfoData;
it = md.LossTotal(:, 1); L = md.LossTotal(:, 2);
ts = info.Timestep(:); Ru = min(max(info.sqR_U(:), 0), 1); Rv = min(max(info.sqR_V(:), 0), 1);
N = numel(it); frames = [round(linspace(2000, N, 150)), repmat(N, 1, 15)];
fig = figure("Visible", "off", "Position", [100 100 1000 560], "Color", "w");
tl = tiledlayout(fig, 3, 1, "TileSpacing", "compact", "Padding", "compact");
title(tl, "Transient PI-MGN training monitor (monitorTransient.mat)")
step = max(1, floor(N/20000));             % thin the 421k points for plotting speed
for n = 1:numel(frames)
    k = frames(n); idx = 1:step:k;
    ax = nexttile(tl, 1); cla(ax); semilogy(ax, it(idx), max(L(idx), 1e-30), "Color", [0.16 0.47 0.84]);
    xlim(ax, [1 N]); ylim(ax, [min(L(L > 0)) max(L)]); grid(ax, "on"); ylabel(ax, "loss |Ma + Ku - F|^2")
    title(ax, sprintf("iteration %d / %d, physical time step t = %.3f s", k, N, ts(k)))
    ax = nexttile(tl, 2); cla(ax); hold(ax, "on")
    plot(ax, it(idx), Ru(idx), "Color", [0.16 0.47 0.84], "DisplayName", "R^2 u_x");
    plot(ax, it(idx), Rv(idx), "Color", [0.92 0.41 0.20], "DisplayName", "R^2 u_y");
    xlim(ax, [1 N]); ylim(ax, [0 1]); grid(ax, "on"); box(ax, "on"); ylabel(ax, "R^2 (clipped to [0,1])")
    legend(ax, "Location", "southeast")
    ax = nexttile(tl, 3); cla(ax); plot(ax, it(idx), ts(idx), "k");
    xlim(ax, [1 N]); ylim(ax, [0 0.31]); grid(ax, "on"); ylabel(ax, "time step [s]"); xlabel(ax, "iteration")
    writeFrame(fig, outFile, n == 1, 0.08);
end
close(fig)
end

function writeFrame(fig, outFile, first, delay)
img = print(fig, "-RGBImage", "-r90");
[A, map] = rgb2ind(img, 256);
if first
    imwrite(A, map, outFile, "gif", "LoopCount", Inf, "DelayTime", delay);
else
    imwrite(A, map, outFile, "gif", "WriteMode", "append", "DelayTime", delay);
end
end
