%% Training-evolution GIFs for the static PI-MGN runs of train_static_cosine_ensemble.m
% Left : MGN-predicted deformed mesh coloured by |u| (member 1), FEM deformed mesh in black,
%        undeformed mesh in light grey (deflections magnified).
% Right: FE-residual loss and 1 - R^2 vs epoch, drawn up to the current epoch.
% Writes gifs/static_<run>.gif for every results_cosine_ensemble_*.mat in this folder.
files = dir("results_cosine_ensemble_*.mat");
if ~exist("gifs", "dir"); mkdir gifs; end
for f = 1:numel(files)
    R = load(files(f).name);
    run = erase(erase(files(f).name, "results_cosine_ensemble_"), ".mat");
    makeGif(R, run, fullfile("gifs", "static_" + run + ".gif"));
    fprintf("wrote gifs/static_%s.gif\n", run);
end

function makeGif(R, run, outFile)
magn = 5;                                   % deflection magnification
h = R.hist{1}; S = R.snaps{1}; ep = R.snapEpochs;
x = R.x(:); y = R.y(:); faces = R.IX';
uf = R.u_fem(:); vf = R.v_fem(:);
cmax = max(hypot(uf, vf));
err = max(1 - h(:, 2), 1e-12);
titles = struct("layer", "LayerNorm", "batch", "BatchNorm (encoders)", ...
                "layer_warmstart", "LayerNorm, warm start from netStaticPretrained.mat", ...
                "layer_rawinputs", "LayerNorm, no input normalization");
name = run; if isfield(titles, run); name = titles.(run); end

fig = figure("Visible", "off", "Position", [100 100 1100 480], "Color", "w");
tl = tiledlayout(fig, 2, 2, "TileSpacing", "compact", "Padding", "compact");
for k = [1:numel(ep), repmat(numel(ep), 1, 15)]      % hold the final frame
    up = S(1:2:end, k); vp = S(2:2:end, k); e = ep(k);
    ax = nexttile(tl, 1, [2 1]); cla(ax); hold(ax, "on")
    patch(ax, "Faces", faces, "Vertices", [x y], "FaceColor", "none", "EdgeColor", [.85 .85 .85]);
    patch(ax, "Faces", faces, "Vertices", [x + magn*up, y + magn*vp], "FaceVertexCData", hypot(up, vp), ...
          "FaceColor", "interp", "EdgeColor", [.3 .3 .3], "LineWidth", 0.5);
    patch(ax, "Faces", faces, "Vertices", [x + magn*uf, y + magn*vf], "FaceColor", "none", ...
          "EdgeColor", "k", "LineWidth", 1.5, "LineStyle", "--");
    colormap(ax, turbo); clim(ax, [0 cmax]); cb = colorbar(ax); cb.Label.String = "|u| (MGN)";
    axis(ax, "equal"); xlim(ax, [-0.1 1.1 + magn*max(uf)]); ylim(ax, [min(y + magn*vf) - 0.1, 1.1]);
    box(ax, "on"); grid(ax, "on")
    title(ax, sprintf("Epoch %d / %d, deflection x%d", e, size(h, 1), magn))
    subtitle(ax, sprintf("R^2 = %.5f   (black dashed: FEM)", h(e, 2)))

    ax2 = nexttile(tl, 2); cla(ax2);
    semilogy(ax2, 1:e, h(1:e, 1), "Color", [0.16 0.47 0.84], "LineWidth", 1.5); hold(ax2, "on")
    xlim(ax2, [1 size(h, 1)]); ylim(ax2, [min(h(:, 1)) max(h(:, 1))]); grid(ax2, "on")
    ylabel(ax2, "FE residual loss"); title(ax2, name)

    ax3 = nexttile(tl, 4); cla(ax3);
    semilogy(ax3, 1:e, err(1:e), "Color", [0.92 0.41 0.20], "LineWidth", 1.5); hold(ax3, "on")
    xlim(ax3, [1 size(h, 1)]); ylim(ax3, [min(err) 2]); grid(ax3, "on")
    ylabel(ax3, "1 - R^2 vs FEM"); xlabel(ax3, "epoch (cosine-annealed learning rate)")

    img = print(fig, "-RGBImage", "-r90");
    [A, map] = rgb2ind(img, 256);
    if k == 1
        imwrite(A, map, outFile, "gif", "LoopCount", Inf, "DelayTime", 0.08);
    else
        imwrite(A, map, outFile, "gif", "WriteMode", "append", "DelayTime", 0.08);
    end
end
close(fig)
end
