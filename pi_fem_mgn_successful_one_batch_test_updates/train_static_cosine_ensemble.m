%% PI-MGN static cantilever plate: cosine annealing, feature normalization, warm start, ensemble
% Extends global_fem_large_mesh_test_loss_adjust.m (same physics loss R = K_uu*u - F_uu) with:
%   - input normalization  : z-score of node and edge features
%   - output normalization : network predicts u/uScale with uScale = rho*|g|*L^2/E
%   - cosine annealing     : lr = lrMin + 0.5*(lr0 - lrMin)*(1 + cos(pi*epoch/epochs))
%   - normalization type   : "layer" (LayerNorm) or "batch" (BatchNorm) inside the MGN MLPs
%   - warm start           : optionally initialise member 1 from netStaticPretrained.mat
%   - ensemble             : nEnsemble independently seeded MGNs, prediction = mean
% Runs on CPU (no gpuArray) so it also works on machines without a CUDA GPU.
%
% Usage (from this folder):  cfg = struct("normType","layer","warmStart",true); train_static_cosine_ensemble
if ~exist("cfg", "var"); cfg = struct(); end
def = struct("normType", "layer", "warmStart", false, "nEnsemble", 3, "epochs", 3000, ...
             "lr0", 1e-3, "lrMin", 1e-6, "normalizeInputs", true, "mesh", "gmsh_nen_25_numel_16.m", ...
             "nSnapshots", 150);   % log-spaced prediction snapshots per member (for make_training_gifs.m)
f = fieldnames(def);
for k = 1:size(f, 1); if ~isfield(cfg, f{k}); cfg.(f{k}) = def.(f{k}); end; end
addpath(genpath(pwd))

%% Mesh, material, FEM reference
run(cfg.mesh)
[x, y, LM, IX, ID, srlist, edgesNodesMatrix, numnp, ndf, nElem, nen, numed] = processGmsh(msh);  % not "numel": would shadow the builtin
E = 7; nu = 0.25; g = -0.1; rho = 1; L = 1;
D = D_mat(E, nu);
nodesLeftBdry = find(x == 0);
fixed_nodes = [nodesLeftBdry; nodesLeftBdry];
f_x = [0; g*rho];
[K, F_uu, K_uu] = globalAssembly(numnp, ndf, nElem, x, y, IX, LM, f_x, D, fixed_nodes, ID);
[u_fem, v_fem] = solveFEM(ID, fixed_nodes, numnp, ndf, K, F_uu, x, y);

%% Features (6 node features, matching netStaticPretrained.mat): [x, y, dirX, dirY, fx, fy]
dirMaskX = ismember(1:numnp, fixed_nodes(1,:)); dirMaskY = ismember(1:numnp, fixed_nodes(2,:));
dirMaskRev = reshape([double(~dirMaskX); double(~dirMaskY)], [], 1);
Fn = reshape(F_uu, 2, []) / max(abs(F_uu));
nodeRaw = [x; y; double(dirMaskX); double(dirMaskY); Fn];
direc = [x(srlist(1,:)) - x(srlist(2,:)); y(srlist(1,:)) - y(srlist(2,:))];
edgeRaw = [direc; vecnorm(direc)];
if cfg.normalizeInputs && ~cfg.warmStart
    zs = @(A) (A - mean(A, 2)) ./ max(std(A, 0, 2), 1e-8);   % constant rows (e.g. fx = 0) -> 0
    nodeRaw = zs(nodeRaw); edgeRaw = zs(edgeRaw);
end
nodeAttr = dlarray(nodeRaw, "CUB"); edgeAttr = dlarray(edgeRaw, "CUB");
edgeMask = dlarray(ones(1, numed), "CUB"); srl = dlarray(srlist, "CUB");
enm = dlarray(edgesNodesMatrix, "CUB"); nodeMask = dlarray(ones(1, numnp), "CUB");
uScale = rho*abs(g)*L^2/E;                                   % output normalization
K_in = dlarray(K_uu); F_in = dlarray(F_uu); dirIn = dlarray(dirMaskRev);

%% Train ensemble
accFun = dlaccelerate(@modelloss);
UV = zeros(2*numnp, cfg.nEnsemble); hist = cell(cfg.nEnsemble, 1);
snapEpochs = unique(round(logspace(0, log10(cfg.epochs), cfg.nSnapshots)));
snaps = cell(cfg.nEnsemble, 1);
for m = 1:cfg.nEnsemble
    rng(m)
    clearCache(accFun)
    warm = cfg.warmStart && m == 1 && cfg.normType == "layer" && numnp == 25;
    if warm
        s = load("netStaticPretrained.mat"); net = s.net;    % warm start (same 25-node architecture)
        uS = 1;                                              % pretrained net outputs raw displacements
    else
        net = meshGraphNetwork(6, 3, numnp, numed, 1, 2, normType = cfg.normType);
        uS = uScale;
    end
    avgG = []; avgSq = []; h = zeros(cfg.epochs, 3); snaps{m} = zeros(2*numnp, numel(snapEpochs));
    for ep = 1:cfg.epochs
        lr = cfg.lrMin + 0.5*(cfg.lr0 - cfg.lrMin)*(1 + cos(pi*ep/cfg.epochs));   % cosine annealing
        [loss, grads, UVp, state] = dlfeval(accFun, net, nodeAttr, edgeAttr, srl, edgeMask, enm, nodeMask, ...
                                            dirIn, K_in, F_in, uS, numnp);
        net.State = state;                                   % BatchNorm running statistics
        [net, avgG, avgSq] = adamupdate(net, grads, avgG, avgSq, ep, lr, 0.9, 0.999);
        up = extractdata(UVp);
        h(ep, :) = [extractdata(loss), errorRelative(u_fem, v_fem, up(1:2:end)', up(2:2:end)'), lr];
        [isSnap, kSnap] = ismember(ep, snapEpochs);
        if isSnap; snaps{m}(:, kSnap) = up; end
    end
    UV(:, m) = up; hist{m} = h;
    fprintf("member %d (%s norm, warm start = %d): R^2 = %.5f, rel. L2 error = %.3f%%\n", m, cfg.normType, ...
        warm, h(end, 2), 100*norm(up - reshape([u_fem; v_fem], [], 1))/norm(reshape([u_fem; v_fem], [], 1)));
end

%% Ensemble prediction
uvE = mean(UV, 2);
ref = reshape([u_fem; v_fem], [], 1);
relErr = norm(uvE - ref)/norm(ref);
fprintf("ENSEMBLE (%d members, %s): R^2 = %.5f, rel. L2 error = %.3f%%\n", cfg.nEnsemble, cfg.normType, ...
    errorRelative(u_fem, v_fem, uvE(1:2:end)', uvE(2:2:end)'), 100*relErr);
suffix = ""; if cfg.warmStart; suffix = "_warmstart"; end
if ~cfg.normalizeInputs; suffix = suffix + "_rawinputs"; end
save("results_cosine_ensemble_" + cfg.normType + suffix + ".mat", "cfg", "hist", "UV", "relErr", ...
     "snaps", "snapEpochs", "x", "y", "IX", "srlist", "u_fem", "v_fem")

function [loss, gradients, UV_pred, state] = modelloss(net, nodeAttr, edgeAttr, srlist, edgeMask, ...
    edgesNodesMatrix, nodeMask, dirMaskRev, K, F, uScale, numnp)
[out, state] = forward(net, nodeAttr, edgeAttr, srlist, edgeMask, edgesNodesMatrix, nodeMask);
UV_pred = uScale*reshape(permute(stripdims(out), [1, 3, 2]), [], 1).*dirMaskRev;   % strong Dirichlet
loss = sum((K*UV_pred - F).^2, "all")/numnp;                                         % FE residual
gradients = dlgradient(loss, net.Learnables);
end
