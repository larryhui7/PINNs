function [K_el, F_el] = element(x_el, y_el, f_x, D)
%ELEMENT Generates element stiffness matrix and force vector
%   Uses simple 2x2 Gauss numerical integration, for a 2D bilinear element,
%   Jacobian determinant (j_el), Shape function matrix (N) and Strain-displacement matrix (B) are
%   pregenerated as functions in the file "generate_interpolations.mlx"

nen = 4; % number of nodes in an element (implied, does not actually change)
ndf = 2; % degrees of freedom per node (implied, does not actually change)
ngauss = 2; % number of gauss points per direction (implied, does not actually change)

% Gauss points and weights
gp = [-sqrt(1/3), sqrt(1/3)];
gw = [1, 1]; 

% Initiate matrices
K_el = zeros(nen*ndf);
F_el = zeros(nen*ndf, 1);
N_loc = zeros(ndf, nen*ndf);

% Cycle through all Gauss points and add matrices and vectors together per
% each
for i = 1:ngauss
    for j = 1:ngauss
                % Pull specific Gauss points and weights in 2D
                gp1 = gp(i); gp2 = gp(j);
                gw1 = gw(i); gw2 = gw(j);
                % Pull shape functions at i,j G-points and reshape
                N = N_gen(gp1, gp2); N_loc(1, 1:2:end) = N; N_loc(2, 2:2:end) = N;
                % Pull strain-displacement at i,j G-points
                B_loc = B_el(gp1, gp2, x_el, y_el);
                % Pull jacobian determinant at the i,j's gauss point
                j_loc = j_el(gp1, gp2, x_el, y_el); 
                % Evaluate stiffness matrix at i,j's G-point and add to
                % total element stiffness matrix
                K_loc = transpose(B_loc)*D*B_loc*j_loc*gw1*gw2;
                K_el = K_el + K_loc;
                % Evaluate external force vector at i,j's G-point and add to
                % total external force vector
                F_loc = transpose(N_loc)*f_x*j_loc*gw1*gw2;
                F_el = F_el + F_loc;

    end
end
end

