function [K, F, K_uu] = globalAssembly(numnp, ndf, numel, x, y, IX, LM, f_x, D, fixed_nodes, ID)

K = zeros(numnp*ndf);
F = zeros(numnp*ndf, 1);
% Cycle though elements and assign them into global stifffness matrix and force vector 
for iel = 1:numel
    nodes_el = IX(:, iel); % global nodes of the element in the propert order
    % Pull element x and y positions in the proper order
    x_el = x(nodes_el)'; y_el = y(nodes_el)'; 
    % Calculate local element K and F arrays
    [K_el, F_el] = element(x_el, y_el, f_x, D);

    % Assign K_el and F_el into corresponding K and F positions
    dofs_el = LM(:, iel); % global degrees of freedom of the element in the proper order
    K(dofs_el, dofs_el) = K(dofs_el, dofs_el) + K_el;
    F(dofs_el) = F(dofs_el) + F_el;
end
K_uu = K;
fixed_x_dof = ID(1, fixed_nodes(1, :));
fixed_y_dof = ID(2, fixed_nodes(2, :));
fixed_dof = [fixed_x_dof, fixed_y_dof];

K_uu(fixed_dof, :) = zeros(size(K_uu(fixed_dof, :))); % Set all dirichlet dofs to zero
K_uu(:, fixed_dof) = zeros(size(K_uu(:, fixed_dof))); % Set all dirichlet dofs to zero

F(fixed_dof) = zeros(length(fixed_dof),1); 
end

