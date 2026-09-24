function [u_fem,v_fem,x_def,y_def] = solveFEM(ID,fixed_nodes,numnp,ndf,K,F,x,y)
fixed_x_dof = ID(1, fixed_nodes(1, :));
fixed_y_dof = ID(2, fixed_nodes(2, :));
fixed_dof = [fixed_x_dof, fixed_y_dof];
free_dof = sort(setdiff(1:numnp*ndf, fixed_dof));

K_uu = K(free_dof, free_dof);
F_u = F(free_dof);
uv_fem = zeros(numnp*ndf, 1);

uv_fem(free_dof) = inv(K_uu)*F_u;
u_fem = uv_fem(1:2:end)';
v_fem = uv_fem(2:2:end)';

x_def = x + u_fem;
y_def = y + v_fem;
end