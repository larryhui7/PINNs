function Displacement = solveFEM(free_dof,numnp,ndf,K,F)
K_uu = K(free_dof, free_dof);
F_u = F(free_dof);
uv_fem = zeros(numnp*ndf, 1);

uv_fem(free_dof) = inv(K_uu)*F_u;


Displacement.ux = uv_fem(1:2:end);
Displacement.uy = uv_fem(2:2:end);
Displacement.Magnitude = sqrt(Displacement.ux.^2 + Displacement.uy.^2);

end