function Displacement = solveFEMTransient(free_dof, numnp, ndf, K, F, M, tlist)


beta = 1/4; gamma = 0.5; % Trapesoidal rule: beta = 1/4; gamma = 0.5; 
%% Initialization
dt = tlist(2) - tlist(1);
nsteps = length(tlist);
M_uu = M(free_dof, free_dof);
K_uu = K(free_dof, free_dof);
F_u = F(free_dof, :);
uv_fem = zeros(numnp*ndf, nsteps);
u_n = zeros(length(free_dof), 1); % displacement
v_n = zeros(length(free_dof), 1); % velocity

% Precalculate inverses
M_inv = inv(M_uu);
A_inv = inv(1/(beta*dt^2)*M_uu + K_uu);

%% Cycle: perform first step before cycle
% General Newmark solution: (1/(beta*dt^2)*M_uu + K_uu)*u_n_plus_one = F_n_plus_one + ...
% M_uu*((u_n + v_n*dt)/(beta*dt^2) + (1 - 2*beta)/(2*beta)*a_n)
% => A*u_n_plus_one = B
a_n = M_inv*(F_u - K_uu*u_n);
B = F_u + M_uu*((u_n + v_n*dt)/(beta*dt^2) + (1 - 2*beta)/(2*beta)*a_n);
u_n1 = A_inv*B; % displacement of the next timestep
uv_fem(free_dof, 2) = u_n1;
a_n1 = (u_n1 - u_n - v_n*dt)/(beta*dt^2) - (1 - 2*beta)/(2*beta)*a_n;
v_n1 = v_n + ((1-gamma)*a_n + gamma*a_n1)*dt;

for n = 2:(nsteps-1)
    u_n = u_n1;
    v_n = v_n1;
    a_n = a_n1;

    B = F_u + M_uu*((u_n + v_n*dt)/(beta*dt^2) + (1 - 2*beta)/(2*beta)*a_n);
    u_n1 = A_inv*B; % displacement of the next timestep, resets it
    uv_fem(free_dof, n+1) = u_n1;
    a_n1 = (u_n1 - u_n - v_n*dt)/(beta*dt^2) - (1 - 2*beta)/(2*beta)*a_n;
    v_n1 = v_n + ((1-gamma)*a_n + gamma*a_n1)*dt;
end

Displacement.ux = uv_fem(1:2:end, :);
Displacement.uy = uv_fem(2:2:end, :);
Displacement.Magnitude = sqrt(Displacement.ux.^2 + Displacement.uy.^2);

end