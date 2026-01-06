syms t v1 R1 real
syms offset_angle real

% Define angular displacement
theta = (v1/R1)*t;

% Offset angle (30 degrees in radians)
offset_angle = sym(pi/6);

% Base rotation (same as satellite 2’s initial)
R_base = [ 0  0 -1;
          -1  0  0;
           0  1  0];

% Offset rotation about x-axis
c1 = cos(offset_angle);
s1 = sin(offset_angle);
R_offset = [1  0   0;
            0  c1 -s1;
            0  s1  c1];

% Initial orientation and position
R01_0 = R_offset * R_base;
p01_0 = R_offset * [0; R1; 0];

% Orbit axis in frame {0}
w = R_offset * [0;0;1];
w_hat = [  0    -w(3)  w(2);
          w(3)   0    -w(1);
         -w(2)  w(1)   0];

% Rodrigues' formula for relative rotation
w_norm = simplify(norm(w));
I = eye(3);
R_rel = I + (sin(w_norm*theta)/w_norm)*w_hat + ...
          ((1-cos(w_norm*theta))/(w_norm^2))*(w_hat*w_hat);

% Net orientation and position
R_net = simplify(R_rel * R01_0);
p_net = simplify(R_rel * p01_0);

% Homogeneous transformation matrix
T01 = [R_net, p_net; 0 0 0 1]
R