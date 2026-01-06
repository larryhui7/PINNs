// Gmsh project created on Thu Apr 10 10:53:17 2025
//+
Point(1) = {0, 0, 0, 1.0};
//+
Point(2) = {1, 1, 0, 1.0};
//+
Point(3) = {0, 1, 0, 1.0};
//+
Point(4) = {1, 0, 0, 1.0};
//+
Line(1) = {3, 2};
//+
Line(2) = {2, 4};
//+
Line(3) = {4, 1};
//+
Line(4) = {1, 3};
//+
Curve Loop(1) = {4, 1, 2, 3};
//+
Plane Surface(1) = {1};
