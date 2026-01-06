// Gmsh project created on Wed Apr 09 03:30:55 2025
Point(1) = {0, 0, 0, 1.0};
Point(2) = {1, 0, 0, 1.0};
Point(3) = {1, 1, 0, 1.0};
Point(4) = {0, 1, 0, 1.0};
Point(5) = {0.5, 0.5, 0, 1.0};

Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};//+
Point(6) = {0.5, 0.625, 0, 1.0};
//+
Point(7) = {0.5, 0.375, 0, 1.0};
//+
Circle(5) = {7, 5, 6};
//+
Circle(6) = {6, 5, 7};
//+
Curve Loop(1) = {3, 4, 1, 2};
//+
Curve Loop(2) = {5, 6};
//+
Plane Surface(1) = {1, 2};
