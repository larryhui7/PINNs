function sqR = errorRelative(U_exact, V_exact, U, V)

SS = sum(U_exact.^2 + V_exact.^2, "all");
SS_Err = sum((U_exact - U).^2 + (V_exact - V).^2, "all");
sqR = 1 - SS_Err/SS;

end