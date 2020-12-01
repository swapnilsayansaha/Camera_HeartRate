%how to find optimal weights for rPPG models

a = optimvar('a');
b = optimvar('b');
c = optimvar('c');
prob = optimproblem;
prob.Objective = mean(sqrt((GT - (a*CHROM + b*ICA + c*POS)).^2)) + sqrt(mean((GT - (a*CHROM + b*ICA + c*POS)).^2));
prob.Constraints.cons1 = a + b + c == 1;
prob.Constraints.cons2 = a <= 1;
prob.Constraints.cons3 = a >= 0;
prob.Constraints.cons4 = b <= 1;
prob.Constraints.cons5 = b >= 0;
prob.Constraints.cons6 = c <= 1;
prob.Constraints.cons7 = c >= 0;
x0.a = 1/3;
x0.b = 1/3;
x0.c = 1/3;
sol = solve(prob,x0)