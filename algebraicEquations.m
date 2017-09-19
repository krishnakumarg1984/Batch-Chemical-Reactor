function [algebraiceqn_residual] = algebraicEquations(z,x,param)
    algebraiceqn_residual(1) = param.Q_plus - x(6) + 10^(-z(1)) - z(2) - z(3) - z(4);
    algebraiceqn_residual(2) = z(2) - param.K2*x(1)/(param.K2 + 10^(-z(1)));
    algebraiceqn_residual(3) = z(3) - param.K3*x(3)/(param.K3 + 10^(-z(1)));
    algebraiceqn_residual(4) = z(4) - param.K1*x(5)/(param.K1 + 10^(-z(1)));
end
