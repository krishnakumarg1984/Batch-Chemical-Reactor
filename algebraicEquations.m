function [algebraiceqn_residual] = algebraicEquations(Z,X,param)
    algebraiceqn_residual1 = param.Q_plus - X(6) + 10^(-Z(1)) - Z(2) - Z(3) - Z(4);
    algebraiceqn_residual2 = Z(2) - param.K2*X(1)/(param.K2 + 10^(-Z(1)));
    algebraiceqn_residual3 = Z(3) - param.K3*X(3)/(param.K3 + 10^(-Z(1)));
    algebraiceqn_residual4 = Z(4) - param.K1*X(5)/(param.K1 + 10^(-Z(1)));

    algebraiceqn_residual = [algebraiceqn_residual1;algebraiceqn_residual2;algebraiceqn_residual3;algebraiceqn_residual4];
end