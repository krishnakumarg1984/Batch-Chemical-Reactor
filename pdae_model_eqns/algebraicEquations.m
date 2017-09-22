function [algebraiceqn_residual] = algebraicEquations(Z,X,model_params)
    algebraiceqn_residual1 = model_params.Q_plus - X(6) + 10^(-Z(1)) - Z(2) - Z(3) - Z(4);
    algebraiceqn_residual2 = Z(2) - model_params.K2*X(1)/(model_params.K2 + 10^(-Z(1)));
    algebraiceqn_residual3 = Z(3) - model_params.K3*X(3)/(model_params.K3 + 10^(-Z(1)));
    algebraiceqn_residual4 = Z(4) - model_params.K1*X(5)/(model_params.K1 + 10^(-Z(1)));

    algebraiceqn_residual = [algebraiceqn_residual1;algebraiceqn_residual2;algebraiceqn_residual3;algebraiceqn_residual4];
end