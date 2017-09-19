
%% Jacobian and EKF related (future code TBD with Davide)
% import casadi.*
% % Define the symbolic variables.
% xsym  = SX.sym('x',[sum(n_diff)+sum(n_alg),1]);
% xpsym = SX.sym('xp',[sum(n_diff)+sum(n_alg),1]);
% cj    = SX.sym('cj',1);

% Get the model equations written in an implicit form in a symbolic way.
% [dx_tot, ~, ~] = batchChemReactorModel(0,xsym,xpsym);

%% EKF related
% %% Initial Conditions for Differential States (ekf)
% x0_ekf = 1.6;
% x1_ekf = 8.3;
% x2_ekf = 0;
% x3_ekf = 0;
% x4_ekf = 0;
% x5_ekf = 0.014;

% %% EKF required parameters
%
% sigma_1 = 0.02; % std dev of Gaussian noise added to output1
% sigma_2 = 0.02; % std dev of Gaussian noise added to output2
% sigma_3 = 0.02; % std dev of Gaussian noise added to output3
% sigma_4 = 0.01; % std dev of Gaussian noise added to output4
%
% P0 = diag(0.003*ones(1,6))            ; % Tuning parameters of the EKF
% Q  = diag(0.0001*ones(1,6))           ; % Tuning parameters of the EKF
% R  = diag([0.0004,0.0004,0.0001,0.0001]); % Tuning parameters of the EKF

