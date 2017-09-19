%% Parameter Values of Model
alpha_1    = 1.3708e12;  % kg/gmol/h
E1_over_R  = 9.2984e3;   % K
alpha_m1   = 1.6215e20;  % 1/h
Em1_over_R = 1.3108e4;   % K
alpha_2    = 5.2282e12   % kg g/mol/h
E2_over_R  = 9.5999e3;   % K
K1         = 2.575e-16;  % gmol/kg
K2         = 4.876e-14;  % gmol/kg
K3         = 1.7884e-16; % gmol/kg
Q_plus     = 0.0131;     % gmol/kg

%% Initial Conditions for Differential States (truth)
x0_truth = 1.5776;
x1_truth = 8.32;
x2_truth = 0;
x3_truth = 0;
x4_truth = 0;
x5_truth = 0.0142;

%% Initial Conditions for Differential States (ekf)
x0_ekf = 1.6;
x1_ekf = 8.3;
x2_ekf = 0;
x3_ekf = 0;
x4_ekf = 0;
x5_ekf = 0.014;

%% EKF required parameters
Ts = 60; % sec

sigma_1 = 0.02; % std dev of Gaussian noise added to output1
sigma_2 = 0.02; % std dev of Gaussian noise added to output2
sigma_3 = 0.02; % std dev of Gaussian noise added to output3
sigma_4 = 0.01; % std dev of Gaussian noise added to output4

P0 = diag(0.003*ones(1,6))            ; % Tuning parameters of the EKF
Q  = diag(0.0001*ones(1,6))           ; % Tuning parameters of the EKF
R  = diag(0.0004,0.0004,0.0001,0.0001); % Tuning parameters of the EKF
