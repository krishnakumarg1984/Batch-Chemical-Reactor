% vim: set nospell nowrap textwidth=0 wrapmargin=0 formatoptions-=t:
clear;close;clc;
t0 = 0;
tf = 0.35*3600;

% Define absolute and relative tolerances
opt.AbsTol      = 1e-6;
opt.RelTol      = 1e-6;

% Define algebraic and differential variables.
% id:1-> differential variables,
% id:0-> algebraic variables.
n_diff = 6;
n_alg  = 4;
id = [ones(n_diff,1);zeros(n_alg,1)];

%% Parameter Values of Model
param.alpha_1    = 1.3708e12;  % kg/gmol/h
param.E1_over_R  = 9.2984e3;   % K
param.alpha_m1   = 1.6215e20;  % 1/h
param.Em1_over_R = 1.3108e4;   % K
param.alpha_2    = 5.2282e12   % kg g/mol/h
param.E2_over_R  = 9.5999e3;   % K
param.K1         = 2.575e-16;  % gmol/kg
param.K2         = 4.876e-14;  % gmol/kg
param.K3         = 1.7884e-16; % gmol/kg
param.Q_plus     = 0.0131;     % gmol/kg

% Define the structure to be passed to the residual function
ida_user_data.param  = param;
ida_user_data.t0     = t0;
ida_user_data.tf     = tf;
ida_user_data.n_diff = n_diff;
ida_user_data.n_alg  = n_alg;

%% Initial Conditions for Differential States (truth)
x0_truth = 1.5776;
x1_truth = 8.32;
x2_truth = 0;
x3_truth = 0;
x4_truth = 0;
x5_truth = 0.0142;

x_init = [x0_truth;x1_truth;x2_truth;x3_truth;x4_truth;x5_truth];

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
R  = diag([0.0004,0.0004,0.0001,0.0001]); % Tuning parameters of the EKF

%% Begin core computations


z_initguess = zeros(n_alg,1);
[z_init,fun_val] = fsolve(@(z) initialise_algebraic_conditions(z,param,x_init),z_initguess);
% [init_point,~,~,~,~] = fsolve(@algebraicStates,x0_alg,opt_fsolve,ce_init,cs_average_init,Q_init,T_init,film_init,param{i});
clear z_initguess;

% Build the initial values array for the integrator
XZ0 = [x_init;z_init]; % XZ is the augmented vector of differential and algebraic variables
XZp0 = zeros(size(XZ0)); % p stands for time-derivative (or "prime"); This contains the derivatives of both states and algebraic variables. This is needed by IDA. However, the actual model equations only use the first n_diff indices

% Define the options for Sundials
options = IDASetOptions('RelTol', opt.RelTol,...
    'AbsTol'        , opt.AbsTol,...
    'MaxNumSteps'   , 1500,...
    'VariableTypes' , id,...
    'UserData'      , ida_user_data,...
    'LinearSolver'  , 'Dense',...
    );

IDAInit(@batchChemReactorModel,t0,XZ0,XZp0,options);

% import casadi.*
% % Define the symbolic variables.
% xsym    = SX.sym('x',[sum(n_diff)+sum(n_alg),1]);
% xpsym   = SX.sym('xp',[sum(n_diff)+sum(n_alg),1]);
% cj      = SX.sym('cj',1);

% Get the model equations written in an implicit form in a symbolic way.
% [dx_tot, ~, ~] = batchChemReactorModel(0,xsym,xpsym);