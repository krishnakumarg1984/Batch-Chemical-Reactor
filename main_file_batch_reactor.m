% vim: set nospell nowrap textwidth=0 wrapmargin=0 formatoptions-=t:
%% Basic settings
% NOTE: 'Z', 'X' etc. stand for vectors, whereas 'z' , 'x' etc. represent scalars
clear;close;clc; format long g;
set(0,'defaultaxesfontsize',12,'defaultaxeslinewidth',2,'defaultlinelinewidth',2.5,'defaultpatchlinewidth',2,'DefaultFigureWindowStyle','docked');

%% Constant Model Parameters
model_params.alpha_1    = 1.3708e12/3600; % kg/gmol/s
model_params.E1_over_R  = 9.2984e3;       % K
model_params.alpha_m1   = 1.6215e20/3600; % 1/s
model_params.Em1_over_R = 1.3108e4;       % K
model_params.alpha_2    = 5.2282e12/3600; % kg g/mol/s
model_params.E2_over_R  = 9.5999e3;       % K
model_params.K1         = 2.575e-16;      % gmol/kg
model_params.K2         = 4.876e-14;      % gmol/kg
model_params.K3         = 1.7884e-16;     % gmol/kg
model_params.Q_plus     = 0.0131;         % gmol/kg

%% User-entered data: Simulation Conditions
load time_profile; load Temp_profile; % Load the apriori available input (Temperature (degC) vs time(sec)) profile for he given chemical reactor problem

n_diff = 6; n_alg  = 4; % How many differential and algebraic variables in this DAE problem
Z_init_guess = zeros(n_alg,1);  % user's initial guess for algebraic variables (this will be refined by fsolve before time-stepping)

% Initial Conditions for Differential States (truth)
% The word "truth" is used because we are going to implement an EKF soon,
% and we will need this model to serve as out "experiment", i.e. "truth"
x0_init_truth = 1.5776;
x1_init_truth = 8.32;
x2_init_truth = 0;
x3_init_truth = 0;
x4_init_truth = 0;
x5_init_truth = 0.0142;

% Assemble into a vector of differential states (init, i.e. at time t=0)
X_init_truth = [x0_init_truth;x1_init_truth;x2_init_truth;x3_init_truth;x4_init_truth;x5_init_truth];
clear x0_init_truth x1_init_truth x2_init_truth x3_init_truth x4_init_truth x5_init_truth;

% Specify simulation interval
t0 = 0;          % initial time at start of simulation
tf = 0.35*3600;  % simulation end time

% Define absolute and relative tolerances for time-stepping solver (IDA)
opt_IDA.AbsTol = 1e-7;
opt_IDA.RelTol = 1e-7;

% 'fsolve' is used to solve for the initial values of algebraic variables by keeping the differential variables constant at their initial values.
opt_fsolve             = optimset;
opt_fsolve.Display     = 'off';
opt_fsolve.FunValCheck = 'on';
% opt_fsolve.TolX      = 1e-9;

[Z_init_fsolve_refined,~,~,~,~] = fsolve(@algebraicEquations,Z_init_guess,opt_fsolve,X_init_truth,model_params);
clear Z_init_guess opt_fsolve;

% Build the initial values array for the integrator
XZ0  = [X_init_truth;Z_init_fsolve_refined]; % XZ is the (combined) augmented vector of differential and algebraic variables
XZp0 = zeros(size(XZ0)); % 'p' in the variable name stands for time-derivative (or "prime"); This vector contains the derivatives of both states and algebraic variables. This is needed by IDA. However, the actual model equations only use the first n_diff indices
clear X_init_truth Z_init_fsolve_refined;

%% Other required settings to be configured for the given problem
% Tell IDA how to identify the algebraic and differential variables in the combined XZ (differential+algebraic state) vector.
% id:1-> differential variables,id:0-> algebraic variables.
id = [ones(n_diff,1);zeros(n_alg,1)];
clear n_alg;

% Additional user-data that may be passed to IDA as additional parameters
ida_user_data_struct.model_params = model_params;
% ida_user_data.t0                = t0;
% ida_user_data.tf                = tf;
ida_user_data_struct.n_diff       = n_diff;
% ida_user_data_struct.n_alg      = n_alg;
ida_user_data_struct.time_profile = time_profile;
ida_user_data_struct.Temp_profile = Temp_profile;

clear model_params n_diff time_profile Temp_profile;

% Set a few desired options for Sundials IDA
ida_options_struct = IDASetOptions('RelTol', opt_IDA.RelTol,...
    'AbsTol'        , opt_IDA.AbsTol,...
    'MaxNumSteps'   , 1500,...
    'VariableTypes' , id,...
    'UserData'      , ida_user_data_struct,...
    'LinearSolver'  , 'Dense');

clear id ida_user_data;

%% Initialise (IDA) solver and prepare for simulation
IDAInit(@batchChemReactorModel,t0,XZ0,XZp0,ida_options_struct); % does it not call the stateequation? (only algebraic????? Not sure)
[~, combined_diff_alg_result_vector_t0, ~] = IDACalcIC(t0+0.1,'FindAlgebraic'); % (Find consistent initial conditions) might have to change the 10 to a different horizon

t                       = t0; % t denotes the running 'adaptive' time used by IDA. initialise simulation time to t0
sim_results_matrix(:,1) = combined_diff_alg_result_vector_t0; % Rows of this matrix represents the different quantities (components) of the overall state vector (diff & alg) being solved for columns of this matrix advance in time.
sim_time                = t;
iter                    = 2;

%% IMPORTANT: Actual time-domain Simulation (i.e. time-stepping is implemented here)
while(t<tf)
    [~, t, soln_vec_at_t]      = IDASolve(tf,'OneStep');
    sim_results_matrix(:,iter) = soln_vec_at_t;
    sim_time                   = [sim_time t];
    iter                       = iter + 1;
end
clear iter;

%% Post-process simulation results in order to retain only the samples at mulitples of sampling time, Ts
[sim_time, unique_simtime_indices, ~] = unique(sim_time,'stable');
sim_results_matrix = sim_results_matrix(:,unique_simtime_indices);

sampling_instants_vector = 0:Ts:round(sim_time(end));
for output = 1:length(sim_results_matrix(:,1))   % which output are we re-sampling
    sim_results_sampled = interp1(sim_time,sim_results_matrix(output,:),sampling_instants_vector);
end
clear sim_time sim_results_matrix n_alg;

%% Plot simulation results
for plot_no = 1:length(sim_results_matrix(:,1))
    figure(plot_no);clf;
    plot(sim_time,sim_results_matrix(plot_no,:),'o');
end

%% Jacobian and EKF related (future)
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
% Ts = 60; % sec
%
% sigma_1 = 0.02; % std dev of Gaussian noise added to output1
% sigma_2 = 0.02; % std dev of Gaussian noise added to output2
% sigma_3 = 0.02; % std dev of Gaussian noise added to output3
% sigma_4 = 0.01; % std dev of Gaussian noise added to output4
%
% P0 = diag(0.003*ones(1,6))            ; % Tuning parameters of the EKF
% Q  = diag(0.0001*ones(1,6))           ; % Tuning parameters of the EKF
% R  = diag([0.0004,0.0004,0.0001,0.0001]); % Tuning parameters of the EKF

