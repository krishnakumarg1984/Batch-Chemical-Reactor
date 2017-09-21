% This code implements the Batch Chemical Reactor (non-linear set of DAEs)
% discussed in the journal article:
% "Applying the extended Kalman filter to systems described by nonlinear
% differential-algebraic equations", V.M. Becerra, P.D., Roberts, G.W.
% Griffiths, Control Engineering Practice, 2001 pp 267-281

% Authors: Krishnakumar Gopalakrishnan, Davide M. Raimondo
% License: MIT License

%% Basic settings for MATLAB IDE, plotting, numerical display etc.
% NOTE: In this problem statement/code, 'X', 'Z' etc. are vectors, whereas 'x' , 'z' etc. represent scalar quantities
clear;close all;clc; format long g;
set(0,'defaultaxesfontsize',12,'defaultaxeslinewidth',2,'defaultlinelinewidth',2.5,'defaultpatchlinewidth',2,'DefaultFigureWindowStyle','docked');

%% Constant Model Parameters for this chemical reactor
model_params.alpha_1    = 1.3708e12/3600; % kg/gmol/s
model_params.E1_over_R  = 9.2984e3;       % K
model_params.alpha_m1   = 1.6215e20/3600; % 1/s
model_params.Em1_over_R = 1.3108e4;       % K
model_params.alpha_2    = 5.2282e12/3600; % kg g/mol/s
model_params.E2_over_R  = 9.5999e3;       % K
model_params.K1         = 2.575e-16;      % gmol/kg  (appears only in algebaric equations)
model_params.K2         = 4.876e-14;      % gmol/kg  (appears only in algebaric equations)
model_params.K3         = 1.7884e-16;     % gmol/kg  (appears only in algebaric equations)
model_params.Q_plus     = 0.0131;         % gmol/kg  (appears only in algebaric equations)

%% User-entered data: Simulation Conditions
load time_profile; load Temp_profile; % Load the apriori available input (Temperature (degC) vs time(sec)) profile for he given chemical reactor problem
Ts = 60;                              % How often is simulation results needed ?

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

% Assemble these initial components into a vector representing initial values of differential states (init, i.e. at time t=0)
X_init_truth = [x0_init_truth;x1_init_truth;x2_init_truth;x3_init_truth;x4_init_truth;x5_init_truth];
clear x0_init_truth x1_init_truth x2_init_truth x3_init_truth x4_init_truth x5_init_truth;

% Specify simulation interval
t0 = 0;          % initial time at start of simulation
tf = 0.35*3600;  % simulation end time

% Define absolute and relative tolerances for time-stepping solver (IDA)
opt_IDA.AbsTol = 1e-6;
opt_IDA.RelTol = 1e-6;

% 'fsolve' is used to solve for the initial values of algebraic variables by keeping the differential variables constant at their initial values.
opt_fsolve             = optimset;
opt_fsolve.Display     = 'off';
opt_fsolve.FunValCheck = 'on';
% opt_fsolve.TolX      = 1e-9;

[Z_init_fsolve_refined,~,~,~,~] = fsolve(@algebraicEquations,Z_init_guess,opt_fsolve,X_init_truth,model_params);
clear Z_init_guess opt_fsolve;

% Build the initial (for t=0) combined (i.e. augmented diff & alg) vectors & the vector of their derivatives to be used by the IDA integrator
XZ0  = [X_init_truth;Z_init_fsolve_refined]; % XZ is the (combined) augmented vector of differential and algebraic variables
XZp0 = zeros(size(XZ0));                     % 'p' in this variable name stands for time-derivative (i.e. "prime"); This vector contains the derivatives of both states and algebraic variables. This is needed by IDA. However, the actual model equations (in this paper) only need the first n_diff variables (i.e. only the portion of this combined vector containing only the time-derivatives). Please refer to the function that implements the model equations if you need further clarification
clear X_init_truth Z_init_fsolve_refined;

%% Other required settings to be configured for the given problem
% Tell IDA how to identify the algebraic and differential variables in the combined XZ (differential+algebraic state) vector.
% id:1-> differential variables,id:0-> algebraic variables.
id = [ones(n_diff,1);zeros(n_alg,1)];

% Additional user-data that may be passed to IDA as additional parameters
ida_user_data_struct.model_params = model_params;
ida_user_data_struct.n_diff       = n_diff;
ida_user_data_struct.time_profile = time_profile;
ida_user_data_struct.Temp_profile = Temp_profile;


%% Analytical Jacobian for automatic differentiation using CasADi
% Import casadi framework
import casadi.*
% Define the symbolic variables.
XZsym    = SX.sym('x',[sum(n_diff)+sum(n_alg),1]);
XZpsym   = SX.sym('xp',[sum(n_diff)+sum(n_alg),1]);
cj      = SX.sym('cj',1);

clear model_params n_diff n_alg time_profile Temp_profile;

% % Get the model equations written in an implicit form in a symbolic way.
[residuals_vector_symbolic, ~, ~] = batchChemReactorModel(0,XZsym,XZpsym,ida_user_data_struct);

% Evaluate the Jacobian matrix. (Please refer to the Sundials guide for
% further information about the Jacobian structure).
J = jacobian(residuals_vector_symbolic,XZsym) + cj*jacobian(residuals_vector_symbolic,XZpsym);

% Define a function for the Jacobian evaluation for a given set of
% differential and algebraic variables.
JacFun = Function('fJ',{XZsym,cj},{J});

% Store the function into a structure such that IDA will use it for the
% evaluation of the Jacobian matrix (see the definition of the function
% djacfn at the end of this file).
ida_user_data_struct.fJ = JacFun;

% Define the options for Sundials
ida_options_struct = IDASetOptions('RelTol', opt_IDA.RelTol,...
    'AbsTol'        , opt_IDA.AbsTol,...
    'MaxNumSteps'   , 1500,...
    'VariableTypes' , id,...
    'UserData'      , ida_user_data_struct,...
    'JacobianFn'    , @djacfn,...
    'LinearSolver'  , 'Dense');

% clear model_params n_diff n_alg time_profile Temp_profile;
clear id ida_user_data_struct;

%% Initialise (IDA) solver and prepare for time-stepping
IDAInit(@batchChemReactorModel,t0,XZ0,XZp0,ida_options_struct); % does it not call the stateequation? (only algebraic????? Not sure)
clear ans XZ0 XZp0 opt_IDA;

[~, combined_diff_alg_result_vector_t0, ~] = IDACalcIC(t0 + 0.1,'FindAlgebraic'); % (Find consistent initial conditions) might have to change the 10 to a different horizon

t                       = t0; % t denotes the running 'adaptive' time used by IDA. initialise simulation time to t0
sim_results_matrix(:,1) = combined_diff_alg_result_vector_t0; % Rows of this matrix represents the different quantities (components) of the overall state vector (diff & alg) being solved for columns of this matrix advance in time.
sim_time_ida                = t;

clear t0 combined_diff_alg_result_vector_t0;

%% IMPORTANT: Actual time-domain Simulation (i.e. time-stepping) is implemented here
time_step_iter = 2;

tic;
while(t<tf)
    [~, t, soln_vec_at_t] = IDASolve(tf,'OneStep');
    sim_results_matrix(:,time_step_iter) = soln_vec_at_t;
    sim_time_ida              = [sim_time_ida t];
    time_step_iter        = time_step_iter + 1;
end
toc;

clear time_step_iter soln_vec_at_t;

%% Post-process simulation results in order to retain only the samples at mulitples of sampling time, Ts
[sim_time_ida, unique_simtime_indices, ~] = unique(sim_time_ida,'stable'); % de-dup sim-times (sometimes IDA producdes two data points at same time-step. This cannot be accepted by interp1 below)
sim_results_matrix = sim_results_matrix(:,unique_simtime_indices);
clear unique_simtime_indices;

sim_time_final = 0:Ts:round(sim_time_ida(end));
for augmented_state_vector_row = 1:length(sim_results_matrix(:,1))   % which output are we re-sampling
    sim_results_resampled(augmented_state_vector_row,:) = interp1(sim_time_ida,sim_results_matrix(augmented_state_vector_row,:),sim_time_final);
end
clear sim_time_ida sim_results_matrix n_alg augmented_state_vector_row;

%% Plot simulation results (all components of the augmented state vector, i.e. differential and algebraic states)
for plot_no = 1:length(sim_results_resampled(:,1))
    figure(plot_no);clf;
    plot(sim_time_final/3600,sim_results_resampled(plot_no,:),'o-');
    if plot_no<=6
        label_str = ['State x_' num2str(plot_no-1)];
    else
        label_str = ['State z_' num2str(plot_no-7)];
    end
    xlabel('Time [hours]'); ylabel(label_str);
    title(['Sim result: ' label_str]);axis square;
end

%% Adjust figure properties to match the graph reported in paper
% figure(1); ylim([0.7 1.6]); xlim([0 0.35]);
% % clear plot_no label_str;
%
% This function is used to evaluate the Jacobian Matrix of the P2D model.
function [J, flag, new_data] = djacfn(t, y, yp, rr, cj, data)

% Extract the function object (representing the Jacobian)
fJ    = data.fJ;

% Evaluate the Jacobian with respect to the present values of the states and their time derivatives.
try
    J = full(fJ(y,cj));
catch
    J = [];
end

% Return dummy values
flag        = 0;
new_data    = [];
end
% vim: set nospell nowrap textwidth=0 wrapmargin=0 formatoptions-=t: