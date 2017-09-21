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
n_inputs = 1;
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
XZsym   = SX.sym('XZ',[sum(n_diff)+sum(n_alg),1]);
XZpsym  = SX.sym('XZp',[sum(n_diff)+sum(n_alg),1]);
cj      = SX.sym('cj',1);

clear model_params n_diff n_alg time_profile Temp_profile;

% % Get the model equations written in an implicit form in a symbolic way.
[overall_residuals_vector_symbolic, ~, ~] = batchChemReactorModel(0,XZsym,XZpsym,ida_user_data_struct);

% Evaluate the Jacobian matrix. (Please refer to the Sundials guide for
% further information about the Jacobian structure).
J = jacobian(overall_residuals_vector_symbolic,XZsym) + cj*jacobian(overall_residuals_vector_symbolic,XZpsym);

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
clear ans XZ0 XZp0 opt_IDA;

t_local_start = t0;
t_local_finish_actual = t_local_start + Ts;

% Build the initial (for t=0) combined (i.e. augmented diff & alg) vectors & the vector of their derivatives to be used by the IDA integrator
XZ_t_local_finish_actual  = [X_init_truth;Z_init_fsolve_refined]; % XZ is the (combined) augmented vector of differential and algebraic variables
XZp_t_local_finish = zeros(size(XZ_t_local_finish_actual));                     % 'p' in this variable name stands for time-derivative (i.e. "prime"); This vector contains the derivatives of both states and algebraic variables. This is needed by IDA. However, the actual model equations (in this paper) only need the first n_diff variables (i.e. only the portion of this combined vector containing only the time-derivatives). Please refer to the function that implements the model equations if you need further clarification
clear X_init_truth Z_init_fsolve_refined;

IDAInit(@batchChemReactorModel,t_local_start,XZ_t_local_finish_actual,XZp_t_local_finish,ida_options_struct); % does it not call the stateequation? (only algebraic????? Not sure)
[~, overall_system_vector_at_t0, ~] = IDACalcIC(t_local_start + 0.1,'FindAlgebraic'); % (Find consistent initial conditions) might have to change the 10 to a different horizon

differential_states_results_matrix_stored = nan(6,ceil(tf/Ts));
differential_states_results_matrix_stored(:,1) = overall_system_vector_at_t0(1:6);

algebraic_states_results_stored_matrix = nan(4,ceil(tf/Ts));
algebraic_states_results_stored_matrix(:,1) = overall_system_vector_at_t0(7:end);

measurement_outputs_matrix_stored = nan(4,ceil(tf/Ts));
measurement_outputs_matrix_stored(:,1) = overall_system_vector_at_t0(1:4);


t_local_finish_actual_vector = zeros(ceil(tf/Ts),1);
t_local_finish_actual_vector(1) = t_local_finish_actual;
iter = 0;
while (t_local_finish_actual < tf)
    iter = iter + 1;
    IDAInit(@batchChemReactorModel,t_local_start,XZ_t_local_finish_actual,XZp_t_local_finish,ida_options_struct); % does it not call the stateequation? (only algebraic????? Not sure)
    [~, t_local_finish_actual, XZ_t_local_finish_actual] = IDASolve(t_local_finish_actual,'Normal');
    XZp_t_local_finish = IDAGet('DerivSolution',t_local_finish_actual,1)';
    t_local_start = t_local_finish_actual;
    t_local_finish_actual = t_local_start + Ts;
    algebraic_states_results_stored_matrix(:,iter) = XZ_t_local_finish_actual(7:end);
    differential_states_results_matrix_stored(:,iter) = XZ_t_local_finish_actual(1:6);
    measurement_outputs_matrix_stored(:,iter) = XZ_t_local_finish_actual(1:4);
    t_local_finish_actual_vector(iter) = t_local_finish_actual;
end
clear time_step_iter soln_vec_at_t;

%% Plot simulation results (all components of the augmented state vector, i.e. differential and algebraic states)
for plot_no = 1:10
    figure(plot_no);clf;
    if plot_no<=6
        plot(t_local_finish_actual_vector(1:end-1)/3600,differential_states_results_matrix_stored(plot_no,1:end-1),'o-');
        label_str = ['State Variable x_' num2str(plot_no-1)];
    else
        plot(t_local_finish_actual_vector(1:end-1)/3600,algebraic_states_results_stored_matrix(plot_no-6,1:end-1),'o-');
        label_str = ['Algebraic Variable z_' num2str(plot_no-7)];
    end
    xlabel('Time [hours]'); ylabel(label_str);
    title(['Sim result: ' label_str]);axis square;
end
% Adjust figure properties to match the graph reported in paper
figure(1); ylim([0.7 1.6]); xlim([0 0.35]);
% clear plot_no label_str;

%% Extract the individual jacobians from the overall system jacobian
Fx = J(1:6,1:6);
Fz = J(1:6,7:end);
gx = J(7:end,1:6);
gz = J(7:end,7:end);

%% Function to Compute the Jacobian of the complete augmented system
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