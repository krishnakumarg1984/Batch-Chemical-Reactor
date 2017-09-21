% This code implements the Batch Chemical Reactor (non-linear set of DAEs)
% discussed in the journal article:
% "Applying the extended Kalman filter to systems described by nonlinear
% differential-algebraic equations", V.M. Becerra, P.D., Roberts, G.W.
% Griffiths, Control Engineering Practice, 2001 pp 267-281

% Authors: Krishnakumar Gopalakrishnan, Davide M. Raimondo
% License: MIT License

%% Basic settings for MATLAB IDE, plotting, numerical display etc.
% NOTE: In this problem statement/code, 'X', 'Z' etc. are vectors, whereas 'x' , 'z' etc. represent scalar quantities
clear;close all;clc; format long g; format compact;
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
n_inputs = 1;           % How many input variables
Z_init_guess = zeros(n_alg,1);  % user's initial guess for algebraic variables (this will be refined by fsolve before time-stepping)

% Initial Conditions for Differential States (truth) The word "truth" is used because we are going to implement an EKF soon, and we will need this model to serve as out "experiment", i.e. "truth"
x0_init_truth = 1.5776;
x1_init_truth = 8.32;
x2_init_truth = 0;
x3_init_truth = 0;
x4_init_truth = 0;
x5_init_truth = 0.00142;

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

[Z_init_truth_fsolve_refined,~,~,~,~] = fsolve(@algebraicEquations,Z_init_guess,opt_fsolve,X_init_truth,model_params);

%% Other required settings to be configured for the given problem
% Tell IDA how to identify the algebraic and differential variables in the combined XZ (differential+algebraic state) vector.
% id:1-> differential variables,id:0-> algebraic variables.
id = [ones(n_diff,1);zeros(n_alg,1)];

% Additional user-data that may be passed to IDA as additional parameters
user_data_struct.model_params = model_params;
user_data_struct.n_diff       = n_diff;
user_data_struct.time_profile = time_profile;
user_data_struct.Temp_profile = Temp_profile;

%% Analytical Jacobian (using CasADi) for automatic differentiation for time-stepping (not for EKF linearisation)
% Import casadi framework
import casadi.*
% Define the symbolic variables.
XZsym  = SX.sym('XZ',[sum(n_diff)+sum(n_alg),1]);
XZpsym = SX.sym('XZp',[sum(n_diff)+sum(n_alg),1]);
Usym   = SX.sym('U',n_inputs);
cj     = SX.sym('cj',1);

user_data_struct.Usym = Usym;

[sym_XZ_residuals_vector_IDA, ~, ~] = batchChemReactorModel_IDA(0,XZsym,XZpsym,user_data_struct); % Get the model equations in implicit form in a symbolic way
sym_Jac_Diff_algebraic_States_and_stateDerivs_IDA = jacobian(sym_XZ_residuals_vector_IDA,XZsym) + cj*jacobian(sym_XZ_residuals_vector_IDA,XZpsym); % Compute the  symbolic Jacobian (Please refer to the Sundials' IDA user guide for further information about the Jacobian structure).

% Define a function for the Jacobian evaluation for a given set of
% differential and algebraic variables.
JacFun = Function('fJ',{XZsym,cj},{sym_Jac_Diff_algebraic_States_and_stateDerivs_IDA});

% Store the function into a structure such that IDA will use it for the
% evaluation of the Jacobian matrix (see the definition of the function
% djacfn at the end of this file).
user_data_struct.fJ = JacFun;

% Define the options for Sundials
ida_options_struct = IDASetOptions('RelTol', opt_IDA.RelTol,...
    'AbsTol'        , opt_IDA.AbsTol,...
    'MaxNumSteps'   , 1500,...
    'VariableTypes' , id,...
    'UserData'      , user_data_struct,...
    'JacobianFn'    , @djacfn,...
    'LinearSolver'  , 'Dense');

% clear model_params n_diff n_alg time_profile Temp_profile;
clear id ida_user_data_struct;

%% Jacobian matrices For EKF linearisation of the model
[sym_XZ_residuals_vector_ekf, ~, ~] = batchChemReactorModel_ekf(0,XZsym,XZpsym,user_data_struct); % Get the model equations in implicit form in a symbolic way
sym_Jac_ekf_Fg_wrt_XZ = jacobian(sym_XZ_residuals_vector_ekf,XZsym); % jacobian of the combined F and g system with respect to X and Z vector
sym_Jac_ekf_Fg_wrt_u = jacobian(sym_XZ_residuals_vector_ekf,Usym); % jacobian of the combined F and g system with respect to the input vector, U

Fx = sym_Jac_ekf_Fg_wrt_XZ(1:n_diff,1:n_diff);         % no. of diff eqns x no. of diff variables (states)
Fz = sym_Jac_ekf_Fg_wrt_XZ(1:n_diff,n_diff+1:end);     % no. of diff eqns x no. of algebraic variables
gx = sym_Jac_ekf_Fg_wrt_XZ(n_diff+1:end,1:n_diff);     % no. of algebraic eqns x no. of diff variables (states)
gz = sym_Jac_ekf_Fg_wrt_XZ(n_diff+1:end,n_diff+1:end); % no. of algebraic. eqns x no. of algebraic variables

Fu = sym_Jac_ekf_Fg_wrt_u(1:n_diff,1:n_inputs);        % no. of diff eqns x no. of inputs
gu = sym_Jac_ekf_Fg_wrt_u(n_diff+1:end,1:n_inputs);    % no. of algebraic eqns x no. of inputs

A_symbolic = Fz*inv(gz)*gx - Fx;
A_ekf_function = Function('A_ekf',{XZsym,Usym},{A_symbolic}); % This function will later enable numerical evaluation of the appropriate symbolic Jacobian for a given set of differential and algebraic variables.. This helps in variable name correspondence

%% Initialise time-step and prepare for time-stepping of the true model that produces "experimental" data
% clear ans opt_IDA;
t_local_start = t0;
t_local_finish = t_local_start + Ts;

% sneaky little way to get the state+algebraic time derivatives at initial time
% Build the initial (for t=0) combined (i.e. augmented diff & alg) vectors & the vector of their derivatives to be used by the IDA integrator
XZ_truth_t_local_finish  = [X_init_truth;Z_init_truth_fsolve_refined]; % XZ is the (combined) augmented vector of differential and algebraic variables
XZp_truth_zeros_init_guess = zeros(size(XZ_truth_t_local_finish));      % 'p' in this variable name stands for time-derivative (i.e. "prime"); This vector contains the derivatives of both states and algebraic variables. This is needed by IDA. However, the actual model equations (in this paper) only need the first n_diff variables (i.e. only the portion of this combined vector containing only the time-derivatives). Please refer to the function that implements the model equations if you need further clarification
XZp_truth_t_local_finish = -1*(batchChemReactorModel_IDA(0,XZ_truth_t_local_finish,XZp_truth_zeros_init_guess,user_data_struct)); % Evaluate the residual vector at t=0, and multiply it by -1
IDAInit(@batchChemReactorModel_IDA,t_local_start,XZ_truth_t_local_finish,XZp_truth_t_local_finish,ida_options_struct); % does it not call the stateequation? (only algebraic????? Not sure)
[~, XZ_truth_t_local_finish, ~] = IDACalcIC(t_local_start + 0.1,'FindAlgebraic'); % (Find consistent initial conditions) might have to change the 10 to a different horizon
% XZp_truth_t_local_finish = IDAGet('DerivSolution',t_local_finish,1)';

clear X_init_truth Z_init_truth_fsolve_refined;

diff_states_truth_results_matrix_stored = nan(6,ceil(tf/Ts));
diff_states_truth_results_matrix_stored(:,1) = XZ_truth_t_local_finish(1:6);

alg_states_truth_results_stored_matrix = nan(4,ceil(tf/Ts));
alg_states_truth_results_stored_matrix(:,1) = XZ_truth_t_local_finish(7:end);

measurement_outputs_matrix_stored = nan(4,ceil(tf/Ts));
measurement_outputs_matrix_stored(:,1) = XZ_truth_t_local_finish(1:4);

t_local_finish_vector = zeros(ceil(tf/Ts),1);
t_local_finish_vector(1) = t_local_finish;

%% EKF initialisation
x0_init_ekf = 1.6;
x1_init_ekf = 8.3;
x2_init_ekf = 0;
x3_init_ekf = 0;
x4_init_ekf = 0;
x5_init_ekf = 0.0014;

sigma_1 = 0.02; % std dev of Gaussian noise added to output1
sigma_2 = 0.02; % std dev of Gaussian noise added to output2
sigma_3 = 0.02; % std dev of Gaussian noise added to output3
sigma_4 = 0.01; % std dev of Gaussian noise added to output4

X_init_ekf = [x0_init_ekf;x1_init_ekf;x2_init_ekf;x3_init_ekf;x4_init_ekf;x5_init_ekf];
[Z_init_ekf_fsolve_refined,~,~,~,~] = fsolve(@algebraicEquations,Z_init_guess,opt_fsolve,X_init_ekf,model_params);
clear Z_init_guess opt_fsolve model_params;

XZ_ekf_t_local_finish = [X_init_ekf;Z_init_ekf_fsolve_refined];
XZp_ekf_zeros_init_guess = zeros(size(XZ_ekf_t_local_finish));    % 'p' in this variable name stands for time-derivative (i.e. "prime"); This vector contains the derivatives of both states and algebraic variables. This is needed by IDA. However, the actual model equations (in this paper) only need the first n_diff variables (i.e. only the portion of this combined vector containing only the time-derivatives). Please refer to the function that implements the model equations if you need further clarification
XZp_ekf_t_local_finish = -1*(batchChemReactorModel_IDA(0,XZ_ekf_t_local_finish,XZp_ekf_zeros_init_guess,user_data_struct)); % Evaluate the residual vector at t=0, and multiply it by -1
IDAInit(@batchChemReactorModel_IDA,t_local_start,XZ_ekf_t_local_finish,XZp_ekf_t_local_finish,ida_options_struct); % does it not call the stateequation? (only algebraic????? Not sure)
[~, XZ_ekf_t_local_finish, ~] = IDACalcIC(t_local_start + 0.1,'FindAlgebraic'); % (Find consistent initial conditions) might have to change the 10 to a different horizon
% XZp_ekf_t_local_finish = IDAGet('DerivSolution',t_local_finish,1)';

T_degC_init = interp1(time_profile,Temp_profile,t_local_start);  % Temperature at time t (degC)  % For time-stepping by IDA (or even by IDACalcIC), a symbolic 'U' is not acceptable. 
A = full(A_ekf_function(XZ_ekf_t_local_finish,T_degC_init));
phi = expm(A*Ts);
P = diag(0.003*ones(1,6));                % Tuning parameters of the EKF
S = sqrtm(P);
Q = diag(0.0001*ones(1,6));               % Tuning parameters of the EKF
L = sqrtm(Q);
R  = diag([0.0004,0.0004,0.0001,0.0001]); % Tuning parameters of the EKF

%% Time-stepper code
k = 1;      % iteration (sample number)

while (t_local_finish < tf)
    
    % IDA time-stepping
    IDAInit(@batchChemReactorModel_IDA,t_local_start,XZ_truth_t_local_finish,XZp_truth_t_local_finish,ida_options_struct); % does it not call the stateequation? (only algebraic????? Not sure)
    [~, t_local_finish, XZ_truth_t_local_finish] = IDASolve(t_local_finish,'Normal');
    XZp_truth_t_local_finish = IDAGet('DerivSolution',t_local_finish,1)';
    
    alg_states_truth_results_stored_matrix(:,k+1) = XZ_truth_t_local_finish(7:end);
    diff_states_truth_results_matrix_stored(:,k+1) = XZ_truth_t_local_finish(1:6);
    measurement_outputs_matrix_stored(:,k+1) = XZ_truth_t_local_finish(1:4);
    t_local_finish_vector(k+1) = t_local_finish;
    
    
    % EKF steps
    L1 = [phi*S L];
    R1 = lq(L1);
    S = R1(1:n_diff,1:n_diff);
    
    % Compute apriori state estimate x_hat_minus at the current time-sample
    IDAInit(@batchChemReactorModel_IDA,t_local_start,XZ_ekf_t_local_finish,XZp_ekf_t_local_finish,ida_options_struct); % does it not call the stateequation? (only algebraic????? Not sure)
    [~, t_local_finish, XZ_ekf_t_local_finish] = IDASolve(t_local_finish,'Normal');
    XZp_ekf_t_local_finish = IDAGet('DerivSolution',t_local_finish,1)';

    k = k + 1;
    t_local_start = t_local_finish;
    t_local_finish = t_local_start + Ts;
    

end
clear time_step_iter soln_vec_at_t;

%% Plot simulation results (all components of the augmented state vector, i.e. differential and algebraic states)
for plot_no = 1:10
    figure(plot_no);clf;
    if plot_no<=6
        plot(t_local_finish_vector(1:end-1)/3600,diff_states_truth_results_matrix_stored(plot_no,1:end-1),'o-');
        label_str = ['State Variable x_' num2str(plot_no-1)];
    else
        plot(t_local_finish_vector(1:end-1)/3600,alg_states_truth_results_stored_matrix(plot_no-6,1:end-1),'o-');
        label_str = ['Algebraic Variable z_' num2str(plot_no-7)];
    end
    xlabel('Time [hours]'); ylabel(label_str);
    title(['Sim result: ' label_str]);axis square;
end
% Adjust figure properties to match the graph reported in paper
figure(1); ylim([0.7 1.6]); xlim([0 0.35]);
% clear plot_no label_str;


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
flag     = 0;
new_data = [];
end
% vim: set nospell nowrap textwidth=0 wrapmargin=0 formatoptions-=t: