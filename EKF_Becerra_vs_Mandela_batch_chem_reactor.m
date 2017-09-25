% This code implements the Batch Chemical Reactor (non-linear set of DAEs)
% discussed in the journal article:
% "Applying the extended Kalman filter to systems described by nonlinear
% differential-algebraic equations", V.M. Becerra, P.D., Roberts, G.W.
% Griffiths, Control Engineering Practice, 2001 pp 267-281

% However, the special "Mandela EKF" proposed in "Recursive state
% estimation techniques for nonlinear differential algebraic systems", Ravi
% Kumar Mandela, Raghunathan Rengaswamy, Shankar Narasimhan, Lakshmi N.
% Sridhar, Chem. Eng. Science, 2010, issue 65, pp4548-4556 is also
% implemented for comparison purposes.

% Authors: Krishnakumar Gopalakrishnan, Davide M. Raimondo
% License: MIT License

%% Basic settings for MATLAB IDE, plotting, numerical display etc.
% NOTE: In this problem statement/code, 'X', 'Z' etc. are vectors, whereas 'x' , 'z' etc. represent scalar quantities
clear;clc; format short g; format compact;
close all;
set(0,'defaultaxesfontsize',12,'defaultaxeslinewidth',2,'defaultlinelinewidth',2.5,'defaultpatchlinewidth',2,'DefaultFigureWindowStyle','docked');
tic;
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
Ts = 60; % How often is simulation results needed ?
enable_sensor_noise = 1;
enable_process_noise = 1;

% Specify simulation interval
t0 = 0;          % initial time at start of simulation
tf = 0.35*3600;  % simulation end time

n_diff = 6; n_alg  = 4; % How many differential and algebraic variables in this DAE problem
n_inputs = 1;           % How many input variables
n_outputs = 4;          % How many output variables
Z_init_guess = zeros(n_alg,1);  % user's initial guess for algebraic variables (this will be refined by fsolve before time-stepping)

% Initial Conditions for Differential States (truth) The word "truth" is used because we are going to implement an EKF soon, and we will need this model to serve as out "experiment", i.e. "truth"
x0_init_truth = 1.5776;
x1_init_truth = 8.32;
x2_init_truth = 0;
x3_init_truth = 0;
x4_init_truth = 0;
x5_init_truth = 0.00142;
% x5_init_truth = 0.0142;

% Assemble these initial components into a vector representing initial values of differential states (init, i.e. at time t=0)
X_init_truth = [x0_init_truth;x1_init_truth;x2_init_truth;x3_init_truth;x4_init_truth;x5_init_truth];
clear x0_init_truth x1_init_truth x2_init_truth x3_init_truth x4_init_truth x5_init_truth;

% Define absolute and relative tolerances for time-stepping solver (IDA)
opt_IDA.AbsTol = 1e-6;
opt_IDA.RelTol = 1e-6;

% 'fsolve' is used to solve for the initial values of algebraic variables by keeping the differential variables constant at their initial values.
opt_fsolve             = optimset;
opt_fsolve.Display     = 'off';
opt_fsolve.FunValCheck = 'on';
% opt_fsolve.TolX      = 1e-6;
% opt_fsolve.TolFun      = 1e-6;

[Z_init_truth_fsolve_refined,~,~,~,~] = fsolve(@algebraicEquations,Z_init_guess,opt_fsolve,X_init_truth,model_params);

%% Other required settings to be configured for the given problem
% Tell IDA how to identify the algebraic and differential variables in the combined XZ (differential+algebraic state) vector.
id = [ones(n_diff,1);zeros(n_alg,1)]; % id:1-> differential variables,id:0-> algebraic variables.

% Additional user-data that may be passed to IDA as additional parameters
user_data_struct.model_params = model_params;
user_data_struct.n_diff       = n_diff;
user_data_struct.time_profile = time_profile;
user_data_struct.Temp_profile = Temp_profile;
user_data_struct.n_outputs    = n_outputs;
user_data_struct.Ts           = Ts;
user_data_struct.enable_process_noise = enable_process_noise;

%% Analytical Jacobian (using CasADi) for automatic differentiation for time-stepping (not for EKF linearisation)
% Import casadi framework
import casadi.*
% Define the symbolic variables.
XZsym  = SX.sym('XZ',[sum(n_diff)+sum(n_alg),1]);
XZpsym = SX.sym('XZp',[sum(n_diff)+sum(n_alg),1]);
cj     = SX.sym('cj',1);

user_data_struct.process_noise_flag = 'noise_free';
[sym_XZ_residuals_vector_IDA, ~, ~] = batchChemReactorModel_IDA(0,XZsym,XZpsym,user_data_struct); % Get the model equations in implicit form in a symbolic way
sym_Jac_Diff_algebraic_States_and_stateDerivs_IDA = jacobian(sym_XZ_residuals_vector_IDA,XZsym) + cj*jacobian(sym_XZ_residuals_vector_IDA,XZpsym); % Compute the  symbolic Jacobian (Please refer to the Sundials' IDA user guide for further information about the Jacobian structure).

JacFun = Function('fJ',{XZsym,cj},{sym_Jac_Diff_algebraic_States_and_stateDerivs_IDA}); % Define a function for the Jacobian evaluation for a given set of differential and algebraic variables.
user_data_struct.fJ = JacFun; % Store the function into a structure such that IDA will use it for the evaluation of the Jacobian matrix (see the definition of the function djacfn at the end of this file).

%% EKF linearisation of the model
Usym   = SX.sym('U',n_inputs);
user_data_struct.Usym = Usym;
[sym_XZ_residuals_vector_ekf, ~, ~, sym_rhs_stateeqn] = batchChemReactorModel_ekf(0,XZsym,XZpsym,user_data_struct); % Get the model equations in implicit form in a symbolic way
sym_Jac_ekf_Fg_wrt_XZ = jacobian(sym_XZ_residuals_vector_ekf,XZsym); % jacobian of the combined F and g system with respect to X and Z vector
sym_Jac_ekf_Fg_wrt_u = jacobian(sym_XZ_residuals_vector_ekf,Usym); % jacobian of the combined F and g system with respect to the input vector, U

Fx = sym_Jac_ekf_Fg_wrt_XZ(1:n_diff,1:n_diff);         % no. of diff eqns x no. of diff variables (states)
Fz = sym_Jac_ekf_Fg_wrt_XZ(1:n_diff,n_diff+1:end);     % no. of diff eqns x no. of algebraic variables
gx = sym_Jac_ekf_Fg_wrt_XZ(n_diff+1:end,1:n_diff);     % no. of algebraic eqns x no. of diff variables (states)
gz = sym_Jac_ekf_Fg_wrt_XZ(n_diff+1:end,n_diff+1:end); % no. of algebraic. eqns x no. of algebraic variables

Fu = sym_Jac_ekf_Fg_wrt_u(1:n_diff,1:n_inputs);        % no. of diff eqns x no. of inputs
gu = sym_Jac_ekf_Fg_wrt_u(n_diff+1:end,1:n_inputs);    % no. of algebraic eqns x no. of inputs

A_ekf_symbolic = Fz*inv(gz)*gx - Fx;
A_ekf_fcn = Function('A_ekf',{XZsym,Usym},{A_ekf_symbolic}); % This function will later enable numerical evaluation of the appropriate symbolic Jacobian for a given set of differential and algebraic variables.. This helps in variable name correspondence

sym_h_vector_ekf = outputFunction(XZsym,user_data_struct);
sym_Jac_ekf_h_wrt_XZ = jacobian(sym_h_vector_ekf,XZsym);  % only the first n_diff states are present in the output vector
hx = sym_Jac_ekf_h_wrt_XZ(:,1:n_diff);                    % Be careful - The matrix indices/slices used are only applicable for this problem (batch chemical reactor)
C_ekf_symbolic = hx;
C_ekf_fcn = Function('C_ekf',{XZsym},{C_ekf_symbolic}); % This function will later enable numerical evaluation of the appropriate symbolic Jacobian for a given set of differential and algebraic variables.. This helps in variable name correspondence

%% Mandela EKF linearisation of the model
fx = jacobian(sym_rhs_stateeqn,XZsym(1:n_diff));     % A_MandelaEKF
fz = jacobian(sym_rhs_stateeqn,XZsym(n_diff+1:end)); % B_MandelaEKF
A_aug_MandelaEKF_symbolic = [fx fz; -inv(gz)*gx*fx -inv(gz)*gx*fz];
A_aug_MandelaEKF_fcn = Function('A_Mandela_EKF',{XZsym,Usym},{A_aug_MandelaEKF_symbolic}); % This function will later enable numerical evaluation of the appropriate symbolic Jacobian for a given set of differential and algebraic variables.. This helps in variable name correspondence

Gamma_bottom_MandelaEKF_symbolic = -inv(gz)*gx;
Gamma_bottom_MandelaEKF_fcn = Function('Gamma_bottom_Mandela_EKF',{XZsym,Usym},{Gamma_bottom_MandelaEKF_symbolic});

H_aug_MandelaEKF_symbolic = sym_Jac_ekf_h_wrt_XZ;
H_aug_MandelaEKF_fcn = Function('H_aug_MandelaEKF',{XZsym},{H_aug_MandelaEKF_symbolic});

%% EKF parameterisation
P = diag(0.003*ones(1,n_diff));          % Tuning parameters of the EKF
Q = diag(0.0001*ones(1,n_diff));       % Tuning parameters of the EKF
% Q = diag(1e-5*ones(1,n_diff));           % Tuning parameters of the EKF
user_data_struct.chol_Q =  chol(Q);
% Q = 1*diag(1e-6*ones(1,n_diff));       % Tuning parameters of the EKF
R = diag([0.0004,0.0004,0.0001,0.0001]); % Tuning parameters of the EKF

S = chol(P,'lower'); L = chol(Q,'lower');D = chol(R,'lower');

%% Mandela EKF parameterisation
P_MandelaEKF = diag([0.004*ones(1,n_diff) zeros(1,n_alg)]);

%% Initialise time-step and prepare for time-stepping of the true model that produces "experimental" data
% clear ans opt_IDA;
t_local_start = t0;
t_local_finish = t_local_start + Ts;

%% EKF initialisation (common to square root EKF and Mandela EKF)
x0_init_ekf = 1.6;
x1_init_ekf = 8.3;
x2_init_ekf = 0;
x3_init_ekf = 0;
x4_init_ekf = 0;
x5_init_ekf = 0.0014;

X_init_ekf = [x0_init_ekf;x1_init_ekf;x2_init_ekf;x3_init_ekf;x4_init_ekf;x5_init_ekf];
[Z_init_ekf_fsolve_refined,~,~,~,~] = fsolve(@algebraicEquations,Z_init_guess,opt_fsolve,X_init_ekf,model_params);
clear Z_init_guess;

%% Compute consistent state and algebraic initial variables for square-roor EKF
XZ_ekf_t_local_finish = [X_init_ekf;Z_init_ekf_fsolve_refined];

XZp_ekf_zeros_init_guess = zeros(size(XZ_ekf_t_local_finish));    % 'p' in this variable name stands for time-derivative (i.e. "prime"); This vector contains the derivatives of both states and algebraic variables. This is needed by IDA. However, the actual model equations (in this paper) only need the first n_diff variables (i.e. only the portion of this combined vector containing only the time-derivatives). Please refer to the function that implements the model equations if you need further clarification

XZp_ekf_t_local_finish = -1*(batchChemReactorModel_IDA(0,XZ_ekf_t_local_finish,XZp_ekf_zeros_init_guess,user_data_struct)); % Evaluate the residual vector at t=0, and multiply it by -1
XZp_ekf_t_local_finish(n_diff+1:end) = 0;  % The previous line gave correct value of state derivatives, but returns an incorrect calculation of algebraic time-derivatives

user_data_struct.process_noise_flag = 'noise_free';
ida_options_struct = compute_updated_ida_options(opt_IDA,id,user_data_struct);
IDAInit(@batchChemReactorModel_IDA,t_local_start,XZ_ekf_t_local_finish,XZp_ekf_t_local_finish,ida_options_struct); % does it not call the stateequation? (only algebraic????? Not sure)
[~, XZ_ekf_t_local_finish, ~] = IDACalcIC(t_local_start + 0.1,'FindAlgebraic'); % (Find consistent initial conditions) might have to change the 10 to a different horizon

state_vars_ekf_estimates_stored = nan(n_diff,ceil(tf/Ts));
state_vars_ekf_estimates_stored(:,1) = XZ_ekf_t_local_finish(1:n_diff);

%% EKF Mandela initialisation
XZ_MandelaEKF_t_local_finish = [X_init_ekf;Z_init_ekf_fsolve_refined];
XZp_MandelaEKF_t_local_finish = -1*(batchChemReactorModel_IDA(0,XZ_MandelaEKF_t_local_finish,XZp_ekf_zeros_init_guess,user_data_struct)); % Evaluate the residual vector at t=0, and multiply it by -1
XZp_MandelaEKF_t_local_finish(n_diff+1:end) = 0;  % The previous line gave correct value of state derivatives, but returns an incorrect calculation of algebraic time-derivatives

user_data_struct.process_noise_flag = 'noise_free';
ida_options_struct = compute_updated_ida_options(opt_IDA,id,user_data_struct);
IDAInit(@batchChemReactorModel_IDA,t_local_start,XZ_MandelaEKF_t_local_finish,XZp_MandelaEKF_t_local_finish,ida_options_struct);
[~, XZ_MandelaEKF_t_local_finish, ~] = IDACalcIC(t_local_start + 0.1,'FindAlgebraic'); % (Find consistent initial conditions) might have to change the 10 to a different horizon

state_vars_MandelaEKF_estimates_stored = nan(n_diff,ceil(tf/Ts));
state_vars_MandelaEKF_estimates_stored(:,1) = XZ_MandelaEKF_t_local_finish(1:n_diff);

%% True plant initialisation
XZ_truth_t_local_finish  = [X_init_truth;Z_init_truth_fsolve_refined]; % XZ is the (combined) augmented vector of differential and algebraic variables
XZp_truth_zeros_init_guess = zeros(size(XZ_truth_t_local_finish));      % 'p' in this variable name stands for time-derivative (i.e. "prime"); This vector contains the derivatives of both states and algebraic variables. This is needed by IDA. However, the actual model equations (in this paper) only need the first n_diff variables (i.e. only the portion of this combined vector containing only the time-derivatives). Please refer to the function that implements the model equations if you need further clarification
XZp_truth_t_local_finish = -1*(batchChemReactorModel_IDA(0,XZ_truth_t_local_finish,XZp_truth_zeros_init_guess,user_data_struct)); % Evaluate the residual vector at t=0, and multiply it by -1
XZp_truth_t_local_finish(n_diff+1:end) = 0;  % The previous line gave correct value of state derivatives, but returns an incorrect calculation of algebraic time-derivatives
% XZp_truth_t_local_finish = -1*(batchChemReactorModel_IDA(0,XZ_truth_t_local_finish,[zeros(n_diff,1);Z_init_truth_fsolve_refined],user_data_struct)); % Evaluate the residual vector at t=0, and multiply it by -1
user_data_struct.process_noise_flag = 'pseudo_white_noise';
ida_options_struct = compute_updated_ida_options(opt_IDA,id,user_data_struct);
IDAInit(@batchChemReactorModel_IDA,t_local_start,XZ_truth_t_local_finish,XZp_truth_t_local_finish,ida_options_struct); % does it not call the stateequation? (only algebraic????? Not sure)
[~, XZ_truth_t_local_finish, ~] = IDACalcIC(t_local_start + 0.1,'FindAlgebraic'); % (Find consistent initial conditions) might have to change the 10 to a different horizon

clear X_init_truth Z_init_truth_fsolve_refined;

state_vars_truth_results_stored = nan(n_diff,ceil(tf/Ts));
state_vars_truth_results_stored(:,1) = XZ_truth_t_local_finish(1:n_diff);

alg_vars_truth_results_stored = nan(n_alg,ceil(tf/Ts));
alg_vars_truth_results_stored(:,1) = XZ_truth_t_local_finish(n_diff+1:end);

measurement_outputs_stored = nan(n_outputs,ceil(tf/Ts));
measurement_outputs_stored(:,1) = XZ_truth_t_local_finish(1:n_outputs);

sim_time_vector = zeros(ceil(tf/Ts),1);
sim_time_vector(1) = t0;

%% Time-stepper code
T_degC_init = interp1(time_profile,Temp_profile,t0);  % Temperature at time t (degC)  % For time-stepping by IDA (or even by IDACalcIC), a symbolic 'U' is not acceptable.
A = full(A_ekf_fcn(XZ_ekf_t_local_finish,T_degC_init));
phi = expm(A*Ts);

k = 1;      % iteration (sample number)
measured_outputs = nan(n_outputs,1);

while (t_local_finish < tf)
    IDAInit(@batchChemReactorModel_IDA,t_local_start,XZ_truth_t_local_finish,XZp_truth_t_local_finish,ida_options_struct); % does it not call the stateequation? (only algebraic????? Not sure)
    [~, t_local_finish, XZ_truth_t_local_finish] = IDASolve(t_local_finish,'Normal');
    XZp_truth_t_local_finish = IDAGet('DerivSolution',t_local_finish,1);
    
    state_vars_truth_results_stored(:,k+1) = XZ_truth_t_local_finish(1:n_diff);
    alg_vars_truth_results_stored(:,k+1) = XZ_truth_t_local_finish(n_diff+1:end);
    
    true_model_outputs = outputFunction(XZ_truth_t_local_finish,user_data_struct);
    measured_outputs = true_model_outputs;   % Temporaily turn off sensor noise
    if enable_sensor_noise == 1
        measured_outputs = measured_outputs + chol(R)*randn(n_outputs,1); % Measurements are corrupted by zero mean, gaussian noise
    end
    
    measurement_outputs_stored(:,k+1) = measured_outputs;
    sim_time_vector(k+1) = t_local_finish;
    
    % EKF steps
    L1 = [phi*S L];
    R1 = lq(L1);
    S = R1(1:n_diff,1:n_diff);
    
    % Compute apriori state estimate x_hat_minus at the current time-sample
    user_data_struct.process_noise_flag = 'noise_free';
    ida_options_struct = compute_updated_ida_options(opt_IDA,id,user_data_struct);
    IDAInit(@batchChemReactorModel_IDA,t_local_start,XZ_ekf_t_local_finish,XZp_ekf_t_local_finish,ida_options_struct);
    [~, t_local_finish, XZ_ekf_t_local_finish] = IDASolve(t_local_finish,'Normal');
    %     XZp_ekf_t_local_finish = IDAGet('DerivSolution',t_local_finish,1)';
    C = full(C_ekf_fcn(XZ_ekf_t_local_finish));
    L2 = [D C*S;zeros(n_diff,n_alg) S];
    R2 = lq(L2);
    W = R2(1:n_outputs,1:n_outputs);
    K_bar = R2(n_outputs+1:end,1:n_outputs); % dim: nx x ny
    S = R2(n_outputs+1:end,n_outputs+1:end); % dim: nx x ny
    
    K = K_bar/W; % Kalman gain
    
    % update (differential variables), i.e. states using measurement and kalman gain
    XZ_ekf_t_local_finish(1:n_diff) = XZ_ekf_t_local_finish(1:n_diff) + K*(measured_outputs - outputFunction(XZ_ekf_t_local_finish,user_data_struct));
    state_vars_ekf_estimates_stored(:,k+1) = XZ_ekf_t_local_finish(1:n_diff);
    
    % update algebraic variables & state derivatives
    [XZ_ekf_t_local_finish(n_diff+1:end),~,~,~,~] = fsolve(@algebraicEquations,XZ_ekf_t_local_finish(n_diff+1:end),opt_fsolve,state_vars_ekf_estimates_stored(:,k+1),model_params);
    XZp_ekf_t_local_finish = -1*(batchChemReactorModel_IDA(0,XZ_ekf_t_local_finish,[zeros(n_diff,1);XZ_ekf_t_local_finish(n_diff+1:end)],user_data_struct)); % Evaluate the residual vector at t=0, and multiply it by -1
    
    T_degC_at_t = interp1(time_profile,Temp_profile,t_local_finish);  % Temperature at time, t (in degC)
    A = full(A_ekf_fcn(XZ_ekf_t_local_finish,T_degC_at_t));
    phi = expm(A*Ts);
    
    %% Mandela EKF steps
    user_data_struct.process_noise_flag = 'noise_free';
    ida_options_struct = compute_updated_ida_options(opt_IDA,id,user_data_struct);
    IDAInit(@batchChemReactorModel_IDA,t_local_start,XZ_MandelaEKF_t_local_finish,XZp_MandelaEKF_t_local_finish,ida_options_struct);
    [~, ~, XZ_MandelaEKF_t_local_finish] = IDASolve(t_local_finish,'Normal');
    A_aug_MandelaEKF = full(A_aug_MandelaEKF_fcn(XZ_MandelaEKF_t_local_finish,T_degC_at_t));
    phi_MandelaEKF = expm(A_aug_MandelaEKF*Ts);
    Gamma_MandelaEKF = [eye(n_diff);full(Gamma_bottom_MandelaEKF_fcn(XZ_MandelaEKF_t_local_finish,T_degC_at_t))];
    P_MandelaEKF = phi_MandelaEKF*P_MandelaEKF*phi_MandelaEKF' + Gamma_MandelaEKF*Q*Gamma_MandelaEKF';
    H_aug_MandelaEKF_at_present_oppoint = full(H_aug_MandelaEKF_fcn(XZ_MandelaEKF_t_local_finish));
    K_aug_MandelaEKF = P_MandelaEKF*H_aug_MandelaEKF_at_present_oppoint'*inv(H_aug_MandelaEKF_at_present_oppoint*P_MandelaEKF*H_aug_MandelaEKF_at_present_oppoint' + R);
    XZ_MandelaEKF_t_local_finish = XZ_MandelaEKF_t_local_finish + K_aug_MandelaEKF*(measured_outputs - outputFunction(XZ_MandelaEKF_t_local_finish,user_data_struct));
    %     XZ_MandelaEKF_t_local_finish(1:n_diff) = mandela_augmented_states_update(1:n_diff);
    state_vars_MandelaEKF_estimates_stored(:,k+1) = XZ_MandelaEKF_t_local_finish(1:n_diff);
    
    [XZ_MandelaEKF_t_local_finish(n_diff+1:end),~,~,~,~] = fsolve(@algebraicEquations,XZ_MandelaEKF_t_local_finish(n_diff+1:end),opt_fsolve,state_vars_MandelaEKF_estimates_stored(:,k+1),model_params);
    XZp_MandelaEKF_t_local_finish = -1*(batchChemReactorModel_IDA(0,XZ_MandelaEKF_t_local_finish,[zeros(n_diff,1);XZ_MandelaEKF_t_local_finish(n_diff+1:end)],user_data_struct));
    P_MandelaEKF = (eye(n_diff+n_alg) - K_aug_MandelaEKF*H_aug_MandelaEKF_at_present_oppoint)*P_MandelaEKF;
    %% Prepare for next iteration
    k = k + 1;
    
    t_local_start = t_local_finish;
    t_local_finish = t_local_start + Ts;
    user_data_struct.process_noise = 'pseudo_white_noise';
    ida_options_struct = compute_updated_ida_options(opt_IDA,id,user_data_struct);
    
    %     fprintf('The simulation time is now : %3.5f seconds\n',t);
end
clear time_step_iter soln_vec_at_t;

%% Plot truth and estimated results
close all;
for plot_no = 1:n_diff
    figure(plot_no);clf;
    plot(sim_time_vector/3600,state_vars_truth_results_stored(plot_no,:),'s-','linewidth',1.5);hold on;
    plot(sim_time_vector/3600,state_vars_ekf_estimates_stored(plot_no,:),'ro-','linewidth',2);
    plot(sim_time_vector/3600,state_vars_MandelaEKF_estimates_stored(plot_no,:),'kx-','linewidth',2.5);
    hold off;
    label_str = ['State Variable x_' num2str(plot_no-1)];
    xlabel('Time [hours]'); ylabel(label_str);xlim([0 tf/3600]);
    title(['Sim result: ' label_str]);axis square;
    legend('truth','Becerra sqrt EKF','Mandela EKF','location','best');
end

% % Adjust figure properties to match the graph reported in paper
% figure(1); ylim([0.7 1.6]); xlim([0 0.35]);
% clear plot_no label_str;
toc;shg;
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

function ida_options_struct = compute_updated_ida_options(opt_IDA,id,user_data_struct)
ida_options_struct = IDASetOptions('RelTol', opt_IDA.RelTol,...
    'AbsTol'        , opt_IDA.AbsTol,...
    'MaxNumSteps'   , 1500,...
    'VariableTypes' , id,...
    'UserData'      , user_data_struct,...
    'JacobianFn'    , @djacfn,...
    'LinearSolver'  , 'Dense');
end
% vim: set nospell nowrap textwidth=0 wrapmargin=0 formatoptions-=t:
