% This code implements a modification of the Batch Chemical Reactor (non-linear set of DAEs)
% discussed in the journal article:
% "Applying the extended Kalman filter to systems described by nonlinear
% differential-algebraic equations", V.M. Becerra, P.D., Roberts, G.W.
% Griffiths, Control Engineering Practice, 2001 pp 267-281

% However, the special "Mandela EKF" proposed in "Recursive state
% estimation techniques for nonlinear differential algebraic systems", Ravi
% Kumar Mandela, Raghunathan Rengaswamy, Shankar Narasimhan, Lakshmi N.
% Sridhar, Chem. Eng. Science, 2010, issue 65, pp4548-4556 is implemented
% instead of the Becerra EKF in order to understand the system's
% performance when using only algebraic measurements

% Authors: Krishnakumar Gopalakrishnan, Davide M. Raimondo
% License: MIT License

%% Basic settings for MATLAB IDE, plotting, numerical display etc.
% NOTE: In this problem statement/code, 'X', 'Z' etc. are vectors, whereas 'x' , 'z' etc. represent scalar quantities
clear;clc; format short g; format compact;
close all;
set(0,'defaultaxesfontsize',12,'defaultaxeslinewidth',2,'defaultlinelinewidth',2.5,'defaultpatchlinewidth',2,'DefaultFigureWindowStyle','docked');
% tic;
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
t0 = 0;          % initial time at start of simulation [sec]
tf = 0.35*3600;  % simulation end time [sec]

n_diff = 6; n_alg  = 4; % no. of differential and algebraic variables in this DAE problem
n_inputs = 1;           % no. of input variables
n_outputs = 1;          % no. of output variables

X_init_truth = [1.5776;8.32;0;0;0;0.00142]; % Vector representing initial values of differential states (init, i.e. at time t=0)
Z_init_guess = zeros(n_alg,1);  % user's initial guess for algebraic variables (this will be refined by fsolve before time-stepping)

% Define absolute and relative tolerances for time-stepping solver (IDA)
opt_IDA.AbsTol = 1e-6;
opt_IDA.RelTol = 1e-6;

% 'fsolve' is used to solve the system of algebraic equations (g(x,z) = 0) and obtain the algebraic variables by keeping the differential variables constant at their latest values.
opt_fsolve             = optimset;
opt_fsolve.Display     = 'off';
opt_fsolve.FunValCheck = 'on';
% opt_fsolve.TolX      = 1e-6;
% opt_fsolve.TolFun    = 1e-6;

[Z_init_truth_fsolve_refined,~,~,~,~] = fsolve(@algebraicEquations,Z_init_guess,opt_fsolve,X_init_truth,model_params);

%% Set up a few other required settings
id = [ones(n_diff,1);zeros(n_alg,1)]; % 1-> differential variables, 0-> algebraic variables. % Tell IDA how to identify the algebraic and differential variables in the combined XZ (differential+algebraic state) vector.

% Additional user-data that needs to be passed to IDA as additional parameters for this given (DAE) problem
user_data_struct.model_params = model_params;
user_data_struct.n_diff       = n_diff;
user_data_struct.time_profile = time_profile;
user_data_struct.Temp_profile = Temp_profile;
user_data_struct.Ts           = Ts;
user_data_struct.enable_process_noise = enable_process_noise;

%% Analytical Jacobian of noise-free system (using CasADi's automatic differentiation) 
% This section is mainly used for time-stepping (not for EKF linearisation)
import casadi.*
XZsym  = SX.sym('XZ', [sum(n_diff)+sum(n_alg),1]);
XZpsym = SX.sym('XZp',[sum(n_diff)+sum(n_alg),1]);
cj     = SX.sym('cj',1);

user_data_struct.process_noise_flag = 'noise_free';
[sym_XZ_residuals_vector_IDA, ~, ~] = batchChemReactorModel_IDA(0,XZsym,XZpsym,user_data_struct); % Get the model's residual (equations in implicit form [F g] = 0) in a symbolic way
sym_Jac_Diff_algebraic_States_and_stateDerivs_IDA = jacobian(sym_XZ_residuals_vector_IDA,XZsym) + cj*jacobian(sym_XZ_residuals_vector_IDA,XZpsym); % Compute the  symbolic Jacobian (Please refer to the Sundials' IDA user guide for further information about the Jacobian structure).
clear sym_XZ_residuals_vector_IDA;

JacFun_IDA = Function('fJ',{XZsym,cj},{sym_Jac_Diff_algebraic_States_and_stateDerivs_IDA}); % Define a function for the Jacobian evaluation for a given set of differential and algebraic variables.
user_data_struct.fJ = JacFun_IDA; % Store the function into a structure such that IDA will use it for the evaluation of the Jacobian matrix (see the definition of the function djacfn at the end of this file).
clear cj sym_Jac_Diff_algebraic_States_and_stateDerivs_IDA JacFun_IDA;

%% EKF linearisation of the model
Usym   = SX.sym('U',n_inputs);
user_data_struct.Usym = Usym;
[sym_XZ_residuals_vector_ekf, ~, ~, sym_rhs_stateeqn] = batchChemReactorModel_ekf(0,XZsym,XZpsym,user_data_struct); % the "_ekf" model includes symbolic inputs, whereas the "_ida" model cannot accept symbolic inputs, U
sym_Jac_ekf_Fg_wrt_XZ = jacobian(sym_XZ_residuals_vector_ekf,XZsym); % jacobian of the combined (implicit) F and g system (i.e. written as residuals) with respect to X and Z vector
clear sym_XZ_residuals_vector_ekf;

gx = sym_Jac_ekf_Fg_wrt_XZ(n_diff+1:end,1:n_diff);     % no. of algebraic eqns x no. of diff variables (states)
gz = sym_Jac_ekf_Fg_wrt_XZ(n_diff+1:end,n_diff+1:end); % no. of algebraic. eqns x no. of algebraic variables
clear sym_Jac_ekf_Fg_wrt_XZ;

fx = jacobian(sym_rhs_stateeqn,XZsym(1:n_diff));     % A_MandelaEKF
fz = jacobian(sym_rhs_stateeqn,XZsym(n_diff+1:end)); % B_MandelaEKF

A_aug_MandelaEKF_symbolic = [fx fz; -inv(gz)*gx*fx -inv(gz)*gx*fz];
A_aug_MandelaEKF_fcn = Function('A_Mandela_EKF',{XZsym,Usym},{A_aug_MandelaEKF_symbolic}); % This function will later enable numerical evaluation of the appropriate symbolic Jacobian for a given set of differential and algebraic variables.. This helps in variable name correspondence
clear fx fz A_aug_MandelaEKF_symbolic;

Gamma_bottom_MandelaEKF_symbolic = -inv(gz)*gx;
Gamma_bottom_MandelaEKF_fcn = Function('Gamma_bottom_Mandela_EKF',{XZsym,Usym},{Gamma_bottom_MandelaEKF_symbolic});
clear Usym gx gz Gamma_bottom_MandelaEKF_symbolic;

sym_h_vector_ekf = outputFunction_only_algebraic_vars(XZsym,user_data_struct);
n_outputs = length(sym_h_vector_ekf);
user_data_struct.n_outputs    = n_outputs;
sym_Jac_ekf_h_wrt_XZ = jacobian(sym_h_vector_ekf,XZsym);
clear sym_h_vector_ekf;
H_aug_MandelaEKF_symbolic = sym_Jac_ekf_h_wrt_XZ;
clear sym_Jac_ekf_h_wrt_XZ;
H_aug_MandelaEKF_fcn = Function('H_aug_MandelaEKF',{XZsym},{H_aug_MandelaEKF_symbolic});
clear XZsym XZpsym H_aug_MandelaEKF_symbolic sym_Jac_ekf_h_wrt_XZ;

%% Set initial time for simulation time-stepping and prepare for time-stepping of the true model that produces "experimental" data
t_local_start = t0;
t_local_finish = t_local_start + Ts;

%% EKF parameterisation & initialisation
Q = diag(0.0001*ones(1,n_diff));         % Tuning parameters of the EKF
% R = diag([0.0004,0.0004,0.0001,0.0001]); % Tuning parameters of the EKF
R = diag(0.01*ones(n_outputs,1)); % Tuning parameters of the EKF
P_MandelaEKF = diag([0.004*ones(1,n_diff) zeros(1,n_alg)]);

X_init_ekf = [1.6;8.3;0;0;0;0.0014];
[Z_init_ekf_fsolve_refined,~,~,~,~] = fsolve(@algebraicEquations,Z_init_guess,opt_fsolve,X_init_ekf,model_params);

XZ_MandelaEKF_t_local_finish = [X_init_ekf;Z_init_ekf_fsolve_refined]; % initial value of augmented vector
clear X_init_ekf Z_init_ekf_fsolve_refined Z_init_guess;

XZp_MandelaEKF_zeros_init_guess = zeros(size(XZ_MandelaEKF_t_local_finish));
user_data_struct.process_noise_flag = 'noise_free';
ida_options_struct = compute_updated_ida_options(opt_IDA,id,user_data_struct);
XZp_MandelaEKF_t_local_finish = -1*(batchChemReactorModel_IDA(0,XZ_MandelaEKF_t_local_finish,XZp_MandelaEKF_zeros_init_guess,user_data_struct)); % Evaluate the residual vector at t=0, and multiply it by -1
XZp_MandelaEKF_t_local_finish(n_diff+1:end) = 0;  % Although, the previous line of code did return the correct value for state derivatives, unfortunately it returns an (arbitrary, probably incorrect) calculation of algebraic time-derivatives
clear XZp_MandelaEKF_zeros_init_guess;

IDAInit(@batchChemReactorModel_IDA,t_local_start,XZ_MandelaEKF_t_local_finish,XZp_MandelaEKF_t_local_finish,ida_options_struct);
[~, XZ_MandelaEKF_t_local_finish, ~] = IDACalcIC(t_local_start + 0.1,'FindAlgebraic'); % (Find consistent initial conditions) might have to change the 10 to a different horizon

estimated_state_vars_MandelaEKF_stored = nan(n_diff,ceil(tf/Ts));
estimated_state_vars_MandelaEKF_stored(:,1) = XZ_MandelaEKF_t_local_finish(1:n_diff);

%% True plant initialisation (ensure that this section is executed just prior to time-stepping loop, because of the requirement of consistent noise signal)

XZ_truth_t_local_finish  = [X_init_truth;Z_init_truth_fsolve_refined];  % XZ is the (combined) augmented vector of differential and algebraic variables
clear X_init_truth Z_init_truth_fsolve_refined;

XZp_truth_zeros_init_guess = zeros(size(XZ_truth_t_local_finish));      % 'p' in this variable name stands for time-derivative (i.e. "prime"); This vector contains the derivatives of both states and algebraic variables. This is needed by IDA. However, the actual model equations (in this paper) only need the first n_diff variables (i.e. only the portion of this combined vector containing only the time-derivatives). Please refer to the function that implements the model equations if you need further clarification

user_data_struct.process_noise_flag = 'pseudo_white_noise';
ida_options_struct = compute_updated_ida_options(opt_IDA,id,user_data_struct);
XZp_truth_t_local_finish = -1*(batchChemReactorModel_IDA(0,XZ_truth_t_local_finish,XZp_truth_zeros_init_guess,user_data_struct)); % Evaluate the residual vector at t=0, and multiply it by -1
clear XZp_truth_zeros_init_guess;

XZp_truth_t_local_finish(n_diff+1:end) = 0;  % Although, the previous line of code did return the correct value for state derivatives, unfortunately it returns an (arbitrary, probably incorrect) calculation of algebraic time-derivatives
IDAInit(@batchChemReactorModel_IDA,t_local_start,XZ_truth_t_local_finish,XZp_truth_t_local_finish,ida_options_struct); % does it not call the stateequation? (only algebraic????? Not sure)
[~, XZ_truth_t_local_finish, ~] = IDACalcIC(t_local_start + 0.1,'FindAlgebraic'); % (Find consistent initial conditions) might have to change the 10 to a different horizon

state_vars_truth_results_stored = nan(n_diff,ceil(tf/Ts));
state_vars_truth_results_stored(:,1) = XZ_truth_t_local_finish(1:n_diff);

alg_vars_truth_results_stored = nan(n_alg,ceil(tf/Ts));
alg_vars_truth_results_stored(:,1) = XZ_truth_t_local_finish(n_diff+1:end);

sim_time_vector = nan(ceil(tf/Ts),1);
sim_time_vector(1) = t0;

%% Time-stepper code
T_degC_init = interp1(time_profile,Temp_profile,t0);  % Temperature at time t (degC)  % For time-stepping by IDA (or even by IDACalcIC), a symbolic 'U' is not acceptable.
clear t0;

k = 1;      % iteration number (sample number)

while (t_local_finish < tf)
    IDAInit(@batchChemReactorModel_IDA,t_local_start,XZ_truth_t_local_finish,XZp_truth_t_local_finish,ida_options_struct); % does it not call the stateequation? (only algebraic????? Not sure)
    [~, ~, XZ_truth_t_local_finish] = IDASolve(t_local_finish,'Normal');
    XZp_truth_t_local_finish = IDAGet('DerivSolution',t_local_finish,1);  % Will be used as initial derivatives for next time-step, i.e. in the code couple of lines above
    
    state_vars_truth_results_stored(:,k+1) = XZ_truth_t_local_finish(1:n_diff);
    alg_vars_truth_results_stored(:,k+1) = XZ_truth_t_local_finish(n_diff+1:end);
    
    true_model_outputs = outputFunction_only_algebraic_vars(XZ_truth_t_local_finish,user_data_struct);
    
    measured_outputs = true_model_outputs;   % Temporaily turn off sensor noise (i.e. set outputs to be same as model outputs for now)
    if enable_sensor_noise == 1
        measured_outputs = measured_outputs + chol(R)*randn(n_outputs,1); % Measurements are corrupted by zero mean, gaussian noise
    end
        
    %% EKF steps
    user_data_struct.process_noise_flag = 'noise_free';
    ida_options_struct = compute_updated_ida_options(opt_IDA,id,user_data_struct);
    IDAInit(@batchChemReactorModel_IDA,t_local_start,XZ_MandelaEKF_t_local_finish,XZp_MandelaEKF_t_local_finish,ida_options_struct);
    [~, ~, XZ_MandelaEKF_t_local_finish] = IDASolve(t_local_finish,'Normal');
    
    T_degC_at_t_local_finish = interp1(time_profile,Temp_profile,t_local_finish);  % Temperature at current time, t (degC)
    A_aug_MandelaEKF = full(A_aug_MandelaEKF_fcn(XZ_MandelaEKF_t_local_finish,T_degC_at_t_local_finish));
    phi_MandelaEKF = expm(A_aug_MandelaEKF*Ts);
    Gamma_MandelaEKF = [eye(n_diff);full(Gamma_bottom_MandelaEKF_fcn(XZ_MandelaEKF_t_local_finish,T_degC_at_t_local_finish))];
    P_MandelaEKF = phi_MandelaEKF*P_MandelaEKF*phi_MandelaEKF' + Gamma_MandelaEKF*Q*Gamma_MandelaEKF';
    H_aug_MandelaEKF_at_present_oppoint = full(H_aug_MandelaEKF_fcn(XZ_MandelaEKF_t_local_finish));
    K_aug_MandelaEKF = P_MandelaEKF*H_aug_MandelaEKF_at_present_oppoint'*inv(H_aug_MandelaEKF_at_present_oppoint*P_MandelaEKF*H_aug_MandelaEKF_at_present_oppoint' + R);
    XZ_MandelaEKF_t_local_finish = XZ_MandelaEKF_t_local_finish + K_aug_MandelaEKF*(measured_outputs - outputFunction(XZ_MandelaEKF_t_local_finish,user_data_struct));
    estimated_state_vars_MandelaEKF_stored(:,k+1) = XZ_MandelaEKF_t_local_finish(1:n_diff);
    
    [XZ_MandelaEKF_t_local_finish(n_diff+1:end),~,~,~,~] = fsolve(@algebraicEquations,XZ_MandelaEKF_t_local_finish(n_diff+1:end),opt_fsolve,estimated_state_vars_MandelaEKF_stored(:,k+1),model_params);
    XZp_MandelaEKF_t_local_finish = -1*(batchChemReactorModel_IDA(0,XZ_MandelaEKF_t_local_finish,[zeros(n_diff,1);XZ_MandelaEKF_t_local_finish(n_diff+1:end)],user_data_struct));
    P_MandelaEKF = (eye(n_diff+n_alg) - K_aug_MandelaEKF*H_aug_MandelaEKF_at_present_oppoint)*P_MandelaEKF;
    
    %% Prepare for next iteration
    sim_time_vector(k+1) = t_local_finish;  
    k = k + 1;
    
    t_local_start = t_local_finish;
    t_local_finish = t_local_start + Ts;
    user_data_struct.process_noise = 'pseudo_white_noise';
    ida_options_struct = compute_updated_ida_options(opt_IDA,id,user_data_struct);
    
end

clear time_step_iter soln_vec_at_t model_params opt_fsolve id A_aug_MandelaEKF_fcn Gamma_bottom_MandelaEKF_fcn;
clear H_aug_MandelaEKF_fcn opt_IDA XZ_MandelaEKF_t_local_finish;
clear time_profile Temp_profile user_data_struct;
clear t_local_start t_local_finish;
clear ans Q R ida_options_struct measured_outputs;
clear k XZ_truth_t_local_finish true_model_outputs T_degC_at_t_local_finish K_aug_MandelaEKF;
clear A_aug_MandelaEKF phi_MandelaEKF Gamma_MandelaEKF P_MandelaEKF H_aug_MandelaEKF_at_present_oppoint;

%% Plot truth and estimated results
close all;
for plot_no = 1:n_diff
    figure(plot_no);clf;
    plot(sim_time_vector/3600,state_vars_truth_results_stored(plot_no,:),'s-','linewidth',1.5);hold on;
    plot(sim_time_vector/3600,estimated_state_vars_MandelaEKF_stored(plot_no,:),'kx-','linewidth',2.5);
    hold off;
    label_str = ['State Variable x_' num2str(plot_no-1)];
    xlabel('Time [hours]'); ylabel(label_str);xlim([0 tf/3600]);
    title(['Sim result: ' label_str]);axis square;
    legend('truth','Mandela EKF','location','best');
end

% Adjust figure properties to match the graph reported in paper
figure(1); ylim([0.7 1.6]); xlim([0 0.35]);shg;
clear plot_no label_str;
% toc;

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
