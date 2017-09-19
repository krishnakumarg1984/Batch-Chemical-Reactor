% vim: set nospell nowrap textwidth=0 wrapmargin=0 formatoptions-=t:
function [overall_residual_vector, flag, new_data] = batchChemReactorModel(t,XZ,XZp,ida_user_data_struct) % x_tot contains both x (differential states), z (algebraic variables) and derivates of states
    % This function implements all the model equations, returning their
    % residual. This function will be repeatedly called by IDA in a time-stepping loop.
    % Note. X: vector of differential (time-derivative) states, Z: vector of algebraic states

    %% dummy variables for IDA
    flag     = 0;  % These two variables are not used but required by IDA(s) solver.
    new_data = []; % These two variables are not used but required by IDA(s) solver.

    %% Unpack data from the 'UserData' structure into various fields
    model_params = ida_user_data_struct.model_params;
    n_diff       = ida_user_data_struct.n_diff;
    time_profile = ida_user_data_struct.time_profile;
    Temp_profile = ida_user_data_struct.Temp_profile;

    X  = [XZ(1);XZ(2);XZ(3);XZ(4);XZ(5);XZ(6)]; % state vector (differential variables only)
    Z  = [XZ(7);XZ(8);XZ(9);XZ(10)];            % Build the array of algebraic variables
    Xp = XZp(1:n_diff);                         % retain only the first n_diff components of the combined derivative vector. Only these are required in the model equations below

    %% Compute dynamically varying coefficients in the model equations (i.e. those coeffs which are function of time, t)
    T_degC = interp1(time_profile,Temp_profile,t);  % Temperature at time t (degC)
    k1     = model_params.alpha_1*exp(-model_params.E1_over_R/(T_degC+273));
    k2     = model_params.alpha_2*exp(-model_params.E2_over_R/(T_degC+273));
    k3     = k1;
    km1    = model_params.alpha_m1*exp(-model_params.Em1_over_R/(T_degC+273));
    km3    = 0.5*km1;

    %% Actual model equations of the non-linear DAE model
    % Components of residual vector of differential state variables (i.e. time-domain derivative variables)
    % Refer to paper for the equations themselves
    res_X_dot(1) = Xp(1) - (-k2*X(2)*Z(2));
    res_X_dot(2) = Xp(2) - (-k1*X(2)*X(6) + km1*Z(4) - k2*X(2)*Z(2));
    res_X_dot(3) = Xp(3) - (k2*X(2)*Z(2) + k3*X(4)*X(6) - km3*Z(3));
    res_X_dot(4) = Xp(4) - (-k3*X(4)*X(6) + km3*Z(3));
    res_X_dot(5) = Xp(5) - (k1*X(2)*X(6) - km1*Z(4));
    res_X_dot(6) = Xp(6) - (-k1*X(2)*X(6) + km1*Z(4) -k3*X(4)*X(6) + km3*Z(3));

    % Vector of residuals of algebraic states is computed and returned by the following function
    % (remember: this is a nice way to re-use this function, which was initially employed for refining algebraic guess   % initially using 'fsolve', before we let the model evolve in time)
    res_Z = algebraicEquations(Z,X,model_params); % returns the residuals of the algebraic variables (i.e. implements the algebraic equations)

    %% Assemble the overall augmented residual vector of the system [n_diff+n_alg x 1] column vector (the first n_diff components are residuals of differential variables and the rest of the components are residuals of algebraic variables)
    overall_residual_vector = [res_X_dot';res_Z']; % (NOTE: transposing a quick solution for scalar ode/dae systems)

end