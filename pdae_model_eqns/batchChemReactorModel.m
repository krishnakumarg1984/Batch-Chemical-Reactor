% vim: set nospell nowrap textwidth=0 wrapmargin=0 formatoptions-=t:
function [overall_XZ_residual_vector] = batchChemReactorModel(~,XZ,XZp,user_data_struct,T_degC) % x_tot contains both x (differential states), z (algebraic variables) and derivates of states
    % This function actually performs the implementation of all the model equations, returning their residuals.
    % Note. X: vector of differential (time-derivative) states, Z: vector of algebraic states

    %% Unpack data from the 'UserData' structure into various fields
    model_params = user_data_struct.model_params;
    n_diff       = user_data_struct.n_diff;

    X  = [XZ(1);XZ(2);XZ(3);XZ(4);XZ(5);XZ(6)]; % state vector (differential variables only)
    Z  = [XZ(7);XZ(8);XZ(9);XZ(10)];            % Build the array of algebraic variables
    Xp = XZp(1:n_diff);                         % retain only the first n_diff components of the combined derivative vector. Only these are required in the model equations below

    %% Compute dynamically varying coefficients in the model equations (i.e. those coeffs which are function of time, t)
    k1     = model_params.alpha_1*exp(-model_params.E1_over_R/(T_degC+273));
    k2     = model_params.alpha_2*exp(-model_params.E2_over_R/(T_degC+273));
    k3     = k1;
    km1    = model_params.alpha_m1*exp(-model_params.Em1_over_R/(T_degC+273));
    km3    = 0.5*km1;

    % overall_residual_vector = [];
    %% Actual model equations of the non-linear DAE model
    % Components of residual vector of differential state variables (i.e. time-domain derivative variables)
    % Refer to paper for the equations themselves
    rhs1 = -k2*X(2)*Z(2)                                      + user_data_struct.process_noise(1);
    rhs2 = -k1*X(2)*X(6) + km1*Z(4) - k2*X(2)*Z(2)            + user_data_struct.process_noise(2); 
    rhs3 =  k2*X(2)*Z(2) + k3*X(4)*X(6) - km3*Z(3)            + user_data_struct.process_noise(3);
    rhs4 = -k3*X(4)*X(6) + km3*Z(3)                           + user_data_struct.process_noise(4);
    rhs5 =  k1*X(2)*X(6) - km1*Z(4)                           + user_data_struct.process_noise(5);
    rhs6 = -k1*X(2)*X(6) + km1*Z(4) - k3*X(4)*X(6) + km3*Z(3) + user_data_struct.process_noise(6); 
    
    res_X_dot1 = Xp(1) - rhs1;
    res_X_dot2 = Xp(2) - rhs2;
    res_X_dot3 = Xp(3) - rhs3;
    res_X_dot4 = Xp(4) - rhs4;
    res_X_dot5 = Xp(5) - rhs5;
    res_X_dot6 = Xp(6) - rhs6;

    res_X_dot = [res_X_dot1;res_X_dot2;res_X_dot3;res_X_dot4;res_X_dot5;res_X_dot6];

    % Vector of residuals of algebraic states is computed and returned by the following function
    % (remember: this is a nice way to re-use this function, which was initially employed for refining algebraic guess   % initially using 'fsolve', before we let the model evolve in time)
    res_Z = algebraicEquations(Z,X,model_params); % returns the residuals of the algebraic variables (i.e. implements the algebraic equations)

    %% Assemble the overall augmented residual vector of the system [n_diff+n_alg x 1] column vector (the first n_diff components are residuals of differential variables and the rest of the components are residuals of algebraic variables)
    overall_XZ_residual_vector = [res_X_dot;res_Z];
    
end
