% vim: set nospell nowrap textwidth=0 wrapmargin=0 formatoptions-=t:
function [overall_XZ_residual_vector, flag, new_data] = batchChemReactorModel_IDA(t,XZ,XZp,ida_user_data_struct) % x_tot contains both x (differential states), z (algebraic variables) and derivates of states
    % This function aims to return the residual of the combined diff+alg states residual.
    % residual. This code will be repeatedly called by IDA in a time-stepping loop.
    % Note. X: vector of differential (time-derivative) states, Z: vector of algebraic states
    
    %% dummy variables for IDA
    flag     = 0;  % These two variables are not used but required by IDA(s) solver.
    new_data = []; % These two variables are not used but required by IDA(s) solver.

    %% Evaluate temperature at given time, t (running sim time)
    time_profile = ida_user_data_struct.time_profile;
    Temp_profile = ida_user_data_struct.Temp_profile;
    
    %% Compute dynamic input coefficients (i.e. those input coeffs which are function of time, t)
%         T_degC = ida_user_data_struct.Usym;
    T_degC = interp1(time_profile,Temp_profile,t,'linear','extrap');  % Temperature at time t (degC)  % For time-stepping by IDA (or even by IDACalcIC), a symbolic 'U' is not acceptable. 
    
    %% Assemble the overall augmented residual vector of the system [n_diff+n_alg x 1] column vector (the first n_diff components are residuals of differential variables and the rest of the components are residuals of algebraic variables)
    [overall_XZ_residual_vector] = batchChemReactorModel(t,XZ,XZp,ida_user_data_struct,T_degC);

end
