% vim: set nospell nowrap textwidth=0 wrapmargin=0 formatoptions-=t:
function [dx_tot, flag, new_data] = batchChemReactorModel(t,XZ,XZp,ida_user_data) % x_tot contains both x (differential states), z (algebraic variables) and derivates of states

    flag     = 0;  % These two variables are not used but required by IDA(s) solver.
    new_data = []; % These two variables are not used but required by IDA(s) solver.

    % Retreive data from the UserData field of IDA(s)
    param  = ida_user_data.param;
    t0     = ida_user_data.t0;
    tf     = ida_user_data.tf;
    n_diff = ida_user_data.n_diff;
    % n_alg  = ida_user_data.n_alg;

    residual_array = []; % Empty the total array of residuals.

    Xp = XZp(1:n_diff);

    %% Extract the state variables, algebraic variables and state derivatives
    %from the corresponding combined vectors passed in
    x0 = XZ(1);
    x1 = XZ(2);
    x2 = XZ(3);
    x3 = XZ(4);
    x4 = XZ(5);
    x5 = XZ(6);

    z0 = XZ(7);
    z1 = XZ(8);
    z2 = XZ(9);
    z3 = XZ(10);

    x0_dot = Xp(1);
    x1_dot = Xp(2);
    x2_dot = Xp(3);
    x3_dot = Xp(4);

    Z0 = [z0;z1;z2;z3]; % Build the array of initial conditions for the algebraic equations (calculated apriori by fsolve)

    res_Z = algebraicStates(Z0,ce,cs_barrato,Q,T,film,param); % Get the residuals from the algebraic equations