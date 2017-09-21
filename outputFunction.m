function h = outputFunction(XZ,user_data_struct)

    X = XZ(1:user_data_struct.n_diff); % extract the 1st n_diff variables from the user data struct 
    h = X(1:user_data_struct.n_outputs); % in this specific example, the 1st four states (differential variables) themselves were chosen as the outputs. Normally, this is a complicated non-linear function.
end