function h = outputFunction_only_algebraic_vars(XZ,user_data_struct) % h is a 'vector valued' function.

    h = [XZ(user_data_struct.n_diff+1); XZ(user_data_struct.n_diff+2); XZ(user_data_struct.n_diff+4)];
end