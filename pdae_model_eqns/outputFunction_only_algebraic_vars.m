function h = outputFunction_only_algebraic_vars(XZ,user_data_struct) % h is a 'vector valued' function.

    h = XZ(user_data_struct.n_diff+3);
%     h = [h; XZ(user_data_struct.n_diff+4)];
end