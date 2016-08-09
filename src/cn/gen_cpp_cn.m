% Matlab function which generates a c++ header that provides data for Color Names features
% color names data taken from https://github.com/ihpdep/samf
function gen_cpp_cn()
    %load w2crs variable 32768x10
    load w2crs.mat  

    fid = fopen('cn_data.cpp', 'w');
    fprintf(fid, '#include "cnfeat.hpp" \n\n');

    fprintf(fid, 'float CNFeat::p_id2feat[32768][10] = { \n');

    for id = 1:size(w2crs,1)
        fprintf(fid, '\t{');
        fprintf(fid, '%f, ', w2crs(id,1:end-1));
        fprintf(fid, '%f', w2crs(id,end));
        if (id < size(w2crs,1))
            fprintf(fid, '},\n');
        else
            fprintf(fid, '}\n');
        end
    end

    fprintf(fid, '\t}; //static p_id2feat[32768][10]\n');
    fclose(fid);

end