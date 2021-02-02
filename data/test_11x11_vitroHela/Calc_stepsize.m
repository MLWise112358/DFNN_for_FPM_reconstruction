%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Calcute the step-size for the next iteration
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Calculate current cost function (it can also be incrementially
% accumulated without additional computation)

Err_now = 0;

for num = 1 : LED_num_x*LED_num_y
    
    kx = round(kxky_index(num,1));
    ky = round(kxky_index(num,2));
    
    Subspecturm = F(Fcenter_Y+ky-fix(M/2): Fcenter_Y+ky+ceil(M/2)-1, ...
        Fcenter_X+kx-fix(N/2):Fcenter_X+kx+ceil(M/2)-1);
    Abbr_Subspecturm = Subspecturm.*Aperture_fun;
    
    Curr_lowres = (ifft2(fftshift(Abbr_Subspecturm)));
    Err_now = Err_now + sum(sum((abs(Curr_lowres)-(RAW(:,:,Image_num_index(num)))).^2));
    
end

if((Err_bef-Err_now)/Err_bef<0.01)
    
    % Reduce the stepsize when no sufficient progress is made
    Alpha = Alpha / 2;
    
    % Stop the iteration when Alpha is less than 0.001(convergenced)
    if(Alpha<0.001)
        Alpha = 0;
    end
    
end

Err_bef = Err_now;
