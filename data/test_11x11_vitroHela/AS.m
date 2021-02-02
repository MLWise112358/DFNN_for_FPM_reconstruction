%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Adaptive step-size iteration (PIE with adpative alpha)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for num = 1 : LED_num_x*LED_num_y
    
    
    % Calcute the step-size for the next iteration
    if(num ==1 && iter==1)
        Alpha = 1;
        Err_bef = inf;
    elseif(num ==1 && iter>1)
        eval Calc_stepsize;
    end
    
    % Get the subspecturm
    kx = round(kxky_index(num,1));
    ky = round(kxky_index(num,2));
    Subspecturm = F(Fcenter_Y+ky-fix(M/2): Fcenter_Y+ky+ceil(M/2)-1, Fcenter_X+kx-fix(N/2):Fcenter_X+kx+ceil(M/2)-1);
%     Subspecturm = double(Subspecturm);
    Abbr_Subspecturm = (1/Mag_image)^2.*Subspecturm.*CTF; %
    %加入pupil的原始子图
    Uold = ifft2(fftshift(Abbr_Subspecturm.*pupil));
    compensate =  Mag_image^2;
    if num > 49
        RAW_compensate = RAW(:,:,Image_num_index(num))*compensate * 1.0;
    else
        RAW_compensate = RAW(:,:,Image_num_index(num))*compensate * 1.0;
    end
    Unew = RAW_compensate.*(Uold./abs(Uold));
    
    Abbr_Subspecturm_corrected = fftshift(fft2(Unew)).*CTF./pupil;
%     Subspecturm = F(Fcenter_Y+ky-fix(M/2): Fcenter_Y+ky+ceil(M/2)-1, Fcenter_X+kx-fix(N/2):Fcenter_X+kx+ceil(M/2)-1);
%     Subspecturm = double(Subspecturm);
    
%     Subspecturm = Subspecturm + (Alpha.*conj(pupil)./((abs(pupil).^2 + eps.^2))).*(Abbr_Subspecturm_corrected - Abbr_Subspecturm.*(pupil));
%     pupil = pupil + (1.*conj(Subspecturm)./(max(max(abs(Subspecturm).^2)))).*(Abbr_Subspecturm_corrected - Abbr_Subspecturm.*(pupil));

%     加入光瞳函数修正的EPRY算法
    Subspecturm = Subspecturm + Alpha.*conj(pupil)./(max(max(abs(pupil).^2))).*(Abbr_Subspecturm_corrected - Abbr_Subspecturm.*pupil);%
%     pupil = pupil + Alpha.*conj(Subspecturm)./(max(max(abs(Subspecturm).^2))).*(Abbr_Subspecturm_corrected - Abbr_Subspecturm.*pupil);%
% 
%     Subspecturm = Subspecturm + Alpha*(abs(pupil).*conj(pupil)./(max(max(pupil)).*(abs(pupil).^2 + 1e-8))).*(Abbr_Subspecturm_corrected - Abbr_Subspecturm.*(pupil));
%     pupil = pupil + Alpha*(abs(Subspecturm).*conj(Subspecturm)./(max(max(Subspecturm)).*abs(Subspecturm).^2 + 1e-8)).*(Abbr_Subspecturm_corrected - Abbr_Subspecturm.*(pupil));
   
    
%     %加入光瞳函数修正的Newton算法
%     Abbr_Subspecturm_corrected = fftshift(fft2(Unew)).*CTF./pupil;
%     Subspecturm = F(Fcenter_Y+ky-fix(M/2): Fcenter_Y+ky+ceil(M/2)-1, Fcenter_X+kx-fix(N/2):Fcenter_X+kx+ceil(M/2)-1);
%     Subspecturm = double(Subspecturm);
%     
%     Subspecturm = Subspecturm + Alpha.*sum(sum(abs(pupil))).*conj(pupil)./(max(max(abs(pupil))).*(sum(sum(abs(pupil).^2))+eps)).*(Abbr_Subspecturm_corrected - Abbr_Subspecturm.*pupil);
%     pupil = pupil + Alpha.*sum(sum(abs(Subspecturm))).*conj(Subspecturm)./(max(max(abs(Subspecturm))).*(sum(sum(abs(Subspecturm).^2))+eps)).*(Abbr_Subspecturm_corrected - Abbr_Subspecturm.*pupil);
%     
    F(Fcenter_Y+ky-fix(M/2): Fcenter_Y+ky+ceil(M/2)-1, Fcenter_X+kx-fix(N/2):Fcenter_X+kx+ceil(M/2)-1) = Subspecturm;

    
    
    Result = ifft2(fftshift(F));
%     Result_pupil = pupil;
end

