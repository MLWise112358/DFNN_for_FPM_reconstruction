%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Adaptive step-size iteration (PIE with adpative alpha)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for num = 1 : LED_num_x*LED_num_y
%     Alpha= 1;
%     Calcute the step-size for the next iteration
    if(num ==1 && iter==1)
        Alpha = 1;
        Err_bef = inf;
    elseif(num ==1 && iter>1)
        eval Calc_stepsize;
    end
% %     
    % Get the subspecturm
    kx = round(kxky_index(num,1));
    ky = round(kxky_index(num,2));
    Subspecturm = F(Fcenter_Y+ky-fix(M/2): Fcenter_Y+ky+ceil(M/2)-1, Fcenter_X+kx-fix(N/2):Fcenter_X+kx+ceil(M/2)-1);
%     Subspecturm1 = F1(Fcenter_Y+ky-fix(M/2): Fcenter_Y+ky+ceil(M/2)-1, Fcenter_X+kx-fix(N/2):Fcenter_X+kx+ceil(M/2)-1);
    Subspecturm = double(Subspecturm);
%     Subspecturm1 = double(Subspecturm1);
    Abbr_Subspecturm = (1/Mag_image)^2.*Subspecturm.*CTF;
%     Abbr_Subspecturm1 = Subspecturm1.*Aperture_fun;

%     % Real space modulus constraint
%     Uold = ifft2(fftshift(Abbr_Subspecturm));
    %加入pupil的原始子图
    Uold = ifft2(fftshift(Abbr_Subspecturm.*pupil));
    
%     Uold1 = ifft2(fftshift(Abbr_Subspecturm1));
%     RAW(:,:,Image_num_index(num)) = RAW(:,:,Image_num_index(num))-dark_edge;
    
    compensate = Mag_image^2;
%     compensate = (sum(sum((abs(Uold)./max(max(abs(Uold))))^2)))/(sum(sum(RAW(:,:,Image_num_index(num))./max(max(RAW(:,:,Image_num_index(num)))))))*2.5;
%     compensate = mean2(abs(Uold1).^2)/mean2(RAW(:,:,Image_num_index(num)));
%     y = fix(num/LED_num_x) -  (LED_num_x-1)/2;
%     x = rem(num, LED_num_x) - (LED_num_x-1)/2 - 1;
%     if(rem(num, LED_num_x) == 0)
%         y = y - 1;
%         x = (LED_num_x-1)/2;
%     end
%     L = (x^2+y^2)*5000^2;
%     
%     
%     
%     if L>(8*5000^2)
% %         compensate_fit = (sqrt(LED2stage^2 + L)/(LED2stage))^6;
%         RAW1 = RAW(:,:,Image_num_index(num))-dark_center;
%         RAW1(RAW1<0) = 0;
%     else
% %         compensate_fit = (sqrt(LED2stage^2 + L)/(LED2stage))^6*3;
%         RAW1 = RAW(:,:,Image_num_index(num))-dark_edge;
%         RAW1(RAW1<0) = 0;
%     end
% % 
% % %     compensate = compensate*compensate_fit;
% %     compensate = compensate_fit;
% %         
%     compensate = mean2(abs(Uold1).^2)/mean2(RAW1);
%     
    RAW_compensate = RAW(:,:,Image_num_index(num)).*compensate;
%     RAW_compensate = RAW1.*compensate;
%     
%     delta = mean2(RAW_compensate) - mean2(Uold);
%     RAW_compensate = RAW_compensate - 0.5*delta;
%     
    Unew = RAW_compensate.*exp(1i.*angle(Uold));
%     Unew = RAW(:,:,Image_num_index(num)).*(Uold./abs(Uold));
    % Fourier space constraint and object function update
    
%     %没有光瞳函数修正的
%     Abbr_Subspecturm_corrected = fftshift(fft2(Unew));
%     Subspecturm = F(Fcenter_Y+ky-fix(M/2): Fcenter_Y+ky+ceil(M/2)-1, Fcenter_X+kx-fix(N/2):Fcenter_X+kx+ceil(M/2)-1);
%     Subspecturm = double(Subspecturm);
%     
%     W = Alpha*abs(Aperture_fun)./max(max(abs(Aperture_fun)));
%     
%     invP = conj(Aperture_fun)./((abs(Aperture_fun)).^2+eps.^2);
%     Subspecturmnew = (W.*Abbr_Subspecturm_corrected + (1-W).*(Abbr_Subspecturm)).*invP;
%     Subspecturmnew(Aperture==0) = Subspecturm(Aperture==0);
% %     
%     % Fourier sperturm replacement
%     F(Fcenter_Y+ky-fix(M/2): Fcenter_Y+ky+ceil(M/2)-1, Fcenter_X+kx-fix(N/2):Fcenter_X+kx+ceil(M/2)-1) = Subspecturmnew;
    
    %加入光瞳函数修正的EPRY算法
    Abbr_Subspecturm_corrected = fftshift(fft2(Unew)).*CTF;
%     Subspecturm = F(Fcenter_Y+ky-fix(M/2): Fcenter_Y+ky+ceil(M/2)-1, Fcenter_X+kx-fix(N/2):Fcenter_X+kx+ceil(M/2)-1);
%     Subspecturm = double(Subspecturm);
    
%     Subspecturm = Subspecturm + Alpha.*conj(pupil)./(max(max(abs(pupil).^2))).*(Abbr_Subspecturm_corrected - Abbr_Subspecturm.*(pupil));
    Subspecturm = Subspecturm + Alpha.*conj(pupil)./(abs(pupil).^2 + eps).*abs(pupil)./(max(max(pupil))).*(Abbr_Subspecturm_corrected - Abbr_Subspecturm);% *pupil
%     pupil = pupil + Alpha.*conj(Subspecturm)./(max(max(abs(Subspecturm).^2))).*(Abbr_Subspecturm_corrected - Abbr_Subspecturm.*(pupil));
    F(Fcenter_Y+ky-fix(M/2): Fcenter_Y+ky+ceil(M/2)-1, Fcenter_X+kx-fix(N/2):Fcenter_X+kx+ceil(M/2)-1) = Subspecturm;
    Inverse FT to get the reconstruction
    
    Result = ifft2(fftshift(F));
    Result_pupil = 1;
%     pupil = pupil.*CTF;
end

