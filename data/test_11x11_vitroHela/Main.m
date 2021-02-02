%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Main file to implement the adaptive step-size strategy for Fourier
% ptychographic reconstruction algorithm
%
% Related Reference:
% Adaptive step-size strategy for noise-robust Fourier ptychographic microscopy
% C. Zuo, J. Sun, and Q Chen, submitted to Optics Express
%
% last modified on 03/08/2016
% by Chao Zuo (surpasszuo@163.com, zuochao@njust.edu.cn)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clc;
clear;
close all;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Initialize experimental parameter and HR image
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
eval prapare_for_iterive;
eval Initialize_experimental_parameter;
eval Initialize_image_num_index;
eval Initialize_HR_image;
pupil = 1;

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %% Reconstruction by Gerchberg-Saxton (PIE with alpha =1)
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% % Run for 8 itertions (which is sufficient to `converge')
% Total_iter_num = 18;
% 
% % Reconstruction and display the result
% figure
% for iter = 1:Total_iter_num
%     
%     % Reconstruction by one iteration of Gerchberg-Saxton
%     eval GS;
%     
%     % Show the Fourier spectrum
%     subplot(1,2,1)
%     imshow(log(abs(F)+1),[0, max(max(log(abs(F)+1)))/2]);
%     title('Fourier spectrum');
%     
%     % Show the reconstructed amplitude
%     subplot(1,2,2)
%     imshow((abs(Result)),[]);
%     title(['Iteration No. = ',int2str(iter)]);
%     
%     pause(0.01);
% end
% 
% % Save the result
% Result_GS = Result;
% % 
% % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % %% Reconstruction by the adaptive step-size strategy (adaptive alpha)
% % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% Re-initialize HR image
eval Initialize_HR_image;

Total_iter_num = 30;
tic
% Reconstruction and display the result
for iter = 1:Total_iter_num
    
    % Reconstruction by one iteration of adaptive step-size strategy
    eval AS;
    
    % Show the Fourier spectrum
    subplot(2,2,1)
    imshow(log(abs(F)+1),[0, max(max(log(abs(F)+1)))/2]);
    title('Fourier spectrum');
    
    % Show the reconstructed amplitude
    subplot(2,2,2)
    imshow((abs(Result)),[]);
    title(['Iteration No. = ',int2str(iter), '  \alpha = ',num2str(Alpha)]);
    
    subplot(2,2,3);
    imshow(abs(pupil),[]);
    subplot(2,2,4);
    imshow(-1*angle(pupil),[]);
    
    % Stop the iteration when the algorithm converges
    if(Alpha == 0) break; end
    
    pause(0.01);
end
toc

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Compare the two results
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure
Result_AS = ifft2(fftshift(F));

% subplot(1,2,1)
% imshow(abs(Result_GS),[]);
% title('Gerchberg-Saxton');
% 
% subplot(1,2,2)
amp = -abs(Result_AS);
title('Adaptive step-size');
a = amp-min(min(amp));
a = a/max(max(a));
imwrite(a, 'AS_amplitude.png');
imshow(a,[]);
figure;
% imshow(-1.*angle(Result_AS),[-0.8, 1.3]);%-0.5,1.3
ang = -1.*angle(Result_AS);
% ang(ang<=-1.4)=-1.4;ang(ang>=1.8)=1.8;
ang = ang-min(min(ang));
ang = ang/max(max(ang));
% ang(ang<=0.3)=0.3;ang(ang>=0.7)=0.7;
ang = ang-min(min(ang));
imshow(ang/max(max(ang)));
imshow(-angle(Result_AS), [-0.6, 1.6])
% imwrite(ang/max(max(ang)), 'AS_phase.png');

% r = RAW(:,:,25);
% r = r - min(min(r));
% r = r ./ max(max(r)) .* 255;
% imwrite(uint8(r), 'ÖÐÐÄRAW.png');

