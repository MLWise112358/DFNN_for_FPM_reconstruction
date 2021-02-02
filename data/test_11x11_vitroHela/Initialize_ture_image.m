% real image for quantitative evaluation
Abbr = peaks(M)/7;
Aperture_real = abs(Aperture_fun).*exp(1i.*Abbr);
figure
subplot(1,2,1)
imshow(angle(Aperture_real),[-1,1]);
colorbar
subplot(1,2,2)
imshow(abs(Aperture_real),[0,2]);
colorbar

load Amplitude
load Phase;
% Intensity=im2double(imread('intensityp1.tif'))+0.1;
% Phase=im2double(imread('phasep1.png'))*0.3;
% Intensity(Intensity>1) = 1;
% 
% Intensity=imresize(Intensity,[Hi_res_M,Hi_res_N]);
% Phase=imresize(Phase,[Hi_res_M,Hi_res_N]);