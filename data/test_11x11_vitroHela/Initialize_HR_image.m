%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Initialize HR image
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Upsample the central low-resolution image
A = imresize(RAW(:,:,LED_center),Mag_image,'cubic');

% Rescale to image for energy conservation
scale = Mag_image.^2;
A = A./scale;
[Hi_res_M,Hi_res_N] = size(A);

% Initialize HR image use the amplitude of central low-resolution image
F = fftshift(fft2(A));

Fcenter_X = fix(Hi_res_M/2)+1;
Fcenter_Y = fix(Hi_res_N/2)+1;

