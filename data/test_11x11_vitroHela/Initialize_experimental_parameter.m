%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Initialize experimental parameter
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Load raw dataset (USAF target)
% load RAW;

% Raw image size
[M,N] = size(RAW(:,:,1));

% LED number
LED_num_x = 11;
LED_num_y = 11;
Total_Led = LED_num_x*LED_num_y;
LED_center = (LED_num_x*LED_num_y+1)/2;

% obj NA and magnification
NA = 0.2;
Mag = 8.1458;

% System parameters
LED2stage = 67.5e3;
LEDdelta = 4e3;
Pixel_size = 6.5/Mag;
Lambda = 0.514;
k = 2*pi/Lambda;
kmax=1/Lambda*NA;

% Upsampling ratio
Mag_image = 3.0;
Pixel_size_image = Pixel_size/Mag_image;
Pixel_size_image_freq = 1/Pixel_size_image/(M*Mag_image);
kmax = kmax/Pixel_size_image_freq;

% Create pupil mask
[x, y] = meshgrid ...
    ((-fix(M/2):ceil(M/2)-1) ...
    ,(-fix(N/2):ceil(N/2)-1));
[Theta,R] = cart2pol(x,y);
Aperture = ~(R>kmax);
% Aperture = conv2(Aperture,Aperture, 'same') .* Aperture;
% Aperture = Aperture./max(max(Aperture));
Aperture_fun = double(Aperture);
CTF = Aperture_fun;