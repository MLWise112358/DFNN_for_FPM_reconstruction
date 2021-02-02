%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Initialize image index (updating based on the NA order)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% initialize image index
Image_num_index = zeros(1,LED_num_x*LED_num_y);
kxky_index = zeros(LED_num_x*LED_num_y,2);
kxky_delta = zeros(LED_num_x*LED_num_y,2);
loop_x = (LED_num_x + 1)/2;

%% Update sequence
for loop = 1:loop_x
    if (loop == 1)
        num = 1;
        Image_num_index(num) = LED_center;
        last_index = Image_num_index(num);
        num = num + 1;
    else
        Image_num_index(num) = last_index - 1;
        last_index = Image_num_index(num);
        num = num + 1;
        Image_num_index(num:num+loop*2-4) = (last_index-LED_num_x : -LED_num_x :last_index-LED_num_x-(loop*2-4)*LED_num_x);
        num = num+loop*2-4;
        last_index = Image_num_index(num);
        num = num + 1;
        Image_num_index(num:num+loop*2-3) = (last_index+1 : 1 :last_index+1+(loop*2-3));
        num = num+loop*2-3;
        last_index = Image_num_index(num);
        num = num + 1;
        Image_num_index(num:num+loop*2-3) = (last_index+LED_num_x : LED_num_x :last_index+LED_num_x+(loop*2-3)*LED_num_x);
        num = num+loop*2-3;
        last_index = Image_num_index(num);
        num = num + 1;
        Image_num_index(num:num+loop*2-3) =  (last_index-1 : -1 :last_index-1-(loop*2-3));
        num = num+loop*2-3;
        last_index = Image_num_index(num);
        num = num + 1;
    end
end

%% spectrum location for each illumination
for num = 1 : LED_num_x*LED_num_y
    y_num = fix(Image_num_index(num)/LED_num_x)-(LED_num_x-1)/2;
    x_num = rem(Image_num_index(num),LED_num_x)-(LED_num_x-1)/2-1;
    if(rem(Image_num_index(num),LED_num_x)==0)
        x_num =(LED_num_x - 1)/2;
        y_num = y_num  - 1;
    end
    
    distance = sqrt((y_num*LEDdelta - 1*(S(2)/2-nstarty-imsize/2)*Pixel_size).^2+(x_num*LEDdelta + 1*(S(1)/2-nstartx-imsize/2)*Pixel_size).^2);
    theta = atan2(distance,LED2stage);
    kr = 1/Lambda * sin(theta);
    theta = atan2((y_num*LEDdelta - 1*(S(2)/2-nstarty-imsize/2)*Pixel_size),(x_num*LEDdelta + 1*(S(1)/2-nstartx-imsize/2)*Pixel_size));

%     distance = sqrt(y_num.^2+x_num.^2)*LEDdelta;
%     theta = atan2(distance,LED2stage);
%     kr = 1/Lambda * sin(theta);
%     theta = atan2(y_num,x_num);
    kxky_index(num,2) = round (kr*sin(theta)/Pixel_size_image_freq);
    kxky_index(num,1) = round (kr*cos(theta)/Pixel_size_image_freq);
    kxky_delta(num,2) =  (kr*sin(theta)/Pixel_size_image_freq) - kxky_index(num,2);
    kxky_delta(num,1) =  (kr*cos(theta)/Pixel_size_image_freq) - kxky_index(num,1);
end