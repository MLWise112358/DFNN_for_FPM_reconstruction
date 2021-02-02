clc;
clear;
close all;

n=121;
% nstartx = 383;
% nstarty = 1009;
% nstartx = 1442;
% nstarty = 1664;
nstartx = 1010;
nstarty = 1030;
% nstartx = 1330;
% nstarty = 1430;
imsize = 150; %200 150
% I0=zeros(2298,2200);
% [x,y]=meshgrid(-1100:1:1099,-1149:1:1148);
% I0=((x.^2+y.^2)<1022.^2);
% I2=zeros(1,225);
%  I3=sum(sum(I0));
for i = 1:n
    imagename = strcat('test ',' (',num2str(i),')','.tif');
    I = (double(imread(imagename)));
%     I=I(:,:,1);
%     RAW(:,:,i)=I(1077:1077+255,91:91+255);
%     RAW(:,:,i)=I(1166:1166+255,647:647+255);
    RAW(:,:,i)=I(nstartx:nstartx+imsize-1,nstarty:nstarty+imsize-1);%边缘 (383,1009) %中心 (953, 865) 
%     RAW(:,:,i)=I(953:953+255,865:865+255);%中心
%     I1(:,:,i)= I1(:,:,i).*I0;
%     I2(1,i)=mean2(I1(:,:,i))*(2298*2200)/I3;
    
%     Imgs(:,:,i) = I(1336:1463,2012:2139,1);
%     figure;imshow(I{i},[]);
end
RAW = RAW - min(min(RAW)) / 2.5;
RAW(RAW < 0) = 0;
S = size(I);

clear i I