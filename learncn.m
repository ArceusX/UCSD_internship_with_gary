% Convolutional network with one hidden feature map layer
% Take image, add noise, then see how well network can denoise
epsinit=0.01; eta=0.01;
filter_size=5; nfeature=10; 
w1=epsinit*randn(filter_size,filter_size,nfeature); 
w2=epsinit*randn(filter_size,filter_size,nfeature);
tmax=500;
rmserr=zeros(tmax,1);

input_image=double(imread('baboon.png'))/255; 
label_image=input_image; % label_image used for training

% each iteration of training done with a randomly selected patch of the input image,
out_size=[6 6];  % output patch size
in_size=out_size+(filter_size-1)*2;  % 2 connection layers

for t=1:tmax
    in_offset=floor(rand(1,2).*(size(input_image)-in_size+1)); % random offset for input patch
    out_offset=in_offset+filter_size-1; 
    input_patch=input_image((1:in_size(1))+in_offset(1),(1:in_size(2))+in_offset(2));
    label_patch=label_image((1:out_size(1))+out_offset(1),(1:out_size(2))+out_offset(2));
    
    x0=zeros(in_size);
    x0=input_patch+0.1*randn(size(x0));  % corrupted image
    x1=zeros([in_size-filter_size+1,nfeature]);
    
    for ifeature=1:nfeature
        x1(:,:,ifeature)=conv2(x0,w1(:,:,ifeature),'valid');
    end
    x1=1./(1+exp(-x1));
    x2=zeros(out_size);
    for ifeature=1:nfeature
        x2=x2+conv2(x1(:,:,ifeature),w2(:,:,ifeature),'valid');
    end
    x2=1./(1+exp(-x2));
    
    err=label_patch-x2;
    rmserr(t)=sqrt(mean(err(:).^2));  % rms error
    
    if ~rem(t,100)  % plot average of rms error every 100 iterations
        subplot(2,2,1)
        plot(conv2(rmserr(1:t),ones(100,1),'valid')/100);
        title('rms error vs. time')
        drawnow
    end
    
    delta2=err.*x2.*(1-x2);
    for ifeature=1:nfeature
       delta1(:,:,ifeature)=conv2(delta2,w2(end:-1:1,end:-1:1,ifeature),'full');
    end
    delta1=delta1.*x1.*(1-x1);
    
    for ifeature=1:nfeature
        w2(:,:,ifeature)=w2(:,:,ifeature)+eta*conv2(x1(end:-1:1,end:-1:1,ifeature),delta2,'valid');
        w1(:,:,ifeature)=w1(:,:,ifeature)+eta*conv2(x0(end:-1:1,end:-1:1),delta1(:,:,ifeature),'valid');
    end
end

xx0=zeros(size(input_image));
xx0=input_image+0.1*randn(size(xx0));  % corrupted image
xx1=zeros([size(xx0)-filter_size+1,nfeature]);
% Take convolution
for ifeature=1:nfeature
    xx1(:,:,ifeature)=conv2(xx0,w1(:,:,ifeature),'valid');
end
xx1=1./(1+exp(-xx1));
xx2=zeros(size(xx0)-2*(filter_size-1));
for ifeature=1:nfeature
    xx2=xx2+conv2(xx1(:,:,ifeature),w2(:,:,ifeature),'valid');
end
xx2=1./(1+exp(-xx2));
    
subplot(2,2,2)
imagesc(xx0);
title('corrupted image')
subplot(2,2,3)
imagesc(xx2);
title('restored image')
subplot(2,2,4)
imagesc(input_image);
title('original image');
colormap gray