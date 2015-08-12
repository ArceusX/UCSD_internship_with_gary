f = inline('1./(1+exp(-x))');   % definition of sigmoid
load mnistabridged.mat

[n,m]=size(train);  % number of pixels and number of examples
eta1=0.1;  % learning rate 1
eta2=0.1;  % learning rate 2
epsinit=0.01;  % parameter controlling magnitude of initial conditions
w(:,1:2)=epsinit.*randn(n,1);   % initialization with Gaussian random values
b(:,1:2)=epsinit.*randn;

tmax=1000;
errsq(:,1:2)=zeros(tmax,1);   % squared error on training set

for t=1:tmax
  i=ceil(m*rand);
  x=double(train(:,i))/255;   % uint8 -> double, normalize max value to one
  y=double(trainlabels(i)== 2);
  actual(:,1:2)=f(w(:,1:2)'.*x+b(:,1:2));
  error(:,1:2)=y-actual(:,1:2);
  slope(:,1:2)=actual(:,1:2)*(1-actual(:,1:2));
  errsq(t,1:2)=error(:,1:2)'*error(:,1:2);
  delta(:,1:2)=slope(:,1:2)*error(:,1:2);  % error signal scaled by derivative
  w(:,1)=w(:,1)+eta1*delta(:,1)*x;
  w(:,2)=w(:,2)+eta2*delta(:,2)*x;
  b=b+eta*delta;
  if rem(t,100)==0
    subplot(2,2,1)
    imagesc(reshape(w,28,28))
    title(sprintf('w, t=%d',t))
    axis image off
    colormap hot
    
    subplot(2,2,2)
    imagesc(reshape(w,28,28))
    title(sprintf('w, t=%d',t))
    axis image off
    colormap hot
    
    subplot(2,2,3)
    plot(1:t,cumsum(errsq(1:t,1))./(1:t)')
    drawnow
    
    subplot(2,2,4)
    plot(1:t,cumsum(errsq(1:t,2))./(1:t)')
    drawnow
  end
end 