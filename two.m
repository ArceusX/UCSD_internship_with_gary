%Test how changing the learning rate affects performance
f = inline('1./(1+exp(-x))');   % definition of sigmoid
load mnistabridged.mat

[n,m]=size(train);  % number of pixels and number of examples
eta=[0.1 0.5];  % learning rate
epsinit=0.01;  % parameter controlling magnitude of initial conditions
w=epsinit*randn(n,1);   % initialization with Gaussian random values
b=epsinit*randn;

tmax=1000;
errsq=zeros(tmax,1);   % squared error on training set

for z=1:2
    for t=1:tmax
      i=ceil(m*rand);
      x=double(train(:,i))/255;   % uint8 -> double, normalize max value to one
      y=double(trainlabels(i)==2);
      actual=f(w'*x+b);
      error=y-actual;
      slope=actual*(1-actual);
      errsq(t)=error'*error;
      delta=slope*error;  % error signal scaled by derivative
      w=w+eta(z)*delta*x;
      b=b+eta(z)*delta;
      if rem(t,100)==0
        subplot(2,2,2*z-1)
        imagesc(reshape(w,28,28))
        title(sprintf('w, t=%d',t))
        ylim([0 60])
        axis image off
        colormap hot
        subplot(2,2,2*z)
        plot(1:t,sum(errsq(1:t))./(1:t)')
        ylim([0 60])
        drawnow
      end
    end
end