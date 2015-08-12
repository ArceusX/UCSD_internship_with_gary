%Two hidden layers
f = inline('1./(1+exp(-x))');   % sigmoid
load mnistabridged.mat
[n,m]=size(train); % number of pixels and number of max examples
n0=784; n1=25; n2=10; n3=10;
m1=50;  % number of examples
eta=0.1;  % learning rate
epsinit=0.01;  % parameter controlling magnitude of initial conditions
W1=epsinit*randn(n1,n0); W2=epsinit*randn(n2,n1); W3=epsinit*randn(n3,n2);
b1=epsinit*randn(n1,1); b2=epsinit*randn(n2,1); b3=epsinit*randn(n3,1);
trainlabels(trainlabels==0)=10; % convention: tenth output signals a zero
testlabels(testlabels==0)=10;  % convention: tenth output signals a zero
nepoch=10;
errsq = zeros(nepoch,1);
[n_t,m_t]=size(test);
test_n=randperm(m_t,50);
perf=0;
for j=test_n
    x_t=double(test(:,j))/255;
    y_t(testlabels(j))=1;
end

for iepoch=1:nepoch
    h=randperm(m,m1);
    for i=h
        x0=double(train(:,i))/255;   % normalize max value to one
        y=zeros(n3,1); y(trainlabels(i))=1; % output vector
        x1=f(W1*x0+b1);
        x2=f(W2*x1+b2);
        x3=f(W3*x2+b3);
        delta3=(y-x3).*x3.*(1-x3);
        delta2=(W3'*delta3).*x2.*(1-x2);  % error signal scaled by derivative
        delta1=(W2'*delta2).*x1.*(1-x1);
        W3=W3+eta*delta3*x2';
        W2=W2+eta*delta2*x1';
        W1=W1+eta*delta1*x0';
        b3=b3+eta*delta3;
        b2=b2+eta*delta2;
        b1=b1+eta*delta1;
        err=y_t'-f(W3'*x3+b3);
        errsq(iepoch)=err'*err;
    end
    if rem(iepoch,2)==0;
        plot(1:iepoch,sum(errsq(1:iepoch))./(1:iepoch)');
    end     
end