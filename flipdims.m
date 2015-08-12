function fx = flipdims(x)
% flip every dimension of an array
% E.g. used with convn to perform cross correlation
	sz = size(x); fx = x;
	for dim=1:length(sz), fx = flipdim(fx,dim);, end
end