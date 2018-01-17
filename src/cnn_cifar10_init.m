function net = cnn_cifar10_init(IM1, IM2, F1, F2, K, D, H1, H2, S_CONV, P_S1, P_S2, S_P, PAD_P, numEpochs, varargin)

% K: # of class
% D: dimension of the data
% H: Hidden units 
% numEpochs: number of epochs
% IM1, IM2: size of image
% S: stride
% F1, F2: filter size

% CNN_MNIST_LENET Initialize a CNN similar for MNIST
opts.batchNormalization = true ;
opts.networkType = 'simplenn' ;
opts = vl_argparse(opts, varargin) ;

rng('default');
rng(0) ;

f=1/100 ;
net.layers = {} ;

im3 = IM1 - F1 + 1; 
im4 = IM2 - F2 + 1;
F3 = im3/S_P;
F4 = im4/S_P;

% F3 = IM1 - F1 + 1; 
% F4 = IM2 - F2 + 1;
% net.layers{end+1} = struct('type', 'relu') ;
% 
% net.layers{end+1} = struct('type', 'pool', ...
%                            'method', 'avg', ...
%                            'pool', [8 8], ...
%                            'stride', 8, ...
%                            'pad', 0) ;
% 
% net.layers{end+1} = struct('type', 'conv', ...
%                            'weights', {{f*randn(1,1,256,K, 'single'), zeros(1, K, 'single')}}, ...
%                            'stride', 1, ...
%                            'pad', 0) ;
% net.layers{end+1} = struct('type', 'softmaxloss') ;


net.layers{end+1} = struct('type', 'conv', ...
                           'weights', {{f*randn(F1, F2, D, H1, 'single'), zeros(1, H1, 'single')}}, ...
                           'stride', S_CONV, ...
                           'pad', 0) ;
                           %'weights', {{f*randn(F1, F2, D, H, 'single'), zeros(1, H, 'single')}}, ...

net.layers{end+1} = struct('type', 'pool', ...
                           'method', 'avg', ...
                           'pool', [P_S1 P_S2], ...
                           'stride', S_P, ...
                           'pad', PAD_P) ;
                           
net.layers{end+1} = struct('type', 'relu') ;


%net.layers{end+1} = struct('type', 'conv', ...
%                           'weights', {{f*randn(1,1,H1,H2, 'single'), zeros(1, H2, 'single')}}, ...
%                           'stride', 1, ...
%                           'pad', 0) ;


net.layers{end+1} = struct('type', 'conv', ...
                           'weights', {{f*randn(F3,F4,H2,K, 'single'), zeros(1, K, 'single')}}, ...
                           'stride', 1, ...
                           'pad', 0) ;
net.layers{end+1} = struct('type', 'softmaxloss') ;




% optionally switch to batch normalization
% if opts.batchNormalization
%   net = insertBnorm(net, 1) ;
%   net = insertBnorm(net, 4) ;
%   net = insertBnorm(net, 7) ;
% end

% Meta parameters
%net.meta.inputSize = [28 28 1] ;
net.meta.inputSize = [IM1 IM2 D] ;
%net.meta.inputSize = [32 32 1] ;
net.meta.trainOpts.learningRate = 0.001 ;
net.meta.trainOpts.numEpochs = numEpochs;
net.meta.trainOpts.batchSize = 100 ;

% Fill in default values
net = vl_simplenn_tidy(net) ;

% Switch to DagNN if requested
switch lower(opts.networkType)
  case 'simplenn'
    % done
  case 'dagnn'
    net = dagnn.DagNN.fromSimpleNN(net, 'canonicalNames', true) ;
    net.addLayer('top1err', dagnn.Loss('loss', 'classerror'), ...
      {'prediction', 'label'}, 'error') ;
    net.addLayer('top5err', dagnn.Loss('loss', 'topkerror', ...
      'opts', {'topk', 5}), {'prediction', 'label'}, 'top5err') ;
  otherwise
    assert(false) ;
end

% --------------------------------------------------------------------
function net = insertBnorm(net, l)
% --------------------------------------------------------------------
assert(isfield(net.layers{l}, 'weights'));
ndim = size(net.layers{l}.weights{1}, 4);
layer = struct('type', 'bnorm', ...
               'weights', {{ones(ndim, 1, 'single'), zeros(ndim, 1, 'single')}}, ...
               'learningRate', [1 1 0.05], ...
               'weightDecay', [0 0]) ;
net.layers{l}.biases = [] ;
net.layers = horzcat(net.layers(1:l), layer, net.layers(l+1:end)) ;
