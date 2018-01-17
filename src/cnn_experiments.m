% Experiment with the cnn_mnist_fc_bnorm
clc;
clear;

tr_file = 'cifar10_L127_D7_10_train.mat';
te_file = 'cifar10_L127_D7_10_test.mat';

% tr_file = 'cifar10_L126_D2_train.mat';
% te_file = 'cifar10_L126_D2_test.mat';

H1_max = 20;
%H1 = 10:2:H_max;
H1 = [20 50 100 200 400];
H1 = 100;
H2_max = 20;
%H2 = 10:2:H_max;
H2 = [20 50 100 200 400 500];
H2 = 200;
K = 10;
D = 256; 
IM1 = 8;
IM2 = 8;

tr_size = 5000*K;
te_size = 1000*K;
label_count = tr_size + te_size;

size_h1 = size(H1,2);
size_h2 = size(H2,2);

F1 = [3,5];
F2 = [3,5];

size_f1 = size(F1,2);
size_f2 = size(F2,2);

S_CONV = 1;
P_S1 = 2;
P_S2 = 2;
S_P = 2;
PAD_P = 0;

numEpochs = 50;



size_h1 = 1;
size_h2 = 1;
size_f1 = 1;
size_f2 = 1;

net_tr_error=zeros(size_h1, size_h2, size_f1, numEpochs);
net_val_error=zeros(size_h1, size_h2, size_f1, numEpochs);
numEpochs = 50;

for i = 1:size_h1
    h1 = H1(i);
    
    for j = 1:size_h2
        h2 = H2(j);
       
        for k = 1:size_f1
            f1 = F1(k);
            f2 = f1;
            if f1 == 5
                numEpochs = 30;
            else
                numEpochs = 50;
            end
            h_str = sprintf('data/h1_%dh2_%df1_%df2_%d', h1, h2, f1, f2);
            if (i > 1)|| (j > 1) || (k > 1)
                if(~exist(h_str, 'dir'))
                    mkdir(h_str);
                end
                dst=sprintf('%s/imdb.mat', h_str);
                copyfile(src,dst);
            end
            [net_bn, info_bn] = cnn_cifar10(tr_file, te_file, tr_size, ...
                te_size, label_count, IM1, IM2, f1, f2, K, D, h1, h2, S_CONV, ...
                P_S1, P_S2, S_P, PAD_P, numEpochs, ...
                'expDir', h_str, 'batchNormalization', true, 'errorFunction', 'multiclass');
            if i == 1 && j == 1 && k == 1
                src=sprintf('%s/imdb.mat', h_str);
            end
            if (i > 1)|| (j > 1) || (k > 1)
                data_file = sprintf('%s/imdb.mat', h_str);
                delete(data_file);
            end
            fig_filename1 = sprintf('data/h1_%dh2_%df1_%df2_%d.fig', h1, h2, f1, f2);
            fig_filename2 = sprintf('data/h1_%dh2_%df1_%df2_%d.png', h1, h2, f1, f2);
            saveas(gcf, fig_filename1);
            saveas(gcf, fig_filename2);

            value = getfield(info_bn, 'train');
            value2 = arrayfun(@(x) x.top1err, value);
            value2 = value2';
            %change
            tmp = size(value2,1);
            net_tr_error(i,j, k, 1:tmp)=value2;

            value = getfield(info_bn, 'val');
            value2 = arrayfun(@(x) x.top1err, value);
            value2 = value2';
            %change
            net_val_error(i,j, k, 1:tmp)=value2;
        end
    end

end

save('data/parameter.mat', 'tr_size', 'te_size', 'IM1', 'IM2', 'F1', 'F2', 'K', ...
    'D', 'H1', 'H2', 'S_CONV', 'P_S1', 'P_S2', 'S_P', 'PAD_P', 'numEpochs', ...
    'net_tr_error', 'net_val_error', 'info_bn', 'net_bn');





% [net_fc, info_fc] = cnn_mnist(...
%   'expDir', 'data/mnist-baseline', 'batchNormalization', false);
% 
% figure(1) ; clf ;
% subplot(1,2,1) ;
% 
% value = getfield(info_fc, 'val');
% value2 = arrayfun(@(x) numel(x.objective), value);
% value2 = value2';
% semilogy(value2, 'o-') ; hold all ;
% %semilogy((info_fc.val.objective)', 'o-') ; hold all ;
% value = getfield(info_bn, 'val');
% value2 = arrayfun(@(x) numel(x.objective), value);
% value2 = value2';
% semilogy(value2, '+--') ;
% %semilogy(info_bn.val.objective, '+--') ;
% xlabel('Training samples [x 10^3]'); ylabel('energy') ;
% grid on ;
% h=legend('BSLN', 'BNORM') ;
% set(h,'color','none');
% title('objective') ;
% subplot(1,2,2) ;
% 
% value = getfield(info_fc, 'val');
% value2 = arrayfun(@(x) numel(x.top1err), value);
% value2 = value2';
% plot(value2, 'o-') ; hold all ;
% value = getfield(info_bn, 'val');
% value2 = arrayfun(@(x) numel(x.top1err), value);
% value2 = value2';
% plot(value2, '+--') ;
% 
% %plot(info_fc.val.error', 'o-') ; hold all ;
% %plot(info_bn.val.error', '+--') ;
% 
% h=legend('BSLN-val','BSLN-val-5','BNORM-val','BNORM-val-5') ;
% grid on ;
% xlabel('Training samples [x 10^3]'); ylabel('error') ;
% set(h,'color','none') ;
% title('error') ;
% drawnow ;



% semilogy(info_fc.val.objective', 'o-') ; hold all ;
% semilogy(info_bn.val.objective', '+--') ;
% xlabel('Training samples [x 10^3]'); ylabel('energy') ;
% grid on ;
% h=legend('BSLN', 'BNORM') ;
% set(h,'color','none');
% title('objective') ;
% subplot(1,2,2) ;
% plot(info_fc.val.error', 'o-') ; hold all ;
% plot(info_bn.val.error', '+--') ;
% h=legend('BSLN-val','BSLN-val-5','BNORM-val','BNORM-val-5') ;
% grid on ;
% xlabel('Training samples [x 10^3]'); ylabel('error') ;
% set(h,'color','none') ;
% title('error') ;
% drawnow ;