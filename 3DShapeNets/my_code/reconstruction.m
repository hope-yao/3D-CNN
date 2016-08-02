% use construction test the accuracy of RBM
function reconstruction()

% reset(gpuDevice(1));
run('setup_paths.m')
model = load('pretrained_model_pot_sal_itr100_myth_16_6_2_16_5_2_16_4_1_1200_4000_d10.mat');
filename2 = './my_code/more/pot_train_sal.mat';
% filename2 = './volumetric_data/chair/30/train/chair_000000182_1.mat';

figure;
subplot(1,3,1);reconstruct_one(model.model,filename2,1,0);title('1st rbm')
subplot(1,3,2);reconstruct_one(model.model,filename2,2,0);title('2nd rbm')
subplot(1,3,3);reconstruct_one(model.model,filename2,3,0);title('3rd rbm')

% goto_rbm = 3; %reconstruction to which RBM
% figure;cnt=1;
% n=32*0;
% for i=1+n:4:32+n
%     subplot(2,4,cnt);reconstruct_one(model.model, filename2, goto_rbm, i);title(strcat(num2str(i),'-th filter'))
%     cnt = cnt + 1;
% end
% figure;cnt=1;
% for i=2+n:4:32+n
%     subplot(2,4,cnt);reconstruct_one(model.model, filename2, goto_rbm, i);title(strcat(num2str(i),'-th filter'))
%     cnt = cnt + 1;
% end
% figure;cnt=1;
% for i=3+n:4:32+n
%     subplot(2,4,cnt);reconstruct_one(model.model, filename2, goto_rbm, i);title(strcat(num2str(i),'-th filter'))
%     cnt = cnt + 1;
% end
% figure;cnt=1;
% for i=4+n:4:32+n
%     subplot(2,4,cnt);reconstruct_one(model.model, filename2, goto_rbm, i);title(strcat(num2str(i),'-th filter'))
%     cnt = cnt + 1;
% end

end

function reconstruct_one(model,filename,goto_rbm,test_)
run('kernels.m')
goto_l = goto_rbm+1;

data = load(filename);
batch = data.instance;
batch = reshape(batch,[1,size(batch,1),size(batch,2),size(batch,3)]);
% propagate upwards
l = 2;
stride = model.layers{l}.stride;
hidden_presigmoid = myConvolve2(kConv_forward2, batch, model.layers{l}.w, stride, 'forward');
hidden_presigmoid = bsxfun(@plus, hidden_presigmoid, permute(model.layers{l}.c, [2,3,4,5,1]));
hidden_prob = sigmoid(hidden_presigmoid);
% hidden_sample = single(hidden_prob > rand(size(hidden_prob)));
hidden_sample = single(hidden_prob > 0.5);
chain = hidden_sample;

while l<goto_l
    % propagate upwards
    l = l+1;
    stride = model.layers{l}.stride;
    hidden_presigmoid = myConvolve(kConv_forward_c, chain, model.layers{l}.w, stride, 'forward');
    hidden_presigmoid = bsxfun(@plus, hidden_presigmoid, permute(model.layers{l}.c, [2,3,4,5,1]));
    hidden_prob = sigmoid(hidden_presigmoid);
    %     chain = single(hidden_prob > rand(size(hidden_prob)));
    chain = single(hidden_prob > 0.5);
end

if test_
    num_channel = size(model.layers{goto_l}.w,1);
    for i=1:num_channel
        if i~=test_
            %         chain(1,:,:,:,i) = 0;
            model.layers{goto_l}.w(i,:,:,:,:) = 0;
        else
            model.layers{goto_l}.w(i,:,:,:,:) = num_channel * model.layers{goto_l}.w(i,:,:,:,:);            
        end
    end
end

while l>2
    % PROPDOWN
    stride = model.layers{l}.stride;
    %     NOTICE! there is big difference between 0 and 1. Possibly the GPU
    %     code is erronous
    visible_presigmoid = myConvolve(kConv_backward_c, chain, model.layers{l}.w, stride, 'backward');
    visible_presigmoid = bsxfun(@plus, visible_presigmoid, permute(model.layers{l}.b, [5,1,2,3,4]));
    chain = sigmoid(visible_presigmoid);
    l = l-1;
end

% PROPDOWN
stride = model.layers{l}.stride;
% it seems there is no difference between myConvolve2 and myConvolve
visible_presigmoid = myConvolve2(kConv_backward, chain, model.layers{l}.w, stride, 'backward');
visible_presigmoid = bsxfun(@plus, visible_presigmoid, permute(model.layers{l}.b, [5,1,2,3,4]));
visible_prob = sigmoid(visible_presigmoid);

% cnt = 1;
% for t=0.7:-0.1:0.2
%     subplot(2,3,cnt);
%     show_sample(squeeze(visible_prob(1,:,:,:)),t)
%     title(strcat('threshold ',num2str(t)))
%     [~,name,~] = fileparts(filename);
%     savefig(strcat(strcat(name,'_l',num2str(goto_l)),'_',num2str(t),'.fig'));
%     %     close
%     cnt = cnt + 1;
% end

show_sample(squeeze(visible_prob(1,:,:,:)),0.5)
end


function [y] = sigmoid(x)
y = 1 ./ (1 + exp(-x));
end