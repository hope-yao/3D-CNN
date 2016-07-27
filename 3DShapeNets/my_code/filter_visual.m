function filter_visual()
load('./pretrained_model_nosal_48_6_2_160_5_2_32_4_1_1200_4000_d10.mat');
w = rec_conv(model,3); % filter of which rbm
save('w_projected.mat','w');
% plot_filter(w)
for i=1:10:32
    figure;show_sample(squeeze(w(1,:,:,:)),0.0002)
end
end

function plot_filter(w)

figure;
assert(size(w,1)>=4);
for i=1:4
    subplot(2,2,i)
    vol3d('cdata', squeeze(w(i,:,:,:)), 'xdata', [0 1], 'ydata', [0 1], 'zdata', [0 1]);
    colormap(bone(256));
    alphamap([0 linspace(0.1, 0, 2)]);
    %     axis([0.1,0.9,0.1,0.9,0.1,0.9])
    set(gcf, 'color', 'w');
    view(3);
end

end


function mat2 = rec_conv(model,layer_idx)
kernels

mat2 = (model.layers{layer_idx+1}.w); % upper layer
while layer_idx~=1
    mat1 = (model.layers{layer_idx}.w); % lower layer
    stride = model.layers{layer_idx}.stride;
    if(layer_idx==2)
        mat2 = myConvolve(kConv_backward, mat2, mat1, stride, 'backward');
    else
        mat2 = myConvolve(kConv_backward_c, mat2, mat1, stride, 'backward');
    end
    layer_idx = layer_idx - 1;
end
end

