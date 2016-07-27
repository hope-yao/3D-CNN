clear
figure;
a=1;b=0;
load('./my_code/more/pot_train_new.mat');

saliency = zeros(size(instance));
for x=1:30
    for y=1:30
        for z=1:30
            if instance(x,y,z)~=0
                if (y<8&&z<13)||(y<7&&z>10) || (y>22&&z>17)||(y>23&&z>9)
                    saliency(x,y,z)=a;
                else
                    saliency(x,y,z)=b;
                end
            end
        end
    end
end
instance=saliency;save('pot_train_saliency.mat','instance')
instance=reshape(saliency,[1,30,30,30]);
subplot(2,2,1);show_sample(instance,0.5);title('pot train');

load('./my_code/more/cup_train.mat');
saliency = zeros(size(instance));
for x=1:30
    for y=1:30
        for z=1:30
            if instance(x,y,z)~=0
                if y>20
                    saliency(x,y,z)=a;
                else
                    saliency(x,y,z)=b;
                end
            end
        end
    end
end
instance=saliency;save('cup_train_saliency.mat','instance')
instance=reshape(saliency,[1,30,30,30]);
subplot(2,2,4);show_sample(instance,0.5);title('cup train');


load('./my_code/more/pot_test_new.mat');
saliency = zeros(size(instance));
for x=1:30
    for y=1:30
        for z=1:30
            if instance(x,y,z)~=0
                if y>23||y<7
                    saliency(x,y,z)=a;
                else
                    saliency(x,y,z)=b;
                end
            end
        end
    end
end
instance=saliency;save('pot_test_saliency.mat','instance')
instance=reshape(saliency,[1,30,30,30]);
subplot(2,2,2);show_sample(instance,0.5);title('pot test');


load('./my_code/more/cup_test.mat');
saliency = zeros(size(instance));
for x=1:30
    for y=1:30
        for z=1:30
            if instance(x,y,z)~=0
                if y>20&&z>9
                    saliency(x,y,z)=a;
                else
                    saliency(x,y,z)=b;
                end
            end
        end
    end
end
instance=saliency;save('cup_test_saliency.mat','instance')
instance=reshape(saliency,[1,30,30,30]);
subplot(2,2,3);show_sample(instance,0.5);title('cup test');


