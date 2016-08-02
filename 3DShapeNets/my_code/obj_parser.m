function instance=obj_parser(obj_file)
% from obj to voxel

fid_obj = fopen(obj_file,'r');

vv = []; ff = [];
vcount = 1;  fcount = 1;
tline = fgetl(fid_obj);
while ischar(tline)
    if length(tline)>1
        if (strcmp(tline(1:2),'v ')==1)
            vv = [vv; str2num(tline(3:length(tline)))];
            vcount = vcount + 1;
        elseif (strcmp(tline(1:2),'f ')==1)
            tmpidx1 = [];  tmpidx2 = [];
            for j=1:length(tline)
                if strcmp(tline(j),' ')==1
                    tmpidx1 = [tmpidx1, j];
                end
                if strcmp(tline(j),'/')==1
                    tmpidx2 = [tmpidx2, j];
                end
            end
            v1 = str2num(tline(tmpidx1(1)+1:tmpidx2(1)-1));
            v2 = str2num(tline(tmpidx1(2)+1:tmpidx2(3)-1));
            v3 = str2num(tline(tmpidx1(3)+1:tmpidx2(5)-1));
            ff = [ff; [v1,v2,v3] ];
            fcount = fcount + 1;
        end
    end
    tline = fgetl(fid_obj);
end

faces = ff; vertices = vv;
vertices = vertices - repmat(mean(vertices,1),size(vertices,1),1);
FV.faces = faces;
FV.vertices = vertices;

ll = 100; d = 90;
% ll = 30; d = 26;
% instance=polygon2voxel(FV,[30 30 30],'auto');
instance=polygon2voxel(FV,[d d d],'auto');
pad_size = (ll - d)/2;
instance = padarray(instance, [pad_size, pad_size, pad_size]);
instance = int8(instance);

[pathstr,name,~] = fileparts(obj_file);
matfilename = fullfile(pathstr,[name '.mat']);
save(matfilename,'instance');

data = reshape(instance,[1,ll,ll,ll]);
show_sample(squeeze(data),0.1)
% figure;plot3D(instance,'timed',0.1)

