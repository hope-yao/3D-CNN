

off_data = off_loader('~/Documents/ModelNet10/ModelNet10/bed/train/bed_0001.off',0);
instance = polygon2voxel(off_data, [300, 300, 300], 'auto');

figure
p = patch(isosurface(instance,0.1));
set(p,'FaceColor','red','EdgeColor','none');
daspect([1,1,1])
grid on; axis tight
camlight
lighting gouraud;
set(gcf,'Color','white');
set(gca,'position',[0,0,1,1],'units','normalized');
axis equal