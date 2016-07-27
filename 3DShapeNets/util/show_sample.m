function show_sample(the_sample ,threshold)
% a simple visualization tool using isosurface to show each 3D sample.
% the_sample is in 3D
    
    p = patch(isosurface(the_sample,threshold));
    set(p,'FaceColor','red','EdgeColor','none');
    daspect([1,1,1])
    grid on; axis tight
    camlight 
    lighting gouraud;
    set(gcf,'Color','white');

    axis equal
    axis vis3d
%     view(2);
    axis([0 30 0 30 0 30])
%     axis tight;
%     pause;
%     close(gcf);
end
