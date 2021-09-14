clear;figure(1);clf;colormap jet
simdir = 'data/sia_hershel_bulkley_bn_0.5';
load([simdir '/static.mat'])
[x,y] = ndgrid(x,y);
B     = B - min(B(:));
for it = 10:60:810
    load([simdir '/step_' int2str(it) '.mat'])
    if it == 10
        Yic = 0*H;
    end
    Yic(2:end-1,2:end-1) = 0.25*(Yi(1:end-1,1:end-1)+Yi(2:end,1:end-1)...
        +                        Yi(1:end-1,2:end  )+Yi(2:end,2:end ));
    Yic([1 end],:) = Yic([2 end-1],:);
    Yic(:,[1 end]) = Yic(:,[2 end-1]);
    subplot(121);surf(x,y,B+H,H);shading interp;axis image;colorbar('Location','southoutside');
%     set(gca,'DataAspectRatio',[1 1 1/10])
    xlabel('x');ylabel('y');zlabel('z')
    lighting gouraud;material([0.5 0.5 0.5]);view(3);camlight(30,90)
    title(sprintf('# it = %d, time = %g/%g\nn_{iter}/n_x = %.3f, mass balance = %e',it,time,ttot,iter/nx,sum(H(:)-H0(:))/max(abs(H(:)))/(nx*ny)))
    Bd = B(fix(end/2),:);
    Yf = Yic(fix(end/2),:);
    Hy = H(fix(end/2),:)-Yf;
    subplot(122);area(y(fix(end/2),:), [Bd; Yf; Hy]',-5,'LineWidth',1);
    xlabel('y')
    ylabel('z')
    colororder({'#bebada','#ffffb3','#8dd3c7'})
    axis image
%     set(gca,'DataAspectRatio',[1 1/10 1])
    ylim([-5 max(B(:))])
    legend({'Bedrock','Yielded region','Plug flow'},'Location','northwest')
    drawnow
end