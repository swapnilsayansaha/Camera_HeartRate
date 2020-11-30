function [z] = RGBSkinSeg(VidROI)
x=VidROI;
[k1,k2,~]=size(x);
x=double(x);
for i=1:k1
    for j=1:k2
        r=x(i,j,1);
        g=x(i,j,2);
        b=x(i,j,3);
        m=max([r g b]);
        n=min([r g b]);
        if  ((r>70)&&(g>40)&&(b>20)&&((m-n)>15)&&(abs(r-g)>5)&&(r>g)&&(r>b))
            msk(i,j)=1;
        else
            msk(i,j)=0;
        end
    end
end

for i=1:3
    z(:,:,i)=x(:,:,i).*msk;
end
z = z/255;
end

