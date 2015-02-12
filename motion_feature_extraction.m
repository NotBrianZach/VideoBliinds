%% PART B of Video-BLIINDS: Computing the Motion Model


function [mean_Coh10x10 G] = motion_feature_extraction(frames)


%% Step 1: On the Luminance planes: Motion Vector Computation
 
mblock = 10;

for x=1:size(frames,3)-1
    x
    tic  
    
    imgP = double(frames(:,:,x+1));
    imgI = double(frames(:,:,x));
    
    [motionVects10x10(:,:,x) temp] = motionEstNTSS(imgP,imgI,mblock,mblock+floor(mblock/2));
    toc
end

%% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% Step 2: Compute coherency via the structure tensor of motion vectors

row = size(frames,1);
col = size(frames,2);

mblock=10; 

h = fspecial('gaussian',[5,5]);

for x=1:size(motionVects10x10,3)
    
    Ix(:,:,x) = reshape(motionVects10x10(1,1:floor(row/mblock)*floor(col/mblock),x),floor(col/mblock),floor(row/mblock))';
    Iy(:,:,x) = reshape(motionVects10x10(2,1:floor(row/mblock)*floor(col/mblock),x),floor(col/mblock),floor(row/mblock))';
    Upper_Left(:,:,x) = imfilter(Ix(:,:,x).*Ix(:,:,x),h,'symmetric');
    Lower_Right(:,:,x) = imfilter(Iy(:,:,x).*Iy(:,:,x),h,'symmetric');
    Off_diag(:,:,x) = imfilter(Ix(:,:,x).*Iy(:,:,x),h,'symmetric');
end

for x=1:size(motionVects10x10,3)
    
    count = 0;
    for i=1:size(Upper_Left,1)
        for j=1:size(Upper_Left,2)
            count = count+1;
            temp_tensor(1,1) = Upper_Left(i,j,x); 
            temp_tensor(2,2) = Lower_Right(i,j,x);
            temp_tensor(1,2) = Off_diag(i,j,x);
            temp_tensor(2,1) = Off_diag(i,j,x);
            Eig10x10(:,count,x) = eig(temp_tensor);
        end
    end
end

num(:,:) = ((Eig10x10(2,:,:)-Eig10x10(1,:,:)).^2);
den(:,:) = (Eig10x10(2,:,:)+Eig10x10(1,:,:)).^2;
Coh10x10 = num./den;
Coh10x10(den==0) = 0; 

mean_Coh10x10 = mean(Coh10x10(:));
%% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% Step 3: Compute global motion characterization

nFrames = size(motionVects10x10,3);

for x = 1:size(motionVects10x10,3)
    mode10x10(x) = mode(sqrt((motionVects10x10(1,:,x).^2)+(motionVects10x10(2,:,x).^2)));
    mean10x10(x) = mean(sqrt((motionVects10x10(1,:,x).^2)+(motionVects10x10(2,:,x).^2)));
    motion_diff(x) = abs(mean10x10(x)-mode10x10(x));
end

G = mean(motion_diff)/(1+mean(mode10x10));



