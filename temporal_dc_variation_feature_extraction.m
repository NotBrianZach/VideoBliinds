%% Computing the DC temporal variation
%% feature a.k.a. the DC feature

function dt_dc_measure1 = temporal_dc_variation_feature_extraction(frames)


% mblock = 5;
% 
% row = size(frames,1);
% col = size(frames,2);
% nFrames = size(frames,3);
% 
% dct_diff5x5 = zeros(mblock^2,floor(row/mblock)*floor(col/mblock),nFrames-1);
% 
% for x=1:nFrames-1
%     x
%     mbCount = 0;
%     for i = 1 : mblock : row-mblock+1
%         for j = 1 : mblock : col-mblock+1
%             
%             mbCount = mbCount+1;
%             
%             temp = dct2(frames(i:i+mblock-1,j:j+mblock-1,x+1) - frames(i:i+mblock-1,j:j+mblock-1,x));
%             
%             dct_diff5x5(:,mbCount,x) = temp(:);
%             clear temp
%         end
%     end
% end
% 
% for x = 1:size(dct_diff5x5,3)-1
%     dt_dc(x) = abs(mean(dct_diff5x5(1,:,x+1))-mean(dct_diff5x5(1,:,x)));
% end
% 
% dt_dc_measure2 = mean(dt_dc);
% 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
%%%%%%%%%%%%%%%%%%%  SECOND DC MEASURE %%%%%%%%%%%%%%%%%%%%%%
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


mblock = 16;
h=fspecial('gaussian',mblock);

for x=1:size(frames,3)-1
   % x
    %tic;
    imgP = double(frames(:,:,x+1));
    imgI = double(frames(:,:,x));
%     'imgP'
%     size(imgP)
%     'imgI'
%     size(imgI)
    [motion_vects16x16(:,:,x) temp] = motionEstNTSS(imgP,imgI,mblock,7);
%     'done'
    %toc;
end
motion_vects16x16(1,31:90)

mbsize = 16;
row = size(frames,1);
col = size(frames,2);

for x=1:size(frames,3)-1
    x;
    mbCount = 1;
    for i = 1 : mbsize : row-mbsize+1
        for j = 1 : mbsize : col-mbsize+1
%             'hi, frames'
%             size(frames)
%             'motion_vects'
%             size(motion_vects16x16)
            dct_motion_comp_diff(i:i+mbsize-1,j:j+mbsize-1,x) = dct2(frames(i:i+mbsize-1,j:j+mbsize-1,x+1)-frames(i+motion_vects16x16(1,mbCount,x):i+mbsize-1+motion_vects16x16(1,mbCount,x),j+motion_vects16x16(2,mbCount,x):j+mbsize-1+motion_vects16x16(2,mbCount,x),x));
            mbCount = mbCount+1;
        end
    end
end



for i = 1:size(frames,3)-1
    temp = im2col(dct_motion_comp_diff(:,:,i),[16,16],'distinct');
    temp
    std_dc(i) = std(temp(1,:));
end
%std_dc
%'im2colDistinct motion comp diff'
%size(dct_motion_comp_diff)
clear *motion*


for i = 1:length(std_dc)-1
    dt_dc_temp(i) = abs(std_dc(i+1)-std_dc(i));
end
'dt_dc_temp'
size(dt_dc_temp)
dt_dc_measure1 = mean(dt_dc_temp);
 