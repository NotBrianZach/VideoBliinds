%% Michele A. Saad, Blind Prediction of Natural Video Quality
%% The IEEE Transactions on Image Processing, January 2014
%% Video BLIINDS Algorithm Code

function [dt_dc_measure2 geo_ratio_features] = NSS_spectral_ratios_feature_extraction(frames)

%% PART A of Video-BLIINDS: Computing the NSS DCT features:

%% Step 1: Compute local (5x5 block-based) DCT of frame differences

mblock = 5;

row = size(frames,1);
col = size(frames,2);
nFrames = size(frames,3);

dct_diff5x5 = zeros(mblock^2,floor(row/mblock)*floor(col/mblock),nFrames-1);

for x=1:nFrames-1
    mbCount = 0;
    for i = 1 : mblock : row-mblock+1
        for j = 1 : mblock : col-mblock+1
            
            mbCount = mbCount+1;
            
            temp = dct2(frames(i:i+mblock-1,j:j+mblock-1,x+1) - frames(i:i+mblock-1,j:j+mblock-1,x));
            
            dct_diff5x5(:,mbCount,x) = temp(:);
            clear temp
        end
    end
end


%% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Step 2: Computing gamma of dct difference frequencies

g=[0.03:0.001:10];
r=gamma(1./g).*gamma(3./g)./(gamma(2./g).^2);

for y=1:size(dct_diff5x5,3)
    
    for i=1:mblock*mblock
        temp = dct_diff5x5(i,:,y);
        mean_gauss=mean(temp);
        var_gauss=var(temp);
        mean_abs=mean(abs(temp-mean_gauss))^2;
        rho=var_gauss/(mean_abs+0.0000001);

        gamma_gauss=11;
        for x=1:length(g)-1
            if rho<=r(x) && rho>r(x+1)
               gamma_gauss=g(x);
               break
            end
        end
       gama_freq(i)=gamma_gauss;
    end    
    gama_matrix{y}=col2im(gama_freq',[5,5],[5,5],'distinct'); 
end



% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
%%Step 3: Separate gamma frequency bands


for x=1:length(gama_matrix)
    freq_bands(:,x)=zigzag(gama_matrix{x})';

end

lf_gama5x5 = freq_bands(2:(((mblock*mblock)-1)/3)+1,:);
mf_gama5x5 = freq_bands((((mblock*mblock)-1)/3)+2:(((mblock*mblock)-1)/3)*2+1,:);
hf_gama5x5 = freq_bands((((mblock*mblock)-1)/3)*2+2:end,:);

geomean_lf_gam = geomean(lf_gama5x5);
geomean_mf_gam = geomean(mf_gama5x5);
geomean_hf_gam = geomean(hf_gama5x5);
 
geo_high_ratio = geomean(geomean_hf_gam./(0.1 + (geomean_mf_gam + geomean_lf_gam)/2));
geo_low_ratio = geomean(geomean_mf_gam./(0.1 + geomean_lf_gam));  
geo_HL_ratio = geomean(geomean_hf_gam./(0.1 + geomean_lf_gam));
geo_HM_ratio = geomean(geomean_hf_gam./(0.1 + geomean_mf_gam));
geo_hh_ratio = geomean( ((geomean_hf_gam + geomean_mf_gam)/2)./(0.1 + geomean_lf_gam));

geo_ratio_features = [geo_HL_ratio geo_HM_ratio geo_hh_ratio geo_high_ratio geo_low_ratio];



%%
%%
%%
%%
%%
%%

for x = 1:size(dct_diff5x5,3)-1
    dt_dc(x) = abs(mean(dct_diff5x5(1,:,x+1))-mean(dct_diff5x5(1,:,x)));
end

dt_dc_measure2 = mean(dt_dc);


