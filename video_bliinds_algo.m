%% Compute Video BLIINDS Features
load('movieFrames.mat');
%load('frames_modelparameters.mat');
niqe_features = compute_niqe_features(squeeze(initFrames));
dt_dc_measure1 = temporal_dc_variation_feature_extraction(initFrames);
[dt_dc_measure2 geo_ratio_features] = NSS_spectral_ratios_feature_extraction(initFrames);
[mean_Coh10x10 G] = motion_feature_extraction(initFrames);
% niqe_features = compute_niqe_features(frames);
% dt_dc_measure1 = temporal_dc_variation_feature_extraction(frames);
% [dt_dc_measure2 geo_ratio_features] = NSS_spectral_ratios_feature_extraction(frames);
% [mean_Coh10x10 G] = motion_feature_extraction(frames);

features_test = [niqe_features log(1+dt_dc_measure1) log(1+dt_dc_measure2) log(1+geo_ratio_features) log(1+mean_Coh10x10) log(1+G)];
    
%%

fid = fopen('features_test.txt', 'w+');
fprintf(fid,'%d ',features_test(1,1:end));
fprintf(fid,'\n');
fclose(fid);

system('predictR.r')

%% Reading data from a file
 
%predicted_dmos=textread('predicted_dmos.txt');

