# Computing the DC temporal variation
# feature a.k.a. the DC feature
import os
import sys
import numpy as np
import scipy.ndimage
import scipy.signal
import scipy.io
from time import clock, time
import pickle
from PIL import Image
import glob
import time
import re
import skimage
from skimage.feature import hog
from skimage import data, color, exposure
from sklearn.externals import joblib

#current path to image files:"C:/Users/Zach/Desktop/movieFrames/frames_2003.77.00000074.bmp"
#path to niqe.py "C:/Users/Zach/Desktop/VideoBLIINDS_Code_MicheleSaad/niqe_features.py"
def sorted_nicely(l): 
	""" Sort the given iterable in the way that humans expect.""" 
	convert = lambda text: int(text) if text.isdigit() else text 
	alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
	return sorted(l, key = alphanum_key)

# gamma_range = np.arange(0.2, 10, 0.001)
# a = scipy.special.gamma(2.0/gamma_range)
# a *= a
# b = scipy.special.gamma(1.0/gamma_range)
# c = scipy.special.gamma(3.0/gamma_range)
# prec_gammas = a/(b*c)

def gauss_window(lw, sigma):#fspecial
    sd = float(sigma)
    lw = int(lw)
    weights = [0.0] * (2 * lw + 1)
    weights[lw] = 1.0
    sum = 1.0
    sd *= sd
    for ii in range(1, lw + 1):
        tmp = np.exp(-0.5 * float(ii * ii) / sd)
        weights[lw + ii] = tmp
        weights[lw - ii] = tmp
        sum += 2.0 * tmp
    for ii in range(2 * lw + 1):
        weights[ii] /= sum
    return weights
    
def fspecial(lw, sigma):#fspecial
    sd = float(sigma)
    lw = int(lw)
    weights = [0.0] * (2 * lw + 1)
    weights[lw] = 1.0
    sum = 1.0
    sd *= sd
    for ii in range(1, lw + 1):
        tmp = np.exp(-0.5 * float(ii * ii) / sd)
        weights[lw + ii] = tmp
        weights[lw - ii] = tmp
        sum += 2.0 * tmp
    for ii in range(2 * lw + 1):
        weights[ii] /= sum
    return weights
frames = skimage.io.imread("C:/Users/Zach/Desktop/movieFrames/frames_2003.77.00000074.bmp", as_grey=1)
def temporal_dc_variation_feature_extraction(frames):
'''
computes dt_dc_measure 1
'''  

mblock = 16;
h=fspecial('gaussian',mblock);

# for x=1:size(frames,3)-1
    # x
    # tic  
    # imgP = double(frames(:,:,x+1));
    # imgI = double(frames(:,:,x));
    # [motion_vects16x16(:,:,x) temp] = motionEstNTSS(imgP,imgI,mblock,7);
    # toc
# end


# mbsize = 16;
# row = size(frames,1);
# col = size(frames,2);

# for x=1:size(frames,3)-1
    # x;
    # mbCount = 1;
    # for i = 1 :mbsize : row-mbsize+1
        # for j = 1 :mbsize : col-mbsize+1
            # dct_motion_comp_diff(i:i+mbsize-1,j:j+mbsize-1,x) = dct2(frames(i:i+mbsize-1,j:j+mbsize-1,x+1)-frames(i+motion_vects16x16(1,mbCount,x):i+mbsize-1+motion_vects16x16(1,mbCount,x),j+motion_vects16x16(2,mbCount,x):j+mbsize-1+motion_vects16x16(2,mbCount,x),x));
            # mbCount = mbCount+1;
        # end
    # end
# end



# for i = 1:size(frames,3)-1
    # temp = im2col(dct_motion_comp_diff(:,:,i),[16,16],'distinct');
    # std_dc(i) = std(temp(1,:));
# end
# clear *motion*



# for i = 1:length(std_dc)-1
    # dt_dc_temp(i) = abs(std_dc(i+1)-std_dc(i));

# end

# dt_dc_measure1 = mean(dt_dc_temp);

# def nss_spectral_ratios_feature_extraction(frames):
# '''
# computes dt_dc_measure2 and geo_ratio_features
# '''

# %% PART A of Video-BLIINDS: Computing the NSS DCT features:

# %% Step 1: Compute local (5x5 block-based) DCT of frame differences

# mblock = 5;

# row = size(frames,1);
# col = size(frames,2);
# nFrames = size(frames,3);

# dct_diff5x5 = zeros(mblock^2,floor(row/mblock)*floor(col/mblock),nFrames-1);

# for x=1:nFrames-1
    # mbCount = 0;
    # for i = 1 : mblock : row-mblock+1
        # for j = 1 : mblock : col-mblock+1
            
            # mbCount = mbCount+1;
            
            # temp = dct2(frames(i:i+mblock-1,j:j+mblock-1,x+1) - frames(i:i+mblock-1,j:j+mblock-1,x));
            
            # dct_diff5x5(:,mbCount,x) = temp(:);
            # clear temp
        # end
    # end
# end


# %% 
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# %% Step 2: Computing gamma of dct difference frequencies

# g=[0.03:0.001:10];
# r=gamma(1./g).*gamma(3./g)./(gamma(2./g).^2);

# for y=1:size(dct_diff5x5,3)
    
    # for i=1:mblock*mblock
        # temp = dct_diff5x5(i,:,y);
        # mean_gauss=mean(temp);
        # var_gauss=var(temp);
        # mean_abs=mean(abs(temp-mean_gauss))^2;
        # rho=var_gauss/(mean_abs+0.0000001);

        # gamma_gauss=11;
        # for x=1:length(g)-1
            # if rho<=r(x) && rho>r(x+1)
               # gamma_gauss=g(x);
               # break
            # end
        # end
       # gama_freq(i)=gamma_gauss;
    # end    
    # gama_matrix{y}=col2im(gama_freq',[5,5],[5,5],'distinct'); 
# end



# % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# % 
# %%Step 3: Separate gamma frequency bands


# for x=1:length(gama_matrix)
    # freq_bands(:,x)=zigzag(gama_matrix{x})';

# end

# lf_gama5x5 = freq_bands(2:(((mblock*mblock)-1)/3)+1,:);
# mf_gama5x5 = freq_bands((((mblock*mblock)-1)/3)+2:(((mblock*mblock)-1)/3)*2+1,:);
# hf_gama5x5 = freq_bands((((mblock*mblock)-1)/3)*2+2:end,:);

# geomean_lf_gam = geomean(lf_gama5x5);
# geomean_mf_gam = geomean(mf_gama5x5);
# geomean_hf_gam = geomean(hf_gama5x5);
 
# geo_high_ratio = geomean(geomean_hf_gam./(0.1 + (geomean_mf_gam + geomean_lf_gam)/2));
# geo_low_ratio = geomean(geomean_mf_gam./(0.1 + geomean_lf_gam));  
# geo_HL_ratio = geomean(geomean_hf_gam./(0.1 + geomean_lf_gam));
# geo_HM_ratio = geomean(geomean_hf_gam./(0.1 + geomean_mf_gam));
# geo_hh_ratio = geomean( ((geomean_hf_gam + geomean_mf_gam)/2)./(0.1 + geomean_lf_gam));

# geo_ratio_features = [geo_HL_ratio geo_HM_ratio geo_hh_ratio geo_high_ratio geo_low_ratio];



# %%
# %%
# %%
# %%
# %%
# %%

# for x = 1:size(dct_diff5x5,3)-1
    # dt_dc(x) = abs(mean(dct_diff5x5(1,:,x+1))-mean(dct_diff5x5(1,:,x)));
# end

# dt_dc_measure2 = mean(dt_dc);

