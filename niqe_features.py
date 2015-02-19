
import os
import sys
import numpy as np
import scipy.integrate
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

def sorted_nicely(l): 
	""" Sort the given iterable in the way that humans expect.""" 
	convert = lambda text: int(text) if text.isdigit() else text 
	alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
	return sorted(l, key = alphanum_key)

gamma_range = np.arange(0.2, 10, 0.001)
a = scipy.special.gamma(2.0/gamma_range)
a *= a
b = scipy.special.gamma(1.0/gamma_range)
c = scipy.special.gamma(3.0/gamma_range)
prec_gammas = a/(b*c)

def gauss_window(lw, sigma):
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
avg_window = gauss_window(3, 7.0/6.0)

def extract_ggd_features(imdata):
	nr_gam = 1/prec_gammas
	sigma_sq = np.average(imdata**2)
	E = np.average(np.abs(imdata))
    
	rho = sigma_sq/E**2
	pos = np.argmin(np.abs(nr_gam - rho));
	return gamma_range[pos], np.sqrt(sigma_sq)

def extract_aggd_features(imdata):
	#flatten imdata
	imdata.shape = (len(imdata.flat),)
	imdata2 = imdata*imdata
	left_data = imdata2[imdata<0]
	right_data = imdata2[imdata>0]
	left_mean_sqrt = np.sqrt(np.average(left_data))
	right_mean_sqrt = np.sqrt(np.average(right_data))

	gamma_hat = left_mean_sqrt/right_mean_sqrt
	#solve r-hat norm
	r_hat = (np.average(np.abs(imdata))**2) / (np.average(imdata2))
	rhat_norm = r_hat * (((gamma_hat**3 + 1)*(gamma_hat + 1)) / ((gamma_hat**2 + 1)**2))

	#solve alpha by guessing values that minimize ro
	pos = np.argmin(np.abs(prec_gammas - rhat_norm));
	alpha = gamma_range[pos]

	gam1 = scipy.special.gamma(1.0/alpha)
	gam2 = scipy.special.gamma(2.0/alpha)
	gam3 = scipy.special.gamma(3.0/alpha)

	aggdratio = np.sqrt(gam1) / np.sqrt(gam3)
	bl = aggdratio * left_mean_sqrt
	br = aggdratio * right_mean_sqrt

	#mean parameter
	N = (br - bl)*(gam2 / gam1)*aggdratio
	return (alpha, N, bl, br, left_mean_sqrt, right_mean_sqrt)

extend_mode = 'constant'#'nearest'#'wrap'
def calc_image(image):
	w, h = np.shape(image)
	mu_image = np.zeros((w, h))
	var_image = np.zeros((w, h))
	image = np.array(image).astype('float')
	scipy.ndimage.correlate1d(image, avg_window, 0, mu_image, mode=extend_mode) 
	scipy.ndimage.correlate1d(mu_image, avg_window, 1, mu_image, mode=extend_mode) 
	scipy.ndimage.correlate1d(image**2, avg_window, 0, var_image, mode=extend_mode) 
	scipy.ndimage.correlate1d(var_image, avg_window, 1, var_image, mode=extend_mode) 
	var_image = np.sqrt(np.abs(var_image - mu_image**2))
	return (image - mu_image)/(var_image + 0.003), var_image, mu_image#0.00001)

def paired_p(new_im):
	#new_im /= 0.353257 #make the RV unit variance
	hr_shift = np.roll(new_im, 1, axis=1)
	hl_shift = np.roll(new_im, -1, axis=1)
	v_shift = np.roll(new_im, 1, axis=0)
	vr_shift = np.roll(hr_shift, 1, axis=0)
	vl_shift = np.roll(hl_shift, 1, axis=0)
	
	D1_img = vr_shift*new_im
	D2_img = vl_shift*new_im
	H_img = hr_shift*new_im
	V_img = v_shift*new_im

	#remove stuff close to 0
	H_img = H_img[np.abs(H_img) > 1e-2]
	V_img = V_img[np.abs(V_img) > 1e-2]
	D1_img = D1_img[np.abs(D1_img) > 1e-2]
	D2_img = D2_img[np.abs(D2_img) > 1e-2]

	return (V_img, H_img, D1_img, D2_img) 

def paired_d(new_im):
	#axis 0 is vertical shift (positive goes down)
	#axis 1 is horiz shift (positive goes right)
	hr_shift = np.roll(new_im, 1, axis=1)
	hl_shift = np.roll(new_im, -1, axis=1)
	vd_shift = np.roll(new_im, 1, axis=0)
	vu_shift = np.roll(new_im, -1, axis=0)
	vur_shift = np.roll(hr_shift, -1, axis=0)
	vul_shift = np.roll(hl_shift, -1, axis=0)
	vdl_shift = np.roll(hl_shift, 1, axis=0)
	vdr_shift = np.roll(hr_shift, 1, axis=0)

	#compute log stuff
	hr_shift = np.log(np.abs(hr_shift) + 0.1)
	hl_shift = np.log(np.abs(hl_shift) + 0.1)
	vd_shift = np.log(np.abs(vd_shift) + 0.1)
	vu_shift = np.log(np.abs(vu_shift) + 0.1)
	vur_shift = np.log(np.abs(vur_shift) + 0.1)
	vul_shift = np.log(np.abs(vul_shift) + 0.1)
	vdl_shift = np.log(np.abs(vdl_shift) + 0.1)
	vdr_shift = np.log(np.abs(vdr_shift) + 0.1)
	new_im = np.log(np.abs(new_im) + 0.1) 

	#D1
	item1 = vd_shift - new_im
	item2 = hr_shift - new_im
	item3 = vdr_shift - new_im
	item4 = vur_shift - new_im

	item5 = (hr_shift + hl_shift) - (vu_shift + vd_shift)

	item6 = (new_im + vdr_shift) - (vd_shift + hr_shift)

	item7 = (vul_shift + vdr_shift) - (vdl_shift +  vur_shift)

	item1 = item1[np.abs(item1) > 1e-9]
	item2 = item2[np.abs(item2) > 1e-9]
	item3 = item3[np.abs(item3) > 1e-9]
	item4 = item4[np.abs(item4) > 1e-9]
	item5 = item5[np.abs(item5) > 1e-9]
	item6 = item6[np.abs(item6) > 1e-9]
	item7 = item7[np.abs(item7) > 1e-9]
	
	return (item1, item2, item3, item4, item5, item6, item7)

def pp_extract(imgname1):
	img = skimage.io.imread(imgname1, as_grey=1)
	img2 = scipy.misc.imresize(img, 0.5, interp='bilinear', mode='F')
	img3 = scipy.misc.imresize(img, 0.25, interp='bilinear', mode='F')

	m_image, _, _ = calc_image(img)
	m_image2, _, _ = calc_image(img2)
	m_image3, _, _ = calc_image(img3)
	pps11, pps12, pps13, pps14 = paired_p(m_image)   
	pps21, pps22, pps23, pps24 = paired_p(m_image2)   
	pps31, pps32, pps33, pps34 = paired_p(m_image3)   

	alpha11, N11, bl11, br11, lsq11, rsq11 = extract_aggd_features(pps11)
	alpha12, N12, bl12, br12, lsq12, rsq12 = extract_aggd_features(pps12)
	alpha13, N13, bl13, br13, lsq13, rsq13 = extract_aggd_features(pps13)
	alpha14, N14, bl14, br14, lsq14, rsq14 = extract_aggd_features(pps14)

	alpha21, N21, bl21, br21, lsq21, rsq21 = extract_aggd_features(pps21)
	alpha22, N22, bl22, br22, lsq22, rsq22 = extract_aggd_features(pps22)
	alpha23, N23, bl23, br23, lsq23, rsq23 = extract_aggd_features(pps23)
	alpha24, N24, bl24, br24, lsq24, rsq24 = extract_aggd_features(pps24)

	alpha31, N31, bl31, br31, lsq31, rsq31 = extract_aggd_features(pps31)
	alpha32, N32, bl32, br32, lsq32, rsq32 = extract_aggd_features(pps32)
	alpha33, N33, bl33, br33, lsq33, rsq33 = extract_aggd_features(pps33)
	alpha34, N34, bl34, br34, lsq34, rsq34 = extract_aggd_features(pps34)

	pp_features = np.array([
			alpha11, N11, lsq11, rsq11, #6, 7, 8, 9 (V)
			alpha12, N12, lsq12, rsq12, #10, 11, 12, 13 (H)
			alpha13, N13, lsq13, rsq13,#14, 15, 16, 17 (D1)
			alpha14, N14, lsq14, rsq14,#18, 19, 20, 21 (D2)
			alpha21, N21, lsq21, rsq21, #6, 7, 8, 9 (V)
			alpha22, N22, lsq22, rsq22, #10, 11, 12, 13 (H)
			alpha23, N23, lsq23, rsq23,#14, 15, 16, 17 (D1)
			alpha24, N24, lsq24, rsq24,#18, 19, 20, 21 (D2)
			alpha31, N31, lsq31, rsq31, #6, 7, 8, 9 (V)
			alpha32, N32, lsq32, rsq32, #10, 11, 12, 13 (H)
			alpha33, N33, lsq33, rsq33,#14, 15, 16, 17 (D1)
			alpha34, N34, lsq34, rsq34,#18, 19, 20, 21 (D2)
	])

	return pp_features


def pp_extract_diff(imgname1, imgname2):
	im1 = skimage.io.imread(imgname1, as_grey=1)
	im12 = scipy.misc.imresize(im1, 0.5, interp='bilinear', mode='F')
	im13 = scipy.misc.imresize(im1, 0.25, interp='bilinear', mode='F')

	im2 = skimage.io.imread(imgname2, as_grey=1)
	im22 = scipy.misc.imresize(im2, 0.5, interp='bilinear', mode='F')
	im23 = scipy.misc.imresize(im2, 0.25, interp='bilinear', mode='F')

	img = im2 - im1
	img2 = im22 - im12
	img3 = im23 - im13

	m_image, _, _ = calc_image(img)
	m_image2, _, _ = calc_image(img2)
	m_image3, _, _ = calc_image(img3)
	pps11, pps12, pps13, pps14 = paired_p(m_image)   
	pps21, pps22, pps23, pps24 = paired_p(m_image2)   
	pps31, pps32, pps33, pps34 = paired_p(m_image3)   

	alpha11, N11, bl11, br11, lsq11, rsq11 = extract_aggd_features(pps11)
	alpha12, N12, bl12, br12, lsq12, rsq12 = extract_aggd_features(pps12)
	alpha13, N13, bl13, br13, lsq13, rsq13 = extract_aggd_features(pps13)
	alpha14, N14, bl14, br14, lsq14, rsq14 = extract_aggd_features(pps14)

	alpha21, N21, bl21, br21, lsq21, rsq21 = extract_aggd_features(pps21)
	alpha22, N22, bl22, br22, lsq22, rsq22 = extract_aggd_features(pps22)
	alpha23, N23, bl23, br23, lsq23, rsq23 = extract_aggd_features(pps23)
	alpha24, N24, bl24, br24, lsq24, rsq24 = extract_aggd_features(pps24)

	alpha31, N31, bl31, br31, lsq31, rsq31 = extract_aggd_features(pps31)
	alpha32, N32, bl32, br32, lsq32, rsq32 = extract_aggd_features(pps32)
	alpha33, N33, bl33, br33, lsq33, rsq33 = extract_aggd_features(pps33)
	alpha34, N34, bl34, br34, lsq34, rsq34 = extract_aggd_features(pps34)

	pp_features = np.array([
			alpha11, N11, lsq11, rsq11, #6, 7, 8, 9 (V)
			alpha12, N12, lsq12, rsq12, #10, 11, 12, 13 (H)
			alpha13, N13, lsq13, rsq13,#14, 15, 16, 17 (D1)
			alpha14, N14, lsq14, rsq14,#18, 19, 20, 21 (D2)
			alpha21, N21, lsq21, rsq21, #6, 7, 8, 9 (V)
			alpha22, N22, lsq22, rsq22, #10, 11, 12, 13 (H)
			alpha23, N23, lsq23, rsq23,#14, 15, 16, 17 (D1)
			alpha24, N24, lsq24, rsq24,#18, 19, 20, 21 (D2)
			alpha31, N31, lsq31, rsq31, #6, 7, 8, 9 (V)
			alpha32, N32, lsq32, rsq32, #10, 11, 12, 13 (H)
			alpha33, N33, lsq33, rsq33,#14, 15, 16, 17 (D1)
			alpha34, N34, lsq34, rsq34,#18, 19, 20, 21 (D2)
	])

	return pp_features

def pd_extract(imgname):
	img = skimage.io.imread(imgname, as_grey=1)
	img2 = scipy.misc.imresize(img, 0.5, interp='bilinear', mode='F')
	img3 = scipy.misc.imresize(img, 0.25, interp='bilinear', mode='F')

	m_image, _, _ = calc_image(img)
	m_image2, _, _ = calc_image(img2)
	m_image3, _, _ = calc_image(img3)
	pds11, pds12, pds13, pds14, pds15, pds16, pds17 = paired_d(m_image)
	pds21, pds22, pds23, pds24, pds25, pds26, pds27 = paired_d(m_image2)
	pds31, pds32, pds33, pds34, pds35, pds36, pds37 = paired_d(m_image3)

	alphap11, sqp11 = extract_ggd_features(pds11)
	alphap12, sqp12 = extract_ggd_features(pds12)
	alphap13, sqp13 = extract_ggd_features(pds13)
	alphap14, sqp14 = extract_ggd_features(pds14)
	alphap15, sqp15 = extract_ggd_features(pds15)
	alphap16, sqp16 = extract_ggd_features(pds16)
	alphap17, sqp17 = extract_ggd_features(pds17)

	alphap21, sqp21 = extract_ggd_features(pds21)
	alphap22, sqp22 = extract_ggd_features(pds22)
	alphap23, sqp23 = extract_ggd_features(pds23)
	alphap24, sqp24 = extract_ggd_features(pds24)
	alphap25, sqp25 = extract_ggd_features(pds25)
	alphap26, sqp26 = extract_ggd_features(pds26)
	alphap27, sqp27 = extract_ggd_features(pds27)

	alphap31, sqp31 = extract_ggd_features(pds31)
	alphap32, sqp32 = extract_ggd_features(pds32)
	alphap33, sqp33 = extract_ggd_features(pds33)
	alphap34, sqp34 = extract_ggd_features(pds34)
	alphap35, sqp35 = extract_ggd_features(pds35)
	alphap36, sqp36 = extract_ggd_features(pds36)
	alphap37, sqp37 = extract_ggd_features(pds37)

	pd_features = np.array([
			alphap11, sqp11,#54, 55
			alphap12, sqp12,#56, 57
			alphap13, sqp13,#58, 59
			alphap14, sqp14,#60, 61
			alphap15, sqp15,#62, 63
			alphap16, sqp16,#64, 65
			alphap17, sqp17,#66, 67
			alphap21, sqp21,#54, 55
			alphap22, sqp22,#56, 57
			alphap23, sqp23,#58, 59
			alphap24, sqp24,#60, 61
			alphap25, sqp25,#62, 63
			alphap26, sqp26,#64, 65
			alphap27, sqp27,#66, 67
			alphap31, sqp31,#54, 55
			alphap32, sqp32,#56, 57
			alphap33, sqp33,#58, 59
			alphap34, sqp34,#60, 61
			alphap35, sqp35,#62, 63
			alphap36, sqp36,#64, 65
			alphap37, sqp37,#66, 67
	])
	return pd_features

def mscn_extract(imgname1):
	img = skimage.io.imread(imgname1, as_grey=1)
	img2 = scipy.misc.imresize(img, 0.5, interp='bilinear', mode='F')
	img3 = scipy.misc.imresize(img, 0.25, interp='bilinear', mode='F')

	m_image, _, _ = calc_image(img)
	m_image2, _, _ = calc_image(img2)
	m_image3, _, _ = calc_image(img3)

	m_image = m_image[np.abs(m_image) > 1e-2]
	m_image2 = m_image2[np.abs(m_image2) > 1e-2]
	m_image3 = m_image3[np.abs(m_image3) > 1e-2]

	alpha_m1, sq_m1 = extract_ggd_features(m_image)
	alpha_m2, sq_m2 = extract_ggd_features(m_image2)
	alpha_m3, sq_m3 = extract_ggd_features(m_image3)

	mscn_features = np.array([
			alpha_m1, sq_m1, #0, 1
			alpha_m2, sq_m2, #0, 1
			alpha_m3, sq_m3, #0, 1
	])

	return mscn_features

def mscn_extract_diff(imgname1, imgname2):
	im1 = skimage.io.imread(imgname1, as_grey=1)
	im12 = scipy.misc.imresize(im1, 0.5, interp='bilinear', mode='F')
	im13 = scipy.misc.imresize(im1, 0.25, interp='bilinear', mode='F')

	im2 = skimage.io.imread(imgname2, as_grey=1)
	im22 = scipy.misc.imresize(im2, 0.5, interp='bilinear', mode='F')
	im23 = scipy.misc.imresize(im2, 0.25, interp='bilinear', mode='F')

	img = im2 - im1
	img2 = im22 - im12
	img3 = im23 - im13

	m_image, _, _ = calc_image(img)
	m_image2, _, _ = calc_image(img2)
	m_image3, _, _ = calc_image(img3)

	m_image = m_image[np.abs(m_image) > 1e-2]
	m_image2 = m_image2[np.abs(m_image2) > 1e-2]
	m_image3 = m_image3[np.abs(m_image3) > 1e-2]

	alpha_m1, sq_m1 = extract_ggd_features(m_image)
	alpha_m2, sq_m2 = extract_ggd_features(m_image2)
	alpha_m3, sq_m3 = extract_ggd_features(m_image3)

	mscn_features = np.array([
			alpha_m1, sq_m1, #0, 1
			alpha_m2, sq_m2, #0, 1
			alpha_m3, sq_m3, #0, 1
	])

	return mscn_features

#compute coefficents to 3 files (temporal scales)
#get the input folder
img_input = sys.argv[1]
mscn_f = mscn_extract(img_input)
pp_f = pp_extract(img_input)
features = np.hstack((mscn_f, pp_f))

print features
