from __future__ import division
#change all division signs to python 3.x behavior

# Computing the DC temporal variation
# feature a.k.a. the DC feature
import os
import sys
import numpy as np
from numpy import zeros
from numpy import ones
from numpy import floor
import math
#import scipy.ndimage
import scipy.signal
import scipy.io
from time import clock, time
import pickle
from PIL import Image
import glob
import time
import re
import skimage
import pickle
from skimage.feature import hog
from skimage import data, color, exposure
from sklearn.externals import joblib
from scipy.fftpack import dct

#frames = [skimage.io.imread("C:/Users/Zach/Desktop/movieFrames/frames_2003.77.00000075.bmp", as_grey=1)]
#frames.append(skimage.io.imread("C:/Users/Zach/Desktop/movieFrames/frames_2003.77.00000076.bmp", as_grey=1))
#frames = [skimage.io.imread("/Users/brian/Desktop/Abe.png", as_grey=1)]
def imread_convert(f):
	    im = skimage.io.imread(f)#, as_grey=1)
	    im = im[:, :, 0]
            return im
#path = "/Users/brian/Desktop/VideoBliinds"
#files = os.listdir(path)
#wantedFiles = filter(,files)
#re.search()
#f1 = skimage.io.ImageCollection("/Users/brian/Desktop/VideoBliinds/*(0000000[6-9]|0000001[0-9]|0000002[0-9]|0000003[0-9]|0000004[0-5]).bmp", load_func=imread_convert)
f1 = skimage.io.ImageCollection("/Users/brian/Desktop/VideoBliinds/*.bmp", load_func=imread_convert)
#f1 = skimage.io.ImageCollection("/Users/brian/Desktop/VideoBliinds/*5.bmp", load_func=imread_convert)
#frames.append(skimage.io.imread("/Users/brian/Desktop/alien2.jpg", as_grey=1))
#f2 = np.array(skimage.io.imread("/Users/brian/Desktop/Abe.png", as_grey=1))
#f1 = np.expand_dims(f1,axis=3)
#f2 = np.expand_dims(f2,axis=3)
#print(f1.shape)
#frames = np.concatenate((f1,f2),axis=2)
frames = f1.concatenate()
frames = np.swapaxes(frames,0,2)
frames = np.swapaxes(frames,0,1)
print "frames shape"
print(frames.shape)

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


#mblock = 16;
#h=fspecial('gaussian',mblock);

mblock = 16
h=fspecial(8,.5) #our filter will operate on 17 blocks at a time I think 8 + center + 8
#in the matlab they used a 16 block filter





def costFuncMAD(currentBlk,refBlk):
    return np.average(np.abs(currentBlk.astype(np.float) - refBlk.astype(np.float)))
    
# % Finds the indices of the cell that holds the minimum cost
# %
# % Input
# %   costs : The matrix that contains the estimation costs for a macroblock
# %
# % Output
# %   dx : the motion vector component in columns
# %   dy : the motion vector component in rows
# %
# % Written by Aroh Barjatya

#function [dx, dy, min] = minCost(costs)
def minCost(costs):
    row, col = costs.shape
    idx = np.argmin(costs)
    dx = idx%row
    dy = int(idx/row)
    return dx, dy, costs[dy, dx]

def motionEstNTSS(imgP, imgI, mbSize, p):
    row, col = imgI.shape#right
    row1, col1 = imgP.shape
    vectors = np.zeros((2,row*col/mbSize**2),np.float)#right
    costs = np.ones((3, 3), float) * 65537#right
    allcosts = np.empty(shape=(1,3,3))
    costfile = open("./pythoncosts", "wb");

    L = int(np.log(p+1)/np.log(2.0))
    stepMax = pow(2,(L-1))#right

    computations = 0;
    mbCount = 0;
    for i in range(0, row-mbSize+1, mbSize):
        for j in range(0, col-mbSize+1, mbSize):
            x = j
            y = i
            costs[1,1] = costFuncMAD(imgP[i:i+mbSize,j:j+mbSize], imgI[i:i+mbSize,j:j+mbSize])
            stepSize = int(stepMax)
            computations += 1
            for m in xrange(-stepSize,stepSize+1, stepSize):        
                for n in xrange(-stepSize,stepSize+1,stepSize):
                    refBlkVer = y + m  # % row/Vert co-ordinate for ref block
                    refBlkHor = x + n  # % col/Horizontal co-ordinate
                    if ( refBlkVer < 0 or refBlkVer+mbSize > row
                         or refBlkHor < 0 or refBlkHor+mbSize > col):
                         continue
                    costRow = m/stepSize + 2
                    costCol = n/stepSize + 2
                    if (costRow == 2 and costCol == 2):
                        continue
                    costs[costRow-1, costCol-1] = costFuncMAD(imgP[i:i+mbSize,j:j+mbSize],
							imgI[refBlkVer:refBlkVer+mbSize, refBlkHor:refBlkHor+mbSize])
                    computations += 1
        
            dx, dy, min1 = minCost(costs)     #% finds which macroblock in imgI gave us min Cost
            x1 = x + (dx-1)*stepSize
            y1 = y + (dy-1)*stepSize
        
            stepSize = 1
            for m in xrange(-stepSize, stepSize+1, stepSize):        
                for n in xrange(-stepSize, stepSize+1, stepSize):
                    refBlkVer = y + m  # % row/Vert co-ordinate for ref block
                    refBlkHor = x + n  # % col/Horizontal co-ordinate

                    if ( refBlkVer < 0 or refBlkVer+mbSize > row
                         or refBlkHor < 0 or refBlkHor+mbSize > col):
                         continue

                    costRow = m/stepSize + 2
                    costCol = n/stepSize + 2
                    if (costRow == 2 and costCol == 2):
                        continue

                    costs[costRow-1, costCol-1] = costFuncMAD(imgP[i:i+mbSize,j:j+mbSize],
                        imgI[refBlkVer:refBlkVer+mbSize, refBlkHor:refBlkHor+mbSize])
                    computations += 1
            
            
            # % now find the minimum amongst this
            dx, dy, min2 = minCost(costs)     # % finds which macroblock in imgI gave us min Cost

            # % Find the exact co-ordinates of this point
            x2 = x + (dx-1)*stepSize
            y2 = y + (dy-1)*stepSize
            
            # % the only place x1 == x2 and y1 == y2 will take place will be the
            # % center of the search region
            if (x1 == x2 and y1 == y2):
                NTSSFlag = -1# % this flag will take us out of any more computations 
            elif (min2 <= min1):
                x = x2
                y = y2
                NTSSFlag = 1 #% this flag signifies we are going to go into NTSS mode
            else:
                x = x1
                y = y1
                NTSSFlag = 0 #% This value of flag says, we go into normal TSS
    
            if (NTSSFlag == 1):
                    costs = ones((3,3), float) * 65537
                    costs[1,1] = min2
                    stepSize = 1 
                    for m in xrange(-stepSize, stepSize+1, stepSize):
                        for n in xrange(-stepSize, stepSize+1, stepSize):
                            refBlkVer = y + m  # % row/Vert co-ordinate for ref block
                            refBlkHor = x + n  # % col/Horizontal co-ordinate
                            if ( refBlkVer < 0 or refBlkVer + mbSize > row
                                   or refBlkHor < 0 or refBlkHor + mbSize > col):
                                continue
                            
                            if ( (refBlkVer >= i  and refBlkVer <= i) \
                                    and (refBlkHor >= j  and refBlkHor <= j) ):
                                continue
                          
                            costRow = m/stepSize + 2
                            costCol = n/stepSize + 2
                            if (costRow == 2 and costCol == 2):
                                continue
                            
                            costs[costRow-1, costCol-1] = costFuncMAD(imgP[i:i+mbSize,j:j+mbSize],
                                 imgI[refBlkVer:refBlkVer+mbSize, refBlkHor:refBlkHor+mbSize])
                            computations += 1

                    # % now find the minimum amongst this
                    dx, dy, min2 = minCost(costs)
                    x = x + (dx-1)*stepSize
                    y = y + (dy-1)*stepSize           
                    
            elif (NTSSFlag == 0):
                    # % this is when we are going about doing normal TSS business
                    costs = np.ones((3,3), np.float) * 65537;
                    costs[1,1] = min1
                    stepSize = int(stepMax / 2)
                    while(stepSize >= 1):
                        for m in xrange(-stepSize, stepSize+1, stepSize):
                            for n in xrange(-stepSize, stepSize+1, stepSize):
                                refBlkVer = y + m   #% row/Vert co-ordinate for ref block
                                refBlkHor = x + n   #% col/Horizontal co-ordinate
                                if ( refBlkVer < 0 or refBlkVer+mbSize > row \
                                    or refBlkHor < 0 or refBlkHor+mbSize > col):
                                    continue

                                costRow = m/stepSize + 2;
                                costCol = n/stepSize + 2;
                                if (costRow == 2 and costCol == 2):
                                    continue
                                costs[costRow-1, costCol-1] = costFuncMAD(imgP[i:i+mbSize,j:j+mbSize], 
								imgI[refBlkVer:refBlkVer+mbSize, refBlkHor:refBlkHor+mbSize])
                                computations += 1
                
                        # % Now we find the vector where the cost is minimum
                        # % and store it ... this is what will be passed back.
                
                        dx, dy, min = minCost(costs) #    % finds which macroblock in imgI gave us min Cost

                        # % shift the root for search window to new minima point

                        x = x + (dx-1)*stepSize
                        y = y + (dy-1)*stepSize
                    
                        stepSize = int(stepSize / 2)
                        costs[1,1] = costs[dy,dx]
            allcosts = np.append(allcosts,[costs],axis=0)
            vectors[0,mbCount] = y - i  #  % row co-ordinate for the vector
            vectors[1,mbCount] = x - j  #  % col co-ordinate for the vector            
            mbCount += 1
            costs = ones((3,3), np.float) * 65537
    allcosts.dump(costfile)
    motionVect = vectors
    #print 'motionVect'
    #print motionVect
    #NTSSComputations = computations/(mbCount - 1) original, but set mbCount to 0
    NTSSComputations = computations/(mbCount - 1)#however computations also starts at 0 in the matlab code.
    return motionVect, NTSSComputations


#COMPUTING DC COEFFICIENT CONTINUED NOW THAT WE HAVE THE FUNCTION WE NEED
def im2colDistinct(A, size):
    dx, dy = size
    #assert A.shape[0] % dy == 0
    #assert A.shape[1] % dx == 0
    #performing padding instead of the above.
    if ( A.shape[0] % dx != 0 or A.shape[1] % dy != 0 ):
        AA = np.lib.pad(A, (A.shape[0] % dx, A.shape[1] % dy ), 'constant', constant_values=(0,0))
    else:
        AA = A
    
    mblocks = AA.shape[0] / size[0]
    nblocks = AA.shape[1] / size[1]
    mblocks = int(mblocks) 
    nblocks = int(nblocks) 
    #rows = 0:size[0]
    #cols = 0:size[1]
    
    #size inner product
    sizeInnerProduct = 1
    for i in size:
        sizeInnerProduct *= i
    b = np.zeros((sizeInnerProduct, mblocks*nblocks))
    x = np.zeros((sizeInnerProduct,1))
    for i in xrange(0, mblocks-1):
        for j in xrange(0, nblocks-1):
            #x[:] = AA[i * size[0] + rows , j * size[1] + cols]
            #what is this doing?
            #we're inserting new columns into x equal to the slice of A over the first axis
            #from i*size[0] + rows intersected with the slice of A over the second axis
            #from j*size[1] + cols
            #size is the matrix which tells you how big the blocks are.
            #adding i * size[0] to each slice should produce a new slice that starts and ends
            #at numbers that much higher.
            x[:] = AA[(i * size[0]):((i + 1) * size[0]),(j * size[1]):((j + 1) * size[1])]
            print 'x'
            print x
            #TODO- but it seems like the block size of the index slicing wouldn't fit into a single vector!
            #maybe there's some magic sauce in the matlab implementation?
            b[:,i+j*mblocks+1] = x
    return b
    #ncol = (A.shape[0]/dx) * (A.shape[1]/dy)
    #R = np.empty((ncol, dx*dy), dtype=A.dtype)
    #k = 0
    #for i in xrange(0, A.shape[0], dy):
    #    for j in xrange(0, A.shape[1], dx):
    #        R[:,k] = A[i:i+dx, j:j+dy].ravel()
    #        k += 1
    #return R


def temporal_dc_variation_feature_extraction(frames):
    '''
    computes dt_dc_measure 1
    ''' 
    mbsize = 16
    row = frames.shape[0]
    col = frames.shape[1]
        
    motion_vects = zeros(shape=(2,row*col/mbsize**2,frames.shape[2]-1))
    for x in xrange(5,frames.shape[2]-1):#xrange is inclusive at beginning, exclusive at end,start at 6th frame,end 1 early since x+1
        imgP = frames[:,:,x+1]
        imgI = frames[:,:,x]
        motion_vects[:,:,x], temp = motionEstNTSS(imgP,imgI,mblock,7)
    print "Motion vects"
    print motion_vects
    
    motion_vects.dump(open("./pythonarray", "wb"))
    dct_motion_comp_diff = zeros(shape=(row,col,frames.shape[2]))
    for x in xrange(0,frames.shape[2]):
        mbCount = 0
        for i in xrange(0,row-mbsize-1,mbsize):
            for j in xrange(0,col-mbsize-1,mbsize):
                #dct_motion_comp_diff(i:i+mbsize-1,j:j+mbsize-1,x) = dct2(frames(i:i+mbsize-1,j:j+mbsize-1,x+1)-frames(i+motion_vects16x16(1,mbCount,x):i+mbsize-1+motion_vects16x16(1,mbCount,x),j+motion_vects16x16(2,mbCount,x):j+mbsize-1+motion_vects16x16(2,mbCount,x),x));
                #dct_motion_comp_diff[i:i+mbsize,j:j+mbsize-1,x] = dct(dct(np.pad(frames[i:i+mbsize-1,j:j+mbsize-1,x+1]-frames[i+motion_vects[0,mbCount,x]:i+mbsize-1+motion_vects[0,mbCount,x],j+motion_vects[1,mbCount,x]:j+mbsize-1+motion_vects[1,mbCount,x],x]).T).T);
                print frames[i:i+mbsize-1,j:j+mbsize-1,x+1].shape
                print frames[i+motion_vects[0,mbCount,x]:i+mbsize-1+motion_vects[0,mbCount,x],j+motion_vects[1,mbCount,x]:j+mbsize-1+motion_vects[1,mbCount,x],x].shape
                #if (i+motion_vects[0,mbCount,x] < 0 or j+motion_vects[1,mbCount,x] < 0):
                #        #need more if statements.
                #    tempMatrix1 = zeros(mbsize-1,mbsize-1,frames.shape[2])
                #    tempMatrix2 = zeros(mbsize-1,mbsize-1,frames.shape[2])
                #    if (i+motion_vects[0,mbCount,x] < 0 and i+mbsize-1+motion_vects[0,mbCount,x] < 0):
                #            #tempMatrix1 = frames[i+motion_vects[0,mbCount,x]:i+mbsize-1+motion_vects[0,mbCount,x],j+motion_vects[1,mbCount,x]:j+mbsize-1+motion_vects[1,mbCount,x],x].shape
                #            #here we add the negative row values to the max value
                #        dct_motion_comp_diff[i:i+mbsize-1,j:j+mbsize-1,x] += dct(dct((frames[i:i+mbsize-1,j:j+mbsize-1,x+1]-
                #                frames[row + (i+motion_vects[0,mbCount,x]):row + (i+mbsize-1+motion_vects[0,mbCount,x]),
                #                j+motion_vects[1,mbCount,x]:j+mbsize-1+motion_vects[1,mbCount,x],x]).astype(np.float)).T);
                #    else if (i+motion_vects[0,mbCount,x] < 0 and i+mbsize-1+motion_vects[0,mbCount,x] > 0):
                #        dct_motion_comp_diff[i:i+mbsize-1,j:j+mbsize-1,x] += dct(dct((frames[i:i+mbsize-1,j:j+mbsize-1,x+1]-
                #                frames[i+motion_vects[0,mbCount,x]:i+mbsize-1+motion_vects[0,mbCount,x],
                #                j+motion_vects[1,mbCount,x]:j+mbsize-1+motion_vects[1,mbCount,x],x]).astype(np.float)).T);
                #            #tempMatrix1 = frames[i+motion_vects[0,mbCount,x]:i+mbsize-1+motion_vects[0,mbCount,x],j+motion_vects[1,mbCount,x]:j+mbsize-1+motion_vects[1,mbCount,x],x].shape
                #    if (j+motion_vects[0,mbCount,x] < 0 and j+mbsize-1+motion_vects[0,mbCount,x] < 0):
                #        dct_motion_comp_diff[i:i+mbsize-1,j:j+mbsize-1,x] += dct(dct((frames[i:i+mbsize-1,j:j+mbsize-1,x+1]-
                #                frames[i+motion_vects[0,mbCount,x]:i+mbsize-1+motion_vects[0,mbCount,x],
                #                j+motion_vects[1,mbCount,x]:j+mbsize-1+motion_vects[1,mbCount,x],x]).astype(np.float)).T);
                #            #tempMatrix2 = frames[i+motion_vects[0,mbCount,x]:i+mbsize-1+motion_vects[0,mbCount,x],col+(j+motion_vects[1,mbCount,x]):col+(j+mbsize-1+motion_vects[1,mbCount,x]),x].shape
                #            #don't have to worry about indexing over nonexistent stuff, numpy just ignores indeces past the max.
                #    else if (j+motion_vects[0,mbCount,x] < 0 and j+mbsize-1+motion_vects[0,mbCount,x] > 0):
                #        dct_motion_comp_diff[i:i+mbsize-1,j:j+mbsize-1,x] += dct(dct((frames[i:i+mbsize-1,j:j+mbsize-1,x+1]-
                #                frames[i+motion_vects[0,mbCount,x]:i+mbsize-1+motion_vects[0,mbCount,x],
                #                j+motion_vects[1,mbCount,x]:j+mbsize-1+motion_vects[1,mbCount,x],x]).astype(np.float)).T);
                #            #tempMatrix2 = frames[i+motion_vects[0,mbCount,x]:i+mbsize-1+motion_vects[0,mbCount,x],j+motion_vects[1,mbCount,x]:j+mbsize-1+motion_vects[1,mbCount,x],x].shape
                #    dct_motion_comp_diff[i:i+mbsize-1,j:j+mbsize-1,x] += dct(dct((frames[i:i+mbsize-1,j:j+mbsize-1,x+1]-
                #            frames[i+motion_vects[0,mbCount,x]:i+mbsize-1+motion_vects[0,mbCount,x],
                #            j+motion_vects[1,mbCount,x]:j+mbsize-1+motion_vects[1,mbCount,x],x]).astype(np.float)).T);
                #    #TODO some kind of weird array concatenation?
                #else:
                dct_motion_comp_diff[i:i+mbsize-1,j:j+mbsize-1,x] = dct(dct((frames[i:i+mbsize-1,j:j+mbsize-1,x+1]-
                            frames[i+motion_vects[0,mbCount,x]:i+mbsize-1+motion_vects[0,mbCount,x],
                            j+motion_vects[1,mbCount,x]:j+mbsize-1+motion_vects[1,mbCount,x],x]).astype(np.float)).T);
                mbCount = mbCount + 1

    print "dct_motion_comp_diff"
    print dct_motion_comp_diff

    std_dc = zeros(shape=(frames.shape[2]-1))
    for i in xrange(0,frames.shape[2]-1):
        temp = im2colDistinct(dct_motion_comp_diff[:,:,i],(16,16));
        #this returns a 16*16 vector....why? wouldn't I want something that corresponds to the height and width of dct_motion_comp_diff?
        #by sheer coincedence, dct_motion_comp_diff is 16x16, this is set by mbsize.
        
        print 'temp im2colDistinct'
        print temp
        #TODO-current error is here some kind of weird numpy broadcasting thing or something.,
        #also, there may be an error with 
    #print 'std_dc'
    #print std_dc
    #print 'im2col motioncompdiff'
    #print dct_motion_comp_diff.shape
    
    dt_dc_temp = zeros(shape=(frames.shape[2]-1))
    for i in xrange(0,len(std_dc) - 1):
        dt_dc_temp[i] = abs(std_dc[i+1]-std_dc[i])
    print 'dt_dc_temp'
    print dt_dc_temp.shape
    print dt_dc_temp
    
    dt_dc_measure1 = np.mean(dt_dc_temp)    
    print 'dt_dc_measure1'
    print dt_dc_measure1
    

# for i = 1:size(frames,3)-1
    # temp = im2col(dct_motion_comp_diff(:,:,i),[16,16],'distinct');
    # std_dc(i) = std(temp(1,:));
# end
# clear *motion*

# for i = 1:length(std_dc)-1
    # dt_dc_temp(i) = abs(std_dc(i+1)-std_dc(i));

# end

#dt_dc_measure1 = mean(dt_dc_temp);
    
temporal_dc_variation_feature_extraction(frames)
    
    
    

def nss_spectral_ratios_feature_extraction(frames):
# '''
# computes dt_dc_measure2 and geo_ratio_features
# '''

# %% PART A of Video-BLIINDS: Computing the NSS DCT features:

# %% Step 1: Compute local (5x5 block-based) DCT of frame differences

    mblock = 5

    row = len(frames[0])
    col = len(frames[1])
    nFrames = len(frames[2])

    #dct_diff5x5 = zeros(mblock^2,floor(row/mblock)*floor(col/mblock),nFrames-1);
    dct_diff5x5 = zeros(mblock**2, math.floor(row/mblock)*math.floor(col/mblock),nFrames-1)

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
    for x in xrange(len(nFrames)-1):
        mbCount = 0
        for i in xrange( mblock, row-mblock+1):
            for j in xrange(mblock, col-mblock+1):
                mbCount = mbCount+1
                temp = dct2(frames[i:i+mblock-1,j:j+mblock-1,x+1] - frames[i:i+mblock-1,j:j+mblock-1,x])
                dct_diff5x5[:,mbCount,x] = temp[:]

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

