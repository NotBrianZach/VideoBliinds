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
from skimage.feature import hog
from skimage import data, color, exposure
from sklearn.externals import joblib
from scipy.fftpack import dct

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
frames = [skimage.io.imread("C:/Users/Zach/Desktop/movieFrames/frames_2003.77.00000075.bmp", as_grey=1)]
frames.append(skimage.io.imread("C:/Users/Zach/Desktop/movieFrames/frames_2003.77.00000076.bmp", as_grey=1))

def temporal_dc_variation_feature_extraction(frames):
    '''
    computes dt_dc_measure 1
    ''' 
    print 5 

#mblock = 16;
#h=fspecial('gaussian',mblock);

mblock = 16
h=fspecial(8,.5) #our filter will operate on 17 blocks at a time I think 8 + center + 8
#in the matlab they used a 16 block filter



# for x=1:size(frames,3)-1
    # x
    # tic  
    # imgP = double(frames(:,:,x+1));
    # imgI = double(frames(:,:,x));
    # [motion_vects16x16(:,:,x) temp] = motionEstNTSS(imgP,imgI,mblock,7);
    # toc
# end


# def motionEstNTSS(imgP, imgI,mbSize,p):
    # return 5, 3

motion_vects16x16 = []
# for x in xrange(len(frames) - 1):#xrange is inclusive at beginning, exclusive at end
    # print x
    # imgP = double(frames[x+1])
    # imgI = double(frames[x])
    # imgP = frames[x+1]
    # imgI = frames[x]
    # motion_vects16x16[x] = [][]
    # motion_vects16x16, temp = motionEstNTSS(imgP,imgI,mblock,7)
    # toc
    
# print frames[0].size
# print frames[0].shape
# print len(frames)
    

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




def costFuncMAD(currentBlk,refBlk, n):
# % Computes the Mean Absolute Difference (MAD) for the given two blocks
# % Input
# %       currentBlk : The block for which we are finding the MAD
# %       refBlk : the block w.r.t. which the MAD is being computed
# %       n : the side of the two square blocks
# %
# % Output
# %       cost : The MAD for the two blocks
# %
# % Written by Aroh Barjatya

    # err = 0;
    err = 0
    # for i = 1:n
    for i in xrange(0,n - 1):
        # for j = 1:n
        for j in xrange(0,n - 1):
            err = err + abs((currentBlk[i,j] - refBlk[i,j]))
            # err = err + abs((currentBlk(i,j) - refBlk(i,j)));
        # end
    # end
    cost = err / (n*n)
    return cost
    
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
    # [row, col] = size(costs);
    row, col = costs.shape

    # % we check whether the current
    # % value of costs is less then the already present value in min. If its
    # % indeed smaller then we swap the min value with the current one and note
    # % the indices.

    min = 65537

    # for i = 1:row
        # for j = 1:col
            # if (costs(i,j) < min)
                # min = costs(i,j);
                # dx = j; dy = i;
            # end
        # end
    # end
    for i in xrange(1,row+1):
        for j in  xrange(1,col+1):
            if (costs[i,j] < min):
                min = costs(i,j)
                dx = j 
                dy = i
                return min,dx,dy





def motionEstNTSS(imgP, imgI,mbSize,p):
# % Computes motion vectors using *NEW* Three Step Search method
# %
# % Based on the paper by R. Li, b. Zeng, and M. L. Liou
# % IEEE Trans. on Circuits and Systems for Video Technology
# % Volume 4, Number 4, August 1994 :  Pages 438:442
# %
# % Input
# %   imgP : The image for which we want to find motion vectors
# %   imgI : The reference image
# %   mbSize : Size of the macroblock
# %   p : Search parameter  (read literature to find what this means)
# %
# % Ouput
# %   motionVect : the motion vectors for each integral macroblock in imgP
# %   NTSScomputations: The average number of points searched for a macroblock
# %
# % Written by Aroh Barjatya


# function [motionVect, NTSScomputations] = motionEstNTSS(imgP, imgI, mbSize, p)

# [row col] = size(imgI);
    row, col = imgI.shape
    print "row, col"
    print row
    print col
    row1, col1 = imgP.shape
    print "row1, col1"
    print row1
    print col1
# vectors = zeros(2,row*col/mbSize^2);
# costs = ones(3, 3) * 65537;

    vectors = zeros((2,row*col/mbSize^2),float)
    costs = ones((3, 3), float) * 65537


# % we now take effectively log to the base 2 of p
# % this will give us the number of steps required

# L = floor(log10(p+1)/log10(2));   
# stepMax = 2^(L-1);

# computations = 0;
    L = floor(math.log10(p+1)/ math.log10(2))
    stepMax = pow(2,(L-1))

    computations = 0;

# % we start off from the top left of the image
# % we will walk in steps of mbSize
# % for every macroblock that we look at we will look for
# % a close match p pixels on the left, right, top and bottom of it

# mbCount = 1;
# for i = 1 : mbSize : row-mbSize+1
    # for j = 1 : mbSize : col-mbSize+1
    mbCount = 1;

    for i in range(0, mbSize, row-mbSize):
        for j in range(0, mbSize, col-mbSize):
        
        # % the NEW three step search starts
   
        # x = j;
        # y = i;
            x = j;
            y = i;
        
        # % In order to avoid calculating the center point of the search
        # % again and again we always store the value for it from the
        # % previous run. For the first iteration we store this value outside
        # % the for loop, but for subsequent iterations we store the cost at
        # % the point where we are going to shift our root.
        # %
        # % For the NTSS, we find the minimum first in the far away points
        # % we then find the minimum for the close up points
        # % we then compare the minimums and which ever is the lowest is where
        # % we shift our root of search. If the minimum is the center of the
        # % current window then we stop the search. If its one of the
        # % immediate close to the center then we will do the second step
        # % stop. And if its in the far away points, then we go doing about
        # % the normal TSS approach
        # % 
        # % more details in the code below or read the paper/literature
        
        # costs(2,2) = costFuncMAD(imgP(i:i+mbSize-1,j:j+mbSize-1), ...
                                    # imgI(i:i+mbSize-1,j:j+mbSize-1),mbSize);
        # stepSize = stepMax; 
        # computations = computations + 1;
            print i
            print mbSize
            print j
            costs = costFuncMAD(imgP[i:i+mbSize-1,j:j+mbSize-1],
                                    imgI[i:i+mbSize-1,j:j+mbSize-1], mbSize)
            stepSize = int(stepMax)
            computations = computations + 1

        # % This is the calculation of the outer 8 points
        # % m is row(vertical) index
        # % n is col(horizontal) index
        # % this means we are scanning in raster order
        # for m = -stepSize : stepSize : stepSize        
            # for n = -stepSize : stepSize : stepSize
                # refBlkVer = y + m;   % row/Vert co-ordinate for ref block
                # refBlkHor = x + n;   % col/Horizontal co-ordinate
                # if ( refBlkVer < 1 || refBlkVer+mbSize-1 > row ...
                     # || refBlkHor < 1 || refBlkHor+mbSize-1 > col)
                     # continue;
                # end

                # costRow = m/stepSize + 2;
                # costCol = n/stepSize + 2;
                # if (costRow == 2 && costCol == 2)
                    # continue
                # end
                # costs(costRow, costCol ) = costFuncMAD(imgP(i:i+mbSize-1,j:j+mbSize-1), ...
                    # imgI(refBlkVer:refBlkVer+mbSize-1, refBlkHor:refBlkHor+mbSize-1), mbSize);
                # computations = computations + 1;
            # end
        # end
        
        
        # % This is the calculation of the outer 8 points
        # % m is row(vertical) index
        # % n is col(horizontal) index
        # % this means we are scanning in raster order
            for m in xrange(-stepSize,stepSize, stepSize):        
                for n in xrange(-stepSize,stepSize,stepSize):
                    refBlkVer = y + m   #% row/Vert co-ordinate for ref block
                    refBlkHor = x + n  # % col/Horizontal co-ordinate
                    if ( refBlkVer < 1 or refBlkVer+mbSize-1 > row \
                         or refBlkHor < 1 or refBlkHor+mbSize-1 > col):
                         continue
                    # end

                    costRow = m/stepSize + 2
                    costCol = n/stepSize + 2
                    if (costRow == 2 and costCol == 2):
                        continue
                    # end
                    costs[costRow, costCol] = costFuncMAD(imgP[i:i+mbSize-1,j:j+mbSize-1], \
                        imgI[refBlkVer:refBlkVer+mbSize-1, refBlkHor:refBlkHor+mbSize-1], mbSize);
                    computations = computations + 1
                # end
            # end
        
        
        # % Now we find the vector where the cost is minimum
        # % and store it ... 
        
        # [dx, dy, min1] = minCost(costs);      % finds which macroblock in imgI gave us min Cost
            dx, dy, min1 = minCost(costs)     #% finds which macroblock in imgI gave us min Cost
            
              
        # % Find the exact co-ordinates of this point
            
        # x1 = x + (dx-2)*stepSize;
        # y1 = y + (dy-2)*stepSize;
            x1 = x + (dx-2)*stepSize
            y1 = y + (dy-2)*stepSize
        # % Now find the costs at 8 points right next to the center point
        # % (x,y) still points to the center
        
        # stepSize = 1;
        # for m = -stepSize : stepSize : stepSize        
            # for n = -stepSize : stepSize : stepSize
                # refBlkVer = y + m;   % row/Vert co-ordinate for ref block
                # refBlkHor = x + n;   % col/Horizontal co-ordinate
                # if ( refBlkVer < 1 || refBlkVer+mbSize-1 > row ...
                     # || refBlkHor < 1 || refBlkHor+mbSize-1 > col)
                     # continue;
                # end

                # costRow = m/stepSize + 2;
                # costCol = n/stepSize + 2;
                # if (costRow == 2 && costCol == 2)
                    # continue
                # end
                # costs(costRow, costCol ) = costFuncMAD(imgP(i:i+mbSize-1,j:j+mbSize-1), ...
                    # imgI(refBlkVer:refBlkVer+mbSize-1, refBlkHor:refBlkHor+mbSize-1), mbSize);
                # computations = computations + 1;
            # end
        # end
        
        
        # % Now find the costs at 8 points right next to the center point
        # % (x,y) still points to the center
            
            stepSize = 1
            for m in xrange(-stepSize, stepSize, stepSize):        
                for n in xrange(-stepSize, stepSize, stepSize):
                    refBlkVer = y + m  # % row/Vert co-ordinate for ref block
                    refBlkHor = x + n  # % col/Horizontal co-ordinate
                    if ( refBlkVer < 1 or refBlkVer+mbSize-1 > row \
                         or refBlkHor < 1 or refBlkHor+mbSize-1 > col):
                         continue
                    #end

                    costRow = m/stepSize + 2
                    costCol = n/stepSize + 2
                    if (costRow == 2 and costCol == 2):
                        continue
                    #end
                    costs[costRow, costCol] = costFuncMAD(imgP[i:i+mbSize-1,j:j+mbSize-1], \
                        imgI[refBlkVer:refBlkVer+mbSize-1, refBlkHor:refBlkHor+mbSize-1], mbSize)
                    computations = computations + 1
            
            
            # % now find the minimum amongst this
            
            # [dx, dy, min2] = minCost(costs);      % finds which macroblock in imgI gave us min Cost
            dx, dy, min2 = minCost(costs)     # % finds which macroblock in imgI gave us min Cost

                  
            # % Find the exact co-ordinates of this point

            # x2 = x + (dx-2)*stepSize;
            # y2 = y + (dy-2)*stepSize;
            x2 = x + (dx-2)*stepSize
            y2 = y + (dy-2)*stepSize
            
            # % the only place x1 == x2 and y1 == y2 will take place will be the
            # % center of the search region
            
            # if (x1 == x2 && y1 == y2)
                # % then x and y still remain pointing to j and i;
                # NTSSFlag = -1; % this flag will take us out of any more computations 
            # elseif (min2 <= min1)
                # x = x2;
                # y = y2;
                # NTSSFlag = 1; % this flag signifies we are going to go into NTSS mode
            # else
                # x = x1;
                # y = y1;
                # NTSSFlag = 0; % This value of flag says, we go into normal TSS
            # end
            
            
            # % the only place x1 == x2 and y1 == y2 will take place will be the
            # % center of the search region
            
            if (x1 == x2 and y1 == y2):
                #% then x and y still remain pointing to j and i;
                NTSSFlag = -1# % this flag will take us out of any more computations 
            elif (min2 <= min1):
                x = x2
                y = y2
                NTSSFlag = 1 #% this flag signifies we are going to go into NTSS mode
            else:
                x = x1
                y = y1
                NTSSFlag = 0 #% This value of flag says, we go into normal TSS
        
        # if (NTSSFlag == 1)
            # % Now in order to make sure that we dont calcylate the same
            # % points again which were in the initial center window we take
            # % care as follows
            
            # costs = ones(3,3) * 65537;
            # costs(2,2) = min2;
            # stepSize = 1;
            # for m = -stepSize : stepSize : stepSize        
                # for n = -stepSize : stepSize : stepSize
                    # refBlkVer = y + m;   % row/Vert co-ordinate for ref block
                    # refBlkHor = x + n;   % col/Horizontal co-ordinate
                    # if ( refBlkVer < 1 || refBlkVer+mbSize-1 > row ...
                           # || refBlkHor < 1 || refBlkHor+mbSize-1 > col)
                        # continue;
                    # end
                    
                    # if ( (refBlkVer >= i - 1  && refBlkVer <= i + 1) ...
                            # && (refBlkHor >= j - 1  && refBlkHor <= j + 1) )
                        # continue;
                    # end
                    
                    # costRow = m/stepSize + 2;
                    # costCol = n/stepSize + 2;
                    # if (costRow == 2 && costCol == 2)
                        # continue
                    # end
                    # costs(costRow, costCol ) = costFuncMAD(imgP(i:i+mbSize-1,j:j+mbSize-1), ...
                         # imgI(refBlkVer:refBlkVer+mbSize-1, refBlkHor:refBlkHor+mbSize-1), mbSize);
                    # computations = computations + 1;
                # end
            # end
                
            # % now find the minimum amongst this
        
            # [dx, dy, min2] = minCost(costs);      % finds which macroblock in imgI gave us min Cost
            
            # % Find the exact co-ordinates of this point and stop

            # x = x + (dx-2)*stepSize;
            # y = y + (dy-2)*stepSize;            
            
        # elseif (NTSSFlag == 0)
            # % this is when we are going about doing normal TSS business
            # costs = ones(3,3) * 65537;
            # costs(2,2) = min1;
            # stepSize = stepMax / 2;
            # while(stepSize >= 1)  
                # for m = -stepSize : stepSize : stepSize        
                    # for n = -stepSize : stepSize : stepSize
                        # refBlkVer = y + m;   % row/Vert co-ordinate for ref block
                        # refBlkHor = x + n;   % col/Horizontal co-ordinate
                        # if ( refBlkVer < 1 || refBlkVer+mbSize-1 > row ...
                            # || refBlkHor < 1 || refBlkHor+mbSize-1 > col)
                            # continue;
                        # end

                        # costRow = m/stepSize + 2;
                        # costCol = n/stepSize + 2;
                        # if (costRow == 2 && costCol == 2)
                            # continue
                        # end
                        # costs(costRow, costCol ) = costFuncMAD(imgP(i:i+mbSize-1,j:j+mbSize-1), ...
                                # imgI(refBlkVer:refBlkVer+mbSize-1, refBlkHor:refBlkHor+mbSize-1), mbSize);
                        # computations = computations + 1;
                    
                    # end
                # end
        
                # % Now we find the vector where the cost is minimum
                # % and store it ... this is what will be passed back.
        
                # [dx, dy, min] = minCost(costs);      % finds which macroblock in imgI gave us min Cost
            
            
                # % shift the root for search window to new minima point

                # x = x + (dx-2)*stepSize;
                # y = y + (dy-2)*stepSize;
            
                # stepSize = stepSize / 2;
                # costs(2,2) = costs(dy,dx);
            
            # end
        # end

        # vectors(1,mbCount) = y - i;    % row co-ordinate for the vector
        # vectors(2,mbCount) = x - j;    % col co-ordinate for the vector            
        # mbCount = mbCount + 1;
        # costs = ones(3,3) * 65537;
    # end
    
        if (NTSSFlag == 1):
                # % Now in order to make sure that we dont calculate the same
                # % points again which were in the initial center window we take
                # % care as follows
                
                costs = ones((3,3), Float) * 65537
                costs[2,2] = min2
                stepSize = 1
                for m in xrange(-stepSize, stepSize, stepSize):
                    for n in xrange(-stepSize, stepSize, stepSize):
                        refBlkVer = y + m  # % row/Vert co-ordinate for ref block
                        refBlkHor = x + n  # % col/Horizontal co-ordinate
                        if ( refBlkVer < 1 or refBlkVer + mbSize - 1 > row \
                               or refBlkHor < 1 or refBlkHor + mbSize - 1 > col):
                            continue
                        
                        if ( (refBlkVer >= i - 1  and refBlkVer <= i + 1) \
                                and (refBlkHor >= j - 1  and refBlkHor <= j + 1) ):
                            continue
                      
                        
                        costRow = m/stepSize + 2
                        costCol = n/stepSize + 2
                        if (costRow == 2 and costCol == 2):
                            continue
                        
                        costs[costRow, costCol] = costFuncMAD(imgP[i:i+mbSize-1,j:j+mbSize-1], \
                             imgI[refBlkVer:refBlkVer+mbSize-1, refBlkHor:refBlkHor+mbSize-1], mbSize)
                        computations = computations + 1;

                # % now find the minimum amongst this
            
                dx, dy, min2 = minCost(costs)  #    % finds which macroblock in imgI gave us min Cost
                
                # % Find the exact co-ordinates of this point and stop

                x = x + (dx-2)*stepSize
                y = y + (dy-2)*stepSize           
                
        elif (NTSSFlag == 0):
                # % this is when we are going about doing normal TSS business
                costs = ones((3,3), Float) * 65537;
                costs[2,2] = min1
                stepSize = stepMax / 2
                while(stepSize >= 1):
                    for m in xrange(-stepSize, stepSize, stepSize):
                        for n in xrange(-stepSize, stepSize, stepSize):
                            refBlkVer = y + m   #% row/Vert co-ordinate for ref block
                            refBlkHor = x + n   #% col/Horizontal co-ordinate
                            if ( refBlkVer < 1 or refBlkVer+mbSize-1 > row \
                                or refBlkHor < 1 or refBlkHor+mbSize-1 > col):
                                continue

                            costRow = m/stepSize + 2;
                            costCol = n/stepSize + 2;
                            if (costRow == 2 and costCol == 2):
                                continue
                            costs[costRow, costCol] = costFuncMAD(imgP[i:i+mbSize-1,j:j+mbSize-1], \
                                    imgI[refBlkVer:refBlkVer+mbSize-1, refBlkHor:refBlkHor+mbSize-1], mbSize)
                            computations = computations + 1
            
                    # % Now we find the vector where the cost is minimum
                    # % and store it ... this is what will be passed back.
            
                    dx, dy, min = minCost(costs) #    % finds which macroblock in imgI gave us min Cost

                    # % shift the root for search window to new minima point

                    x = x + (dx-2)*stepSize
                    y = y + (dy-2)*stepSize
                
                    stepSize = stepSize / 2
                    costs[2,2] = costs(dy,dx)

        vectors[1,mbCount] = y - i  #  % row co-ordinate for the vector
        vectors[2,mbCount] = x - j  #  % col co-ordinate for the vector            
        mbCount = mbCount + 1
        costs = ones((3,3), Float) * 65537
    motionVect = vectors
    NTSScomputations = computations/(mbCount - 1)
    return motionVect, NTSSComputations


 #COMPUTING DC COEFFICIENT CONTINUED NOW THAT WE HAVE THE FUNCTION WE NEED
    
    
motion_vects16x16 = []
for x in xrange(len(frames) - 1):#xrange is inclusive at beginning, exclusive at end
    print x
    # imgP = double(frames[x+1])
    # imgI = double(frames[x])
    imgP = frames[x+1]
    imgI = frames[x]
    #motion_vects16x16[x] = [][]
    motion_vects16x16, temp = motionEstNTSS(imgP,imgI,mblock,7)
    #toc
    
print frames[0].size
print frames[0].shape
print len(frames)
    

    
    
# mbsize = 16;
# row = size(frames,1);
# col = size(frames,2);

mbsize = 16;
row = frames.shape[0];
col = frames.shape[1];

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

for x in xrange(1,frames.shape[2]):
    print x
    mbCount = 1
    for i in xrange(1,mbsize + 1,row-mbsize+1):
        for j in xrange(1,mbsize + 1,col-mbsize+1):
            dct_motion_comp_diff[i:i+mbsize-1,j:j+mbsize-1,x] = dct(dct(numpy.pad(frames[i:i+mbsize-1,j:j+mbsize-1,x+1]-frames[i+motion_vects16x16[1,mbCount,x]:i+mbsize-1+motion_vects16x16[1,mbCount,x],j+motion_vects16x16[2,mbCount,x]:j+mbsize-1+motion_vects16x16[2,mbCount,x],x]).T).T);
            mbCount = mbCount + 1
        end
    end
end

# for i = 1:size(frames,3)-1
    # temp = im2col(dct_motion_comp_diff(:,:,i),[16,16],'distinct');
    # std_dc(i) = std(temp(1,:));
# end
# clear *motion*

# for i = 1:length(std_dc)-1
    # dt_dc_temp(i) = abs(std_dc(i+1)-std_dc(i));

# end

#dt_dc_measure1 = mean(dt_dc_temp);

def im2col(Im, block, style='sliding'):
    """block = (patchsize, patchsize)
        first do sliding
    """
    bx, by = block
    Imx, Imy = Im.shape
    Imcol = []
    for j in range(0, Imy):
        for i in range(0, Imx):
            if (i+bx <= Imx) and (j+by <= Imy):
                Imcol.append(Im[i:i+bx, j:j+by].T.reshape(bx*by))
            else:
                break
    return np.asarray(Imcol).T


for i in xrange(1,size(frames,3)):
    temp = im2col(dct_motion_comp_diff[:,:,i],(16,16),'distinct');
    std_dc[i] = std[temp[1,:]];

for i in xrange(1,len(std_dc)):
    dt_dc_temp[i] = abs(std_dc[i+1]-std_dc[i]);

dt_dc_measure1 = mean(dt_dc_temp);    
    
    
    
    
    
    
    
    

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

