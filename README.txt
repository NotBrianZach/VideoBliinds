
Files added by Zach so far: 
matLoader.m
niqe_features.py <-- todd's implementation of niqe feature extraction
dc_features.py <--- this is where the action is.


This is the implementation of Video BLIINDS as described in:

M.A. Saad, A.C. Bovik, and C. Charrier, "Blind Prediction of Natural Video Quality", IEEE Transactions on Image Processing, Vol. 23, no. 3, pp. 1352-1365, March 2014.

Please note, that Video BLIINDS extracts two DC features. There was a slight erratum in the IEEE Transactions on Image Processing paper in which we have missed describing one of the two DC features (which is very related to the other DC feature described). We will be adding a description of the second feature in the code distribution soon.

To run the Video BLIINDS algorithm please run the Matlab script "video_bliinds_algo.m".

You need to have R installed since the prediction requires calling an R script. You also need to have the R package "Kernlab" installed as well.

Please contact Michele Saad (michele.saad@utexas.edu) for any questions.

