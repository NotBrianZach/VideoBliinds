%% Computing the DC temporal variation
%% feature a.k.a. the DC feature
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

def temporal_dc_variation_feature_extraction(frames):
