import numpy as np
import scipy.io as sio

np.set_printoptions(threshold=np.nan)


#content_file = sio.loadmat('./matlaboverallcosts.mat')
#matlabarray = content_file['overallcosts']

content_file = sio.loadmat('./matlaboverallcosts.mat')
matlabcosts= content_file['overallcosts']
#matlabcosts = np.load(matlabcostsarray)
print "matlab costs"
print matlabcosts.shape
#print matlabcosts
#with open('./matlaboverallcosts', 'r') as content_file:
#    matlabcostsarray = content_file['overallcosts']
#    matlabcosts = np.load(matlabcostsarray)
#    print "matlab costs"
#    print matlabcosts.shape
#    print matlabcosts
#    #while (pythoncosts.any()):

with open('./pythonarray', 'r') as content_file:
    pythonarray = np.load(content_file)

with open('./pythoncosts', 'r') as content_file:
    pythoncosts = np.load(content_file)
    #while (pythoncosts.any()):
    print "python costs"
    print pythoncosts.shape
    #print pythoncosts


print "python array shape"
print pythonarray.shape


#print "python costs2"
#print pythoncosts2

print "matlab array shape"
print matlabarray.shape

f = open('finaloutputcomparison','w')
#f.write(np.array_str(pythonarray) + "\n" + np.array_str(matlabarray))
#f.write(np.array_str(pythonarray - matlabarray))
f.write(np.array_str(pythoncosts- matlabcosts))

k = open('pythonout','w')
k.write(np.array_str(pythoncosts))

#g = open('matlabout','w')
#g.write(np.array_str(matlabarray))

#pythonarray = eval(pythonarray)
#if data is a string, it will convert matlab array into numpy array
