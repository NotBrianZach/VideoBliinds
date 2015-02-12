
filenames = dir('C:\Users\Zach\Desktop\movieFrames\*.bmp'); %# get information of all .bmp files in work dir
n  = numel(filenames);    %# number of .bmp files

% for i = 1:n
%     A = imread( filenames(i).name );
% 
%     %# gets full path, filename radical and extension
%     [fpath radical ext] = fileparts( filenames(i).name ); 
% 
%     save([radical '.mat'], 'A');                          
% end
for i = 1:n/200
    initFrames(:,:,i) = rgb2gray(imread( ['C:\Users\Zach\Desktop\movieFrames\' filenames(i).name] ));

    %# gets full path, filename radical and extension
    [fpath radical ext] = fileparts( filenames(i).name ); 
end
    save(['movieFrames.mat'], 'initFrames');                          

% x = load([radical);
% y = load("filename1.mat"_);
% save('combine.mat', 'x', 'y');