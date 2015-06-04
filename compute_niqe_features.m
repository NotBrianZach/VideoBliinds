
function niqe_features = compute_niqe_features(frames)

load('frames_modelparameters.mat')

blocksizerow    = 96;
blocksizecol    = 96;
blockrowoverlap = 0;
blockcoloverlap = 0;
%size(frames,3)
 
for fr = 6 : size(frames,3)-5
    %fr
    [mu qq] = computequality(frames(:,:,fr),blocksizerow,blocksizecol,blockrowoverlap,blockcoloverlap,mu_prisparam,cov_prisparam);
    niqe_feat(fr-5,1:36) = mu;
    niqe_feat(fr-5,37) = qq;
    clear mu qq
end

niqe_features = mean(niqe_feat);
 