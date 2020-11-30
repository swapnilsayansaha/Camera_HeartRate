function [BVP, PR] = POS(VideoFile, FS, StartTime, Duration, WinSec, SkinSegmentTF, useFGTransform)
% POS: The Plane Orthogonal to Skin-Tone (POS) Method from: Wang, W., den Brinker, A. C., Stuijk, S., & de Haan, G. (2017). Algorithmic principles of remote PPG. IEEE Transactions on Biomedical Engineering, 64(7), 1479-1491. DOI: 10.1109/TBME.2016.2609282
%
%   Inputs:
%       VideoFile               = Video file path.
%       FS                      = Video framerate (fps).
%       StartTime               = Timepoint at which to start process (default = 0 seconds).
%       Duration                = Duration of the time window to process (default = 60 seconds).
%       WinSec                  = Window length in seconds.
%       SkinSegmentTF           = Use skin segment (true or false)
%       useFGTransform          = Use color distortion filter (true or false)

%   Outputs:
%       BVP                     = Processed Blood Volume Pulse (BVP).
%       PR                      = Estimated Pulse Rate (PR) from processed BVP timeseries using peak in periodogram.

% Daniel McDuff, Ethan Blackford, January 2019
% Copyright (c)
% Licensed under the MIT License and the RAIL AI License.
% Edited by Rishav Guha, UCLA

f = waitbar(0,'Frames processed','Name','Running POS...');
%% Parameters

LPF = 0.7; %low cutoff frequency (Hz) - specified as 40 bpm (~0.667 Hz) in reference
HPF = 2.5; %high cutoff frequency (Hz) - specified as 240 bpm in reference


%% Add Backup Functions
if(~license('test', 'image_toolbox') && SkinSegmentTF)
    addpath([cd '\optional\rgb2ycbcr.m']); %GNU GPL rgb2ycbcr.m function
end

%% Load Video:
VidObj = VideoReader(VideoFile);
VidObj.CurrentTime = StartTime;

FramesToRead = floor(Duration*VidObj.FrameRate); %video may be encoded at slightly different frame rate

%% Initialize face tracker:

faceDetector = vision.CascadeObjectDetector();
videoFrame = readFrame(VidObj);
bbox = step(faceDetector, videoFrame);
[~, i] = max(bbox(:,3).*bbox(:,4));
bbox = bbox(i,:);
bboxPoints = bbox2points(bbox(1, :));
points = detectMinEigenFeatures(rgb2gray(videoFrame), 'ROI', bbox);
pointTracker = vision.PointTracker('MaxBidirectionalError', 2);
points = points.Location;
initialize(pointTracker, points, videoFrame);
oldPoints = points;

%% Read Video and Spatially Average:
T = zeros(FramesToRead,1);%initialize time vector
RGB = zeros(FramesToRead,3);%initialize color signal
FN = 0;
while hasFrame(VidObj) && (VidObj.CurrentTime <= StartTime+Duration)
    FN = FN+1;
    T(FN) = VidObj.CurrentTime;
    VidFrame = readFrame(VidObj);
    
    %extract face:
    [points, isFound] = step(pointTracker, VidFrame);
    visiblePoints = points(isFound, :);
    oldInliers = oldPoints(isFound, :);
    if size(visiblePoints, 1) >= 2
        [xform, oldInliers, visiblePoints] = estimateGeometricTransform(...
            oldInliers, visiblePoints, 'similarity', 'MaxDistance', 4); 
        bboxPoints = transformPointsForward(xform, bboxPoints);
        VidROI = VidFrame(ceil(min(bboxPoints(:,2))):ceil(max(bboxPoints(:,2))),ceil(min(bboxPoints(:,1))):ceil(max(bboxPoints(:,1))),:);     
        oldPoints = visiblePoints;
        setPoints(pointTracker, oldPoints);
    end
    
    
    if(SkinSegmentTF)%skin segmentation - originally specified in reference as an OC-SVM from Wang et al. 2015
        YCBCR = rgb2ycbcr(VidROI);
        Yth = YCBCR(:,:,1)>80;
        CBth = (YCBCR(:,:,2)>77).*(YCBCR(:,:,2)<127);
        CRth = (YCBCR(:,:,3)>133).*(YCBCR(:,:,3)<173);
        ROISkin = VidROI.*repmat(uint8(Yth.*CBth.*CRth),[1,1,3]);
        RGB(FN,:) = squeeze(sum(sum(ROISkin,1),2)./sum(sum(logical(ROISkin),1),2));
    else
        RGB(FN,:) = sum(sum(VidROI,2)) ./ (size(VidROI,1)*size(VidROI,2));
    end
    waitbar(FN/FramesToRead,f,sprintf('Frames Processed: %d out of %d',FN,FramesToRead));
end
delete(f);

%% POS:
% Transform from: Wang, W., den Brinker, A. C., Stuijk, S., & de Haan, G. (2017, May). Color-distortion filtering for remote photoplethysmography. In Automatic Face & Gesture Recognition (FG 2017), 2017 12th IEEE International Conference on (pp. 71-78). IEEE.
if useFGTransform
    RGBBase = mean(RGB);
    RGBNorm = bsxfun(@times,RGB,1./RGBBase)-1;
    FF = fft(RGBNorm);
    F = (0:size(RGBNorm,1)-1)*FS/size(RGBNorm,1);
    H = FF*[-1/sqrt(6);2/sqrt(6);-1/sqrt(6)];
    W = (H.*conj(H))./sum(FF.*conj(FF),2);
    FMask = (F >= LPF)&(F <= HPF);
    FMask = FMask + fliplr(FMask);
    W=W.*FMask';%rectangular filter in frequency domain - not specified in original paper
    FF = FF.*repmat(W,[1,3]);
    RGBNorm=real(ifft(FF));
    RGBNorm = bsxfun(@times,RGBNorm+1,RGBBase);
    
    RGB=RGBNorm;
end
     
N = size(RGB,1);%line 0 - RGB is of length N frames
H = zeros(1,N);%line 1 - initialize to zeros of length of video sequence
l = ceil(WinSec*FS);%line 1 - window length equivalent to reference: 32 samples of 20 fps camera (default 1.6s)
for n = 1:N-1%line 2 - loop from first to last frame in video sequence
    %line 3 - spatial averaging was performed when video was read
    m = n - l + 1;%line 4 condition
    if(m > 0)%line 4
        Cn = ( RGB(m:n,:) ./ mean(RGB(m:n,:)) )';%line 5 - temporal normalization
        S = [0, 1, -1; -2, 1, 1] * Cn;%line 6 - projection
        h = S(1,:) + ((std(S(1,:)) / std(S(2,:))) * S(2,:));%line 7 - tuning
        H(m:n) = H(m:n) + (h - mean(h));%line 8 - overlap-adding
    end%line 9 - end if
end%line 10 - end for

BVP=H;

% Estimate Pulse Rate from periodogram
PR = prpsd(BVP,FS,40,240,false);


%% Remove Backup Functions
if(~license('test', 'image_toolbox')&&SkinSegmentTF)%remove path if added
    rmpath([cd '\optional\']);
end

end%end function
