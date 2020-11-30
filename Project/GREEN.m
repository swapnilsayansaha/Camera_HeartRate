function [BVP, PR] = GREEN(VideoFile, FS, StartTime, Duration)
% GREEN: The Green-Channel Method from: Verkruysse, W., Svaasand, L. O., & Nelson, J. S. (2008). Remote plethysmographic imaging using ambient light. Optics express, 16(26), 21434-21445. DOI: 10.1364/OE.16.021434
%
%   Inputs:
%       VideoFile               = Video file path.
%       FS                      = Video framerate (fps).
%       StartTime               = Timepoint at which to start process (default = 0 seconds).
%       Duration                = Duration of the time window to process (default = 60 seconds).

%   Outputs:
%       BVP                     = Processed Blood Volume Pulse (BVP).
%       PR                      = Estimated Pulse Rate (PR) from processed BVP timeseries using peak in periodogram.

% Daniel McDuff, Ethan Blackford, January 2019
% Copyright (c)
% Licensed under the MIT License and the RAIL AI License.
% Edited by Swapnil Sayan Saha, UCLA

f = waitbar(0,'Frames processed','Name','Running GREEN...');
addpath(genpath('tools'))

%% Parameters
LPF = 0.8; %low cutoff frequency (Hz) - 0.8 Hz in reference
HPF = 2.5; %high cutoff frequency (Hz) - both 6.0 Hz and 2.0 Hz used in reference


%% Load Video:
VidObj = VideoReader(VideoFile);
VidObj.CurrentTime = StartTime;
TotalFrames = ceil(VidObj.duration)*FS;
FramesToRead=ceil(Duration*VidObj.FrameRate); %video may be encoded at slightly different frame rate

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
    
    VidROI = RGBSkinSeg(VidROI); %skin segmentation
    RGB(FN,:) = sum(sum(VidROI));%if different size regions are used for different frames, the signals should be normalized by the region size, but not necessary for whole frame processing or constant region size
    waitbar((FN+(StartTime*FS))/TotalFrames,f,sprintf('Frames Processed: %d out of %d',FN+(StartTime*FS),TotalFrames));
end
delete(f)

%% Select BVP Source:
% Green channel
BVP = RGB(:,2);

%% Filter, Normalize
NyquistF = 1/2*FS;
[B,A] = butter(3,[LPF/NyquistF HPF/NyquistF]);%Butterworth 3rd order filter - originally specified in reference with a 4th order butterworth using filtfilt function
BVP_F = filtfilt(B,A,(double(BVP)-mean(BVP)));
BVP = BVP_F;
% Estimate Pulse Rate from periodogram
PR = prpsd(BVP,FS,40,240,0);

end%end function
