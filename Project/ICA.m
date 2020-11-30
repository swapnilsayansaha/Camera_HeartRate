function [BVP, PR] = ICA(VideoFile, FS, StartTime, Duration)
% ICA: The Independent Component Analysis (ICA) Method from: Poh, M. Z., McDuff, D. J., & Picard, R. W. (2010). Non-contact, automated cardiac pulse measurements using video imaging and blind source separation. Optics express, 18(10), 10762-10774. DOI: 10.1364/OE.18.010762
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
f = waitbar(0,'Frames processed','Name','Running ICA...');

addpath(genpath('tools'))

%% Parameters
LPF = 0.7; %low cutoff frequency (Hz) - 0.7 Hz in reference
HPF = 2.5; %high cutoff frequency (Hz) - 4.0 Hz in reference


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
RGB=zeros(FramesToRead,3);%initialize color signal
FN=0;
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

%% Detrend & ICA:
NyquistF = 1/2*FS;
RGBNorm=zeros(size(RGB));
Lambda=100;
for c=1:3
    RGBDetrend= spdetrend(RGB(:,c),Lambda); %M. P. Tarvainen, TBME, 2002
    RGBNorm(:,c) = (RGBDetrend - mean(RGBDetrend))/std(RGBDetrend); %normalize to zero mean and unit variance
end
[~,S] = ica(RGBNorm',3); %JADE ICA - J. F. Cardoso 1997, G. D. Clifford, MIT, 2004

%% Select BVP Source:
% Component with maximum normalized (by total power) power
MaxPx=zeros(1,3);
for c=1:3
    FF = fft(S(c,:));
    FF(1)=[];
    N=length(FF);
    Px = abs(FF(1:floor(N/2))).^2;
    Px=Px/sum(Px);
    MaxPx(c)=max(Px);
end

[~,MaxComp]=max(MaxPx(:));
BVP_I = S(MaxComp,:);

%% Filter, Normalize
%originally specified in reference with 5-point moving average, bandpass
%filter, and cubic-spine interpolation to 256Hz
[B,A] = butter(3,[LPF/NyquistF HPF/NyquistF]);%Butterworth 3rd order filter
BVP_F = filtfilt(B,A,double(BVP_I));

BVP=BVP_F;

% Estimate Pulse Rate from periodogram
PR = prpsd(BVP,FS,40,240,0);

end%end function
