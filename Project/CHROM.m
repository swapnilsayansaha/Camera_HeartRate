function [BVP, PR] = CHROM(VideoFile, FS, StartTime, Duration, WinSec, SkinSegmentTF)
% CHROM: The Chrominance Method from: De Haan, G., & Jeanne, V. (2013). Robust pulse rate from chrominance-based rPPG. IEEE Transactions on Biomedical Engineering, 60(10), 2878-2886. DOI: 10.1109/TBME.2013.2266196
%
%   Inputs:
%       VideoFile               = Video file path.
%       FS                      = Video framerate (fps).
%       StartTime               = Timepoint at which to start process (default = 0 seconds).
%       Duration                = Duration of the time window to process (default = 60 seconds).
%       WinSec                  = Window length in seconds.
%       SkinSegmentTF           = Use skin segment (true or false)

%   Outputs:
%       BVP                     = Processed Blood Volume Pulse (BVP).
%       PR                      = Estimated Pulse Rate (PR) from processed BVP timeseries using peak in periodogram.

% Daniel McDuff, Ethan Blackford, January 2019
% Copyright (c)
% Licensed under the MIT License and the RAIL AI License.
% Edited by Suparno Pal, UCLA

f = waitbar(0,'Frames processed','Name','Running CHROM...');
%% Parameters

LPF = 0.7; %low cutoff frequency (Hz) - specified as 40 bpm (~0.667 Hz) in reference
HPF = 2.5; %high cutoff frequency (Hz) - specified as 240 bpm (~4.0 Hz) in reference


%% Add Backup Functions
if(~license('test', 'image_toolbox') && SkinSegmentTF)
    addpath([cd '\optional\rgb2ycbcr.m']);%GNU GPL rgb2ycbcr.m function
end

%% Load Video:
VidObj = VideoReader(VideoFile);
VidObj.CurrentTime = StartTime;

FramesToRead=floor(Duration*VidObj.FrameRate); %video may be encoded at slightly different frame rate

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
    
    if(SkinSegmentTF)%skin segmentation
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
end%endwhile video
delete(f);
if(~license('test', 'image_toolbox')&&SkinSegmentTF)%remove path if added
    rmpath([cd '\optional\']);
end

%% CHROM:
NyquistF = 1/2*FS;
[B,A] = butter(3,[LPF/NyquistF HPF/NyquistF]);%Butterworth 3rd order filter - originally specified as an a FIR band-pass filter with cutoff frequencies 40-240 BPM

%Window parameters - overlap, add with 50% overlap
WinL = ceil(WinSec*FS);
if(mod(WinL,2))%force even window size for overlap, add of hanning windowed signals
    WinL=WinL+1;
end
NWin = floor((FN-WinL/2)/(WinL/2));
S = zeros(NWin,1);
WinS = 1;%Window Start Index
WinM = WinS+WinL/2;%Window Middle Index
WinE = WinS+WinL-1;%Window End Index

for i = 1:NWin
    TWin = T(WinS:WinE,:);
    
    RGBBase = mean(RGB(WinS:WinE,:));
    RGBNorm = bsxfun(@times,RGB(WinS:WinE,:),1./RGBBase)-1;
    
    % CHROM
    Xs = squeeze(3*RGBNorm(:,1)-2*RGBNorm(:,2));%3Rn-2Gn
    Ys = squeeze(1.5*RGBNorm(:,1)+RGBNorm(:,2)-1.5*RGBNorm(:,3));%1.5Rn+Gn-1.5Bn
    
    Xf = filtfilt(B,A,double(Xs));
    Yf = filtfilt(B,A,double(Ys));
    
    Alpha = std(Xf)./std(Yf);
    
    SWin = Xf - Alpha.*Yf;
    
    SWin = hann(WinL).*SWin;
    %overlap, add Hanning windowed signals
    if(i==1)
        S = SWin;
        TX = TWin;
    else
        S(WinS:WinM-1) = S(WinS:WinM-1)+SWin(1:WinL/2);%1st half overlap
        S(WinM:WinE) = SWin(WinL/2+1:end);%2nd half
        TX(WinM:WinE) = TWin(WinL/2+1:end);
    end
    
    WinS = WinM;
    WinM = WinS+WinL/2;
    WinE = WinS+WinL-1;
end

BVP=S;

% Estimate Pulse Rate from periodogram
PR = prpsd(BVP,FS,40,240,false);

%% Remove Backup Functions
if(~license('test', 'image_toolbox') && SkinSegmentTF)%remove path if added
    rmpath([cd '\optional\']);
end

end%end function
