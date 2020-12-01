% Optimal weighted ensemble rPPG 
% (c) 2020 Swapnil Sayan Saha, Suparno Pal, Rishav Guha and Paawan Garg
% parameters to enter:
% 1. video directory (DataDirectory)
% 2. rough frame rate of video (FS)
% output: average heart rate for the video file.

clear; close all
addpath(genpath([cd '/tools/']))
DataDirectory = '/Users/swapnilsayansaha/Downloads/22/'; %enter video directory here
z = dir(DataDirectory);
FS = 30; %enter rough video frame rate here
window = 2; %2 second window

for j = 1:length(z)
    if(contains(upper(z(j).name),upper('.mp4')) || contains(upper(z(j).name),upper('.avi')))
        fprintf("Processing video with name: %s\n",z(j).name);
        VideoFile =  strcat(DataDirectory, z(j).name);
        dur = floor(mmfileinfo(VideoFile).Duration);
        PR3 = 0;
        %CHROM
        fprintf('Processing video via CHROM algorithm...\n');
        [~, PR1] = CHROM(VideoFile, FS, 0, dur, window, true);
        fprintf('Heart rate from CHROM: %f\n',PR1);
        %
        %POS
        fprintf('Processing video via POS algorithm...\n');
        [~, PR2] = POS(VideoFile, FS, 0, dur, window, true, false);
        fprintf('Heart rate from POS: %f\n',PR2);
        %
        fprintf('Processing video via ICA algorithm...\n');
        %ICA:
        for i = 0:window:dur-window
            [~, PR] = ICA(VideoFile, FS, i, window);
            PR3 = [PR3 PR];
        end
        PR3(:,1) = [];
        PR3 = mean(PR3);
        fprintf('Heart rate from ICA: %f\n',PR3);
        %
        PR = 0.1478*PR1 + 0.7834*PR3 + 0.0687*PR2; %weights found via linear programming
        fprintf("Video: %s - Optimal Heart Rate: %f\n",z(j).name, PR);
    end
end
sound(sin(1:5000));
