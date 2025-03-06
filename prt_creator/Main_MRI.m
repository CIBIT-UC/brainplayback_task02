clear, clc;

outputFolder = fullfile(pwd,'prt');

%% PRT Parameters
PRTParameters = struct();

PRTParameters.FileVersion = 2;
PRTParameters.Resolution = 'Volumes';
PRTParameters.ExperimentName = 'BrainPlayback_Task02';
PRTParameters.BackgroundColor = [0 0 0];
PRTParameters.TextColor = [255 255 255];
PRTParameters.TimeCourseColor = [1 1 1];
PRTParameters.TimeCourseThick = 3;
PRTParameters.ReferenceFuncColor = [0 0 80];
PRTParameters.ReferenceFuncThick = 2;

%% PRT Conditions
condNames = {'Discard','Noise','Noise_InterSong', ...
              'Tension','Peacefulness','JoyfulActivation','Tenderness','Power','Sadness','Wonder','Nostalgia','Transcendence'};

blockDuration = [ 12 18 6 24 24 24 24 24 24 24 24 24 ]; %in volumes (think TR = 1000ms)

blockColor = [0 115 190 ; 216 83 25 ; 236 177 32 ; ...
                            110 115 190 ; 150 83 25 ; 190 177 32 ; ...
                            150 115 190 ; 100 83 25 ; 120 177 32 ; ...
                            120 160 120; 120 220 120; 50 50 50];

PRTParameters.nCond = length(condNames);

PRTConditions = struct();

for c = 1:PRTParameters.nCond
    
    PRTConditions.(condNames{c}).Color = blockColor(c,:);
    PRTConditions.(condNames{c}).BlockDuration = blockDuration(c);
    PRTConditions.(condNames{c}).Intervals = [];
    PRTConditions.(condNames{c}).NumBlocks = 0;
    
end

%% Run MRI D12 R2
SEQ = [ 1 4 3 4 2 5 3 5 2 6 3 6 2 7 3 7 2 8 3 8 2 9 3 9 2 10 3 10 2 11 3 11 2 12 3 12 2 ];

[ PRTConditions_R2 ] = buildIntervals( SEQ , PRTConditions );

generatePRT( PRTParameters , PRTConditions_R2 , 'BP_Task2_R2' , outputFolder );
