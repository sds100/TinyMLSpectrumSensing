function [zigbeeWaveform] = myhelperBluetoothGenerateZigbeeWaveform(wlanInterferenceSource, wlanBBFilename)
%helperBluetoothGenerateWLANWaveform Generate the WLAN waveform to be added
%as an interference to the Bluetooth waveforms
%
%   WLANWAVEFORM =
%   helperBluetoothGenerateWLANWaveform(WLANINTERFERENCESOURCE) generates
%   WLAN waveform (802.11b) using the features of the WLAN Toolbox(TM).
%
%   WLANWAVEFORM =
%   helperBluetoothGenerateWLANWaveform(WLANINTERFERENCESOURCE,
%   WLANBBFILENAME) reads the WLAN waveform (802.11b) from the baseband
%   file, WLANBBFILENAME, and returns the waveform.
%
%   WLANWAVEFORM is the IQ samples of the WLAN waveform. It is a
%   complex-valued column vector.
%
%   WLANINTERFERENCESOURCE is the source for adding the WLAN interference
%   to Bluetooth waveforms. This value is specified as one of these:
%       'Generated'     - WLAN interference is generated using the features
%                         of the WLAN Toolbox(TM)
%       'BasebandFile'  - WLAN interference is specified from the .bb file
%
%   WLANBBFILENAME is the baseband filename from which the WLAN waveform is
%   read. This argument is valid only when the WLANINTERFERENCESOURCE
%   argument value is set to 'BasebandFile'. The default value is
%   'WLANNonHTDSSS.bb'.

%   Copyright 2020 The MathWorks, Inc.

% Validate the number of input arguments
narginchk(1, 2);

% Validate the WLAN interference source
wlanInterferenceSource = validatestring(wlanInterferenceSource, ...
    {'Zigbee','Wlan','Zigbee&Wlan','None'}, mfilename, 'wlanInterferenceSource', 1);

if nargin == 2
    % Validate the WLAN baseband filename
    validateattributes(wlanBBFilename, {'string', 'char'}, {},  mfilename, ...
        'wlanBBFilename', 2);
    if ischar(wlanBBFilename)
        validateattributes(wlanBBFilename, {'char'}, {'vector'}, mfilename, ...
            'wlanBBFilename', 2);
    else
        validateattributes(wlanBBFilename, {'string'}, {'scalar'}, mfilename, ...
            'wlanBBFilename', 2);
    end
else
    % Default baseband filename for generating the WLAN waveform
    wlanBBFilename = 'WLANNonHTDSSS.bb';
end

% Samples per symbol for Bluetooth waveforms
sps = 88;

% Maximum number of IQ samples for a single Bluetooth waveform = number of
% slots * slot duration * samples per symbol (5*625*88)
waveformMaxSize = 88000000;

% Initialize the zigbee waveform with zeros
zigbeeWaveform = complex(zeros(waveformMaxSize, 1));


% Zigbee waveform sample rate in samples per second
zigbeeSampleRate = 2e6;

switch wlanInterferenceSource
    case 'Zigbee'
        NumberpackageZ=180;
        spc = 44;                            % samples per chip
        msgLen = 8*2;                     % length in bits
        Zn1=[1,2,3];
        Zn2=4:6;
        Zn=[Zn1,Zn2];
        idletimelen=42240;
        zigbeeWaveform=[];
        for mz=1:NumberpackageZ
            Random=randi(length(Zn));
            message1 = randi([0 1], msgLen, 1);  % transmitted message
            Waveform = lrwpan.PHYGeneratorOQPSK(message1, spc, '2450 MHz');
            idle=complex(zeros([idletimelen*Random,1]));
            zig=[Waveform;idle];
            zigbeeWaveform=[zigbeeWaveform;zig];
        end
    case 'None'
        NumberpackageZ=300;
        spc = 44;                            % samples per chip
        msgLen = 8*2;                     % length in bits
        Zn1=[1,2,3];
        Zn2=4:6;
        Zn=[Zn1,Zn2];
        idletimelen=42240;
        zigbeeWaveform=[];
        for mz=1:NumberpackageZ
            Random=randi(length(Zn));
            message1 = randi([0 1], msgLen, 1);  % transmitted message
            Waveform = lrwpan.PHYGeneratorOQPSK(message1, spc, '2450 MHz');
            idle=complex(zeros([idletimelen*Random,1]));
            zig=[Waveform;idle];
            zigbeeWaveform=[zigbeeWaveform;zig];
        end
        
        % WLAN signal from .bb file
    case 'BasebandFile'
        % Create the System Object for reading the baseband file
        bbReader = comm.BasebandFileReader('Filename', wlanBBFilename);
        bbInfo = info(bbReader);
        
        % Configure the baseband file reader
        bbReader.SamplesPerFrame = bbInfo.NumSamplesInData;
        
        % Read the WLAN waveform from the baseband file
        wlanRead = bbReader();
        
        % Update the WLAN waveform for modeling the interference
        wlanWaveform = wlanRead(1:min(bbInfo.NumSamplesInData, waveformMaxSize));
        wlanSampleRate = bbReader.SampleRate;
end
end

% LocalWords:  WLAN wlan PSDU sps DSSS
