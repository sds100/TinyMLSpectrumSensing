function wlanWaveform = helperBluetoothGenerateWLANWaveform(wlanInterferenceSource, wlanBBFilename)
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
    {'Generated', 'BasebandFile'}, mfilename, 'wlanInterferenceSource', 1);

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
waveformMaxSize = 275000;

% Initialize the WLAN waveform with zeros
wlanWaveform = complex(zeros(waveformMaxSize, 1));

% Bluetooth waveform sample rate = sps*1e6 in samples per second
bluetoothSampleRate = sps*1e6;

% WLAN waveform sample rate in samples per second
wlanSampleRate = 22e6;

switch wlanInterferenceSource
    % WLAN waveform is generated using the features of the WLAN Toolbox(TM)
    case 'Generated'
        % Add your custom signal generation code here
        
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
% Sampling factor to match the sample rate of Bluetooth and WLAN waveforms
samplingFactor = floor(bluetoothSampleRate/wlanSampleRate);

% Upsample the WLAN waveform to add with Bluetooth waveform
filterCoeffs = rcosdesign(0.35, 4, samplingFactor);
ups = upsample(wlanWaveform, samplingFactor);
wlanWaveform = filter(filterCoeffs, 1, ups);
end

% LocalWords:  WLAN wlan PSDU sps DSSS
