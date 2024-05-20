function [wlanWaveform] = myhelperBluetoothGenerateWLANWaveform(wlanInterferenceSource, wlanBBFilename)
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

% Initialize the WLAN waveform with zeros
wlanWaveform = complex(zeros(waveformMaxSize, 1));


% WLAN waveform sample rate in samples per second
wlanSampleRate = 22e6;

bluetoothSampleRate=88e6;

switch wlanInterferenceSource
    % WLAN waveform is generated using the features of the WLAN Toolbox(TM)
    case 'Wlan'
        % Add your custom signal generation code here
        channelcoding='BCC';
        Modulation=5;
        Guard='Long';
        Bandwith='CBW20';
        %%wlanHTConfig--wifi4
        Config=wlanHTConfig;
        Config.ChannelBandwidth=Bandwith;
        Config.MCS=Modulation;
        Config.GuardInterval=Guard;
        Config.ChannelCoding=channelcoding;
        
        %%%%生成波形
        Numberpackage=400;
        Wn1=[1,2,3];
        Wn2=15:63;
        Wn=[Wn1,Wn2];
        interTime=15e-6;
        wlanWaveform=[];
        for i=1:Numberpackage
            ran=randi(length(Wn));
            intervaltime=interTime*Wn(ran);
            numbits=Config.PSDULength*8*Numberpackage;
            data=randi([0,1],numbits,1);
            wlan=wlanWaveformGenerator(data,Config,"IdleTime",intervaltime,NumPackets=1);
            wlanWaveform=[wlanWaveform;wlan];
        end
       
    case 'None'
        % Add your custom signal generation code here
        channelcoding='BCC';
        Modulation=5;
        Guard='Long';
        Bandwith='CBW20';
        %%wlanHTConfig--wifi4
        Config=wlanHTConfig;
        Config.ChannelBandwidth=Bandwith;
        Config.MCS=Modulation;
        Config.GuardInterval=Guard;
        Config.ChannelCoding=channelcoding;
        
        %%%%生成波形
        Numberpackage=800;
        Wn1=[1,2,3];
        Wn2=15:63;
        Wn=[Wn1,Wn2];
        interTime=15e-6;
        wlanWaveform=[];
        for i=1:Numberpackage
            ran=randi(length(Wn));
            intervaltime=interTime*Wn(ran);
            numbits=Config.PSDULength*8*Numberpackage;
            data=randi([0,1],numbits,1);
            wlan=wlanWaveformGenerator(data,Config,"IdleTime",intervaltime,NumPackets=1);
            wlanWaveform=[wlanWaveform;wlan];
        end
        %%%%Zigbee生成波形
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

if wlanWaveform(1)~=0
    % Sampling factor to match the sample rate of Bluetooth and WLAN
    % waveform 进行上采样
    samplingFactor = floor(bluetoothSampleRate/wlanSampleRate);
    
    % Upsample the WLAN waveform to add with Bluetooth waveform
    filterCoeffs = rcosdesign(0.35, 4, samplingFactor);
    ups = upsample(wlanWaveform, samplingFactor);
    wlanWaveform = filter(filterCoeffs, 1, ups);
end
% if any(real(zigbeeWaveform))
%     % Sampling factor to match the sample rate of Bluetooth and zigbee
%     % waveform 进行上采样
%     samplingFactor = floor(bluetoothSampleRate/zigbeeSampleRate);
%     
%     % Upsample the zigbee waveform to add with Bluetooth waveform
%     filterCoeffs = rcosdesign(0.35, 4, samplingFactor);
%     ups = upsample(zigbeeWaveform, samplingFactor);
%     zigbeeWaveform = filter(filterCoeffs, 1, ups);
% end
end

% LocalWords:  WLAN wlan PSDU sps DSSS
