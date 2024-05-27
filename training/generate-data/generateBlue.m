function [BtWaveform,hopIndex]=generateBlue(simulationTime)

packetType='DM1';
phyMode='BR';
payloadLength=10;

% Bluetooth frequency hopping
frequencyHop = bluetoothFrequencyHop;
frequencyHop.SequenceType = 'Connection Adaptive';
num=randi([10,90]);
address=['9E8B' num2str(num)];
frequencyHop.DeviceAddress = address;

% Configure Bluetooth PHY transmission
phyTx = helperBluetoothPHY;
phyTx.Mode = phyMode;

% Specify as one of 'Zigbee' | 'Wlan' | 'Zigbee&Wlan'|'None'
% wlanInterferenceSource = 'BasebandFile';
% wlanBBFilename = 'WLANNonHTDSSS.bb'; % Default baseband file
wlanInterferenceSource = 'None';
wlanBBFilename = 'WLANNonHTDSSS.bb'; % Default baseband file


% Configure wireless channel
channel = myhelperBluetoothChannel;
channel.EbNo = 40; % Ratio of energy per bit (Eb) to the spectral noise density (No) in dB
channel.SIRW = -20; % Signal to wlaninterference ratio in dB
channel.SIRZ = -10; % Signal to zigbeeinterference ratio in dB


slotTime = 625; % Bluetooth slot duration in microseconds

% Simulation time in terms of slots
numSlots = floor(simulationTime/slotTime);

% Slot duration, including transmission and reception
slotValue = phyTx.slotsRequired(packetType)*2;

% Number of Central transmission slots
numCentralTxSlots = floor(numSlots/slotValue);

% Total number of Bluetooth physical channels
numBluetoothChannels = 79;

% errorsBasic and errorsAdaptive store relevant bit and packet error
% information per channel. Each row stores the channel index, bit errors,
% packet errors, total bits, and BER per channel. errorsBasic and
% errorsAdaptive arrays store these values for basic frequency hopping
% and AFH, respectively.
[errorsBasic, errorsAdaptive] = deal(zeros(numBluetoothChannels,5));

% Initialize first column with channel numbers
[errorsBasic(:,1), errorsAdaptive(:,1)] = deal(0:78);

% Initialize variables for calculating PER and BER
[berBasic, berAdaptive, bitErrors] = deal(0);
badChannels = zeros(1,0);
totalTransmittedPackets = numCentralTxSlots;

% Number of bits per octet
octetLength = 8;

% Sample rate and input clock used in PHY processing
samplePerSymbol = 88;
symbolRate = 1e6;
sampleRate = symbolRate*samplePerSymbol;
inputClock = 0;

% Store hop index
hopIndex = zeros(1, numCentralTxSlots);

% Index to hop index vector
hopIdx = 1;

% Baseband packet structure
basebandData = struct(...
    'LTAddr',1,             ... % Logical transport address
    'PacketType',packetType,... % Packet type
    'Payload',zeros(1,phyTx.MaxPayloadSize), ... % Payload
    'PayloadLength',0,  ... % Payload length
    'LLID',[0; 0],      ... % Logical link identifier
    'SEQN',0,           ... % Sequence number
    'ARQN',1,           ... % Acknowledgment flag
    'IsValid',false);   ... % Flag to identify the status of cyclic redundancy check (CRC) and header error control (HEC)
    
% Bluetooth signal structure
bluetoothSignal = struct(...
    'PacketType',packetType, ... % Packet type
    'Waveform',[],           ... % Waveform
    'NumSamples',[],         ... % Number of samples
    'SampleRate',sampleRate, ... % Sample rate
    'SamplesPerSymbol',samplePerSymbol,      ... % Samples per symbol in Hz
    'Payload',zeros(1,phyTx.MaxPayloadSize), ... % Payload
    'PayloadLength',0, ... % Payload length in bytes
    'SourceID',0,      ... % Source identifier
    'Bandwidth',1,     ... % Bandwidth
    'NodePosition',[0 0 0], ... % Node position
    'CenterFrequency',centerFrequency(phyTx), ... % Center frequency
    'StartTime',0, ... % Waveform start time in microseconds
    'EndTime',0,   ... % Waveform end time in microseconds
    'Duration',0); ... % Waveform duration in microseconds
    
% Clock ticks(one slot is 2 clock ticks)
clockTicks = slotValue*2;

% Set the default random number generator ('twister' type with seed value 0).
% The seed value controls the pattern of random number generation. For high
% fidelity simulation results, change the seed value and average the
% results over multiple simulations.
sprev = rng('shuffle');
k=1;
BtWaveform=[];
data=[];
for slotIdx = 0:slotValue:numSlots-slotValue
    % Update clock
    inputClock = inputClock + clockTicks;
    
    % Frequency hopping
    [channelIndex,~] = nextHop(frequencyHop,inputClock);
    %             channelIndex=channelIndex0(k);
    % PHY transmission
    stateTx = 1; % Transmission state
    TxBits = randi([0 1],payloadLength*octetLength,1);
    basebandData.Payload = TxBits;
    basebandData.PayloadLength = payloadLength;
    
    % Generate whiten initialization vector from clock
    clockBinary = int2bit(inputClock,28,false)';
    whitenInitialization = [clockBinary(2:7)'; 1];
    
    % Update the PHY with request from the baseband layer
    updatePHY(phyTx,stateTx,channelIndex,whitenInitialization,basebandData);
    
    % Initialize and pass elapsed time as zero
    elapsedTime = 0;
    [nextTxTime,btWaveform] = run(phyTx,elapsedTime); % Run PHY transmission
    run(phyTx, nextTxTime); % Update next invoked time
    
    % Channel
    bluetoothSignal.Waveform = btWaveform;
    bluetoothSignal.NumSamples = numel(btWaveform);
    bluetoothSignal.CenterFrequency = centerFrequency(phyTx);
    channel.ChannelIndex = channelIndex;
    bluetoothSignal = run(channel,bluetoothSignal,phyMode,k);
    distortedWaveform = bluetoothSignal.Waveform;
    BtWaveform=[BtWaveform;distortedWaveform];
    
    hopIndex(hopIdx) = channelIndex;
    hopIdx = hopIdx + 1;
    k=k+1;
end
end