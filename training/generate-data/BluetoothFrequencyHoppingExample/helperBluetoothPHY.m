classdef helperBluetoothPHY < helperBluetoothPHYBase
%helperBluetoothPHY Create an object for modeling Bluetooth basic rate or
%enhanced data rate (BR/EDR) physical layer (PHY)
%   BTPHY = helperBluetoothPHY creates an object to model Bluetooth BR/EDR
%   PHY.
%
%   BTPHY = helperBluetoothPHY(Name, Value) creates a Bluetooth BR/EDR PHY
%   object with the specified property Name set to the specified Value. You
%   can specify additional name-value pair arguments in any order as
%   (Name1, Value1, ..., NameN, ValueN).
%
%   helperBluetoothPHY properties:
%
%   Mode                - PHY transmission mode
%   ChannelIndex        - Bluetooth BR/EDR channel index to transmit or
%                         receive the baseband data
%   TxPower             - Signal transmission power in dBm

%   Copyright 2020-2022 The MathWorks, Inc.

properties
    %TxPower Signal transmission power in dBm
    % This property specifies the Tx power as a scalar double value in the
    % range [-20, 20]. This property specifies the signal transmission
    % power in dBm. The default value is 20 dBm.
    TxPower = 20
end

properties (Constant, Hidden)
    % Maximum number of IQ samples for a single Bluetooth waveform = number
    % of slots * slot duration * samples per symbol (5*625*88)
    WaveformMaxSize = 275000
    
    % Maximum number of interfered signals to be stored
    InterferedSignalsCount = 10
    
    % Flow bit, bit for controlling the packets over the ACL logical
    % transport
    Flow = 1 % GO indication
    
    % Flow indicator, control the flow at the L2CAP level
    FlowIndicator = true
    
    % Number of samples per symbol
    SamplesPerSymbol = 88
    
    % Symbol rate
    SymbolRate = 1e6;
end

properties (Access = private)    
    %pTimer Timer for receiving a waveform in microseconds
    pTimer = 0
    
    %pInterferedSignals Buffer to store the interfered signals
    pInterferedSignals
    
    %pState State of the device
    % 0 - Idle
    % 1 - Tx
    % 2 - Rx
    pState = 0

    %pProcessing Flag to specify the waveform is processing in the PHY (to
    %model the waveform duration)
    pProcessing = false
    
    %pWhitenInit Initialization vector for whitening or dewhitening process
    %in PHY
    pWhitenInit = ones(7, 1)
    
    %pSignal Received Bluetooth BR/EDR waveform along with metadata
    pSignal
    
    %pEmptySignal Empty waveform along with metadata
    pEmptySignal
    
    %pFrequencyOffset System object for adding frequency offset to the
    %waveform
    pFrequencyOffset
    
    %pErrorRate System object for calculating the BER
    pErrorRate
end

properties (SetAccess = private, Hidden)
    %Duration Duration for receiving the waveform(in microseconds)
    Duration = 0
    
    %Status Status of the waveform reception (either 0 | 1 | 2 )
    % 0 - Not started
    % 1 - RxStart
    % 2 - RxEnd
    Status = 0
    
    %Decoded Flag to identify that the waveform is decoded
    Decoded = false
    
    %DecodedBasebandData Decoded baseband data from the received waveform
    DecodedBasebandData
    
    %UpdateReceptionSlots Flag to update the number of reception slots to
    %baseband layer. A true value indicates that a valid waveform is
    %received at PHY, and the corresponding slots need to be updated at
    %baseband.
    UpdateReceptionSlots = false
    
    %RecvPktType Received baseband packet type
    RecvPktType = blanks(0)
end

properties (Hidden)
    %WaveformConfig Configuration object for generating Bluetooth BR/EDR
    %waveforms
    WaveformConfig
end

% PHY layer statistics
properties (SetAccess = private)
    %TransmittedSignals Number of signals transmitted
    TransmittedSignals = 0
    
    %TransmissionTime Transmission time in microseconds
    TransmissionTime = 0
    
    %ReceivedSignals Number of signals received
    ReceivedSignals = 0
    
    %TotalCollisions Total number of collisions occurred
    % Multiple signals collide with the actual signal is considered as one
    % collision
    TotalCollisions = 0
    
    %BER Bit error rate
    BER = zeros(1, 0)
    
    %PacketErrors Number of packet errors
    PacketErrors = 0
end

properties (SetAccess = private, Dependent)
    %PER Packet error rate
    PER
end

methods
    % Constructor
    function obj = helperBluetoothPHY(varargin)
        % Number of bits in single octet
        octetLen = 8;
        
        % Apply name-value pairs
        obj@helperBluetoothPHYBase(varargin{:});
        
        % Initialize the Bluetooth signal
        obj.pSignal = struct('Waveform', complex(zeros(obj.WaveformMaxSize, 1)), ...
                                                'PacketType', blanks(0), ...
                                                'NumSamples', 0, ...
                                                'SampleRate', 0, ...
                                                'SamplesPerSymbol', 0, ...
                                                'SourceID', 0, ...
                                                'Bandwidth', 0, ...
                                                'NodePosition', [0 0 0], ...
                                                'CenterFrequency', 0, ...
                                                'StartTime', 0, ...
                                                'EndTime', 0, ...
                                                'Payload', zeros(obj.MaxPayloadSize * octetLen, 1), ...
                                                'PayloadLength', 0);
                       
        % Initialize the buffer for storing the interfered signals
        obj.pInterferedSignals = repmat(obj.pSignal, 1, obj.InterferedSignalsCount);
        % Initialize the empty signal
        obj.pEmptySignal = obj.pSignal;
        % Initialize the decoded baseband data
        obj.DecodedBasebandData = obj.pBasebandData;
        % Create Bluetooth BR/EDR waveform configuration object
        obj.WaveformConfig = bluetoothWaveformConfig;
        % Initialize the frequency offset system object
        obj.pFrequencyOffset = comm.PhaseFrequencyOffset('SampleRate', ...
            obj.SamplesPerSymbol*obj.SymbolRate);
        obj.pErrorRate = comm.ErrorRate;
    end
    
    % Set Tx power in dBm
    function set.TxPower(obj, value)
        validateattributes(value, {'numeric'}, {'scalar', ...
            'real', '>=', -20, '<=', 20}, mfilename, 'TxPower');
        obj.TxPower = value;
    end
    
    function set.PER(obj, value)
        obj.PER = value;
    end
    
    % Calculate the packet error rate as number packet errors divided by
    % the total number of valid Bluetooth signals received
    function value = get.PER(obj)
        value = obj.PacketErrors/obj.ReceivedSignals;
    end
    
    function updatePHY(obj, state, channelIndex, whitenInit, basebandData)
    %updatePHY Update the PHY with the request from the baseband layer
    %
    %   updatePHY(OBJ, STATE, CHANNELINDEX, WHITENINIT) updates the PHY
    %   with the request for the reception (STATE is 2) of the channel
    %   index, CHANNELINDEX, and initialization vector, WHITENINIT, for
    %   whitening or dewhitening process from the baseband layer.
    %
    %   updatePHY(OBJ, STATE, CHANNELINDEX, WHITENINIT, BASEBANDDATA)
    %   updates the PHY with request for the transmission (STATE is 1) of
    %   baseband packet, BASEBANDDATA along with the channel index,
    %   CHANNELINDEX, and initialization vector, WHITENINIT, for whitening
    %   or dewhitening process.
    %
    %   OBJ is an object of type helperBluetoothPHY.
    %
    %   STATE is the state of PHY (transmit - 1 or receive - 2).
    %
    %   CHANNELINDEX is the physical channel for waveform transmission or
    %   reception.
    %
    %   WHITENINIT is the initialization vector for whitening or
    %   dewhitening process in PHY.
    %
    %   BASEBANDDATA is the structure containing these fields:
    %       LTAddr          - Logical transport address of an active
    %                         destination peripheral for a packet in a
    %                         central-to-peripheral or
    %                         peripheral-to-central transmission slot. It
    %                         is a 3-bit value. The value of this field is
    %                         a scalar positive integer.
    %       PacketType      - Type of Bluetooth packet. The value of this
    %                         field is a scalar or a character vector
    %                         containing one of these: {'NULL', 'POLL',
    %                         'HV1', 'HV2', 'HV3', 'DM3', 'DM1', 'DH1',
    %                         'DM5', 'DH3', 'DH5'}.
    %       Payload         - Payload bits from the baseband layer. It is a
    %                         binary column vector.
    %       PayloadLength   - Number of payload bytes. It is a scalar
    %                         positive integer.
    %       LLID            - Logical link identifier
    %       SEQN            - 1-bit sequence number for transmission
    %       ARQN            - 1-bit acknowledgement for previous 
    %                         transmission
    %       IsValid         - Logical flag indicating whether the received
    %                         packet is valid or not based on packet header
    %                         error check (HEC) and cyclic redundancy check
    %                         (CRC).
    
        % Number of input arguments
        narginchk(4, 5);
        
        % Baseband packet is received for transmitting through PHY
        if nargin > 4
            obj.pBasebandData = basebandData;
        end
        
        % Update the request from the baseband
        obj.pState = state;
        obj.ChannelIndex = channelIndex;
        obj.pWhitenInit = whitenInit;
    end
    
    function [nextInvokeTime, btWaveform] = run(obj, elapsedTime, btSignal)
    %run Process the Bluetooth PHY
    %
    %   [NEXTINVOKETIME, BTWAVEFORM] = run(OBJ, ELAPSEDTIME) processes the
    %   Bluetooth PHY transmission.
    %
    %   [NEXTINVOKETIME, BTWAVEFORM] = run(OBJ, ELAPSEDTIME, BTSIGNAL)
    %   processes the Bluetooth PHY reception.
    %
    %   NEXTINVOKETIME returns the time after which the run function must
    %   be invoked again.
    %
    %   BTWAVEFORM is the waveform to be transmitted into physical channel.
    %
    %   OBJ is an object of type helperBluetoothPHY.
    %
    %   ELAPSEDTIME is the time elapsed in microseconds between two
    %   successive calls of this function.
    %
    %   BTSIGNAL is the structure containing these fields:
    %       Waveform         - IQ samples of the received waveform
    %       NumSamples       - Length of the waveform (number of IQ samples)
    %       SampleRate       - Sample rate of the received waveform
    %       PacketType       - Bluetooth packet type
    %       SourceID         - Source node ID
    %       Bandwidth        - Channel bandwidth in MHz
    %       NodePosition     - Source node position
    %       SamplesPerSymbol - Samples per symbol to generate the waveform
    %       CenterFrequency  - Bluetooth channel center frequency in MHz
    %       StartTime        - Simulation time in microseconds at the
    %                          waveform entry
    %       EndTime          - Simulation time in microseconds after the
    %                          waveform duration
    %       Payload          - Actual payload bits used for generating the
    %                          received waveform. It is a binary column
    %                          vector. This is used in calculating the BER
    %       PayloadLength    - Number of payload bytes. It is a scalar,
    %                          integer value
    
        % Initialize
        nextInvokeTime = -1;
        btWaveform = zeros(1, 0);
        obj.Decoded = false;
        obj.UpdateReceptionSlots = false;
        
        % Validate number of input arguments
        narginchk(2, 3);
        
        % No signal is received for PHY processing
        if nargin < 3
            btSignal = obj.pEmptySignal;
        end
        
        % PHY transmission
        if obj.pState == 1
            [nextInvokeTime, btWaveform] = phyTx(obj, elapsedTime);
        % PHY reception
        elseif obj.pState == 2
            nextInvokeTime = phyRx(obj, elapsedTime, btSignal);
        end
    end
end

methods (Access = private)
    function nextInvokeTime = phyRx(obj, elapsedTime, btSignal)
    %phyRx Bluetooth PHY reception
    %
    %   NEXTINVOKETIME = phyRx(OBJ, ELAPSEDTIME, BTSIGNAL) decodes the
    %   received Bluetooth BR/EDR waveform and updates the decoded baseband
    %   data.
    %
    %   NEXTINVOKETIME returns the time after which the run function must
    %   be invoked again.
    %
    %   OBJ is an object of type helperBluetoothPHY.
    %
    %   ELAPSEDTIME is the time elapsed in microseconds between two
    %   successive calls of this function.
    %
    %   BTSIGNAL is the structure containing these fields:
    %       Waveform         - IQ samples of the received waveform
    %       NumSamples       - Length of the waveform (number of IQ samples)
    %       SampleRate       - Sample rate of the received waveform
    %       PacketType       - Bluetooth packet type
    %       SourceID         - Source node ID
    %       Bandwidth        - Channel bandwidth in MHz
    %       NodePosition     - Source node position
    %       SamplesPerSymbol - Samples per symbol to generate the waveform
    %       CenterFrequency  - Bluetooth channel center frequency in MHz
    %       StartTime        - Simulation time in microseconds at the
    %                          waveform entry
    %       EndTime          - Simulation time in microseconds after the
    %                          waveform duration
    %       Payload          - Actual payload bits used for generating the
    %                          received waveform. It is a binary column
    %                          vector. This is used in calculating the BER
    %       PayloadLength    - Number of payload bytes. It is a scalar,
    %                          integer value
    
        % Initialize
        nextInvokeTime = -1;
        basebandData = obj.pBasebandData;
        % Number of bits in single octet
        octetLen = 8;
        
        % Waveform is received
        if obj.pProcessing
            % Update the reception timer
            obj.pTimer = obj.pTimer - elapsedTime;

            % Waveform duration is completed
            if obj.pTimer <= 0
                % Apply interference on the received signal
                btWaveform = applyInterference(obj);
                
                % Create and configure 'bluetoothPhyConfig' object
                rxConfig = bluetoothPhyConfig;
                rxConfig.Mode = obj.Mode;
                rxConfig.WhitenStatus = 'On';
                rxConfig.SamplesPerSymbol = obj.SamplesPerSymbol;
                rxConfig.WhitenInitialization = obj.pWhitenInit;
                rxConfig.DeviceAddress = obj.WaveformConfig.DeviceAddress;
 
                % Get the number of slots required to process the received
                % waveform
                numSlots = slotsRequired(obj, obj.pSignal.PacketType);
                if numSlots == 0
                    % Unsupported packet type is received, calculate number
                    % of slots based on the number of samples in the
                    % received waveform
                    numSlots = ceil(obj.pSignal.NumSamples/(625*obj.SamplesPerSymbol));
                end
                % Update the received waveform with the maximum
                % duration (numSlots * slotDuration * sps)
                btWaveformRx = complex(zeros(numSlots*625*obj.SamplesPerSymbol, 1));
                btWaveformRx(1:obj.pSignal.NumSamples) = btWaveform;
                
                % Demodulate and decode Bluetooth BR/EDR waveform
                [bits, decodedInfo, isValid] = bluetoothIdealReceiver(btWaveformRx, rxConfig);
                
                % Return the decoded baseband data
                basebandData.LTAddr = bit2int(decodedInfo(1).LogicalTransportAddress, 3, false);
                basebandData.PacketType = decodedInfo(1).PacketType;
                basebandData.PayloadLength = decodedInfo(1).PayloadLength;
                basebandData.Payload(1:decodedInfo(1).PayloadLength*octetLen) = bits(1:decodedInfo(1).PayloadLength*octetLen);
                basebandData.LLID = decodedInfo(1).LLID;
                % Sequence number
                basebandData.SEQN = decodedInfo(1).HeaderControlBits(3);
                % Acknowledgement flag
                basebandData.ARQN = decodedInfo(1).HeaderControlBits(2);
                % Valid flag for specifying the CRC or HEC status
                basebandData.IsValid = isValid(1);

                % Update the decoded baseband data
                obj.Decoded = true;
                obj.DecodedBasebandData = basebandData;
                
                % Update the packet errors
                if ~isValid
                    obj.PacketErrors = obj.PacketErrors + 1;
                end
                
                % Calculate the bit error rate (BER)
                txLength = obj.pSignal.PayloadLength * octetLen; % in bits
                rxLength = basebandData.PayloadLength * octetLen; % in bits
                minLength = min(txLength, rxLength);
                if minLength > 0
                    % Calculate the BER for the received payload
                    berVec = obj.pErrorRate(obj.pSignal.Payload(1:minLength), ...
                        basebandData.Payload(1:minLength));
                    if isempty(obj.BER)
                        obj.BER = berVec(1);
                    else
                        obj.BER = (obj.BER + berVec(1))/2;
                    end
                else
                    % Received the packet with empty payload even the
                    % transmitted packet has the payload
                    if txLength > 0
                        obj.PacketErrors = obj.PacketErrors + 1;
                    end
                end
                
                % Update the status to specify the decoding is completed
                obj.Status = 2;
                % Reset the reception flag and reception timer
                obj.pProcessing = false;
                obj.pTimer = 0;
                
                % Remove the old interfered signals
                removeInterferedSignals(obj, obj.pSignal.EndTime);
            else
                % Received an interfered signal
                if btSignal.NumSamples ~= 0
                    % Apply frequency offset
                    frequencyIndex = obj.ChannelIndex - 39; % To visualize as a two sided spectrum
                    release(obj.pFrequencyOffset);
                    obj.pFrequencyOffset.FrequencyOffset = -frequencyIndex*obj.SymbolRate;
                    btSignal.Waveform(1:btSignal.NumSamples) = ...
                        obj.pFrequencyOffset(btSignal.Waveform(1:btSignal.NumSamples));
                    
                    % Calculate the waveform duration in microseconds
                    waveformDuration = btSignal.NumSamples/obj.SamplesPerSymbol;
                    % Update the end time of the received Bluetooth signal
                    btSignal.EndTime = btSignal.StartTime + waveformDuration;
                    
                    % Get an index in the interfered signals buffer to
                    % store the received interfered signal
                    idx = getStoreIndex(obj);
                    
                    if idx ~= 0
                        % Store the interfered waveform along with updated
                        % metadata
                        obj.pInterferedSignals(idx) = btSignal;
                    else
                        disp('Interference buffer is full, ignoring the received waveform.');
                    end
                end
                obj.Status = 0;
            end
            % Update next event timer
            nextInvokeTime = obj.pTimer;
        % No waveform is available for reception
        else
            if btSignal.NumSamples ~= 0    
                % Apply frequency offset
                frequencyIndex = obj.ChannelIndex - 39; % To visualize as a two sided spectrum
                release(obj.pFrequencyOffset);
                obj.pFrequencyOffset.FrequencyOffset = -frequencyIndex*obj.SymbolRate;
                btSignal.Waveform(1:btSignal.NumSamples) = ...
                    obj.pFrequencyOffset(btSignal.Waveform(1:btSignal.NumSamples));
                % Store the received waveform with metadata
                obj.pSignal = btSignal;
                obj.Status = 1;
                obj.pProcessing = true;
                % Update the number of reception slots to baseband layer
                obj.UpdateReceptionSlots = true;
                % Update the received baseband packet type
                obj.RecvPktType = btSignal.PacketType;
                
                % Calculate the waveform duration in microseconds
                obj.Duration = btSignal.NumSamples/obj.SamplesPerSymbol;
                % Set the reception timer to waveform duration time
                obj.pTimer = obj.Duration;
                % Update the end time of the received Bluetooth BR/EDR
                % signal
                obj.pSignal.EndTime = obj.pSignal.StartTime + obj.Duration;
                nextInvokeTime = obj.pTimer;
                obj.ReceivedSignals = obj.ReceivedSignals + 1;
            else
                obj.Status = 0;
            end
        end
    end
    
    function [nextInvokeTime, btWaveform] = phyTx(obj, elapsedTime)
    %phyTx Bluetooth PHY transmission
    %
    %   [NEXTINVOKETIME, BTWAVEFORM] = phyTx(OBJ, ELAPSEDTIME) generates
    %   and returns the Bluetooth BR/EDR waveform along with the next
    %   invoke time in microseconds.
    %
    %   NEXTINVOKETIME returns the time after which the run function must
    %   be invoked again.
    %
    %   BTWAVEFORM returns the IQ samples of the generated waveform.
    %
    %   OBJ is instance of an object of type helperBluetoothPHY.
    %
    %   ELAPSEDTIME is the time elapsed in microseconds between two
    %   successive calls of this function.
    
        % Initialize
        nextInvokeTime = -1;
        btWaveform = zeros(1, 0);
        % Number of bits in single octet
        octetLen = 8;
        
        % Waveform is transmitting
        if obj.pProcessing
            % Update the transmission timer
            obj.pTimer = obj.pTimer - elapsedTime;
            % Waveform duration is completed
            if obj.pTimer <= 0
                obj.Status = 2;
                obj.pProcessing = false;
                obj.pTimer = 0;
                obj.TransmittedSignals = obj.TransmittedSignals + 1;
                obj.TransmissionTime = obj.TransmissionTime + obj.Duration;
                obj.pBasebandData.PacketType = blanks(0);
                obj.Duration = 0;
            else
                obj.Status = 0;
            end
            % Update next event timer
            nextInvokeTime = obj.pTimer;
        % No active transmission
        else
            if ~isempty(obj.pBasebandData.PacketType)
                % Configure the object for generating the Bluetooth
                % waveform
                obj.WaveformConfig.Mode = obj.Mode;
                obj.WaveformConfig.WhitenStatus = 'On';
                obj.WaveformConfig.PacketType = obj.pBasebandData.PacketType;
                obj.WaveformConfig.PayloadLength = obj.pBasebandData.PayloadLength;
                obj.WaveformConfig.LogicalTransportAddress = int2bit(obj.pBasebandData.LTAddr, 3, false);
                % Configure header control bits (Flow, ARQN, SEQN)
                obj.WaveformConfig.HeaderControlBits = [obj.Flow; obj.pBasebandData.ARQN; obj.pBasebandData.SEQN];
                obj.WaveformConfig.FlowIndicator = obj.FlowIndicator;
                obj.WaveformConfig.LLID = obj.pBasebandData.LLID;
                obj.WaveformConfig.WhitenInitialization = obj.pWhitenInit;
                obj.WaveformConfig.SamplesPerSymbol = obj.SamplesPerSymbol;
                
                % Generate waveform for the given baseband data
                txWaveform = bluetoothWaveformGenerator(...
                    obj.pBasebandData.Payload(1:obj.pBasebandData.PayloadLength*octetLen), obj.WaveformConfig);
                
                % Apply frequency offset
                frequencyIndex = obj.ChannelIndex - 39; % To visualize as a two sided spectrum
                release(obj.pFrequencyOffset);
                obj.pFrequencyOffset.FrequencyOffset = frequencyIndex*obj.SymbolRate;
                hoppedWaveform = obj.pFrequencyOffset(txWaveform);
                
                % Apply Tx power
                btWaveform = applyTxPower(obj, hoppedWaveform);
                
                obj.Status = 1;
                obj.pProcessing = true;
                % Calculate the duration to transmit the generated waveform
                obj.Duration = bluetoothPacketDuration(obj.Mode, ...
                    obj.pBasebandData.PacketType, obj.pBasebandData.PayloadLength);
                obj.pTimer = obj.Duration;
                nextInvokeTime = obj.pTimer;
            end
        end
    end
end

methods (Access = private)
    function btWaveform = applyTxPower(obj, btWaveform)
        % Apply Tx power on the given waveform
        scale = 10.^((-30 + obj.TxPower)/20);
        btWaveform = btWaveform * scale;
    end
    
    function storeIdx = getStoreIndex(obj)
    %getStoreIndex Get an index to store the interfered waveform in the
    %buffer
    
        storeIdx = 0;
        for idx = 1:obj.InterferedSignalsCount
            % Get an index having an empty signal
            if (obj.pInterferedSignals(idx).NumSamples == 0)
                storeIdx = idx;
                break;
            end
        end
    end
    
    function interferedIdxs = getInterferedIdxs(obj, startTime, endTime)
    %getInterferedIdxs Get indexes of the interfered signals from stored
    %buffer using interference start and end times
    
        idxs = zeros(1, obj.InterferedSignalsCount);
        idxCount = 1;
        for idx = 1:obj.InterferedSignalsCount
            % Fetch valid signals
            if obj.pInterferedSignals(idx).NumSamples > 0
                % Fetch index of the interfered signals based on the start
                % time and end time
                if (startTime <= obj.pInterferedSignals(idx).EndTime) && ...
                        endTime >= (obj.pInterferedSignals(idx).StartTime)
                    idxs(idxCount) = idx;
                    idxCount = idxCount + 1;
                end
            end
        end
        % Return the interfered indexes
        interferedIdxs = idxs(1:idxCount-1);
    end
    
    function removeInterferedSignals(obj, startTime)
    %removeInterferedSignals Remove the interfered signals from the stored
    %buffer whose end time is less than the start time of the newly
    %processing waveform
        
        for idx = 1:obj.InterferedSignalsCount
            % Remove the waveform
            if obj.pInterferedSignals(idx).NumSamples ~= 0
                if obj.pInterferedSignals(idx).EndTime < startTime
                    obj.pInterferedSignals(idx).NumSamples = 0;
                end
            end
        end
    end
    
    function btWaveform = applyInterference(obj)
    %applyInterference Apply interference on the received signal and
    %return the interfered waveform
    
        % Received waveform
        btWaveform = obj.pSignal.Waveform(1:obj.pSignal.NumSamples);
        
        % Get indices of the interfered Bluetooth signals
        idxs = getInterferedIdxs(obj, obj.pSignal.StartTime, obj.pSignal.EndTime);

        % Add interference, if present
        if numel(idxs) > 0
            obj.TotalCollisions = obj.TotalCollisions + 1;
            for idx = 1:numel(idxs)
                % Fetch interfered signal
                interferedSignal = obj.pInterferedSignals(idxs(idx));
                
                % Interfered waveform
                interferedWaveform = interferedSignal.Waveform(1:interferedSignal.NumSamples);
                
                % Calculate the interference start time and end time
                interferedStartTime = max(obj.pSignal.StartTime, interferedSignal.StartTime);
                interferedEndTime = min(obj.pSignal.EndTime, interferedSignal.EndTime);
                
                % Calculate the interfered duration in microseconds
                interferenceDuration = interferedEndTime - interferedStartTime;
                
                % Calculate sample duration in microseconds
                sampleDuration = (1/interferedSignal.SampleRate)*obj.SymbolRate;
                
                % Get the start index of the interfered samples
                if obj.pSignal.StartTime > interferedSignal.StartTime
                    sampleStartIdx = round((1/sampleDuration)*(abs(obj.pSignal.StartTime - interferedSignal.StartTime))+1);
                    btWaveformStartIdx = 1;
                else
                    sampleStartIdx = 1;
                    btWaveformStartIdx = round((1/sampleDuration)*(abs(obj.pSignal.StartTime - interferedSignal.StartTime))+1);
                end
                % Get end start index of the interfered samples
                sampleEndIdx = round(sampleStartIdx + (1/sampleDuration)*(interferenceDuration))-1;
                btWaveformEndIdx = round(btWaveformStartIdx + (1/sampleDuration)*(interferenceDuration))-1;
                % Corrupt the waveform with interfered Bluetooth samples
                btWaveform(btWaveformStartIdx:btWaveformEndIdx) = btWaveform(btWaveformStartIdx:btWaveformEndIdx) + ...
                    interferedWaveform(sampleStartIdx:sampleEndIdx);
            end
        end
    end
end

methods (Hidden)
    function numSlots = slotsRequired(~, packetType)
        % Get the required slots for processing the received packet
        switch upper(packetType)
            % One-slot packets
            case {'DM1', 'DH1', 'POLL', 'NULL', 'HV1', 'HV2', 'HV3', ...
                    'ID', 'FHS', 'DV', 'EV3', '2-EV3', '3-EV3', 'AUX1', ...
                    '2-DH1', '3-DH1'}
                numSlots = 1;
            % Three-slot packets
            case {'DH3', 'DM3', 'EV4', 'EV5', '2-EV5', '3-EV5', '2-DH3', ...
                    '3-DH3'}
                numSlots = 3;
            % Five-slot packets
            case {'DH5', 'DM5', '2-DH5', '3-DH5'}
                numSlots = 5;
            otherwise
                numSlots = 0;
        end
    end
end
end

% LocalWords:  EDR PHY ACL HEC CRC sps ARQN SEQN
