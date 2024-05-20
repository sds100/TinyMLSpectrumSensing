classdef myhelperBluetoothChannel < handle
    %helperBluetoothChannel Create an object for Bluetooth basic rate or
    %enhanced data rate (BR/EDR) channel model
    %   BTCHANNEL = helperBluetoothChannel creates an object for Bluetooth
    %   BR/EDR channel model. This class models the range propagation loss and
    %   free-space path loss and adds WLAN interference to the resultant
    %   Bluetooth waveform after the path loss.
    %
    %   BTCHANNEL = helperBluetoothChannel(Name, Value) creates a Bluetooth
    %   BR/EDR channel model object with the specified property Name set to the
    %   specified Value. You can specify additional name-value pair arguments
    %   in any order as (Name1, Value1, ..., NameN, ValueN).
    %
    %   helperBluetoothChannel properties:
    %
    %   ChannelIndex            - Bluetooth channel index for receiving
    %                             baseband data
    %   FSPL                    - Enable or disable free-space path loss
    %   NodePosition            - Coordinates (x, y, z) of the node
    %   EbNo                    - Ratio of energy per bit to noise power
    %                             spectral density in dB
    %   SIR                     - Signal to interference ratio in dB
    
    %   Copyright 2020 The MathWorks, Inc.
    
    properties
        %ChannelIndex Bluetooth channel index for receiving baseband data
        %   This property specifies channel index as a scalar integer in the
        %   range [0, 78]. The default value is 0.
        ChannelIndex = 0
        
        %FSPL Enable or disable free-space path loss
        %   This property specifies free-space path loss model as a scalar
        %   logical value. It defines whether to enable or disable the
        %   free-space path loss. As the distance between the Bluetooth devices
        %   increases, the signal strength reduces. The default value is true
        %   (enabled).
        FSPL (1, 1) logical = true
        
        %NodePosition Coordinates (x, y, z) of the node
        %   This property specifies the position of node as a row-vector of
        %   integer values representing the three-dimensional coordinates. This
        %   property does not consider the 'z' value. The default value is [0 0
        %   0].
        NodePosition = [0 0 0]
        
        %EbNo Ratio of energy per bit to noise power spectral density in dB
        %   This property specifies the Eb/No of node as a scalar double value.
        %   This property measures the signal-to-noise ratio (SNR). The default
        %   value is 10 dB.
        EbNo = 10
        
        %SIR Signal to interference ratio in dB
        %   This property specifies the signal-to-interference ratio as a
        %   scalar double value. This property is applicable only when the WLAN
        %   waveform is interfering with the Bluetooth waveform. This property
        %   is computed using the transmitter power and the path loss model.
        %   The default value is 0 dB.
        SIRW = 0
        SIRZ = 0
    end
    
    properties (Constant, Hidden)
        % Path loss exponent (to model the free-space path loss)
        PLExponent = 2
        
        % Maximum number of IQ samples for a single Bluetooth waveform = number
        % of slots * slot duration * samples per symbol (5*625*88)
        WaveformMaxSize = 8800000
    end
    
    properties (Access = private)
        %pWLANWaveform WLAN waveform to be interfered with Bluetooth signals
        pWLANWaveform
        pZWaveform
        
        %pHasInterference Flag to specify whether the WLAN interference is
        %added or not
        pHasInterference = false
        pHasZInterference = false
    end
    
    methods
        % Constructor
        function obj = myhelperBluetoothChannel(varargin)
            % Set name-value pairs
            for idx = 1:2:nargin
                obj.(varargin{idx}) = varargin{idx+1};
            end
            
            % Initialize the WLAN waveform with maximum size of the Bluetooth
            % waveform
            obj.pWLANWaveform = complex(zeros(obj.WaveformMaxSize, 1));
            obj.pZWaveform = complex(zeros(obj.WaveformMaxSize, 1));
        end
        
        % Set channel index
        function set.ChannelIndex(obj, value)
            validateattributes(value, {'numeric'}, {'scalar', ...
                'integer', '>=', 0, '<=', 78}, mfilename, 'ChannelIndex');
            obj.ChannelIndex = value;
        end
        
        % Set node position
        function set.NodePosition(obj, value)
            validateattributes(value, {'numeric'}, {'row', 'real', ...
                'numel', 3}, mfilename, 'NodePosition');
            obj.NodePosition = value;
        end
        
        % Set Eb/No in dB
        function set.EbNo(obj, value)
            validateattributes(value, {'numeric'}, {'scalar', 'real'}, ...
                mfilename, 'EbNo');
            obj.EbNo = value;
        end
        
        % Set signal to interference ratio
        function set.SIRW(obj, value)
            validateattributes(value, {'numeric'}, {'scalar', 'real'}, ...
                mfilename, 'SIR');
            obj.SIRW = value;
        end
        
        % Set signal to interference ratio
        function set.SIRZ(obj, value)
            validateattributes(value, {'numeric'}, {'scalar', 'real'}, ...
                mfilename, 'SIR');
            obj.SIRZ = value;
        end
        
        function addWLANWaveform(obj, wlanWaveform)
            %addWLANWaveform Add WLAN waveform to be interfered with Bluetooth
            
            obj.pWLANWaveform(1:min(obj.WaveformMaxSize, numel(wlanWaveform))) = ...
                wlanWaveform(1:min(obj.WaveformMaxSize, numel(wlanWaveform)));
            obj.pHasInterference = true;
        end
        
        function addZWaveform(obj, ZWaveform)
            %addWLANWaveform Add WLAN waveform to be interfered with Bluetooth
            %waveforms
            
            obj.pZWaveform(1:min(obj.WaveformMaxSize, numel(ZWaveform))) = ...
                ZWaveform(1:min(obj.WaveformMaxSize, numel(ZWaveform)));
            obj.pHasZInterference = true;
        end
        
        function btSignalOut = run(obj, btSignalIn, phyMode,K)
            %run Process the received Bluetooth signal and return the signal after
            %passing through the channel
            %
            %   BTSIGNALOUT = run(OBJ, BTSIGNALIN, PHYMODE) process the received
            %   Bluetooth signal and returns the signal after passing through the
            %   channel.
            %
            %   BTSIGNALIN and BTSIGNALOUT are structures containing these fields:
            %       Waveform:         IQ samples of the received waveform
            %       NumSamples:       Length of the waveform (number of IQ samples)
            %       SampleRate:       Sample rate of the received waveform
            %       PacketType:       Bluetooth packet type
            %       SourceID:         Source node ID
            %       Bandwidth:        Channel bandwidth in MHz
            %       NodePosition:     Source node position
            %       SamplesPerSymbol: Samples per symbol to generate the waveform
            %       CenterFrequency:  Bluetooth channel center frequency in MHz
            %       StartTime:        Simulation time in microseconds at the
            %                         waveform entry
            %       EndTime:          Simulation time in microseconds after the
            %                         waveform duration
            %       Payload:          Payload bits from the baseband layer. It is a
            %                         binary column vector
            %       PayloadLength:    Number of payload bytes. It is a scalar,
            %                         integer value
            %
            %   PHYMODE is the PHY transmission mode specified as character vector
            %   or string scalar.
            
            % Initialize
            btWaveformOut = complex(zeros(1, 0));
            btSignalOut = btSignalIn;
            
            %%%wlan and zigbee legnth
            %         wlanLen=4;
            %         zLen=8800000;
            %         Wlannum=wlanLen/btSignalIn.NumSamples;
            %         Znum=zLen/16000;
            % %         seed1=randi([16000,32000]);
            %         seed2=randi([1,55000-16000]);
            %         zeros1=complex(zeros([seed2,1]));
            %         zeros2=complex(zeros([39000-seed2,1]));
            fadechannel=comm.RayleighChannel("SampleRate",88e6);
            %         if k/11<=1
            %             Zig=[zeros1;obj.pZWaveform(16000*(k-1)+1:16000*k);zeros2];
            %             Zig=fadechannel(Zig);
            %         else
            %             s=floor(k/11);
            %             K=k-11*s+1;
            %             Zig=[zeros1;obj.pZWaveform(16000*(K-1)+1:16000*K);zeros2];
            %             Zig=fadechannel(Zig);
            %         end
            if K>160
                k=K-floor(K/160)*160+1;
            else
                k=K;
            end
            
            
            % Calculate distance between the source and destination nodes in
            % meters
            distance = norm(btSignalIn.NodePosition - obj.NodePosition);
            
            % Get Bluetooth channel index from the given center frequency
            rxChannelIndex = btSignalIn.CenterFrequency - 2402;
            
            % Received Bluetooth waveform
            btWaveformIn = btSignalIn.Waveform(1:btSignalIn.NumSamples);
            
            % Check if channels match
            if obj.ChannelIndex == rxChannelIndex
                % Free-space path loss model enabled
                if obj.FSPL && (distance ~= 0)
                    % Get the linear scaling factor (alpha) after applying
                    % the free-space path loss
                    alpha = applyPathloss(obj, distance, btSignalIn.CenterFrequency*1e6);
                    % Apply free-space path loss
                    btWaveformOut = alpha * btWaveformIn;
                else
                    btWaveformOut = btWaveformIn;
                end
                
                % Add the WLAN interference
                if obj.pHasInterference && obj.pHasZInterference
                    % Calculate the Rx power from the received IQ samples
                    rxPowerdB = 10*log10(var(btWaveformOut));
                    % Calculate the interference power in dB
                    pidBW = rxPowerdB - obj.SIRW;
                    % Interference power in linear scale
                    piScaleW =  10.^(pidBW/20);
                    piScaleW=1;
                    % Add WLAN interference
                    btWaveformOut = btWaveformOut + ...
                        piScaleW * fadechannel(obj.pWLANWaveform(btSignalOut.NumSamples*(k-1)+1:btSignalOut.NumSamples*k));
                    btWaveformOutP=fadechannel(obj.pWLANWaveform(btSignalOut.NumSamples*(k-1)+1:btSignalOut.NumSamples*k));
                    
                    % Calculate the interference power in dB
                    pidBZ = rxPowerdB - obj.SIRZ;
                    % Interference power in linear scale
                    piScaleZ =  10.^(pidBZ/20);
                    piScaleZ=1;
                    % Add WLAN interference
                    btWaveformOut = btWaveformOut + ...
                        piScaleZ * fadechannel(obj.pZWaveform(btSignalOut.NumSamples*(k-1)+1:btSignalOut.NumSamples*k));
                    btWaveformOutP=btWaveformOutP+fadechannel(obj.pZWaveform(btSignalOut.NumSamples*(k-1)+1:btSignalOut.NumSamples*k));
                elseif obj.pHasInterference
                    
                    % Calculate the Rx power from the received IQ samples
                    rxPowerdB = 10*log10(var(btWaveformOut));
                    % Calculate the interference power in dB
                    pidBW = rxPowerdB - obj.SIRW;
                    % Interference power in linear scale
                    piScaleW =  10.^(pidBW/20);
                    % Add WLAN interference
                    btWaveformOut = btWaveformOut + ...
                        piScaleW * fadechannel(obj.pWLANWaveform(btSignalOut.NumSamples*(k-1)+1:btSignalOut.NumSamples*k));
                    btWaveformOutP=fadechannel(obj.pWLANWaveform(btSignalOut.NumSamples*(k-1)+1:btSignalOut.NumSamples*k));
                    
                elseif obj.pHasZInterference
                    % Calculate the Rx power from the received IQ samples
                    rxPowerdB = 10*log10(var(btWaveformOut));
                    % Calculate the interference power in dB
                    pidBZ = rxPowerdB - obj.SIRZ;
                    % Interference power in linear scale
                    piScaleZ =  10.^(pidBZ/20);
                    % Add WLAN interference
                    btWaveformOut = btWaveformOut + ...
                        piScaleZ * fadechannel(obj.pZWaveform(btSignalOut.NumSamples*(k-1)+1:btSignalOut.NumSamples*k));
                    btWaveformOutP=fadechannel(obj.pZWaveform(btSignalOut.NumSamples*(k-1)+1:btSignalOut.NumSamples*k));
                else
                    btWaveformOut = btWaveformOut;
                    %                 btWaveformOutP=complex(zeros([length(btWaveformIn),1]));
                end
                
                % Calculate the SNR
                %             snr = calculateSNR(obj, phyMode, btSignalIn.SamplesPerSymbol, ...
                %                 btSignalIn.PacketType);
                
                % Add AWGN noise to the waveform
                %             btWaveformOutP = awgn(complex(btWaveformOut), snr);
                btWaveformOutP =btWaveformOut;
            end
            
            % Return the waveform after passing through the channel
            btSignalOut.NumSamples = numel(btWaveformOutP);
            btSignalOut.Waveform(1:btSignalOut.NumSamples) = btWaveformOutP;
        end
    end
    
    methods (Access = private)
        function alpha = applyPathloss(obj, distance, centerFrequency)
            %applyPathloss Returns the path loss factor in linear scale after
            %modeling the Bluetooth path loss
            %
            %   ALPHA = applyPathloss(OBJ, DISTANCE, CENTERFREQUENCY) returns
            %   the path loss factor in linear scale after modeling the
            %   Bluetooth path loss.
            %
            %   ALPHA returns the attenuation factor in linear scale to be
            %   applied on the received waveform.
            %
            %   DISTANCE is the distance between transmitter and receiver in
            %   meters.
            %
            %   CENTERFREQUENCY is an integer represents the center frequency
            %   in Hz.
            
            lamda = 3e8/centerFrequency;
            pathLoss = (distance^obj.PLExponent)*(4*pi/lamda)^2;
            pathLossdB = 10*log10(pathLoss); % in dB
            alpha = 10^(-pathLossdB/20);
        end
        
        function snr = calculateSNR(obj, phyMode, sps, packetType)
            %calculateSNR Return the SNR value based on the PHY mode, packet
            %type and sps
            
            % Set code rate based on packet type
            switch packetType
                case {'FHS','DM1','DM3','DM5','HV2','DV','EV4'}
                    codeRate = 2/3;
                case 'HV1'
                    codeRate = 1/3;
                otherwise
                    codeRate = 1;
            end
            
            % Set number of bits per symbol based on the PHY transmission mode
            switch phyMode
                case 'BR'
                    bitsPerSymbol = 1;
                case 'EDR2M'
                    bitsPerSymbol = 2;
                otherwise
                    bitsPerSymbol = 3;
            end
            
            % Calculate SNR
            snr = obj.EbNo + 10*log10(codeRate) + 10*log10(bitsPerSymbol) - 10*log10(sps);
        end
    end
end

% LocalWords:  EDR WLAN PSDU DSSS PHY sps FHS
