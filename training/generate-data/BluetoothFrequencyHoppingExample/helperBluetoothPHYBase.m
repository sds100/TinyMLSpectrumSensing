classdef helperBluetoothPHYBase < handle
%helperBluetoothPHYBase Base class for Bluetooth basic rate or enhanced
%data rate (BR/EDR) physical layer (PHY) modeling
%   BTPHYBASE = helperBluetoothPHYBase creates a base object for Bluetooth
%   BR/EDR PHY modeling.
%
%   BTPHYBASE = helperBluetoothPHYBase(Name, Value) creates a base object
%   for Bluetooth BR/EDR PHY with the specified property Name set to the
%   specified Value. You can specify additional name-value pair arguments
%   in any order as (Name1, Value1, ..., NameN, ValueN).
%
%   helperBluetoothPHYBase methods:
%
%   centerFrequency     - Return the center frequency (in MHz)
%                         corresponding to the given Bluetooth BR/EDR
%                         channel index
%   channelIndex        - Return the Bluetooth channel index corresponding 
%                         to the given channel center frequency (in MHz)
%
%   helperBluetoothPHYBase abstracted methods:
%
%   run                 - Simulate the Bluetooth BR/EDR PHY
%   updatePHY           - Update the PHY with the request from the baseband
%                         layer
%
%   helperBluetoothPHYBase properties:
%
%   Mode                - PHY transmission mode
%   ChannelIndex        - Bluetooth BR/EDR channel index to transmit or
%                         receive the baseband data

%   Copyright 2020-2021 The MathWorks, Inc.

properties
    %Mode PHY transmission mode
    %   This property specifies the physical layer transmission mode as
    %   'BR', 'EDR2M' and 'EDR3M' to indicate basic rate, enhanced data
    %   rate with 2 Mbps and 3 Mbps, respectively. The default value is
    %   'BR'.
    Mode = 'BR'
    
    %ChannelIndex Bluetooth BR/EDR channel index to transmit or receive the
    %baseband data
    %   This property specifies channel index as a scalar integer in the
    %   range [0, 78]. This property defines the channel to transmit or
    %   receive the baseband data. The default value is 0.
    ChannelIndex = 0
end

properties (Constant, Hidden)
    % PHY transmission mode values
    ModeValues = {'BR', 'EDR2M', 'EDR3M'};
    
    % Bluetooth channel bandwidth in MHz
    ChannelBandwidth = 1
    
    % Base frequency (in MHz) is the operating frequency of the channel
    % index 0
    BaseFrequency = 2402
    
    % Maximum size of the information bytes (data from the higher layers)
    % in octets
    MaxPayloadSize = 400
end

properties (Access = protected)
    % Define interface between PHY and baseband layers
    pBasebandData
end

methods
    % Constructor
    function obj = helperBluetoothPHYBase(varargin)
        % Number of bits in single octet
        octetLen = 8;
        
        % Set name-value pairs
        for idx = 1:2:nargin
            obj.(varargin{idx}) = varargin{idx+1};
        end
        
        % To support codegen for variable sized values
        coder.varsize('basebandData.PacketType', [1 3]);
        
        % Initialize the data between PHY and baseband layers
        basebandData = struct('LTAddr', 0, ... % Logical transport address
            'PacketType', blanks(0), ... % Packet type
            'Payload', zeros(obj.MaxPayloadSize * octetLen, 1), ... % Payload
            'PayloadLength', 0, ... % Payload size in bytes
            'LLID', [0; 0], ... % Logical link identifier
            'SEQN', 0, ... % Sequence number
            'ARQN', 0, ... % Acknowledgement flag
            'IsValid', false); % Flag to identify the status of CRC and HEC
        obj.pBasebandData = basebandData;
    end
    
    % Auto-completion for fixed set of option strings
    function v = set(obj, prop)
        v = obj.([prop, 'Values']);
    end
    
    % Set PHY transmission mode
    function set.Mode(obj, value)
        obj.Mode = validatestring(value, obj.ModeValues, ...
            mfilename, 'Mode');
    end
    
    % Set channel index
    function set.ChannelIndex(obj, value)
        validateattributes(value, {'numeric'}, {'scalar', ...
            'integer', '>=', 0, '<=', 78}, mfilename, 'ChannelIndex');
        obj.ChannelIndex = value;
    end
    
    function centerFrequency = centerFrequency(obj)
    %centerFrequency Returns the center frequency in MHz
    %
    %   CENTERFREQUENCY = centerFrequency(OBJ) returns the center
    %   frequency in MHz corresponding to the configured Bluetooth BR/EDR
    %   channel number.
    %
    %   CENTERFREQUENCY returns an integer representing the center
    %   frequency in MHz.
    %
    %   OBJ is an object of type helperBluetoothPHYBase.
        
        % Calculate the center frequency for the given channel index
        centerFrequency = obj.BaseFrequency + obj.ChannelIndex; % in MHz
    end
    
    function channelIndex = channelIndex(obj, centerFrequency)
    %channelIndex Returns the Bluetooth channel index
    %
    %   CHANNELINDEX = channelIndex(OBJ, CENTERFREQUENCY) returns the
    %   channel index corresponding to the given center frequency in MHz.
    %
    %   CHANNELINDEX returns the index of the Bluetooth channel.
    %
    %   CENTERFREQUENCY is an integer representing the center frequency in
    %   MHz.
        
        % Get channel index from the given center frequency
        channelIndex = centerFrequency - obj.BaseFrequency;
    end
end

methods (Abstract)
    [nextInvokeTime, varargout] = run(obj, elapsedTime, varargin)
    %run Process the PHY
    %
    %   [NEXTINVOKETIME, VARARGOUT] = run(OBJ, ELAPSEDTIME, VARARGIN)
    %   processes the received data and returns the output data with the
    %   next invoke time.
    %
    %   NEXTINVOKETIME returns the time after which the run function must
    %   be invoked again.
    %
    %   OBJ is instance of an object of type helperBluetoothPHYBase.
    %
    %   ELAPSEDTIME is the time elapsed in microseconds between two
    %   successive calls of this function.
    
    updatePHY(obj, state, channelIndex, whitenInit, basebandData)
    %updatePHY Update the PHY with the request from the baseband layer
    %
    %   updatePHY(OBJ, STATE, CHANNELINDEX, WHITENINIT, BASEBANDDATA)
    %   updates the PHY based on the requests from the baseband layer.
    %
    %   OBJ is instance of an object of type helperBluetoothPHYBase.
    %
    %   STATE is the state of PHY (transmit or receive).
    %
    %   CHANNELINDEX defines the physical channel for waveform
    %   transmission or reception.
    %
    %   WHITENINIT is the initialization vector for whitening or
    %   de-whitening process in PHY.
    %
    %   BASEBANDDATA is the structure containing these fields:
    %       LTAddr          - Logical transport address of an active
    %                         destination peripheral for a packet in a
    %                         central-to-peripheral or
    %                         peripheral-to-central transmission slot. It
    %                         is a 3-bit value. The value of this field is
    %                         a scalar positive integer
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
end
end

% LocalWords:  EDR PHY Mbps LLID SEQN ARQN CRC HEC
