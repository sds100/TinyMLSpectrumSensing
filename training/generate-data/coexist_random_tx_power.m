function coexist_random_tx_power(snr, file)
    %add transmit power
    zig_Txpower=[-50, -40, -30];
    wifi_Txpower=[-40, -30, -20];
    blue_Txpower=[-42, -48, -51, -54, -57];%also can be -42,-48,-51,-54,-57
    
    %add fade channel with sample rate 88MHz
    fadechannel=comm.RayleighChannel("SampleRate",88e6);
    
    %define path to save data including 7 classes
    foldername={'B','W','Z','BW','ZW','ZB','ZBW'};
    
    % parameters of Short-Time Fourier Transform
    FFT=256;
    Noverlap=FFT/2;
    
    SNR=snr;
    
    % move the central frequency of Zigbee to different values
    ZigBee_Delta_Freq=0;%42e6,10e6,0,-20e6,-50e6 according to 2434，2439，2444，2449
    
    if ~exist(file, 'dir')
        mkdir(file);
    end
    
    %generate data
    disp("Generate Bluetooth waveform");
    % generate Bluetooth waveform
    simulationTime=5*1e5;
    [BlueWaveform,Blue_hopindex]=generateBlue(simulationTime);
    
    disp("Generate WLAN waveform");
    % Generate the WLAN waveform
    wlanWaveform = myhelperBluetoothGenerateWLANWaveform('Wlan');
    
    disp("Generate ZigBee waveform");
    % Generate the Zigbee waveform
    zigbeeWaveform = myhelperBluetoothGenerateZigbeeWaveform('Zigbee');
    
    % limit the length
    len=min([length(wlanWaveform),length(zigbeeWaveform),length(BlueWaveform)]);

    step_size = 500000;

    i = 1;
    rng();

    while i < (len - step_size)
        %confirm the transmit power
        b_Txpower=blue_Txpower(randi(length(blue_Txpower),1));
        z_Txpower=zig_Txpower(randi(length(zig_Txpower),1));
        w_Txpower=wifi_Txpower(randi(length(wifi_Txpower),1));
        
        %turn dB to W
        z_scale = sqrt(1e-3*10^(z_Txpower/10)/bandpower(zigbeeWaveform));
        w_scale = sqrt(1e-3*10^(w_Txpower/10)/bandpower(wlanWaveform));
        b_scale = sqrt(1e-3*10^(b_Txpower/10)/bandpower(BlueWaveform));
        
        start = i;
        last = start + step_size;
        zigbeeWaveform(i:last)=z_scale*zigbeeWaveform(i:last);
        wlanWaveform(i:last)=w_scale*wlanWaveform(i:last);
        BlueWaveform(i:last)=b_scale*BlueWaveform(i:last);
    
        i = i + step_size;
    end
    
    
    for s=1:7
        scen=[1,2,3,4,5,6,7]; %%B-1 W-2 Z-3 BW-4 ZW-5 ZB-6 ZBW-7
        switch scen(s)
            case 1 %B
                disp("Generate B scenario");
                consig=BlueWaveform(1:len);
                channelsig=fadechannel(consig);
                WaveformOut = awgn(complex(channelsig), SNR,'measured');
                %                       figure;
                %                       data=stft(WaveformOut,88e6,'OverlapLength',Noverlap,'FFTLength',FFT);
                
            case 2 %W
                disp("Generate W scenario");
                consig=wlanWaveform(1:len);
                channelsig=fadechannel(consig);
                WaveformOut = awgn(complex(channelsig), SNR,'measured');
                %                       figure;
                %                       stft(WaveformOut(1:200000),88e6,'Window',kaiser(256,5),'OverlapLength',220,'FFTLength',512)
            case 3 %Z
                disp("Generate Z scenario");
                if any(real(zigbeeWaveform))
                    %%%add frequency offset to Zigbee
                    Freq_Sample=88e6;   
                    Simulation_Length=length(zigbeeWaveform);
                    Carrier=exp(1j*(ZigBee_Delta_Freq/Freq_Sample*(1:Simulation_Length)))';
                    zigbeeWaveformcc=zigbeeWaveform.*Carrier;
                end
                consig=zigbeeWaveformcc(1:len);
                channelsig=fadechannel(consig);
                WaveformOut = awgn(complex(channelsig), SNR,'measured');
                %                          figure;
                %                          stft(WaveformOut(1:200000),88e6,'Window',kaiser(256,5),'OverlapLength',Noverlap,'FFTLength',FFT)
            case 4 %BW
                disp("Generate BW scenario");
                consig=wlanWaveform(1:len)+BlueWaveform(1:len);
                channelsig=fadechannel(consig);
                WaveformOut = awgn(complex(channelsig), SNR,'measured');
                %                       figure;
                %                       stft(WaveformOut(1:200000),88e6,'Window',kaiser(256,5),'OverlapLength',Noverlap,'FFTLength',FFT)
            case 5 % ZW
                disp("Generate ZW scenario");
                if any(real(zigbeeWaveform))
                    %%% add frequency offset to Zigbee
                    Freq_Sample=88e6;
                    Simulation_Length=length(zigbeeWaveform);
                    Carrier=exp(1j*(ZigBee_Delta_Freq/Freq_Sample*(1:Simulation_Length)))';
                    zigbeeWaveformcw=zigbeeWaveform.*Carrier;
                end
                consig=wlanWaveform(1:len)+zigbeeWaveformcw(1:len);
                channelsig=fadechannel(consig);
                WaveformOut = awgn(complex(channelsig), SNR,'measured');
                %                          figure;
                %                          stft(WaveformOut(1:200000),88e6,'Window',kaiser(256,5),'OverlapLength',220,'FFTLength',512)
            case 6 %ZB
                disp("Generate ZB scenario");
                if any(real(zigbeeWaveform))
                    %%% add frequency offset to Zigbee
                    Freq_Sample=88e6;
                    Simulation_Length=length(zigbeeWaveform);
                    Carrier=exp(1j*(ZigBee_Delta_Freq/Freq_Sample*(1:Simulation_Length)))';
                    zigbeeWaveformcb=zigbeeWaveform.*Carrier;
                end
                consig=zigbeeWaveformcb(1:len)+BlueWaveform(1:len);
                channelsig=fadechannel(consig);
                WaveformOut = awgn(complex(channelsig), SNR,'measured');
                %                          figure;
                %                          stft(WaveformOut(1:200000),88e6,'Window',kaiser(256,5),'OverlapLength',220,'FFTLength',512)
            case 7 %ZBW
                disp("Generate ZBW scenario");
                if any(real(zigbeeWaveform))
                    %%% add frequency offset to Zigbee
                    Freq_Sample=88e6;
                    Simulation_Length=length(zigbeeWaveform);
                    Carrier=exp(1j*(ZigBee_Delta_Freq/Freq_Sample*(1:Simulation_Length)))';
                    zigbeeWaveformcbw=zigbeeWaveform.*Carrier;
                end
                consig=zigbeeWaveformcbw(1:len)+BlueWaveform(1:len)+wlanWaveform(1:len);
                
                channelsig=fadechannel(consig);
                WaveformOut = awgn(complex(channelsig), SNR,'measured');
                %                          figure;
                %                          stft(WaveformOut(1:200000),88e6,'Window',kaiser(256,5),'OverlapLength',220,'FFTLength',512)
        end
        class=foldername{s};

        % Save as CSV
        % filename=[file '/csv/' class '_SNR' num2str(SNR(snr)) '.csv'];
        % writematrix(WaveformOut,filename);

        % Save as .mat file
        filename=[file '/' 'SNR' num2str(SNR) '_'  class '.mat'];

        % only concat data if the file exists
        if isfile(filename)
            old_data = load(filename);
            WaveformOut = [old_data.WaveformOut;WaveformOut];
        end
        save(filename, 'WaveformOut');
    end
    % stft(TxWaveform,88e6,'Window',kaiser(256,5),'OverlapLength',220,'FFTLength',512)
    % stft(TxWaveform(1:2000000),88e6,'Window',kaiser(256,5),'OverlapLength',220,'FFTLength',512)
    
    disp("Done");
    
    % Plot and save the selected channel index per slot
    % figBasic = figure('Name','Basic frequency hopping');
    % axisBasic = axes(figBasic);
    % xlabel(axisBasic,'Slot');
    % ylabel(axisBasic,'Channel Index');
    % ylim(axisBasic,[0 numBluetoothChannels+3]);
    % title(axisBasic,'Bluetooth Basic Frequency Hopping');
    % hold on;
    % plot(axisBasic,0:slotValue:numSlots-slotValue,hopIndex,'-o');
    % save('..\random-hopIndexzchannel162m.mat','hopIndex')
end
