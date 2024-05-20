load('E:\科研\论文实验\matlab工具箱\WLAN toolbox\IQ\ZBW\iq1.mat');
figure;
plot(real(TxWaveform))
title('ZBW')
window=1024;
overlap=window/2;
[s,f,t,p]=spectrogram(TxWaveform,window,overlap,window,88e6,'centered');
% s=(s-min(min(s)))/(max(max(s))-min(min(s)));
imshow(20*log10((abs(s))),[]);
figure;
imshow(20*log10(p),[]);
% colorbar;
% figure;
% imagesc(t, f, 20*log10((abs(s))));xlabel('Samples'); ylabel('Freqency');
% colorbar;


spectrumAnalyzerBasic = dsp.SpectrumAnalyzer(...
    'Name','Bluetooth Basic Frequency Hopping', ...
    'ViewType','Spectrum and spectrogram', ...
    'TimeResolutionSource','Property', ...
    'TimeResolution',0.0005, ...    % In seconds
    'SampleRate',88e6, ...    % In Hz
    'TimeSpanSource','Property', ...
    'TimeSpan', 0.05, ...           % In seconds
    'FrequencyResolutionMethod', 'WindowLength', ...
    'WindowLength', 512, ...        % In samples
    'AxesLayout', 'Horizontal', ...
    'FrequencyOffset',2441*1e6, ... % In Hz
    'ColorLimits',[-20 15]);


load('E:\科研\论文实验\matlab工具箱\WLAN toolbox\IQ\ZBW\iq1.mat');
figure;
plot(real(TxWaveform))
title('ZBW')

window=1024;
overlap=window/2;
[s,f,t]=spectrogram(TxWaveform,window,overlap,window,88e6);
figure;
imagesc(t, f, 20*log10((abs(s))));xlabel('Samples'); ylabel('Freqency');
colorbar;



subplot(221)
plot(real(TxWaveform(1:1024)))
subplot(222)
plot(real(TxWaveform(1+1024*3:1024*4)))
subplot(223)
plot(real(TxWaveform(1+1024*5:1024*6)))
subplot(224)
plot(real(TxWaveform(1+1024*27:1024*28)))

load('E:\科研\论文实验\matlab工具箱\WLAN toolbox\IQ\Z\iq1.mat');
figure;
plot(real(TxWaveform))
title('Z')
window=1024;
overlap=window/2;
[s,f,t]=spectrogram(TxWaveform,window,overlap,window,88e6);
figure;
imagesc(t, f, 20*log10((abs(s))));xlabel('Samples'); ylabel('Freqency');
colorbar;


load('E:\科研\论文实验\matlab工具箱\WLAN toolbox\IQ\W\iq1.mat');
figure;
plot(real(TxWaveform))
title('W')

load('..\ZB\iq1.mat');
figure;
plot(real(TxWaveform))
title('ZB')

load('..\ZW\iq1.mat');
figure;
plot(real(TxWaveform))
title('ZW')

load('..\BW\iq1.mat');
figure;
plot(real(TxWaveform))
title('BW')


load('..\ZBW\iq1.mat');
figure;
plot(real(TxWaveform))
title('ZBW')
spectrumAnalyzerBasic(TxWaveform)