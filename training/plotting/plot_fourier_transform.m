n = 512
freqs = abs(fft(WaveformOut(1:262144), n))

freqs2 = [freqs(n/2:n); freqs(1:n/2)]
plot(abs(freqs2))