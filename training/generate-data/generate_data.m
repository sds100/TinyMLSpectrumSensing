SNR = [-5, -10, -15, -20];

file='../data/matlab';

% create data files in parallel
for i=1:length(SNR)
    % append 44,000,000 lines by running the script twice
    coexist(SNR(i), file);
    coexist(SNR(i), file);
    coexist(SNR(i), file);
    coexist(SNR(i), file);
    coexist(SNR(i), file);
    coexist(SNR(i), file);
end 