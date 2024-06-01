SNR = [30, 20, 10, 0];

file='../data/matlab-random';

% create data files in parallel
parfor i=1:length(SNR)
    % create 44,000,000 lines by running the script twice
    coexist_random_tx_power(SNR(i), file);
    coexist_random_tx_power(SNR(i), file);
end