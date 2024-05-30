SNR = [30, 20, 10, 0];

% create data files in parallel
parfor i=1:length(SNR)
    % create 44,000,000 lines by running the script twice
    coexist(SNR(i));
    coexist(SNR(i));
end