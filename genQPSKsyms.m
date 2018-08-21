clc;
clear all;
close all;

num_users = 4;
num_syms = 101;
NFFT = 1024;

qam = 'BPSK';
k = 1;
refconst = qammod(0:2^k - 1,2^k);
refconst_chanest = qammod(0:1,2);
nf = modnorm(refconst,'peakpow',1);
nf_chnest = modnorm(refconst_chanest,'peakpow',1);

X_chanest = randi([0 1],NFFT-1,1);
YBPSKChanEst = qammod(X_chanest,2);
YBPSKChanEst = YBPSKChanEst * nf_chnest;
for i = 1:num_users
    X_Input = randi([0 2^k - 1],(NFFT-1)*(num_syms-1),1);
    YQPSKOutput = qammod(X_Input,2^k);
    YQPSKOutput = YQPSKOutput * nf;
    Y_QPSKOutput(:,i) = [YBPSKChanEst;YQPSKOutput];
end

% str = ['C:\Users\Bhargav04\Documents\Massive MIMO programs\OFDM_',qam,'_',int2str(NFFT),'FFT_QAMsyms'];
% 
% if ~exist(str)
%     mkdir(str);
% end
% fid1 = fopen([str,'\Symbols.dat'],'w');
% fid2 = fopen([str,'\Pilots.dat'],'w');
% for u = 1:num_users    
%     for i = 1:length(Y_QPSKOutput)
%         fwrite(fid1,[real(Y_QPSKOutput(i,u)); imag(Y_QPSKOutput(i,u))], 'float32');
%     end
% end
% for i = 1:NFFT-1
%     fwrite(fid2,[real(YBPSKChanEst(i)); imag(YBPSKChanEst(i))], 'float32');
% end
% fclose('all');



