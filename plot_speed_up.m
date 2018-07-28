clc;
clear all;
close all;

speed_up_1024FFT = [1.15E-01	2.14E-01	4.08E-01	1.99E-01	6.61E-01	2.17E-01
1.51E-01	4.26E-01	4.08E-01	4.12E-01	8.83E-01	3.95E-01
2.39E-01	8.91E-01	3.93E-01	8.54E-01	9.18E-01	6.90E-01
5.16E-01	1.55E+00	4.16E-01	1.74E+00	1.13E+00	1.12E+00
8.81E-01	2.27E+00	3.62E-01	2.82E+00	9.72E-01	1.37E+00
];

speed_up_64FFT = [2.42E-02	8.39E-02	4.60E-02	9.00E-02	9.16E-01	8.34E-02
3.14E-02	1.43E-01	6.19E-02	1.63E-01	7.79E-01	1.47E-01
6.66E-02	3.20E-01	9.17E-02	3.68E-01	8.86E-01	3.25E-01
1.00E-01	6.55E-01	1.41E-01	7.80E-01	8.95E-01	6.64E-01
1.78E-01	1.20E+00	1.91E-01	1.41E+00	8.62E-01	1.15E+00
];

% fit1024 = fit(2.^(0:4)',speed_up_1024FFT(:,3),'smoothingspline');
% fit64 = fit(2.^(0:4)',speed_up_64FFT(:,3),'smoothingspline');

% plot(fit1024, 'r--'); hold on;
plot(2.^(0:4)',speed_up_1024FFT(:,2), 'r--^'); hold on;
% plot(fit64, 'b-');
plot(2.^(0:4)',speed_up_64FFT(:,2), 'b--o');
grid on;
% axis([1 16 0 2]);
lgd = legend('1024 FFT','64 FFT');
lgd.Location = 'northwest';
xlabel('Number of antennas');
ylabel('Acceleration (CPU Time/GPU Time)');
% title('Speed up in using GPU for demodulation of uplink OFDM symbols');

figure;
bar([speed_up_1024FFT(end,:);speed_up_64FFT(end,:)]'); hold on;
% bar(, 'b','LineStyle', '--');
grid on;
axis([0 7 0 3]);
lgd = legend('1024 FFT','64 FFT');
% txt = {'Symbol Read';'Channel Estimation';'MRC';'FFT';'Prefix Drop';'Demodulation'};
% set(gca,'xticklabel',txt);
xlabel('Function Index');%: 1 - Symbol read, 2 - Channel Estimation, 3 - MRC, 4 - FFT, 5 - Prefix Drop, 6 - Demodulation');
ylabel('Acceleration (CPU Time/GPU Time)');
lgd.Location = 'northwest';


% text(4.5,2.85,txt,'Interpreter','latex');