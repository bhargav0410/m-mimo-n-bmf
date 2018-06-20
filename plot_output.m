clc;
clear all;

while 1
system('scp root@node21-1:uhd/host/build/examples/Output_cpu.dat .');
system('scp root@node21-1:uhd/host/build/examples/Output_gpu.dat .');
system('scp root@node21-1:uhd/host/build/examples/time_cpu.dat .');
system('scp root@node21-1:uhd/host/build/examples/time_gpu.dat .');
rx1 = read_float_binary('Output_cpu.dat');
rxComp = rx1(1:2:end) + 1i*rx1(2:2:end);
rx2 = read_float_binary('Output_gpu.dat');
rxComp = rx2(1:2:end) + 1i*rx2(2:2:end);

figure(1);
subplot(1,2,1);
plot(rx1(1:2:end),rx1(2:2:end), 'r.'); hold on;
plot(rx2(1:2:end),rx2(2:2:end), 'b.'); hold off;
axis([-2 2 -2 2]);
title('Constellation','fontsize',12);
legend('CPU','GPU');

cputime = read_float_binary('time_cpu.dat');
gputime = read_float_binary('time_gpu.dat');

subplot(1,2,2);
stem(cputime*1e3, 'r--*'); hold on;
stem(gputime*1e3, 'b');
title('CPU and GPU time for various processses','fontsize',12);
xlabel('Index for various processes: 1 - Reading from shared memory, 2 - Channel estimation, 3 - MRC, 4 - FFT, 5 - Prefix drop','fontsize',10);
ylabel('Time (ms)','fontsize',12);
legend('Time (ms) for CPU','Time (ms) for GPU');
hold off;
pause(0.1);
end

