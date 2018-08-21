import socket
import sys
import threading

txrxflag = [0, 0, 0];

num_APs = int(sys.argv[1]);
print("Number of APs:",num_APs);

num_users = int(sys.argv[2]);
print("Number of users:",num_users);

lock = threading.Lock();
	

	
	
def ap_tx_rx(sock, addr, port, txrxflag):
	print("In AP",port-10000);
	
	print("Starting TX");
	sock.sendto('Start_TX'.encode(),(addr, port));
	data, addr = sock.recvfrom(1024);
	print(data.decode());
	if data.decode() == 'Started':
		lock.acquire();
		txrxflag[1] += 1;
		lock.release();
	
	while txrxflag[0] < num_users:
		continue;
	lock.acquire();
	if txrxflag[0] > 0:
		txrxflag[0] = 0;
	lock.release();

	sock.sendto('Stop_TX'.encode(),(addr, port));
	data, addr = sock.recvfrom(1024);
	print(data.decode());
		

def user_tx_rx(sock, addr, port, txrxflag):
	print("In user",port-10000-num_APs);
		
	while txrxflag[1] < num_APs:
		continue;
	lock.acquire();
	if txrxflag[1] > 0:
		txrxflag[1] = 0;
	lock.release();
	
	print("Starting RX");
	sock.sendto('Start_RX'.encode(),(addr, port));
	data, addr = sock.recvfrom(1024);
	print(data.decode());
	if data.decode() == 'Done':
		lock.acquire();
		txrxflag[0] += 1;
		lock.release();
	


def node_tx_rx(addr, port, txrxflag):
	sock = socket.socket(socket.AF_INET,socket.SOCK_DGRAM);
	sock.bind((addr, port));
	print("Addr:",addr,",Port:",port);
	
	data, addr = sock.recvfrom(1024);
	if data.decode() == 'Initialized':
		print(data.decode());
		lock.acquire();
		txrxflag[2] += 1;
		lock.release();
	
	while txrxflag[2] < num_APs + num_users:
		continue;
	
	if (port-10000) < num_APs:
		ap_tx_rx(sock, addr, port, txrxflag);
	else:
		user_tx_rx(sock, addr, port, txrxflag);

	
myThread = [];
def main():
	for node in range(num_APs + num_users):
		myThread.append(threading.Thread(name = "Thread"+str(node), target = node_tx_rx, args = (sys.argv[node + 3], 10000 + node, txrxflag)));
		print("Thread ID:",myThread[node].name);
		myThread[node].start();
	

if __name__ == "__main__":
	main();
	

	
