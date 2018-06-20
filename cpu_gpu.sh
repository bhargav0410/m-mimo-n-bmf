echo Press q to exit

while true; do
	./cpu
	./gpu
	read -t 0.25 -N 1 input
	if [[ $input = "q" ]] || [[ $input = "Q" ]]; then
		break
	fi
done