// 4-qubit GHZ; the cx 0->3 forces routing SWAPs across the dot line.
qubits 4
h 0
cx 0 1
cx 1 2
cx 0 3
rz 2 0.7853981633974483
cxswap 1 2
