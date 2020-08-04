import pennylane as qml

def _mry(q, weights):
    n = len(q)
    for i in range(n):
        qml.RY(weights[i], wires=q[i])
        if (i+1) % 2 == 0:
            qml.CNOT(wires=[q[i], q[i-1]])

def circuit(n, q, weights):
    layer = 0
    ops_total = 0
    while True:
        ops_count = n // 2**layer

        _mry(q[ : n : n // ops_count], weights[ops_total : ops_total+ops_count])
        
        ops_total += ops_count
        layer += 1
        if (ops_count <= 1):
            break

    return qml.expval(qml.PauliZ(0))

