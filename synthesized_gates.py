from qiskit import *

def rzcz(theta, ctrl, targ):
    '''
    ------[RZ(ctrl)]----o(ctrl)----
                        |
                        |
    --------------------o----------
    '''
    circ = QuantumCircuit(2, name=f"rzcz{ctrl}{targ}")
    circ.ry(theta, 0)
    circ.cz(0, 1)
    circ = circ.to_gate()
    return circ


def xrz(theta):
    '''
    ------[X]--RZ(theta)-----
    '''
    circ = QuantumCircuit(1, name="xrz_gate")
    circ.x(0)
    circ.rz(theta, 0)
    circ = circ.to_gate()
    return circ

def sxxsx():
    '''
    ------[Sqrt(X)]----------
    
    ------[X]------[Sqrt(X)]--
    '''
    circ = QuantumCircuit(2, name="xrz_gate")
    circ.sx(0)
    circ.x(1)
    circ.sx(1)
    circ = circ.to_gate()
    return circ


