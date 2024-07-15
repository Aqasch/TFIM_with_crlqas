import numpy as np
import scipy.linalg as la

def pauli_matrices():
    """Return the Pauli matrices."""
    sigma_x = np.array([[0, 1], [1, 0]])
    sigma_y = np.array([[0, -1j], [1j, 0]])
    sigma_z = np.array([[1, 0], [0, -1]])
    identity = np.eye(2)
    return sigma_x, sigma_y, sigma_z, identity

def construct_hamiltonian(J, h, N):
    """Construct the Hamiltonian matrix for the transverse field Ising model.
    J = the hopping matrix element,
    h = strength of magnetic field
    N = number of spins = number of qubits.

    Some information:

    -- At B<1, the system is in the ordered phase. In this phase the ground state breaks the spin-flip symmetry. 
               Thus, the ground state is in fact two-fold degenerate.
    -- At B=1, we observe quantum phase transition.
    -- At B>1, the system is in the disordered phase. Here, the ground state does preserve the spin-flip symmetry, 
               and is nondegenerate.
    """

    sigma_x, _, sigma_z, _ = pauli_matrices()
    H = np.zeros((2**N, 2**N), dtype=complex)
    
    # Interaction terms: -J * sigma_i^z * sigma_{i+1}^z
    for i in range(N - 1):
        term = np.kron(np.eye(2**i), np.kron(sigma_z, sigma_z))
        term = np.kron(term, np.eye(2**(N - i - 2)))
        H -= J * term

    # Transverse field terms: -h * sigma_i^x
    for i in range(N):
        term = np.kron(np.eye(2**i), sigma_x)
        term = np.kron(term, np.eye(2**(N - i - 1)))
        H -= h * term
    
    return H


J = 1
h = 1
N = 2
H = construct_hamiltonian(J, h, N)

# print(np.min(la.eig(H)[0].real))
print('Sum of pauli coeff:', (N-1)*-J+ N*-h)


ham = dict()
ham['hamiltonian'], ham['eigvals'] = H, la.eig(H)[0].real

print(np.min(ham['eigvals']))
exit()

np.savez(f'ham_data/tfim_ham_{N}q_j{J}_h{h}.npz', **ham)

ham = np.load(f"ham_data/tfim_ham_{N}q_j{J}_h{h}.npz")

# print

hamiltonian, eigvals = ham['hamiltonian'], ham['eigvals']

print(hamiltonian, eigvals)
# print(H)

