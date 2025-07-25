import numpy as np
from math import log2, ceil
from pennylane.labs.trotter_error import RealspaceOperator, RealspaceCoeffs, RealspaceSum, RealspaceMatrix
import csv

def vibronic_fragments_modebased(nstates, p, freqs, taylor_coeffs, subtype):

    omegas, lambdas, alphas, betas = freqs, taylor_coeffs[0], taylor_coeffs[1], taylor_coeffs[2]
    n_blocks = 2**ceil(log2(nstates))
    independent_frags = []
    coupling_mats = []

    # Add harmonic potential to the diagonal quadratic coefficients
    for i in range(nstates):
        for r in range(p):
            betas[i, i, r, r] += omegas[r] / 2

    # --- Create fragments for each term type ---

    # 1. Q_i^2 fragments (diagonal quadratic terms)
    for r in range(p):
        frag_Q2 = RealspaceMatrix.zero(n_blocks, p)
        mat_Q2 = np.zeros((nstates, nstates))
        is_frag_nonzero = False
        for i in range(nstates):
            for j in range(nstates):
                h2_ij = betas[i, j, r, r]
                if abs(h2_ij) > 1e-15: # Check for non-zero coefficient
                    is_frag_nonzero = True
                    coeffs2 = np.zeros((p, p))
                    coeffs2[r, r] = h2_ij
                    op2 = RealspaceOperator(p, ("Q", "Q"), RealspaceCoeffs(coeffs2, label=f"beta[{r}][{i},{j}]"))
                    frag_Q2.set_block(i, j, RealspaceSum(p, [op2]))
                    mat_Q2[i, j] = h2_ij
        if is_frag_nonzero:
            independent_frags.append(frag_Q2)
            coupling_mats.append(mat_Q2)

    for r in range(p):
        for s in range(p):
            # Skip diagonal Q_r^2 terms, as they are handled in their own loop
            if r == s:
                continue

            # Each off-diagonal (r, s) pair gets its own potential fragment
            frag_Q_rs = RealspaceMatrix.zero(n_blocks, p)
            mat_Q_rs = np.zeros((nstates, nstates))
            is_frag_nonzero = False
            
            for i in range(nstates):
                for j in range(nstates):
                    # Get the specific coefficient for the Q_r * Q_s term
                    h_rs_ij = betas[i, j, r, s]
                    
                    if abs(h_rs_ij) > 1e-15:
                        is_frag_nonzero = True
                        coeffs_rs = np.zeros((p, p))
                        
                        # Create an asymmetric operator with only the (r, s) entry populated.
                        # This correctly represents a single, non-symmetric beta[i,j,r,s] * Q_r * Q_s term.
                        coeffs_rs[r, s] = h_rs_ij
                        
                        op_rs = RealspaceOperator(p, ("Q", "Q"), RealspaceCoeffs(coeffs_rs, label=f"beta[{r},{s}][{i},{j}]"))
                        frag_Q_rs.set_block(i, j, RealspaceSum(p, [op_rs]))
                        mat_Q_rs[i, j] = h_rs_ij
            
            # Only add the fragment to the list if it's not empty
            if is_frag_nonzero:
                independent_frags.append(frag_Q_rs)
                coupling_mats.append(mat_Q_rs)

    # 3. Q_i fragments (linear terms)
    for r in range(p):
        frag_Q = RealspaceMatrix.zero(n_blocks, p)
        mat_Q = np.zeros((nstates, nstates))
        is_frag_nonzero = False
        for i in range(nstates):
            for j in range(nstates):
                h_ij = alphas[i, j, r]
                if abs(h_ij) > 1e-15: # Check for non-zero coefficient
                    is_frag_nonzero = True
                    coeffs = np.zeros(p)
                    coeffs[r] = h_ij
                    op = RealspaceOperator(p, ("Q",), RealspaceCoeffs(coeffs, label=f"alpha[{r}][{i},{j}]") )
                    frag_Q.set_block(i, j, RealspaceSum(p, [op]))
                    mat_Q[i, j] = h_ij
        if is_frag_nonzero:
            independent_frags.append(frag_Q)
            coupling_mats.append(mat_Q)

    # --- Handle subtypes ---

    if subtype == 'Individual':
        # Add the kinetic energy and electronic couplings term
        alphas_frag = np.zeros((nstates, nstates, p))
        betas_frag = np.zeros((nstates, nstates, p, p))
        kin = kin_frag(nstates, n_blocks, p, omegas) + couplings_to_RealSpaceMatrix(nstates, n_blocks, p, lambdas, alphas_frag, betas_frag)
        independent_frags.append(kin)

        return independent_frags
    
    elif subtype == 'FC':
        # Group fragments by electronic commutation (this logic remains the same)
        remaining = list(range(len(independent_frags)))
        groups = []
        while remaining:
            seed = remaining[0]
            group = [seed]
            rest = []
            for idx in remaining[1:]:
                # Check if the new candidate fragment commutes with all fragments already in the group
                if all(commute(coupling_mats[idx], coupling_mats[m]) for m in group):
                    group.append(idx)
                else:
                    rest.append(idx)
            groups.append(group)
            remaining = rest

        # Merge each commuting set into a single fragment
        grouped_fragments = []
        for group in groups:
            merged = RealspaceMatrix.zero(n_blocks, p)
            for idx in group:
                merged += independent_frags[idx]
            grouped_fragments.append(merged)

        # Add the kinetic energy and electronic couplings term to the grouped list
        alphas_frag = np.zeros((nstates, nstates, p))
        betas_frag = np.zeros((nstates, nstates, p, p))
        kin = kin_frag(nstates, n_blocks, p, omegas) + couplings_to_RealSpaceMatrix(nstates, n_blocks, p, lambdas, alphas_frag, betas_frag)
        grouped_fragments.append(kin)
        
        return grouped_fragments



def count_nonzero_Q_terms(matrix, threshold=1e-15):
    """
    Counts the number of unique linear (Q_i), squared quadratic (Q_i^2),
    and mixed quadratic (Q_i * Q_j) terms in a given matrix object.
    """
    Q_modes = set()
    # For Q_i^2 terms, where indices are the same, e.g., (i, i)
    Qi_squared_modes = set()
    # For Q_i * Q_j terms, where indices are different, e.g., (i, j)
    QiQj_modes = set()

    # Assumes matrix._blocks is a dictionary of objects that have an 'ops' attribute
    for rs_sum in matrix._blocks.values():
        for op in rs_sum.ops:
            # Case 1: Linear terms like ("Q",)
            if op.ops == ("Q",):
                for index, val in op.coeffs.nonzero(threshold).items():
                    # index is a tuple like (i,)
                    Q_modes.add(index[0])
            # Case 2: Quadratic terms like ("Q", "Q")
            elif op.ops == ("Q", "Q"):
                for index, val in op.coeffs.nonzero(threshold).items():
                    # index is a tuple like (i, j)
                    if index[0] == index[1]:
                        # It's a Q_i^2 term if indices are the same
                        Qi_squared_modes.add(index[0])
                    else:
                        # It's a Q_i * Q_j term if indices are different
                        # Sort the tuple to treat (i, j) and (j, i) as the same term
                        QiQj_modes.add(tuple(sorted(index)))

    # Return the three distinct sets of modes
    return len(Q_modes), len(Qi_squared_modes), len(QiQj_modes)

def couplings_to_RealSpaceMatrix(nstates, n_blocks, p, lambdas, alphas, betas):
    '''RealSpaceMatrix for the potential part'''
    #assuming len coeff == 2, it is list of alphas and betas
    rs_matrix = RealspaceMatrix.zero(n_blocks, p)
    for i in range(nstates):
        for j in range(nstates):
                c_op = RealspaceOperator(p, (), RealspaceCoeffs(lambdas[i,j], label="lamdas"))
                l_op = RealspaceOperator(p, ("Q",), RealspaceCoeffs(alphas[i,j], label="alphas"))
                q_op = RealspaceOperator(p, ("Q","Q",), RealspaceCoeffs(betas[i,j], label="betas"))
                rs_sum = RealspaceSum(p, [c_op, l_op, q_op])
                rs_matrix.set_block(i,j, rs_sum)
    return rs_matrix

def commute(matrix1, matrix2):
    commutator = matrix1@matrix2 - matrix2@matrix1
    return not np.any(commutator)


def kin_frag(nstates, n_blocks, p, omegas):
    kin_term = RealspaceOperator(
        p,
        ("P", "P"),
        RealspaceCoeffs(np.diag(omegas) / 2, label="omega"),
    )
    kin_sum = RealspaceSum(p, (kin_term,))
    kin_blocks = {(i, i): kin_sum for i in range(nstates)}
    kin_frag = RealspaceMatrix(n_blocks, p, kin_blocks)

    return kin_frag

def get_norm_value(mol, K, M, fragmentation_scheme):
    """
    Reads a CSV file to find and return a specific norm value.

    Args:
        mol (str): The name of the molecule (e.g., 'no4a_monomer.pkl').
        K (int): The value from the second column.
        M (int): The value from the third column.
        fragmentation_scheme (str): The fragmentation scheme.

    Returns:
        float: The corresponding norm value, or None if not found.
    """
    mol = f'{mol}.pkl'
    filepath=f"model_params/norms_{fragmentation_scheme}.csv"

    try:
        with open(filepath, mode='r', newline='') as data_file:
            reader = csv.reader(data_file)
            for row in reader:
                # Check if the row matches the query
                if not row or len(row) < 4:
                    continue
                
                # Directly compare values after converting types
                if row[0] == mol and int(row[1]) == K and int(row[2]) == M:
                    return float(row[3]) # Return the norm value if found
    except FileNotFoundError:
        print(f"Error: The file '{filepath}' was not found.")
        return None
    except (ValueError, IndexError):
        print(f"Error: A row in '{filepath}' has incorrect format.")
        return None

    print('Norm not found!')

def print_Qubit_Toff(resources):
    '''Function for printing the resources for a circuit'''
    qubit_count = resources.qubit_manager.total_qubits
    toffoli_count = resources.clean_gate_counts.get("Toffoli", 0)
    
    if toffoli_count > 9999:
        toffoli_count = f"{toffoli_count:.3E}"
    
    print(f'Qubit Count = {qubit_count}')
    print(f'Toffoli Count = {toffoli_count}')

def print_Toff(resources):
    '''Function for printing the resources for a circuit'''
    toffoli_count = resources.clean_gate_counts.get("Toffoli", 0)
    
    if toffoli_count > 9999:
        toffoli_count = f"{toffoli_count:.3E}"
 
    print(f'Toffoli Count = {toffoli_count}')