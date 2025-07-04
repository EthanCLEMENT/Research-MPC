import numpy as np
import control as ctrl

# Define continuous time model.
Acont = np.array([[-1.2822, 0, 0.98, 0],
              [0, 0, 1, 0],
              [-5.4293, 0, -1.8366, 0],
              [-128.2, 128.2, 0, 0]])
Bcont = np.array([[-0.3], [0], [-17], [0]])
Ccont = np.array([[1,0,0,0],[0, 1, 0, 0], [0, 0, 1, 0]])
Dcont = np.array([[0], [0]])

# Checking controllability and observability
ctrb_M = ctrl.ctrb(Acont,Bcont)
obsv_M = ctrl.obsv(Acont,Ccont)

print(np.linalg.matrix_rank(ctrb_M))
print(np.linalg.matrix_rank(obsv_M))

eigenvalues_A = np.linalg.eig(Acont)
print(eigenvalues_A)

#PBH test
def is_detectable(A, C, eigvals_A):
    n = A.shape[0]
    detectable = True

    for i in eigvals_A:
        PBH_matrix = np.vstack([i * np.eye(n) - A, C])

        rank_PBH = np.linalg.matrix_rank(PBH_matrix)

        if rank_PBH < n and np.real(i) > 0:
            detectable = False
            print(f"Mode associated with eigenvalue {i} is not detectable.")

    return detectable

detectable_check = is_detectable(Acont, Ccont, eigenvalues_A)
print(detectable_check)
