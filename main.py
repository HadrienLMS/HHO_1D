import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy.integrate import quad

# pd.n_cells: number of cells
# pd.h: cell diameter, uniform for all elements
# pd.K: polynomial degree, equal for all elements

# cell dofs of all the cells first 
# then the edge dofs 

# Github copilot hadrien-beriot_SAGCP

class ProblemDefinition:
    def __init__(self):
        self.domain_length = 0
        self.h = 0
        self.K = 0
        self.n_cells = 0
        self.n_dofs = 0
        self.n_dofs_super = 0

    def initialize(self):
        self.compute_element_size()
        self.compute_n_dofs()
        self.compute_n_dofs_super()
        self.print_header()  
    
    def compute_element_size(self):
        """
        Compute the element size h based on the domain length and the number of cells.
        """
        if self.n_cells > 0:  # Avoid division by zero
            self.h = self.domain_length / self.n_cells
        else:
            raise ValueError("Number of cells must be greater than zero.")

    def compute_n_dofs(self):
        self.n_dofs = self.n_cells*(self.K+1) + (self.n_cells+1)

    def compute_n_dofs_super(self):
        # super convergent basis 
        self.n_dofs_super = self.n_cells*(self.K+2) + (self.n_cells+1)

    def print_header(self):
        """
        Print all attributes in a header-like format.
        """
        print('#' * 50)
        print(f"{'Problem Definition':^50}")
        print('#' * 50)
        print(f"{'Domain Length':<20}: {self.domain_length}")
        print(f"{'Number of Cells':<20}: {self.n_cells}")
        print(f"{'Element Size (h)':<20}: {self.h}")
        print(f"{'order':<20}: {self.K}")
        print(f"{'number of dofs':<20}: {self.n_dofs}")
        print(f"{'number of dofs (super-convergent)':<20}: {self.n_dofs_super}")
        print('#' * 50)

def cell_center(pd,elem):
    x_bar = pd.h*elem + pd.h/2
    return x_bar

def face_centers(pd,elem):
    x_bar = cell_center(pd,elem)
    xF1 = x_bar - pd.h/2
    xF2 = x_bar + pd.h/2
    return xF1, xF2

def integrate(order, h, x_bar):
    p, w = np.polynomial.legendre.leggauss(max(order,2))
    qws = w*h/2 
    qps = x_bar + h/2*p
    return qps, qws, len(qws)

def get_cell_dofs(pd,elem):
    # all cell dofs first, face dofs after
    return elem*(pd.K+1) + np.arange(0,pd.K+1)
    
def get_face_dofs(pd,elem):
    # all cell dofs first, face dofs after
    return pd.n_cells*(pd.K+1) + np.arange(elem, elem+2)

def get_dofs(pd,elem):
    cell_dofs = get_cell_dofs(pd,elem)
    face_dofs = get_face_dofs(pd,elem)
    elem_dofs = np.concatenate((cell_dofs, face_dofs), axis=0)
    return elem_dofs, cell_dofs, face_dofs

def get_cell_dofs_super(pd,elem):
    # all cell dofs first, face dofs after
    return elem*(pd.K+2) + np.arange(0,pd.K+2)
    
def get_face_dofs_super(pd,elem):
    # all cell dofs first, face dofs after
    return pd.n_cells*(pd.K+2) + np.arange(elem, elem+2)

def get_dofs_super(pd,elem):
    cell_dofs_super = get_cell_dofs_super(pd,elem)
    face_dofs_super = get_face_dofs_super(pd,elem)
    elem_dofs_super = np.concatenate((cell_dofs_super, face_dofs_super), axis=0)
    return elem_dofs_super, cell_dofs_super, face_dofs_super

def basis(x, x_bar, h, max_k):
    # Listing 8.1 Possible implementation of a function evaluating the scaled monomial
    # scalar basis and its derivatives in a 1D cell.
    # convert to local coord x_tilde in [-1,1]  
    x_tilde = 2*(x-x_bar)/h 
    phi = np.zeros(max_k+1)
    dphi = np.zeros(max_k+1)
    for k in range(max_k+1):
        phi[k] = x_tilde**k
        dphi[k] = 2*k/h *(x_tilde**(k-1)) 
    return phi, dphi

def get_matrices(pd, order, elem):
    x_bar = cell_center(pd, elem)
    mass = np.zeros((order+1, order+1))
    stiff = np.zeros((order+1, order+1))
    qps, qws, nn = integrate(2*order + 2, pd.h, x_bar)
    for ii in range(nn): 
        phi, dphi = basis(qps[ii], x_bar, pd.h, order)
        mass += qws[ii] * np.outer(phi.T,phi) # Mass matrix
        stiff += qws[ii] * np.outer(dphi.T,dphi) # stiffness matrix
    return mass, stiff
    
def fun_rhs_vector(pd, order, elem, fun):
    rhs = np.zeros((order+1))
    x_bar = cell_center(pd, elem)
    qps, qws, nn = integrate(2*order + 6, pd.h, x_bar)
    for ii in range(nn): 
        phi = basis(qps[ii], x_bar, pd.h, order)[0]
        rhs += qws[ii] * phi * fun(qps[ii])
    return rhs

def hho_reduction(pd, elem, fun):
    # Listing 8.2 Possible implementation of the reduction operator in 1D.
    # Simple L^2 projection at order p !
    # contains also projection on face dofs   
    order = pd.K
    mass = get_matrices(pd, order, elem)[0]
    v_rhs = fun_rhs_vector(pd, order, elem, fun)
    I = np.zeros(pd.K+3)
    I[:pd.K+1] = np.linalg.solve(mass,v_rhs) # Project on the cell
    # Project on faces: in 1D we just need to evaluate the function at the faces
    xF1, xF2 = face_centers(pd, elem)
    I[pd.K+1] = fun(xF1); 
    I[pd.K+2] = fun(xF2); 
    return I

def hho_reconstruction(pd,elem): 
    # Listing 8.3 Possible implementation of the reconstruction operator in 1D.
    
    # R translates a vector from P_d^{k} to P_d^{k+1}
    # if v is a vector of HHO solution then R@v is in P_d^{k+1} 
    # hence we have K = R^T @ K+ @ R, where K+ is the stiffness matrix of order K+1
    # such R it translates the vector of unknowns into a higher degree space 
    # issue is that we have to remove the constant term from the stiffness matrix
    # and then add it back again. So we find first K*
    # K* R = T - F1 - F2

    # A := R.T @ K_star @ R
    #      2x4 @  4x4   @ 4x2 (linear K=0)
    # where K_star is a standard stiffness matrix of order K+1 with constant removed 
            
    x_bar = cell_center(pd, elem)
    xF1, xF2 = face_centers(pd, elem)
    
    # stiffness matrix of order k+1 (3x3 in linear)
    # only for the cell dofs !
    order = pd.K+1
    stiff_mat = get_matrices(pd, order, elem)[1]
    # then obtain K* --> remove the constant function
    # for linear 3x3 --> 2x2 matrix (containing linear and quadratic functions)
    gr_lhs = stiff_mat[1:, 1:] # % Left-hand side

    # Compute the right-hand side T - F1 - F2 
    # higher_degree_cell_dofs* x #elem_dofs
    # for linear k=1, this is a 2x4 matrix 
    gr_rhs = np.zeros((pd.K+1, pd.K+3))  
    
    # Right-hand side, cell part
    # q  is in P_d*^{k+1} --> 1:       same space as K* --> including super convergent, but remove the constant
    # vT is in P_d^k      --> 0:pd.K+1 (take columns up to order k only, including constant)
    gr_rhs[:,0:pd.K+1] = stiff_mat[1:,0:pd.K+1]; # (∇ q, ∇ vT)L2(T ) 
    # Compute T using cell basis functions at the faces vT|F1, vT|F2
    phiF1, dphiF1 = basis(xF1, x_bar, pd.h, pd.K+1)
    phiF2, dphiF2 = basis(xF2, x_bar, pd.h, pd.K+1)
    
    # Right-hand side, boundary part F1 and F2
    # minus comes from the normal, remind that q is in k+1 space
    gr_rhs[:, :pd.K+1] += + np.outer(dphiF1[1:].T,phiF1[:pd.K+1]) # (nT ·∇ q, vT)L2(F1)
    gr_rhs[:, :pd.K+1] += - np.outer(dphiF2[1:].T,phiF2[:pd.K+1]) # (nT ·∇ q, vT)L2(F2)
    # face dofs are at K+1 and K+2 
    gr_rhs[:, pd.K+1] = - dphiF1[1:] # (nT ·∇ q, vF)L2(F1)
    gr_rhs[:, pd.K+2] = + dphiF2[1:] # (nT ·∇ q, vF)L2(F2)
    
    R = np.linalg.solve(gr_lhs,gr_rhs); # Solve problem (up to a constant)
    A = gr_rhs.T@R # Compute (∇RT (·),∇RT (·))L2(T )

    # Note that here we still miss the contribution from the constant term
    return A, R

def get_phi_norm(pd, elem, order):
    phi_norm = np.zeros(order + 1)
    x_bar = cell_center(pd, elem)
    qps, qws, nn = integrate(2*order, pd.h, x_bar)
    for ii in range(nn): 
        phi = basis(qps[ii], x_bar, pd.h, order)[0]
        phi_norm += qws[ii] * phi 
    return phi_norm

def hho_vector_from_sol(pd,v_sol):
    # reconstruct HHO vector in P_d^{k+1}
    # from solution vector in P_d^{k} using reconstruction operator R
    v_hho = np.zeros(pd.n_dofs_super)
    for elem in range(pd.n_cells):
        elem_dofs, cell_dofs, face_dofs = get_dofs(pd,elem)
        elem_dofs_super, cell_dofs_super, face_dofs_super = get_dofs_super(pd,elem)
        A_elem, R_elem = hho_reconstruction(pd, elem)
        vstar = R_elem @ v_sol[elem_dofs]
        vT = v_sol[cell_dofs]
        # need the constant term
        norm_vstar = compute_cell_vector_norm(pd, elem, vstar, pd.K+1, is_constant=False)        
        norm_vT =    compute_cell_vector_norm(pd, elem, vT   , pd.K  , is_constant=True)
        v_hho_T = np.zeros(vstar.shape[0]+1)
        v_hho_T[1:] = vstar
        v_hho_T[0] = (norm_vT - norm_vstar)/pd.h
        v_hho[cell_dofs_super] = v_hho_T
        v_hho[face_dofs_super] = v_sol[face_dofs]

    return v_hho

def compute_cell_vector_norm(pd, elem, vector, order, is_constant = False):
    # compute the norm of a vector on a cell - up to a constant
    if is_constant:
        phi_start_index = 0
    else:
        phi_start_index = 1
    vector_norm = 0
    x_bar = cell_center(pd, elem)
    qps, qws, nn = integrate(2*order + 2, pd.h, x_bar)
    for ii in range(nn): 
        phi = basis(qps[ii], x_bar, pd.h, order)[0]
        vector_norm += qws[ii] * np.dot(vector, phi[phi_start_index:]) 
    return vector_norm

def get_vstar_norm(pd, elem, vstar):
    order = pd.K+1 # super convergent
    vstar_norm = 0
    x_bar = cell_center(pd, elem)
    qps, qws, nn = integrate(2*order, pd.h, x_bar)
    for ii in range(nn): 
        phi = basis(qps[ii], x_bar, pd.h, order)[0]
        vstar_norm += qws[ii] * np.dot(vstar,phi[1:]) 
    return vstar_norm

def hho_stabilization(pd, elem, R):
    """
    Compute the stabilization matrix S for the HHO method.
    """
    x_bar = cell_center(pd, elem)
    xF1, xF2 = face_centers(pd, elem)
    order_sup = pd.K+1
    mass_mat = get_matrices(pd, order_sup, elem)[0]

    # Compute the term tmp1 = uT − Σ RT(ˆuT) - M^-1*Q*R
    M = mass_mat[:pd.K+1,:pd.K+1] # M is the p-th order mass matrix
    Q = mass_mat[:pd.K+1,1:pd.K+2] 
    tmp1 = - np.linalg.solve(M,Q@R)
    tmp1[:pd.K+1, :pd.K+1] += np.eye(pd.K+1)

    # Compute the stabilization matrix S on F1
    phiF1 = basis(xF1, x_bar, pd.h, order_sup)[0] # value of cell basis at F1
    Mi = 1
    Ti = phiF1[1:] # super convergent remove constant
    Ti_tilde = phiF1[:pd.K+1]
    tmp2 = (1 / Mi) * Ti @ R # tmp2 = Σ RT(ˆuT)
    tmp2[pd.K+1] = tmp2[pd.K+1]-1; # tmp2 = Σ RT(ˆuT) − uF
    tmp3 = (1 / Mi) * (Ti_tilde @ tmp1) # tmp3 = Σ(uT − Σ RT(ˆuT))
    Si = tmp2 + tmp3 # Si = Σ RT(ˆuT) − uF + Σ(uT − Σ RT(ˆuT))
    S1 = np.outer(Si*Mi,Si) / pd.h # Accumulate on S

    # Compute the stabilization matrix S on F2
    phiF2 = basis(xF2, x_bar, pd.h, order_sup)[0] # value of cell basis at F1
    Mi = 1
    Ti = phiF2[1:] # remove constant
    Ti_tilde = phiF2[:pd.K+1]
    tmp2 = (1 / Mi) * Ti @ R # tmp2 = Σ RT(ˆuT)
    tmp2[pd.K+2] = tmp2[pd.K+2]-1; # tmp2 = Σ RT(ˆuT) − uF
    tmp3 = (1 / Mi) * (Ti_tilde @ tmp1) # tmp3 = Σ(uT − Σ RT(ˆuT))
    Si = tmp2 + tmp3 # Si = Σ RT(ˆuT) − uF + Σ(uT − Σ RT(ˆuT))
    S2 = np.outer(Si*Mi,Si) / pd.h # Accumulate on S

    S = S1 + S2

    return S

def HHO(pd, v_func, is_plot = False):
    
    # hho reduction --> simple L^2
    I = np.zeros(pd.n_dofs)
    # hho stiffness - using reconstruction R^T @ K* @ R and then adding the constant term
    A = np.zeros((pd.n_dofs,pd.n_dofs))
    S = np.zeros_like(A)
    
    n_sampling = 100
    X = np.zeros(pd.n_cells*n_sampling)
    V_HHO = np.zeros_like(X)
    V_L2 = np.zeros_like(X)
    V_REF = np.zeros_like(X)
    epsilon = 0
    for elem in range(pd.n_cells):
        x_bar = cell_center(pd, elem)        
        elem_dofs, cell_dofs, face_dofs = get_dofs(pd,elem)

        # hho_reduction is a simple L2 projection at order p
        I_elem = hho_reduction(pd, elem, v_func)
        I[elem_dofs] += I_elem

        # Assemble HHO stifness matrix --> A = (∇RT (·),∇RT (·))L2(T ) - still missing the constant term though
        A_elem, R_elem = hho_reconstruction(pd, elem)
        A[np.ix_(elem_dofs, elem_dofs)] += A_elem 
    
        # get the super convergent solution in the cell (up to a constant)
        # I is a solution vector in P_d^k, containing the L2 projection of v(x) in each cell
        # R is a reconstruction matrix from P_d^k to P_d^{k+1} - up to a constant
        v_star = R_elem @ I_elem  

        # now adjust the constant    
        norm_vstar = get_vstar_norm(pd, elem, v_star)        
        xF1, xF2 = face_centers(pd,elem)
        norm_vT, error = quad(v_func, xF1, xF2, epsabs=1e-12, epsrel=1e-12)
        v = np.zeros(v_star.shape[0]+1)
        v[1:] = v_star
        v[0] = (norm_vT - norm_vstar)/pd.h 

        # # let's try to modify R such that it contains the constant term
        # # this implies that we append a row to R at the top
        # R_elem_cst = np.vstack((np.zeros_like(R_elem[0]),R_elem))
        # # then we compute int phi for each shape function (normalization)
        # phi_norm = get_phi_norm(pd, elem, pd.K+1)
        # R_elem_cst[0,0] = -phi_norm[1:]/phi_norm[0]
        # # now we can compute the super convergent solution
        # v_2 = R_elem_cst @ I_elem - 
        # did not succeed in doing it this way, the constant term is not correct

        # also get a classical projection (L^2) for comparison
        mass = get_matrices(pd, pd.K, elem)[0]
        v_rhs = fun_rhs_vector(pd, pd.K, elem, v_func)
        coeffs_L2 = np.linalg.solve(mass,v_rhs)

        # plot it 
        x_plot = np.linspace(xF1,xF2,100)
        for i, x in enumerate(x_plot):
            # order k+1
            phi = basis(x, x_bar, pd.h, pd.K+1)[0]
            V_HHO[elem*n_sampling+i] = np.dot(v,phi)
            # order k (L^2)
            phi = basis(x, x_bar, pd.h, pd.K)[0]
            V_L2[elem*n_sampling+i] = np.dot(coeffs_L2,phi)
            X[elem*n_sampling+i] = x
        
        # target function
        v_ref = [v_func(x) for x in x_plot]
        V_REF[elem*n_sampling:(elem+1)*n_sampling] = v_ref

        # stabilization matrix
        S_elem = hho_stabilization(pd, elem, R_elem)
        S[np.ix_(elem_dofs, elem_dofs)] += S_elem

        # compute the error on the jump
        epsilon += I_elem.T @ S_elem @ I_elem

    # compute and print errors
    error_L2 = np.linalg.norm(V_L2 - V_REF)/np.linalg.norm(V_REF)*100
    error_HHO = np.linalg.norm(V_HHO - V_REF)/np.linalg.norm(V_REF)*100
    error_jump = epsilon/np.linalg.norm(V_REF)*100

    print('error on the jump/stabilization (norm): ', error_jump)
    print('relative L^2 reconstruction error (in percent): ', error_L2)
    print('relative HHO reconstruction error (in percent): ', error_HHO)

    # plot the result 
    if is_plot:
        plt.figure(figsize=(10,6))
        plt.plot(X,V_REF,'r-',label='v(x) - reference')
        plt.plot(X,V_L2,'b--',label='L^2 projection of v(x)')
        plt.plot(X,V_HHO,'g--',label='HHO reconstruction of v(x)')
        plt.legend()
        plt.savefig("single_run.png")  # Save as PNG   

    Stiffness = A + S
    condition_number = np.linalg.cond(Stiffness)
    print(f"Condition number of A + S: {condition_number}")

    # ############################################
    # # Try now a global reduction V_L2 = PHI @ I
    # ############################################
    # n_sampling = 100
    # PHI = np.zeros((pd.n_cells*n_sampling,pd.n_dofs))
    # for elem in range(pd.n_cells):
    #     # from (8.11)    
    #     x_bar = cell_center(pd, elem)
    #     xF1, xF2 = face_centers(pd,elem)
    #     elem_dofs, cell_dofs, face_dofs = get_dofs(pd,elem) 
    #     x_plot = np.linspace(xF1,xF2,n_sampling)
    #     for i, x in enumerate(x_plot):
    #         phi = basis(x, x_bar, pd.h, pd.K)[0]
    #         PHI[elem*n_sampling+i,cell_dofs] = phi
        
    # V_L2 = PHI @ I

    return error_L2, error_HHO, error_jump, Stiffness


###############################################################
###############################################################
###############################################################
###############################################################
if __name__=='__main__':

    pd_single_run = ProblemDefinition()
    pd_single_run.domain_length = 2
    pd_single_run.K = 0
    pd_single_run.n_cells = 2
    pd_single_run.initialize()
    
    # define target function & plot it 
    v_func = lambda x: np.sin(np.pi*x)

    # SINGLE RUN 
    HHO(pd_single_run, v_func, is_plot = False)

    # SINGLE RUN WITH DIRICHLET BC
    Stiffness = HHO(pd_single_run, v_func, is_plot = True)[3]
    rhs_vector = np.zeros(pd_single_run.n_dofs)
    rhs_vector[0] = 0; rhs_vector[-1] = 10
    b_start_dof = pd_single_run.n_cells*(pd_single_run.K+1)
    Stiffness[b_start_dof,:] = 0; Stiffness[b_start_dof,b_start_dof] = 1
    b_end_dof = pd_single_run.n_dofs-1
    Stiffness[b_end_dof,:] = 0; Stiffness[b_end_dof,b_end_dof] = 1
    sol = np.linalg.solve(Stiffness,rhs_vector)
    sol_hho = hho_vector_from_sol(pd_single_run,sol)
    
    condition_number = np.linalg.cond(Stiffness)
    print(f"Stiffness matrix condition number: {condition_number}")   

    n_sampling = 100
    X = np.zeros(pd_single_run.n_cells*n_sampling)
    plot_sol_hho = np.zeros(pd_single_run.n_cells*n_sampling)
    plot_sol     = np.zeros_like(plot_sol_hho)
    for elem in range(pd_single_run.n_cells):
        x_bar = cell_center(pd_single_run, elem)
        xF1, xF2 = face_centers(pd_single_run,elem)
        elem_dofs, cell_dofs, face_dofs = get_dofs(pd_single_run,elem)
        elem_dofs_super, cell_dofs_super, face_dofs_super = get_dofs_super(pd_single_run,elem)
        x_plot = np.linspace(xF1,xF2,n_sampling)
        for i, x in enumerate(x_plot):
            phi = basis(x, x_bar, pd_single_run.h, pd_single_run.K)[0]
            phi_super = basis(x, x_bar, pd_single_run.h, pd_single_run.K+1)[0]
            plot_sol[elem*n_sampling+i] = np.dot(sol[cell_dofs],phi)
            plot_sol_hho[elem*n_sampling+i] = np.dot(sol_hho[cell_dofs_super],phi_super)           
            X[elem*n_sampling+i] = x 
    
    plt.figure(figsize=(10,6))
    plt.plot(X, plot_sol,'b-',label='solution with Dirichlet BC (not super convergent)')
    plt.plot(X, plot_sol_hho,'r-',label='HHO solution with Dirichlet BC')
    plt.legend()
    plt.show()
    plt.pause(0.001)
    plt.savefig("dirichlet_bc.png")  # Save as PNG


    # CONVERGENCE STUDY
    pd_conv_study = ProblemDefinition()
    pd_conv_study.domain_length = 1
    
    #h = np.logspace(0,-2,100)
    N_CELLS = np.array([2,3,4,5,6,7,8,9,10,11,12,14,16,18,20,22,26,28,30,40,50,60,80]).astype(int)
    print(N_CELLS)
    ORDERS = np.array([0,1,2,3])

    ERROR_HHO = np.zeros((len(ORDERS),len(N_CELLS)))
    ERROR_L2 = np.zeros((len(ORDERS),len(N_CELLS)))
    ERROR_jump = np.zeros((len(ORDERS),len(N_CELLS)))

    for i_order, K in enumerate(ORDERS):
        for i_cell, n_cells in enumerate(N_CELLS):
            pd_conv_study.n_cells = n_cells
            pd_conv_study.K = K
            pd_conv_study.initialize()
            error_L2, error_HHO, error_jump = HHO(pd_conv_study, v_func, is_plot = False)[0:3]
            ERROR_HHO[i_order,i_cell] = error_HHO
            ERROR_L2[i_order,i_cell] = error_L2
            ERROR_jump[i_order,i_cell] = error_jump

    # Plotting in log scale
    plt.figure(figsize=(6,10))
    for i_order, K in enumerate(ORDERS):
        plt.loglog(1/N_CELLS, ERROR_HHO[i_order,:],'k',linewidth=2)
        plt.loglog(1/N_CELLS, ERROR_L2[i_order,:], 'b--')
        plt.loglog(1/N_CELLS, ERROR_jump[i_order,:], 'r--')
    plt.xlabel('1 / Number of Cells')
    plt.ylabel('Error (in percent)')
    # plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.savefig("convergence_study.png")
    #plt.pause(0.001)
    #plt.show()
    print(K)