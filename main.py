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
    p, w = sp.special.roots_legendre(max(order,2))
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
    # for linear this is a 2x2 matrix (containing linear and quadratic functions)
    gr_lhs = stiff_mat[1:, 1:] # % Left-hand side

    # Set up local Neumann problem
    # for linear k=1, this is a 2x4 matrix 
    gr_rhs = np.zeros((pd.K+1, pd.K+3)) # q 
    
    # Right-hand side, cell part
    # vT is in P_d^k      --> 0:pd.K+1 (take columns up to order k only, including constant)
    # q  is in P_d*^{k+1} --> 1:       same space as K* --> including super convergent, but remove the constant
    gr_rhs[:,0:pd.K+1] = stiff_mat[1:,0:pd.K+1]; # (∇ vT, ∇ q)L2(T ) 
    # 
    phiF1, dphiF1 = basis(xF1, x_bar, pd.h, pd.K+1)
    phiF2, dphiF2 = basis(xF2, x_bar, pd.h, pd.K+1)
    
    # Right-hand side, boundary part
    # minus comes from the normal 
    gr_rhs[:, :pd.K+1] += + np.outer(dphiF1[1:].T,phiF1[:pd.K+1]) # (vT , nT ·∇ q)L2(F1)
    gr_rhs[:, :pd.K+1] += - np.outer(dphiF2[1:].T,phiF2[:pd.K+1]) # (vT , nT ·∇ q)L2(F2)
    
    gr_rhs[:, pd.K+1] = - dphiF1[1:] # (vF , nT ·∇ q)L2(F1)
    gr_rhs[:, pd.K+2] = + dphiF2[1:] # (vF , nT ·∇ q)L2(F2)
    
    R = np.linalg.solve(gr_lhs,gr_rhs); # Solve problem (up to a constant)
    A = gr_rhs.T@R # Compute (∇RT (·),∇RT (·))L2(T )
    return A, R

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

    # Compute the term tmp1 = uT − Σ RT(ˆuT)
    M = mass_mat[:pd.K+1,:pd.K+1] # M is the p-th order mass matrix
    Q = mass_mat[:pd.K+1,1:pd.K+2]
    tmp1 = - np.linalg.solve(M,Q@R)
    tmp1[:pd.K+1, :pd.K+1] += np.eye(pd.K+1)

    # Compute the stabilization matrix S on F1
    phiF1 = basis(xF1, x_bar, pd.h, order_sup)[0] # value of cell basis at F1
    Mi = 1
    Ti = phiF1[1:] # remove constant
    Ti_tilde = phiF1[:pd.K+1]
    tmp2 = (1 / Mi) * Ti.T @ R # tmp2 = Σ RT(ˆuT)
    tmp2[pd.K+2] = tmp2[pd.K+2]-1; # tmp2 = Σ RT(ˆuT) − uF
    tmp3 = (1 / Mi) * (Ti_tilde @ tmp1) # tmp3 = Σ(uT − Σ RT(ˆuT))
    Si = tmp2 + tmp3 # Si = Σ RT(ˆuT) − uF + Σ(uT − Σ RT(ˆuT))
    S1 = np.outer(Si.T*Mi,Si) / pd.h # Accumulate on S

    # Compute the stabilization matrix S on F2
    phiF2 = basis(xF2, x_bar, pd.h, order_sup)[0] # value of cell basis at F1
    Mi = 1
    Ti = phiF2[1:] # remove constant
    Ti_tilde = phiF2[:pd.K+1]
    tmp2 = (1 / Mi) * Ti.T @ R # tmp2 = Σ RT(ˆuT)
    tmp2[pd.K+2] = tmp2[pd.K+2]-1; # tmp2 = Σ RT(ˆuT) − uF
    tmp3 = (1 / Mi) * (Ti_tilde @ tmp1) # tmp3 = Σ(uT − Σ RT(ˆuT))
    Si = tmp2 + tmp3 # Si = Σ RT(ˆuT) − uF + Σ(uT − Σ RT(ˆuT))
    S2 = np.outer(Si.T*Mi,Si) / pd.h # Accumulate on S

    S = S1 + S2

    return S

###############################################################
if __name__=='__main__':

    pd = ProblemDefinition()
    pd.domain_length = 1
    pd.K = 2
    pd.n_cells = 2
    pd.compute_element_size()
    pd.compute_n_dofs()
    pd.compute_n_dofs_super()
    pd.print_header()

    # define target function & plot it 
    v_func = lambda x: np.sin(np.pi*x)

    # hho reduction --> simple L^2
    I = np.zeros(pd.n_dofs)
    # hho stiffness - using reconstruction R^T @ K* @ R and then adding the constant term
    A = np.zeros((pd.n_dofs,pd.n_dofs))
    S = np.zeros_like(A)
    
    X = []
    V_HHO = []
    V_L2 = []
    V_REF = []
    for elem in range(pd.n_cells):
        x_bar = cell_center(pd, elem)        
        elem_dofs, cell_dofs, face_dofs = get_dofs(pd,elem)

        # hho_reduction is a simple L2 projection at order p
        I_elem = hho_reduction(pd, elem, v_func)
        I[elem_dofs] += I_elem

        # Assemble HHO stifness matrix --> A = (∇RT (·),∇RT (·))L2(T )
        A_elem, R_elem = hho_reconstruction(pd, elem)
        A[np.ix_(elem_dofs, elem_dofs)] = A_elem
    
        # get the super convergent solution in the cell (up to a constant)
        v_star = R_elem @ I_elem  

        # now adjust the constant    
        norm_vstar = get_vstar_norm(pd, elem, v_star)        
        xF1, xF2 = face_centers(pd,elem)
        norm_vT, error = quad(v_func, xF1, xF2, epsabs=1e-12, epsrel=1e-12)
        v = np.zeros(v_star.shape[0]+1)
        v[1:] = v_star
        v[0] = (norm_vT - norm_vstar)/pd.h

        # also get a classical projection (L^2) for comparison
        mass = get_matrices(pd, pd.K, elem)[0]
        v_rhs = fun_rhs_vector(pd, pd.K, elem, v_func)
        coeffs_L2 = np.linalg.solve(mass,v_rhs)

        # plot it 
        x_plot = np.linspace(xF1,xF2,100)
        v_hho = np.zeros_like(x_plot)
        v_L2 = np.zeros_like(x_plot)
        for i, x in enumerate(x_plot):
            # order k+1
            phi = basis(x, x_bar, pd.h, pd.K+1)[0]
            v_hho[i] = np.dot(v,phi)
            # order k (L^2)
            phi = basis(x, x_bar, pd.h, pd.K)[0]
            v_L2[i] = np.dot(coeffs_L2,phi)
        
        # target function
        v_ref = [v_func(x) for x in x_plot]

        X.extend(x_plot.tolist())
        V_HHO.extend(v_hho.tolist())
        V_L2.extend(v_L2.tolist())
        V_REF.extend(v_ref)

        # stabilization matrix
        S_elem = hho_stabilization(pd, elem, R_elem)
        S[np.ix_(elem_dofs, elem_dofs)] = S_elem

    # plot the result 
    plt.figure(figsize=(10,6))
    plt.plot(X,V_REF,'r-',label='v(x) - reference')
    plt.plot(X,V_L2,'b--',label='L^2 projection of v(x)')
    plt.plot(X,V_HHO,'g--',label='HHO reconstruction of v(x)')
    plt.legend()
    plt.savefig("plot.png")  # Save as PNG   

    print(A + S)
    condition_number = np.linalg.cond(A + S)
    print(f"Condition number of A + S: {condition_number}")

