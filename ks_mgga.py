from __future__ import division
import numpy as np
from ks_main import evaluate_atomic_orbital

def evaluate_grad_gto(gto, p, axis=0):
    #compute the value of gaussian gradient density at (x,y,z)

    local_momentum = gto.momentum.copy()

    #evaluate exponential derivative term
    local_momentum[axis] += 1
    A = (p - gto.center) ; L = np.prod(A**local_momentum, axis=1).reshape(-1,1)
    phi = np.sum(-2.0*gto.ex*L*gto.normal*gto.co*np.exp(-gto.ex*np.sum(A*A, axis=1).reshape(-1,1)), axis=1)

    #if angular momentum of axis is not 0, evaluate non-exponential term derivative
    if gto.momentum[axis] != 0:
        local_momentum[axis] -= 2
        L = np.prod(A**local_momentum, axis=1).reshape(-1,1)
        phi += np.sum(gto.momentum[axis]*L*gto.normal*gto.co*np.exp(-gto.ex*np.sum(A*A, axis=1).reshape(-1,1)), axis=1)

    return phi.reshape(-1,1)

def evaluate_grad_ao_axis(basis, p, axis=0):
    #evaluate the GTO gradient of the atomic orbitals in axis direction

    grad_ao_axis = []

    for i in basis:
        grad_ao_axis.append(evaluate_grad_gto(i, p, axis))

    return np.hstack(grad_ao_axis)

def evaluate_ao(basis, p):
    #evaluate ao gradients for all directions

    ngp = p.shape[0] ; nbf = len(basis)
    grad_ao = np.zeros((7, ngp, nbf))

    #get basis
    grad_ao[0] = evaluate_atomic_orbital(basis, p)
    #get gradiants
    for axis in range(3):
        grad_ao[axis+1] = evaluate_grad_ao_axis(basis, p, axis)
    #get Laplacian
    for axis in range(3):
        grad_ao[axis+4] = evaluate_laplace_ao_axis(basis, p, axis)

    return grad_ao

def evaluate_rho_gga(d, grad_ao):
    #evaluate the density and gradient over grid shells - sigmas

    d = d + d.T ; ngp = grad_ao.shape[1]
    grad_rho = np.zeros((6, ngp))
    c = []

    #density
    c.append(np.einsum('pr,rq->pq', grad_ao[0], d, optimize=True))
    grad_rho[0] = np.einsum('pi,pi->p', c[0], grad_ao[0], optimize=True)

    #gradient density
    for axis in range(3):
        grad_rho[axis+1] =  2.0 * np.einsum('pi,pi->p', c[0], grad_ao[axis+1], optimize=True)
        c.append(np.einsum('pr,rq->pq', grad_ao[axis+1], d.T, optimize=True))
        grad_rho[5] += np.einsum('pi,pi->p', c[axis+1], grad_ao[axis+1], optimize=True)


    #Laplacian
    Laplacian = grad_ao[4] + grad_ao[5] + grad_ao[6]
    grad_rho[4] = np.einsum('pi,pi->p', c[0], Laplacian, optimize=True) + grad_rho[5]
    grad_rho[4] *= 2.0 ; grad_rho[5] *= 0.5

    return np.where(np.abs(grad_rho) < 1e-20, 0, grad_rho)

def evaluate_laplace_gto(gto, p, axis=0):
    #compute the value of gaussian gradient density at (x,y,z)

    local_momentum = gto.momentum.copy()

    #evaluate exponential derivative term
    local_momentum[axis] += 2
    A = (p - gto.center) ; L = lambda momentum : np.prod(A**momentum, axis=1).reshape(-1,1)

    phi  = 4.0*np.sum(gto.ex*gto.ex*L(local_momentum)*gto.normal*gto.co*np.exp(-gto.ex*np.sum(A*A, axis=1).reshape(-1,1)), axis=1)
    local_momentum[axis] -= 2 ; factor = 2 * gto.momentum[axis] + 1
    phi -= np.sum(2.0*(factor)*gto.ex*L(local_momentum)*gto.normal*gto.co*np.exp(-gto.ex*np.sum(A*A, axis=1).reshape(-1,1)), axis=1)

    #if angular momentum of axis is not 0, evaluate non-exponential term derivative
    if gto.momentum[axis] > 1:
        local_momentum[axis] -= 2 ; factor = gto.momentum[axis] * (gto.momentum[axis] - 1.0)
        phi += np.sum(factor*L(local_momentum)*gto.normal*gto.co*np.exp(-gto.ex*np.sum(A*A, axis=1).reshape(-1,1)), axis=1)

    return phi.reshape(-1,1)

def evaluate_laplace_ao_axis(basis, p, axis=0):
    #evaluate the GTO gradient of the atomic orbitals in axis direction

    laplace_ao_axis = []

    for i in basis:
        laplace_ao_axis.append(evaluate_laplace_gto(i, p, axis))

    return np.hstack(laplace_ao_axis)
   
def meta_gga_functional(name, rho):
    #evaluate the Made Simple meta-Generalized Gradient Approximation functional

    if name == 'MS':

        rho, dx, dy, dz, laplacian , tau = rho[:6]

        sigma = dx*dx+dy*dy+dz*dz

        #remove zeros - for divide
        epsilon = 1.0e-30
        rho = rho + epsilon   ; sigma = sigma + epsilon 

        #parameters
        KAPPA = 0.29 ; C = 0.28771 ; B = 1.0

        #Local Density Approximation
        ex_lda = -(3.0/4.0) * pow(3.0/np.pi,1.0/3.0) * pow(rho,1.0/3.0)

        #dimensionless gradient (squared)
        p = sigma / (4.0 * pow(3.0,2.0/3.0) * pow(np.pi,4.0/3.0) * pow(rho,8.0/3.0))

        #gradient expansion coefficient
        MU_ge = 10.0/81.0

        #von Weizsacker kinetic energy density
        tau_w = sigma / (8.0 * rho)

        #uniform gas kinetic energy density
        tau_uniform = 0.3 * pow(3*np.pi*np.pi,2.0/3.0) * pow(rho,5.0/3.0)

        #dimensionless inhomogeneity parameter
        alpha = (tau - tau_w)/tau_uniform

        #enhancement factors
        f_lower = 1.0 + KAPPA - KAPPA / (1.0 + (p*MU_ge + C)/KAPPA)
        f_upper = 1.0 + KAPPA - KAPPA / (1.0 + MU_ge*p/KAPPA)
        f_interpolator = pow(1.0-alpha*alpha,3) / (1.0 + pow(alpha,3) + pow(alpha,6)*B)

        f_enhancement = f_interpolator * (f_lower - f_upper) + f_upper

        #libxc compatible so include LDA energy
        ex = ex_lda * f_enhancement

        alpha[abs(alpha) > 100] = 0

        #------potential - rho
        #p-derivatives
        dfl_dp = MU_ge * pow(1.0 + (MU_ge*p + C)/KAPPA,-2)
        dfu_dp = MU_ge * pow(1.0 + MU_ge*p/KAPPA,-2)
        dfa_dp = dfu_dp + f_interpolator * (dfl_dp -  dfu_dp)

        #alpha-derivatives
        da_dn = (-5.0 * tau + 8.0 * tau_w) / (3.0 * tau_uniform * rho)

        #rho-derivative d_dn = d_dp*dp_dn + d_da*da_dn
        dp_dn = -8.0*p/(3.0*rho)  
        dlda_dn = -pow(3.0/np.pi,1.0/3.0)  / (4.0 * pow(rho,2.0/3.0))

        DFx_dn = dfa_dp * dp_dn
        DFx_dn -= 6.0  *  pow(1 - alpha*alpha,2) / (1.0 + pow(alpha,3)  + B * pow(alpha,6)) * (f_lower - f_upper) * alpha * da_dn 
        DFx_dn -= pow(1 - alpha*alpha,3) * 1.0 / pow(1.0 + pow(alpha,3)  + B * pow(alpha,6),2) * \
                  (f_lower - f_upper) * 3.0 * (pow(alpha,2) + 2.0 * B * pow(alpha,5)) * da_dn

        dex_dn = dlda_dn*f_enhancement + ex_lda*DFx_dn
        v_rho_x = rho * dex_dn + ex

        #-------potential - sigma

        #sigma derivatives
        dp_ds = p / sigma
        da_ds = -1.0 / (8.0 * rho * tau_uniform)

        DFx_ds  = dfa_dp * dp_ds
        DFx_ds -= 6.0  *  pow(1 - alpha*alpha,2) / (1.0 + pow(alpha,3)  + B * pow(alpha,6)) * (f_lower - f_upper) * alpha * da_ds 
        DFx_ds -= pow(1 - alpha*alpha,3) * 1.0 / pow(1.0 + pow(alpha,3)  + B * pow(alpha,6),2) * \
                  (f_lower - f_upper) * 3.0 * (pow(alpha,2) + 2.0 * B * pow(alpha,5)) * da_ds

        dex_ds = ex_lda*DFx_ds
        v_sigma_x = rho * dex_ds

        #-------potential - Laplacian
        v_Laplacian_x = np.zeros_like(v_rho_x)

        #-------potential - tau

        #tau derivatives
        dp_dt = 0
        da_dt = 1 / tau_uniform 

        DFx_dt = dfa_dp * dp_dt
        DFx_dt -= 6  *  pow(1 - alpha*alpha,2) / (0.1e1 + pow(alpha,3)  + B * pow(alpha,6)) * (f_lower - f_upper) * alpha * da_dt
        DFx_dt -= pow(1 - alpha*alpha,3) * 0.1e1 / pow(0.1e1 + pow(alpha,3)  + B * pow(alpha,6),2) * \
                (f_lower - f_upper) * 3 * (pow(alpha,2) + 2 * B * pow(alpha,5)) * da_dt

        dex_dt = ex_lda*DFx_dt
        v_tau_x = rho * dex_dt

        #numerical clean-up
        ex[abs(ex) < 1e-8]               = 0
        v_rho_x[abs(v_rho_x) < 1e-8]     = 0
        v_sigma_x[abs(v_sigma_x) < 1e-8] = 0
        v_tau_x[abs(v_tau_x) < 1e-8]     = 0

    return ex , (v_rho_x, v_sigma_x, v_Laplacian_x, v_tau_x)
