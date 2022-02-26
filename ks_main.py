from __future__ import division
from ks_grid import GRID
from ks_util import dii_subspace, out
import ks_mgga as mgga 
import numpy as np

units = {'bohr->angstrom' : 0.52917721092}
elements = {'H':1, 'O':8}

class atom(object):
    #molecule class

    def __init__(self, symbol, center):

        self.symbol = symbol
        self.number = elements[symbol]
        self.center = np.array(center)

class orbital(object):
    #basis class

    def __init__(self, atom, momentum, exponents, coefficients, normalization, atoms):

        self.atom     = atom
        self.momentum = np.array(momentum)
        self.ex       = np.array(exponents)
        self.co       = np.array(coefficients)
        self.normal   = np.array(normalization)
        self.center   = atoms[atom].center

def nuclear_repulsion(mol):
    #nuclear repulsion

    eNuc = 0.0

    atoms = len(mol)
    for i in range(atoms):
        for j in range(i+1, atoms):
            r = np.linalg.norm(mol[i].center-mol[j].center)
            eNuc += mol[i].number*mol[j].number / r

    return eNuc

def evaluate_gto(gto, p):
    #compute the value of gaussian density at (x,y,z)

    A = (p - gto.center) ; L = np.prod(A**gto.momentum, axis=1).reshape(-1,1)

    phi = np.sum(L*gto.normal*gto.co*np.exp(-gto.ex*np.sum(A*A, axis=1).reshape(-1,1)), axis=1)

    return phi.reshape(-1,1)

def evaluate_atomic_orbital(basis, p):
    #evaluate the GTO of the atomic orbitals of the molecule

    ao = []
    for i in basis:
        ao.append(evaluate_gto(i, p))

    return np.hstack(ao)

def evaluate_rho_lda(d, ao):
    #evaluate the density over grid shells

    d = d + d.T
    ao_density = np.einsum('pr,rq->pq', ao, d, optimize=True)

    return np.einsum('pi,pi->p', ao, ao_density, optimize=True)

def get_weighted_potential(vxc, weights, rho):
 
    #weight the potential

    vrho, vgamma = vxc[:2]
    ngrid = vrho.size

    weighted_potential = np.empty((4,ngrid))

    weighted_potential[0]  = weights * vrho
    weighted_potential[1:] = (weights * vgamma * 2) * rho[1:4]
    weighted_potential[0] *= 0.5  

    return weighted_potential


def evaluate_vx_mgga(vxc, ao, weights, rho):
    #construct exchange-correlation matrix

    vrho, vgamma, vLaplace, vtau = vxc

    weighted_p = get_weighted_potential(vxc, weights, rho)
    weighted_ao = np.einsum('npi,np->pi', ao[:4], weighted_p, optimize=True)

    xc = np.einsum('rp,rq->pq', ao[0], weighted_ao, optimize=True)

    weighted_p = (0.25 * weights * vtau).reshape(-1,1)
    for v in ao[1:4]:
        xc += np.einsum('rp,rq->pq', v, weighted_p*v, optimize=True)

    return xc + xc.T

def evaluate_ex_mgga(exc, rho, weights):
    #evaluate exchange-correlation energy

    xc_energy = np.einsum('p,p->', rho*weights, exc, optimize=True)

    return xc_energy

if __name__ == '__main__':

    mesh = 'close' ; functional = 'MS'; DIIS = True ; DIIS_SIZE = 6

    #define the molecule atoms first then basis (sto-3g)
    mol = []
    mol.append(atom('O', [0.0,0.0,0.0])) ; mol.append(atom('H', [0,-0.757 ,0.587])) ; mol.append(atom('H', [0,0.757,0.587]))
    for m in mol:
        m.center /= units['bohr->angstrom']

    orb = []
    orb.append(orbital(0, [0,0,0], [130.7093214, 23.80886605, 6.443608313], [0.1543289672962566, 0.5353281422870151, 0.44463454218921483],   \
                                   [27.551167822078394, 7.681819989204459, 2.882417873168662], mol))
    orb.append(orbital(0, [0,0,0], [5.033151319, 1.169596125, 0.38038896],  [-0.09996722918837482, 0.399512826093505, 0.7001154688886181],   \
                                   [2.394914882501622, 0.8015618386293724, 0.34520813393821864], mol))
    orb.append(orbital(0, [1,0,0], [5.033151319, 1.169596125, 0.38038896],  [0.15591627500155536, 0.6076837186060621, 0.39195739310391],     \
                                   [10.745832634231427, 1.7337440707285054, 0.4258189334467701], mol))
    orb.append(orbital(0, [0,1,0], [5.033151319, 1.169596125, 0.38038896],  [0.15591627500155536, 0.6076837186060621, 0.39195739310391],     \
                                   [10.745832634231427, 1.7337440707285054, 0.4258189334467701], mol))
    orb.append(orbital(0, [0,0,1], [5.033151319, 1.169596125, 0.38038896],  [0.15591627500155536, 0.6076837186060621, 0.39195739310391],     \
                                   [10.745832634231427, 1.7337440707285054, 0.4258189334467701], mol))
    orb.append(orbital(1, [0,0,0], [3.425250914, 0.6239137298, 0.168855404], [0.15432896729459913, 0.5353281422812658, 0.44463454218443965], \
                                   [1.7944418337900938, 0.5003264922111158, 0.1877354618463613], mol))
    orb.append(orbital(2, [0,0,0], [3.425250914, 0.6239137298, 0.168855404], [0.15432896729459913, 0.5353281422812658, 0.44463454218443965], \
                                   [1.7944418337900938, 0.5003264922111158, 0.1877354618463613], mol))

    #output details of molecule
    out([mol, DIIS, DIIS_SIZE, functional, mesh], 'initial')
    #use a reduced version of Harpy's cython integrals
    from ks_aello import aello
    s, t, v, eri = aello(mol, orb)

    #orthogonal transformation matrix
    from scipy.linalg import fractional_matrix_power as fractPow
    x = fractPow(s, -0.5)

    #inital fock is core hamiltonian
    h_core = t + v

    #orthogonal Fock
    fo = np.einsum('rp,rs,sq->pq', x, h_core, x, optimize=True )

    #eigensolve and transform back to ao basis
    eo , co = np.linalg.eigh(fo)
    c = np.einsum('pr,rq->pq', x, co, optimize=True)

    #build our initial density
    nocc = np.sum([a.number for a in mol])//2

    d = np.einsum('pi,qi->pq', c[:, :nocc], c[:, :nocc], optimize=True)

    #SCF conditions
    cycles = 50 ; tolerance = 1e-6
    out([cycles, tolerance], 'cycle')

    #get grid
    grid, weights = GRID(mol, mesh)

    #evaluate basis and derivative over grid
    ao = mgga.evaluate_ao(orb, grid)

    last_cycle_energy = 0.0

    #diis initialisation
    if DIIS: diis = dii_subspace(DIIS_SIZE)

    #SCF loop
    for cycle in range(cycles):

        #build the coulomb integral
        j = 2.0 * np.einsum('rs,pqrs->pq', d, eri, optimize=True)
        k =  np.einsum('rs,prqs->pq', d, eri, optimize=True) 
 
        #evalute density over mesh
        rho = mgga.evaluate_rho_gga(d, ao)

        #evaluate functional over mesh
        exc, vxc = mgga.meta_gga_functional(functional, rho)

        #evaluate energy
        xc_energy = evaluate_ex_mgga(exc, rho[0], weights)

        out([cycle, np.einsum('pq,pq->', d, (2.0*h_core), optimize=True), \
                    np.einsum('pq,pq->', d, (j), optimize=True), \
                    xc_energy, np.sum(rho[0]*weights) ],'scf')
        
        #evaluate potential
        vxc = evaluate_vx_mgga(vxc, ao, weights, rho)

        #if hybrid add HF exchange contribution
        if functional == 'PBE0': vxc -= k * 0.25

        f = h_core + j + vxc

        if (cycle != 0) and DIIS:
            f = diis.build(f, d, s, x)

        #orthogonal Fock and eigen solution
        fo = np.einsum('rp,rs,sq->pq', x, f, x, optimize=True )

        #eigensolve
        eo , co = np.linalg.eigh(fo)
        c = np.einsum('pr,rq->pq', x, co, optimize=True)

        #construct new density
        d = np.einsum('pi,qi->pq', c[:, :nocc], c[:, :nocc], optimize=True)

        #electronic energy
        eSCF = np.einsum('pq,pq->', d, (2.0*h_core + j), optimize=True) + xc_energy

        if abs(eSCF - last_cycle_energy) < tolerance: break
        if DIIS: vector_norm = diis.norm
        else:    vector_norm = ''
        out([cycle, abs(eSCF - last_cycle_energy), vector_norm],'convergence')

        last_cycle_energy = eSCF


    out([eSCF, np.einsum('pq,pq->', d, (2.0*h_core), optimize=True), \
                np.einsum('pq,pq->', d, ( j), optimize=True), \
                evaluate_ex_mgga(exc, rho[0], weights), nuclear_repulsion(mol) ], 'final')

    out([eo, c, np.sum(rho[0]*weights), d, s, mol, orb], 'post')

