![image](https://user-images.githubusercontent.com/73105740/142720205-ddb5a6ad-4d6c-4a1a-8171-9ea1af5eece4.png)
## Density Function Theory Program - kspy-mgga
  The programs **kspy-lda** and **kspy-gaa** showed how to implement a Mura-Knowles/Lebedev grid and use LDA and GGA functionals. This program implements a Meta-Generalized Gradient Approximation (mGGA) based functional. The functional I used was the Sun, Xiao and Ruzsinszky (MS) functional.

### The Grid
  The grid is the same as **kspy-lda** and **kspy-gaa**. The radial grid is a Mura-Knowles radial grid [ME Mura and PJ Knowles 'Improved radial grids for quadrature in density-functional calculations' JCP 104, 9848 (1996); DOI:10.1063/1.471749](https://aip.scitation.org/doi/10.1063/1.471749). The 'coarse' angular grid is of Lebedev orders (11, 15) for period 1 and period 2 respectively. This translates into 50 and 86 points respectively arranged on a spherical shell (VI Lebedev, and DN Laikov, Doklady Mathematics, 'A Quadrature formula for the sphere of the 131st algebraic order of accuracy' Vol. 59, No. 3, (1999)). There are various sources for this data given in the external links of the wikipedia article on Lebedev integration.  A pruning scheme is employed to systematically reduce the number of angular points in regions where dense angular quadrature is not necessary, such as near the nuclei where the charge density is approximately spherically symmetric and at long distance from the nucleus. The pruning scheme I employed was the Treutler-Aldrich scheme [O Treutler and R Ahlrich, 'Efficient molecular numerical integration schemes' JCP 102, 346 (1995); DOI:10.1063/1.469408](https://aip.scitation.org/doi/pdf/10.1063/1.469408). The partitioning of the atomic centered grids to a molecular grid follows a Becke scheme after Stratmann [RE Stratmann, GE Scuseria and MJ Frisch, 'Achieving Linear scaling in exchange-correlation density functional quadratures' CPL 257, 3-4 (1996); DOI:10.1016/009-2614(96)00600-8](https://www.sciencedirect.com/science/article/abs/pii/0009261496006008?via%3Dihub). Finally I have implemented a final radius adjustment during the partition (Becke suggests doing this) using the Bragg radius. A second 'close' grid is also included which is a (50, 75) radial and (29, 29) angular, the latter representing 302 points on each shell. The grid routines are in ks_grid.py.

### The HF Integrals
  To get the DFT SCF started we need an initial density. To do this I use a HF overlap matrix S, and an initial Fock matrix composed of the sum of the 1-electron kinetic and coulomb integrals (core Hamiltonian - T+V). This Fock is then orthogonalised (F<sup>'</sup>) as (S<sup>-0.5</sup>)<sup>T</sup>FS<sup>-0.5</sup>, eigensolve the resulting orthogonal Fock for orbital coefficients C orthogonal, transform back to atomic basis as S<sup>-0.5</sup>C<sup>'</sup>, use resulting ao coefficients to compute a density matrix D<sub>&mu;&nu;</sub> = c<sub>&mu;i</sub>c<sub>i&nu;</sub> where i is over occupied orbitals. This initial density can be used with initial Fock and 2-electron repulsion integrals to form the coulomb integral J and the exchange integral K. To get these integrals I've used a modified version of Harpy's Cython integral package *aello*. I've removed the angular and electric field integrals and additionally the 2-electron repulsions are returned as a tensor rather than a linear array. These are in ks_aello.pyx.

### Molecule and Basis Sets
  The molecule definition is contained in a *mol* object which is itself comprised of objects from an atom class. Each instance of the atom class contains the atom symbol, atomic number and the coordinates of the atom center (array[3]). The molecule is hard coded as H<sub>2</sub>O. The basis is contained in an *orb* object which is itself comprised of objects from a gaussian class. Each instance of the gaussian class contains the atom the Gaussian is centered on, the momentum(array[3]), the exponents (array[primatives]), the coefficients (array[primatives]), the normalisation (array[primatives]) and a copy of the atom center coordinates (array[3]). The momenta are given as s [0,0,0] px [1,0,0] py [0,1,0] and pz [0,0,1]. The basis used is a simple STO-3G so we only require s and p orbitals. The primatives exponent and coefficient values are hard-coded in the __main__ section. (I use the psi4 format of the basis sets from BSE which have some (small) differences from the nwchem format versions as used by eg pyscf. This might lead to numerical differences in values when using high precision).

### The Functional
  We are using the mGGA (meta-Generalized Gradient Approximation) with the MS functional - defined [here](https://aip.scitation.org/doi/10.1063/1.4742312). The functional has been implemented in a way that is compatible with **libxc**. That is to say the LDA energy is included and rather than the pure gradients a contracted version (&sigma;) are passed to the functional evaluator. &sigma; is defined as &Delta;&rho;<sub>i</sub>.&Delta;&rho;<sub>i</sub>, details are given in the **libxc** documentation (https://www.tddft.org/programs/libxc/manual/libxc-5.1.x/). The functional code and gradient evaluation routines are in ks_mgga.py. The functional does not need the Laplacian but routines to calculate it are included in ks_mgga. The &tau; derivatives are also needed for the MS functional. 

### The Density
  The density is determined by first evaluating each Gaussian orbital over the grid (&rho;<sub>g</sub>), then assembling those into a complete basis over the grid (&rho;<sub>b</sub> = &Sigma;&rho;<sub>g</sub>) and finally computing &rho; as the product  &rho;<sub>b</sub>*&rho;<sub>b</sub>D. 

### Convergence
  The diis scheme is taken from harpy (diis.py) with some minor changes. This is implemented as a class in ks_util.

### Output
       ks output
    molecule is             water
    geometry is             OH bonds - 0.96      HOH angle - 104.42
    basis is                STO-3G {psi4 format}
    analytic integration    aello cython - McMurchie-Davidson scheme
    numerical integration   (Mura-Knowles, Lebedev)
                            radial prune is Aldrich-Treutler
                            Becke Partition scheme is Stratmann 
                            Radial adjust is Treutler
                            order: period 1 (10,11) and period 2 (15,15)
    mesh                    close
    functional              MS
    diis                    True   buffer size is  6

    scf control             maximum cycles are  50         convergence tolerance  1e-06

     cycle     1 electron         coulomb         exchange          electrons
                                                                                       ΔE         diis norm
    -------------------------------------------------------------------------------------------------------
        0     -127.35805722     54.98251023    -10.03726549         10.0000 

        1     -117.63957585     42.95247927     -8.85555509         10.0000 
                                                                                     7.859798      0.849618 
        2     -125.27886463     50.95895753     -9.51205034         10.0000 
                                                                                     4.912381      0.677463 
        3     -122.55545128     47.53016881     -9.16305758         10.0000 
                                                                                     1.142798      0.072172 
        4     -122.26664632     47.21072183     -9.13607635         10.0000 
                                                                                     0.117519      0.007506 
        5     -122.23514417     47.17620478     -9.13309844         10.0000 
                                                                                     0.013266      0.000507 
        6     -122.23292948     47.17378281     -9.13289133         10.0000 
                                                                                     0.000970      0.000017 
        7     -122.23285565     47.17370209     -9.13288445         10.0000 
                                                                                     0.000033      0.000000 
        8     -122.23285364     47.17369990     -9.13288426         10.0000 

    final energies (Hartree)
    ------------------------
    one electron        -122.2328536440
    coulomb               47.1736998979  
    exchange              -9.1328842595  
    nuclear repulsion      9.1882584177   
    total electronic     -84.1920380055 

    final total energy   -75.0037795878 


### Test
  The python code for the functional used here was copied into a pyscf eval_xc subroutine (user defined functional) and run with the same grid. The pyscf value is -75.0037795572198 Hartree <sup>*</sup>, this agrees better than the convergence criteria of 1e-6. The value using libxc from pyscf is -75.0037795572198 Hartree. 
  
  **<sup>*</sup>** the basis definition with pyscf (nwchem) differs slightly from the basis here (psi4).

### Installation
  Copy all files to a directory, run

    python3 setup.py build_ext --inplace install --user

then 

    python3 ks_main.py

### Additions 
12 December 2021 - added some simple post SCF properties.

    molecular orbitals
    ------------------
     0  -18.62053 occupied  
     1   -0.86067 occupied  
     2   -0.36899 occupied  
     3   -0.15761 occupied  
     4   -0.06496 homo  
     5    0.37755 lumo 
     6    0.49490 virtual 

    mulliken populations
    --------------------
     0    1.99739   1s  
     1    1.85514   2s  
     2    2.00000   2px 
     3    1.05185   2py 
     4    1.42403   2pz 
     5    0.83579   1s  
     6    0.83579   1s  

    atomic charge
    ---------------
     0   -0.32841 O   
     1    0.16421 H   
     2    0.16421 H   

    dipole momemts (Debye)
    ----------------------
     x= -0.00000 y= 0.00000  z= 1.55889 