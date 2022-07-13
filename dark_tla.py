"""Three-level atom model and integrator.

"""

import numpy as np
import darkhistory.physics as phys
import darkhistory.history.reionization as reion
from scipy.integrate import odeint
from scipy.integrate import solve_ivp
from scipy.misc import derivative
from scipy.interpolate import RegularGridInterpolator
from scipy.special import erf

def norm_compton_cooling_rate(x_be, T_DM, rs, alphaD, m_be, m_bp, xi, thermalized_be_bH=True):
    """ Gamma_c where dlogT_DM/dt = Gamma_c (T_D/T_DM - 1), in 1/s
    """
    T_D = xi*phys.TCMB(rs)
    if thermalized_be_bH:
        CV_fac = x_be/(1+x_be)
    else:
        CV_fac = 1/2 #x_be/(2*x_be)
    pre = 64 * np.pi**3 * alphaD**2 / 135 * T_D**4 * CV_fac
    mass = (1 + (m_be/m_bp)**3) / m_be**3
    return pre * mass / phys.hbar

def compton_cooling_rate(xHII, Tb, rs):
    """ Gamma_c where dlogT_b/dt = Gamma_c (T_CMB/Tb - 1), in 1/s """
    pre = 4 * phys.thomson_xsec * 4 * phys.stefboltz / phys.me
    denom = 3/2 * (1 + phys.chi + xHII)
    return pre * xHII * phys.TCMB(rs)**4 / denom

def norm_be_bH_rate(x_be, T_DM, rs, alphaD, m_be, m_bp, xi, thermalized_be_bH=True):
    """ Gamma_be_bH where dlogT_bH/dt = Gammabe_bH (T_be/T_bH - 1), in 1/s
    """
    T_D = xi*phys.TCMB(rs)
    CV_fac = 1/2 #x_be/(2*x_be)
    pre = 64 * np.pi**3 * alphaD**2 / 135 * T_D**4 * CV_fac
    mass = (1 + (m_be/m_bp)**3) / m_be**3
    return pre * mass / phys.hbar

# See Eqns. (A.1) and (A.2) of Cyr-Racine and Sigurdson #
# Reduced Mass #
def mu_D(alpha, B):
    return 2 * B/alpha**2

# proton mass #
def get_mp(mD, alpha, B):
    mu = mu_D(alpha, B)
    mm = mD + B
    return mm/2 + np.sqrt(mm**2 - 4*mm*mu)/2

# electron mass #
def get_me(mD, alpha, B):
    mu = mu_D(alpha, B)
    mp = mp_D(mD, alpha, B)
    return mu*mp/(mp-mu)

# Binding Energy of Hydrogenic atom #
def get_BD(alphaD, m_be, m_bp):
    """ !!! Double-check """
    mu_D = m_be/(1 + m_be/m_bp)
    return 1/2*alphaD**2 * mu_D

def dark_alpha_recomb(T_DM, alphaD, m_be, m_bp, xi):
    mu_D = m_be*m_bp/(m_be+m_bp)
    B_D = get_BD(alphaD, m_be, m_bp)
    fudge_fac = 1.14
    conversion_fac = (phys.hbar * phys.c)**2 * phys.c
    pre = 0.448 * 64*np.pi/np.sqrt(27*np.pi) * alphaD**2/mu_D**2
    temp = np.sqrt(B_D/T_DM) * np.log(B_D/T_DM)
    return pre * conversion_fac * fudge_fac * temp

def dark_beta_ion(T_D, alphaD, m_be, m_bp, xi):
    mu_D = m_be*m_bp/(m_be+m_bp)
    B_D = get_BD(alphaD, m_be, m_bp)
    recomb = dark_alpha_recomb(T_D, alphaD, m_be, m_bp, xi)

    # de Broglie wavelength
    lam = (
        phys.c * 2*np.pi*phys.hbar
        / np.sqrt(2 * np.pi * mu_D * T_D)
    )
    

    return lam**(-3) * np.exp(-B_D/T_D/4) * recomb / 4

def dark_peebles_C(x_be, rs, alphaD, m_be, m_bp, xi):
    B_D = get_BD(alphaD, m_be, m_bp)
    m_D = m_be + m_bp - B_D
    n_D = phys.rho_DM/m_D * rs**3
    #n_D = phys.nH*rs**3
    T_D = xi * phys.TCMB(rs)

    beta = dark_beta_ion(T_D, alphaD, m_be, m_bp, xi)

    Lya_D = 3/4 * B_D

    dark_sobolev = 2**7 * alphaD**3 * B_D * n_D*phys.c**3 * (1-x_be)/(
            3**7 * 2 * np.pi * phys.hbar * phys.hubble(rs) * Lya_D
    )

    R_D_fac = 3/4 * 2**9 * alphaD**3*B_D/3**8 * (
            1-np.exp(-dark_sobolev)
    )/dark_sobolev

    Lam_2s_1s_fac = 1/4 * (alphaD/phys.alpha)**6 * (
            B_D/phys.rydberg
    ) * phys.width_2s1s_H

    ssum = R_D_fac + Lam_2s_1s_fac

    return ssum/(ssum + beta)
    #return phys.peebles_C(x_be, rs)

def x_be_Saha(rs, alphaD, m_be, m_bp, xi):
    """Saha equilibrium ionization value for H and He.

    Parameters
    ----------
    rs : float
        The redshift in 1+z.
    species : {'HI', 'HeI'}
        The relevant species.

    Returns
    -------
    float
        The Saha equilibrium xe.

    Notes
    -----
    See astro-ph/9909275 and 1011.3758 for details.
    """
    T = xi * phys.TCMB(rs)

    B_D = get_BD(alphaD, m_be, m_bp)
    m_D = m_be + m_bp - B_D
    n_D = phys.rho_DM/m_D * rs**3
    #n_D = phys.nH * rs**3
    mu_D = m_be*m_bp/m_D

    de_broglie_wavelength = phys.c * 2*np.pi*phys.hbar / np.sqrt(2 * np.pi * mu_D * T)

    rhs = (1/de_broglie_wavelength)**3 / n_D * np.exp(-B_D/T)
    a  = 1.
    b  = rhs
    q  = -rhs

    if rhs < 1e8:

        x_be = (-b + np.sqrt(b**2 - 4*a*q))/(2*a)

    else:

        x_be = 1. - a/rhs

    return x_be

def d_x_be_Saha_dz(rs, alphaD, m_be, m_bp, xi):
    """`z`-derivative of the Saha equilibrium ionization value.

    Parameters
    ----------
    rs : float
        The redshift in 1+z.
    species : {'singlet', 'triplet'}

    Returns
    -------
    float
        The derivative of the Saha equilibrium d xe/dz.

    Notes
    -----
    See astro-ph/9909275 and 1011.3758 for details.
    """

    x_be    = x_be_Saha(rs, alphaD, m_be, m_bp, xi)

    T = xi * phys.TCMB(rs)

    B_D = get_BD(alphaD, m_be, m_bp)

    numer = (B_D/T - 3/2)*x_be**2*(1-x_be)
    denom = rs*(2*x_be*(1-x_be) + x_be**2)

    #return phys.d_xe_Saha_dz(rs, 'HI')
    return numer/denom

def F_pi(y):
    """ Eqn. C4 """
    return 3973.6*y**(-0.0222)/(2.012 + y**(0.2412))**6.55

interpfunc = None
def generate_interpfunc():
    global interpfunc
    if interpfunc != None:
        return interpfunc
    else:
        xs=np.array([1e-5,1e-3,1e-1, 10, 19])

        ys1=np.array([42,30, 10, 0.3, 0.15])
        ys2=np.array([44,40, 15, 1, 0.45])
        ys3=np.array([45,46, 38, 5, 2.1])
        ys4=np.array([46,75, 80, 25, 15])

        data = np.array([ys4,ys3,ys2,ys1])
        interpfunc = RegularGridInterpolator(points=(np.array([-5,-3,-1, 1, np.log10(19)]), np.log10([0.06,0.19,0.49,1.0])), values=np.log10(data.T))
        return interpfunc

def F_pr(x, y):
    interpfunc = generate_interpfunc()
    if isinstance(x, np.float):
        if x<1e-5:
            x=1e-5
        elif x>19:
            x=19
    else:
        x[x<1e-5] = 1e-5
        x[x>19] = 19

    if isinstance(y,np.float):
        if y<0.06:
            y=0.06
        elif y>1:
            y=1
    else:
        y[y<0.06] = 0.06
        y[y>1] = 1

    return 10**interpfunc((np.log10(x), np.log10(y)))

def Gam_pi(x_be, T_DM, rs, alphaD, m_be, m_bp, xi, thermalized_be_bH=True):
    B_D  = get_BD(alphaD, m_be, m_bp)
    T_D  = xi * phys.TCMB(rs)
    m_D  = m_be + m_bp - B_D
    mu_D = m_be*m_bp/m_D
    n_D  = phys.rho_DM/m_D * rs**3
    if thermalized_be_bH:
        denom = (3/2 * T_DM * n_D * (1+x_be))
    else:
        denom = (3/2 * T_DM * n_D * (2*x_be))

    # Calculate x_2s using Hyrec's steady stateassumption #
    Lya_D = 3/4 * B_D
    dark_sobolev = 2**7 * alphaD**3 * B_D * n_D*phys.c**3 * (1-x_be)/(
            3**7 * 2 * np.pi * phys.hbar * phys.hubble(rs) * Lya_D)
    R_D = 2**9 * alphaD**3*B_D/3**8 * (1-np.exp(-dark_sobolev))/dark_sobolev

    Lam = (alphaD/phys.alpha)**6 * (B_D/phys.rydberg) * phys.width_2s1s_H
    alpha_B = dark_alpha_recomb(T_DM, alphaD, m_be, m_bp, xi)
    beta_e = dark_beta_ion(T_D, alphaD, m_be, m_bp, xi)
    x_2 = (n_D*x_be**2*alpha_B + (3*R_D + Lam)*(1-x_be)*np.exp(-B_D/T_D))/(
            beta_e + 3/4*R_D + 1/4*Lam)
    x_2s = 1/4*x_2

    pre = alphaD**3 * T_D**2 / (3*np.pi) * np.exp(-B_D/(4*T_D))
    convert = phys.hbar
    return pre * x_2s*n_D * F_pi(T_D/B_D) / denom * convert

def Gam_pr(x_be, T_DM, rs, alphaD, m_be, m_bp, xi, thermalized_be_bH=True):
    B_D = get_BD(alphaD, m_be, m_bp)
    T_D = xi * phys.TCMB(rs)
    m_D = m_be + m_bp - B_D
    mu_D = m_be*m_bp/m_D
    n_D = phys.rho_DM/m_D * rs**3
    if thermalized_be_bH:
        denom = (3/2 * T_DM * n_D * (1+x_be))
    else:
        denom = (3/2 * T_DM * n_D * (2*x_be))

    pre = 2*alphaD**3*np.sqrt(2*np.pi*T_DM)/(3*mu_D**(3/2))
    convert = (phys.hbar * phys.c)**3/phys.hbar

    return pre * x_be**2*n_D**2 * F_pr(T_D/B_D, T_DM/T_D) / denom * convert

def Gam_ff(x_be, T_DM, rs, alphaD, m_be, m_bp, xi, thermalized_be_bH):
    """ free-free absorption - emission (brem) / T_DM in 1/s """
    B_D = get_BD(alphaD, m_be, m_bp)
    T_D = xi * phys.TCMB(rs)
    eps = 1-T_DM/T_D

    m_D = m_be + m_bp - B_D
    mu_D = m_be*m_bp/m_D
    n_D = phys.rho_DM/m_D * rs**3
    g_ff = 1.33
    zeta_3 = 1.20206
    if thermalized_be_bH:
        denom = (3/2 * T_DM * n_D * (1+x_be))
    else:
        denom = (3/2 * T_DM * n_D * (2*x_be))

    pre = 16 * alphaD**3 * g_ff * x_be**2 * n_D**2 / (3*mu_D)**(3/2)
    prnth = np.pi**2 * (1+2*eps)/6 - zeta_3*eps
    convert = (phys.hbar * phys.c)**3/phys.hbar
    return pre * np.sqrt(2*np.pi*T_DM) * prnth / denom * convert

def Gam_R(x_be, T_DM, rs, alphaD, m_be, m_bp, xi, thermalized_be_bH=True):
    """ Rayleigh energy exchange rate / T_DM in 1/s """
    B_D = get_BD(alphaD, m_be, m_bp)
    T_D = xi * phys.TCMB(rs)
    m_D  = m_be + m_bp - B_D
    n_D  = phys.rho_DM/m_D * rs**3
    zeta_9 = 1.00201
    ratio = T_D/T_DM
    if thermalized_be_bH:
        denom = (3/2 * (1+x_be))
    else:
        denom = (3/2 * (1-x_be))

    pre  = 430080 * zeta_9 * alphaD**2 * (1-x_be) / np.pi**2
    temps = (T_D/B_D)**4 * (T_D/m_be)**2 * T_D/m_D * T_D/phys.hbar
    return pre * temps / denom #ratio * convert


def bH_ion_heat_exchange(m_be, m_bp, alphaD, T_bH, T_ion, x_be, rs):
    x_bH = 1-x_be
    m_tot = m_be+m_bp
    B_D = get_BD(alphaD, m_be, m_bp)
    rho_DM  = phys.rho_DM * rs**3

    m =   {'bH': m_tot, 'be': m_be,  'bp': m_bp}
    T =   {'bH': T_bH,     'be': T_ion, 'bp': T_ion}
    x =   {'bH': x_bH,    'be': x_be, 'bp': x_be}
    rho = {'bH': rho_DM*x_bH, 'be': rho_DM*x_be*m_be/m_tot, 'bp': rho_DM*x_be*m_bp/m_tot}
    n =   {'bH': rho_DM*x_bH/m_tot, 'be': rho_DM*x_be/m_tot, 'bp': rho_DM*x_be/m_tot}

    # be-bH, bp-bH,
    parts = ['be', 'bp']
    m_sum = {key: m[key] + m['bH'] for key in parts}
    mu    = {key: m[key]*m['bH']/m_sum[key] for key in parts}
    u_th  = {key: np.sqrt(T[key]/m[key] + T['bH']/m['bH']) for key in parts}


    #cross-section, see Cyr-Racine Eqn.29
    sigma0 = 320*alphaD**2/B_D**2 * (phys.hbar*phys.c)**2

    dotT_ion   = 0
    dotT_bH = 0
    for p1 in parts:
        pre  = 8*np.sqrt(2/np.pi) * rho[p1]*rho['bH']/m_sum[p1]**2 * sigma0 * u_th[p1]

        #if x_be > 0:
        dotT_ion += 2/(3*n[p1])   * pre * phys.c
        dotT_bH  += 2/(3*n['bH']) * pre * phys.c

    return dotT_ion, dotT_bH


def DM_b_heat_exchange(m_be, m_bp, Tb, T_DM, V_pec, xHII, x_be, rs, eps=0, thermalized_be_bH=True):
    #See 1509.00029
    if eps == 0:
        return 0, 0

    rho_DM = phys.rho_DM * rs**3
    nH = phys.nH * rs**3
    nDM = rho_DM/(m_be + m_bp)
    # since n_DM*m_be + n_DM * m_bp = rho_DM

    parts = ['e', 'p', 'be', 'bp']
    m = {'e': phys.me, 'p': phys.mp, 'be': m_be, 'bp': m_bp}
    T = {'e': Tb, 'p': Tb, 'be': T_DM, 'bp': T_DM}
    x = {'e': xHII, 'p': xHII, 'be': x_be, 'bp': x_be}
    n = {'e': xHII*nH, 'p': xHII*nH, 'be': x_be*nDM, 'bp': x_be*nDM}
    rho = {part: n[part]*m[part] for part in parts}

    # heat capacities
    CV_m = 3/2 * (1+phys.chi+xHII)*nH
    if thermalized_be_bH:
        # Only ions contribute
        CV_DM = 3/2 * (1+x_be)*nDM
    else:
        # All dark particles contribute
        CV_DM = 3/2 * (2*x_be)*nDM

    CV = {'e': CV_m, 'p': CV_m, 'be': CV_DM, 'bp': CV_DM}

    # e-be, e-bp, p-be, p-bp
    pairs = ['ebe', 'ebp', 'pbe', 'pbp']
    m_sum = {key: m[key[0]] + m[key[1:]] for key in pairs}
    mu    = {key: m[key[0]]*m[key[1:]]/m_sum[key] for key in pairs}
    u_th  = {key: np.sqrt(T[key[0]]/m[key[0]] + T[key[1:]]/m[key[1:]]) for key in pairs}
    r     = {key: 0 for key in pairs}

    #Coulomb Log: Eqn 2 of Munoz and Loeb
    const = -np.log(4*np.pi*eps**2*phys.alpha**3/9) - 3*np.log(phys.hbar*phys.c)
    logL_DM  = {key: const + np.log(np.divide(T[key[0]]**3, n[key[0]])) for key in pairs}
    logL_b = {key: const + np.log(T[key[1:]]**3 / n[key[1:]]) for key in pairs}

    #cross-section prefactor without the Coulomb log
    prefac = 2*np.pi*phys.alpha**2*eps**2 * (phys.hbar*phys.c)**2
    xsec_0 = {key: prefac/mu[key]**2 for key in pairs}

    dotTb   = 0
    dotT_DM = 0
    for key in pairs:
        p1 = key[0]  # e  or p
        p2 = key[1:] # be or bp

        pre  = rho[p1]*rho[p2]/m_sum[key]**2 * xsec_0[key]/u_th[key]
        term = np.sqrt(2/np.pi)*np.exp(r[key]**2/2)/u_th[key]**2

        #F(r)/r
        if r[key] <= 1e-3:
            #Taylor Expansion
            F_r = 1/3*np.sqrt(2/np.pi) * r[key]**2 - r[key]**4/(5*np.sqrt(2*np.pi))
        else:
            F_r = erf(r[key]/np.sqrt(2))/r[key] - np.sqrt(2/np.pi)*np.exp(-r[key]**2/2)

        if x_be > 0:
            dotTb   += 1/CV[p1] * pre*logL_b[key]  * term*phys.c  #+ m[p2]*F_r)
        if xHII > 0:
            dotT_DM += 1/CV[p2] * pre*logL_DM[key] * term*phys.c  #+ m[p1]*F_r)

    return dotTb, dotT_DM


def get_history(
    rs_vec, init_cond=None, 
    alphaD=phys.alpha, m_be=phys.me, m_bp=phys.mp, xi=1,
    both_sectors = False, eps = 0,
    mxstep = 1000, rtol=1e-4,
    hack = True, thermalized_be_bH=True
):
    """Returns the ionization and thermal history of the IGM and DM.

    Parameters
    ----------
    rs_vec : ndarray
        Abscissa for the solution.
    init_cond : array, optional
        Array containing [initial temperature, initial xHII, initial xHeII, initial xHeIII]. Defaults to standard values if None.
    mxstep : int, optional
        The maximum number of steps allowed for each integration point. See *scipy.integrate.odeint* for more information.
    rtol : float, optional
        The relative error of the solution. See *scipy.integrate.odeint* for more information.
    thermalized_be_bH : bool, optional
        True if we should assume the dark atoms are thermalized with the dark ions, 
        False if we should keep track of T_bH and T_be=T_bp separately.

    Returns
    -------
    list of ndarray
        [temperature solution (in eV), xHII solution, xHeII, xHeIII].

    Notes
    -----
    The actual differential equation that we solve is expressed in terms of y = arctanh(f*(x - f)), where f = 0.5 for x = xHII, and f = nHe/nH * 0.5 for x = xHeII or xHeIII, where nHe/nH is approximately 0.083.

    """
    B_D   = get_BD(alphaD, m_be, m_bp)
    m_D   = m_be + m_bp - B_D
    Lya_D = 3/4 * B_D
    fudge = 1

    # Define conversion functions between x and y. 
    def get_x(y):
            return 0.5 + 0.5*np.tanh(y)

    def tla(rs, var):
        # Returns an array of values for [dT_DM/dz, dy_be/dz].
        # var = [T_DM, x_be]
        #rs = np.exp(logrs)

        nH = phys.nH*rs**3
        n_D  = phys.rho_DM/m_D * rs**3
        #n_D = phys.nH * rs**3 #To match the SM result
        T_D  = phys.TCMB(rs)*xi

        log_TDM, y_be, log_Tb, yHII = var[0], var[1], var[2], var[3]
        T_DM, Tb   = np.exp(log_TDM), np.exp(log_Tb)

        x_be, xHII = get_x(y_be), get_x(yHII)
        V_pec = 0
        baryon, DM_rate = DM_b_heat_exchange(
                m_be, m_bp, Tb, T_DM, V_pec, xHII, x_be, rs, 
                eps=eps, thermalized_be_bH=thermalized_be_bH
        )

        ## Note: when neutrals and ions aren't thermalized, take T_DM to be the ion temperature
        if not thermalized_be_bH:
            log_T_bH = var[4]
            T_bH = np.exp(log_T_bH)
            ion_rate, H_rate = bH_ion_heat_exchange(m_be, m_bp, alphaD, T_bH, T_DM, x_be, rs)
        #else:
        #    T_bH = T_DM

        def dlogTDM_dz(rs):
            eps2 = 1-T_DM/T_D

            # Cooling rate due to adiabatic expansion
            adia  = 2/rs

            #Compton
            comp  = phys.dtdz(rs) * norm_compton_cooling_rate(
                    x_be, T_DM, rs, alphaD, m_be, m_bp, xi
            ) * (T_D/T_DM - 1.)

            #Rayleigh
            if thermalized_be_bH:
                Rayl  = phys.dtdz(rs) * Gam_R(
                        x_be, T_DM, rs, alphaD, m_be, m_bp, xi) * (T_D/T_DM-1)
                recomb = -phys.dtdz(rs) * Gam_pr(x_be, T_DM, rs, alphaD, m_be, m_bp, xi, 
                        thermalized_be_bH=thermalized_be_bH)
            else:
                Rayl, recomb  = 0, 0

            #brem heating + ion heating - recomb cooling
            ff_pi_pr = phys.dtdz(rs) * (
                    Gam_ff(x_be, T_DM, rs, alphaD, m_be, m_bp, xi,
                        thermalized_be_bH=thermalized_be_bH) * eps2
                    + Gam_pi(x_be, T_DM, rs, alphaD, m_be, m_bp, xi, 
                        thermalized_be_bH=thermalized_be_bH)
                    ) + recomb


            # DM-baryon scattering
            dlogTdz_DM_b = phys.dtdz(rs) * DM_rate * (Tb/T_DM-1)

            # dark H dark ion scattering
            if not thermalized_be_bH:
                dlogTdz_ion_bH = phys.dtdz(rs) * ion_rate * (T_bH/T_DM-1)
            else:
                dlogTdz_ion_bH = 0

            deriv = Rayl + comp + adia + ff_pi_pr + dlogTdz_DM_b + dlogTdz_ion_bH
            return fudge*deriv

        def dlogTbH_dz(rs):
            # Cooling rate due to adiabatic expansion
            adia  = 2/rs

            #Rayleigh
            Rayl  = phys.dtdz(rs) * Gam_R(
                    x_be, T_DM, rs, alphaD, m_be, m_bp, xi) * (T_D/T_bH-1)
            # recombination cooling
            recomb = -phys.dtdz(rs) * Gam_pr(x_be, T_DM, rs, alphaD, m_be, m_bp, xi,
                    thermalized_be_bH=thermalized_be_bH)


            dlogTdz_bH_ion = phys.dtdz(rs) * H_rate * (T_DM/T_bH-1)
            deriv = Rayl + recomb + adia + dlogTdz_bH_ion
            return fudge*deriv


        def dybe_dz(rs):
            #rs = np.exp(logrs)
            xD = 1 - x_be

            if x_be > 0.999 and rs > 2000:
                # Use the Saha value. 
                dxdz = d_x_be_Saha_dz(rs, alphaD, m_be, m_bp, xi)
                return 2 * np.cosh(y_be)**2 * dxdz


            peeb_C = dark_peebles_C(x_be, rs, alphaD, m_be, m_bp, xi)
            alpha = dark_alpha_recomb(T_DM, alphaD, m_be, m_bp, xi)
            beta = dark_beta_ion(T_D, alphaD, m_be, m_bp, xi)
            return fudge*2 * np.cosh(y_be)**2 * phys.dtdz(rs) * (
                - peeb_C * (alpha * x_be**2 * n_D 
                    - 4 * beta * xD * np.exp(-Lya_D/T_DM)
                    )
                )


        def dlogTb_dz(rs):
            #if rs > 2e3:
            #    xHII = phys.xe_Saha(rs, 'HI')
            #elif rs<= 1:
            #    xHII = phys.xHII_std(1)
            #else:
            #    xHII = phys.xHII_std(rs)
            hub_rate = phys.hubble(rs)
            comp_rate = compton_cooling_rate(xHII, Tb, rs)

            adia = 2/rs
            comp = phys.dtdz(rs) * comp_rate * (phys.TCMB(rs)/Tb - 1)
            dlogTdz_b_DM = phys.dtdz(rs) * baryon * (T_DM/Tb-1)

            if hack:
                ## The dominant term is compton scattering: couple to T_CMB
                if comp_rate/(hub_rate + DM_rate) > 1e5:
                    return phys.dtdz(rs) * (phys.TCMB(rs)/Tb - 1)*1e4

                ### The dominant term is DM-baryon scattering: couple to T_DM
                #elif DM_rate/(hub_rate + comp_rate) > 1e5:
                #    return phys.dtdz(rs) * (T_DM/Tb - 1)*100

                else:
                    return adia + comp + dlogTdz_b_DM
            else:
                return adia + comp + fudge*dlogTdz_b_DM

        def dyHII_dz(rs):
            ne = xHII * nH
            xHI = 1 - xHII
            T_CMB = phys.TCMB(rs)


            #if not helium_TLA and xHII(yHII) > 0.99 and rs > 1500:
            #    # Use the Saha value. 
            #    return 2 * np.cosh(yHII)**2 * phys.d_xe_Saha_dz(rs, 'HI')


            recomb = phys.alpha_recomb(Tb, 'HI') * ne
            ion = 4*phys.beta_ion(T_CMB, 'HI') * np.exp(-phys.lya_eng/T_CMB)


            if hack:
                if rs>2e3:
                #if ion/recomb > 1e2:
                    return phys.d_xe_Saha_dz(rs, 'HI')
                else:
                    return 2 * np.cosh(yHII)**2 * phys.dtdz(rs) * (
                        - phys.peebles_C(xHII, rs) * (recomb*xHII - ion*xHI)
                        )
            else:
                return 2 * np.cosh(yHII)**2 * phys.dtdz(rs) * (
                    - phys.peebles_C(xHII, rs) * (recomb*xHII - ion*xHI)
                )

        if thermalized_be_bH:
            return np.array([dlogTDM_dz(rs), dybe_dz(rs), 
                dlogTb_dz(rs), dyHII_dz(rs)])
        else:
            #print(np.array([dlogTDM_dz(rs), dybe_dz(rs),
            #    dlogTb_dz(rs), dyHII_dz(rs), dlogTbH_dz(rs)]))
            return np.array([dlogTDM_dz(rs), dybe_dz(rs), 
                dlogTb_dz(rs), dyHII_dz(rs), dlogTbH_dz(rs)])


    if init_cond is None:
        #rs_start = np.exp(logrs_vec[0])
        rs_start = rs_vec[0]
        x_Saha = x_be_Saha(rs_start, alphaD, m_be, m_bp, xi)
        _init_cond = [
            xi*phys.TCMB(rs_start), 
            x_be_Saha(rs_start, alphaD, m_be, m_bp, xi),
            phys.TCMB(rs_start),
            phys.xe_Saha(rs_start, 'HI')
        ]
        if not thermalized_be_bH:
            _init_cond+=[xi*phys.TCMB(rs_start)]

    else:

        _init_cond = np.array(init_cond)

        if init_cond[1] == 1:
            _init_cond[1] = 1 - 1e-12
        if init_cond[3] == 1:
            _init_cond[3] = 1 - 1e-12

    _init_cond[0] = np.log(_init_cond[0])
    _init_cond[1] = np.arctanh(2*(_init_cond[1] - 0.5))
    _init_cond[2] = np.log(_init_cond[2])
    _init_cond[3] = np.arctanh(2*(_init_cond[3] - 0.5))
    if not thermalized_be_bH:
        _init_cond[4] = np.log(_init_cond[4])
    _init_cond = np.array(_init_cond)

    # Note: no reionization model implemented.
    soln = odeint(
            tla, _init_cond, rs_vec, 
            mxstep = mxstep, tfirst=True, rtol=rtol
        )
    #soln = rk(
    #    tla_before_reion, _init_cond,
    #    rs_vec[0], rs_vec[10], step=-.01, vec=None,
    #    rtol=1e-2
    #)

    # Convert from log_T_m to T_m
    soln[:,0] = np.exp(soln[:,0])
    soln[:,1] = 0.5 + 0.5*np.tanh(soln[:,1])
    soln[:,2] = np.exp(soln[:,2])
    soln[:,3] = 0.5 + 0.5*np.tanh(soln[:,3])
    if not thermalized_be_bH:
        soln[:,4] = np.exp(soln[:,4])

    return soln

def rk(deriv, init, t0, t_end, step, vec, rtol=0.001, atol=1e-6):

    # Constants -- Cash-Karp
    ca  = np.array([0, 1./5, 3./10, 3./5, 1, 7./8])
    cb  = np.array([
        [0, 0, 0, 0, 0],
        [1./5, 0, 0, 0, 0],
        [3./40, 9./40, 0, 0, 0],
        [3./10, -9./10, 6./5, 0, 0],
        [-11./54, 5./2, -70./27, 35./27, 0],
        [1631./55296, 175./512, 575./13824, 44275./110592, 253./4096]
    ])
    cc  = np.array([37./378, 0, 250./621, 125./594, 0, 512./1771])
    ccs = np.array([2825./27648, 0, 18575./48384, 13525./55296, 277./14336, 1./4])

    k   = np.zeros((6,init.size))
    yp,yn,ys  = np.zeros(6),np.zeros(6),np.zeros(6)

    # Initialize the loop
    y     = init
    t_cur = t0

    ylist = [init]
    tlist = [t_cur]

    sign = np.sign(t_end-t_cur)
    while sign*t_cur < sign*t_end:
        print(np.log10(t_cur))
        # Let the last step take you just to t_end
        if sign*(t_cur+step) > sign*t_end:
            step=t_end-t_cur
        for i in np.arange(6):
            yp = y+np.dot(cb[i],k[:5])
            k[i] = step*deriv(t_cur+ca[i]*step, yp)

        ys = y + np.dot(ccs,k)
        yn = y + np.dot(cc,k)

        #constraints on relative and absolute error
        delta = np.sum(np.sqrt(np.abs(ys-yn)))
        norm = np.sum(np.sqrt(np.abs(yn)))
        if norm>0:
            Delta = np.sum(np.sqrt(np.abs(ys-yn)))/norm
        else:
            Delta = 0

        if(delta>Delta):
            delta=Delta

        if delta == 0:
            delta = rtol

        if(delta > rtol):
            step = 0.9*step*(rtol/delta)**0.2
        else:
            y = yn
            t_cur += step;

            tlist.append(t_cur)
            ylist.append(y)

            # adjust next step
            step = 0.9*step*(rtol/delta)**0.2
            #if(step>0.05):
            #    step=0.05

    #if vec != None:
    #    return vec, interp1d(np.array(tlist),np.array(ylist))(vec)
    #else:
    return np.array(tlist), np.array(ylist)
