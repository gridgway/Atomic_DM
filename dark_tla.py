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

def norm_compton_cooling_rate(x_be, T_DM, rs, alphaD, m_be, m_bp, xi):
    """ Gamma_c where dlogT_DM/dt = Gamma_c (T_D - T_DM), in 1/s
    """
    T_D = xi*phys.TCMB(rs)
    pre = 64 * np.pi**3 * alphaD**2 / 135 * T_D**4 * x_be/(1+x_be)
    mass = (1 + (m_be/m_bp)**3) / m_be**3
    return pre * mass / phys.hbar

def compton_cooling_rate(xHII, T_m, rs):
    """SM Compton cooling rate. dE/dVdt"""
    ne = xHII * phys.nH*rs**3
    pre = 4 * phys.thomson_xsec * 4 * phys.stefboltz / phys.me
    return pre * (phys.TCMB(rs) - T_m) * phys.TCMB(rs)**4

# See Eqns. (A.1) and (A.2) #
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
    mu_D = m_be * m_bp/(m_be + m_bp)
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

def Gam_pi(x_be, T_DM, rs, alphaD, m_be, m_bp, xi):
    B_D  = get_BD(alphaD, m_be, m_bp)
    T_D  = xi * phys.TCMB(rs)
    m_D  = m_be + m_bp - B_D
    mu_D = m_be*m_bp/m_D
    n_D  = phys.rho_DM/m_D * rs**3
    denom = (3/2 * T_DM * n_D * (1+x_be))

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

def Gam_pr(x_be, T_DM, rs, alphaD, m_be, m_bp, xi):
    B_D = get_BD(alphaD, m_be, m_bp)
    T_D = xi * phys.TCMB(rs)
    m_D = m_be + m_bp - B_D
    mu_D = m_be*m_bp/m_D
    n_D = phys.rho_DM/m_D * rs**3
    denom = (3/2 * T_DM * n_D * (1+x_be))

    pre = 2*alphaD**3*np.sqrt(2*np.pi*T_DM)/(3*mu_D**(3/2))
    convert = (phys.hbar * phys.c)**3/phys.hbar

    return pre * x_be**2*n_D**2 * F_pr(T_D/B_D, T_DM/T_D) / denom * convert

def Gam_ff(x_be, T_DM, rs, alphaD, m_be, m_bp, xi):
    """ free-free absorption - emission (brem) / T_DM in 1/s """
    B_D = get_BD(alphaD, m_be, m_bp)
    T_D = xi * phys.TCMB(rs)
    eps = 1-T_DM/T_D

    m_D = m_be + m_bp - B_D
    mu_D = m_be*m_bp/m_D
    n_D = phys.rho_DM/m_D * rs**3
    g_ff = 1.33
    zeta_3 = 1.20206
    denom = (3/2 * T_DM * n_D * (1+x_be))

    pre = 16 * alphaD**3 * g_ff * x_be**2 * n_D**2 / (3*mu_D)**(3/2)
    prnth = np.pi**2 * (1+2*eps)/6 - zeta_3*eps
    convert = (phys.hbar * phys.c)**3/phys.hbar
    return pre * np.sqrt(2*np.pi*T_DM) * prnth / denom * convert

def Gam_R(x_be, T_DM, rs, alphaD, m_be, m_bp, xi):
    """ Rayleigh energy exchange rate / T_DM in 1/s """
    B_D = get_BD(alphaD, m_be, m_bp)
    T_D = xi * phys.TCMB(rs)
    m_D  = m_be + m_bp - B_D
    n_D  = phys.rho_DM/m_D * rs**3
    zeta_9 = 1.00201
    ratio = T_D/T_DM
    denom = (3/2 * (1+x_be))

    pre  = 430080 * zeta_9 * alphaD**2 * (1-x_be) / np.pi**2
    temps = (T_D/B_D)**4 * (T_D/m_be)**2 * T_D/m_D * T_D/phys.hbar
    return pre * temps / denom #ratio * convert

#!!! Make this correct
def DM_IGM_cooling_rate(m_be, m_bp, T_matter, T_DM, V_pec, xHII, rs, fDM, particle_type, eps=0):
    #See 1509.00029
    if eps == 0:
        return 0

    #!!!
    mDM = min(m_be,m_bp)

    mu_p = mDM*1.22*phys.mp/(mDM + 1.22*phys.mp)

    #u_th and r defined just after eqn. 13.
    u_th_p = np.sqrt(T_matter/phys.mp + T_DM/mDM)
    r_p = V_pec/u_th_p
    #Eqn. 14
    F_p = erf(r_p/np.sqrt(2)) - np.sqrt(2/np.pi)*r_p*np.exp(-r_p**2/2)
    #print(u_th_p, " ", r_p, " ", F_p)

    #Put (13) into (16) into (18) || (19), then put this into rate_x to see that they indeed match
    drag_cooling_term_p  = np.divide(mDM*F_p, r_p, out=np.zeros_like(F_p), where=r_p!=0)
    drag_cooling_term_DM_p = np.divide(phys.mp*F_p, r_p, out=np.zeros_like(F_p), where=r_p!=0)
    #print(drag_cooling_term_p, " ", drag_cooling_term_DM_p)

    #Everything's doubled, one for protons, one for electrons
    u_th_e = np.sqrt(T_matter/phys.me + T_DM/mDM)
    r_e = V_pec/u_th_e
    F_e = erf(r_e/np.sqrt(2)) - np.sqrt(2/np.pi)*r_e*np.exp(-r_e**2/2)

    drag_cooling_term_e  = np.divide(mDM*F_e, r_e, out=np.zeros_like(F_e), where=r_e!=0)
    drag_cooling_term_DM_e = np.divide(phys.me*F_e, r_e, out=np.zeros_like(F_e), where=r_e!=0)

    #Eqn 2, Munoz and Loeb
    xi = np.log(9*T_matter**3/(4*phys.hbar**3*phys.c**3*np.pi*eps**2*phys.alpha**3*xHII*phys.nH*rs**3))

    mu_p = mDM*phys.mp/(mDM + phys.mp)
    mu_e = mDM*phys.me/(mDM + phys.me)

    xsec_0_p = 2*np.pi*phys.alpha**2*eps**2*xi/mu_p**2*phys.hbar**2*phys.c**2
    xsec_0_e = 2*np.pi*phys.alpha**2*eps**2*xi/mu_e**2*phys.hbar**2*phys.c**2

    if particle_type == 'matter':
        rate_p = 2/(3*(1 + phys.nHe/phys.nH + xHII))*(fDM*phys.rho_DM*rs**3*phys.mp*xHII)/(mDM + phys.mp)**2*(
                (xsec_0_p/u_th_p)*(
                    np.sqrt(2/np.pi)*np.exp(-r_p**2/2)*(T_DM - T_matter)/u_th_p**2 + drag_cooling_term_p
                )
            )*phys.c

        rate_e = 2/(3*(1 + phys.nHe/phys.nH + xHII))*(fDM*phys.rho_DM*rs**3*phys.me*xHII)/(mDM + phys.me)**2*(
                (xsec_0_e/u_th_e)*(
                    np.sqrt(2/np.pi)*np.exp(-r_e**2/2)*(T_DM - T_matter)/u_th_e**2 + drag_cooling_term_e
                )
            )*phys.c
    elif particle_type == 'DM':
        rate_p = 2/3*(m_bp*phys.mp*xHII*phys.nH*rs**3)/(m_bp + phys.mp)**2*(
            (xsec_0_p/u_th_p)*(
                np.sqrt(2/np.pi)*np.exp(-r_p**2/2)*(T_matter - T_DM)/u_th_p**2 + drag_cooling_term_DM_p
            )
        )*phys.c

        rate_p += 2/3*(m_be*phys.mp*xHII*phys.nH*rs**3)/(m_be + phys.mp)**2*(
            (xsec_0_p/u_th_p)*(
                np.sqrt(2/np.pi)*np.exp(-r_p**2/2)*(T_matter - T_DM)/u_th_p**2 + drag_cooling_term_DM_p
            )
        )*phys.c

        rate_e = 2/3*(m_bp*phys.me*xHII*phys.nH*rs**3)/(m_bp + phys.me)**2*(
            (xsec_0_e/u_th_e)*(
                np.sqrt(2/np.pi)*np.exp(-r_e**2/2)*(T_matter - T_DM)/u_th_e**2 + drag_cooling_term_DM_e
            )
        )*phys.c

        rate_e += 2/3*(m_be*phys.me*xHII*phys.nH*rs**3)/(m_be + phys.me)**2*(
            (xsec_0_e/u_th_e)*(
                np.sqrt(2/np.pi)*np.exp(-r_e**2/2)*(T_matter - T_DM)/u_th_e**2 + drag_cooling_term_DM_e
            )
        )*phys.c
    else:
        raise TypeError('Invalid particle_type.')

    return rate_p + rate_e


def get_history(
    rs_vec, init_cond=None, 
    alphaD=phys.alpha, m_be=phys.me, m_bp=phys.mp, xi=1,
    both_sectors = False, eps = 0,
    mxstep = 1000, rtol=1e-4
):
    """Returns the ionization and thermal history of the IGM.

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

    # Define conversion functions between x and y. 
    def get_x(y):
            return 0.5 + 0.5*np.tanh(y)

    def tla_before_reion(rs, var):
        # Returns an array of values for [dT_DM/dz, dy_be/dz].
        # var = [T_DM, x_be]
        #rs = np.exp(logrs)

        n_D  = phys.rho_DM/m_D * rs**3
        #n_D = phys.nH * rs**3 #To match the SM result
        T_D  = phys.TCMB(rs)*xi
        def dlogTDM_dz(y_be, log_TDM, rs):
            #rs = np.exp(logrs)

            T_DM = np.exp(log_TDM)
            x_be = get_x(y_be)
            eps = 1-T_DM/T_D

            # Cooling rate due to adiabatic expansion
            adia  = 2/rs

            #Compton
            comp  = phys.dtdz(rs) * norm_compton_cooling_rate(
                    x_be, T_DM, rs, alphaD, m_be, m_bp, xi
            ) * (T_D/T_DM - 1.)

            #Rayleigh
            Rayl  = phys.dtdz(rs) * Gam_R(
                    x_be, T_DM, rs, alphaD, m_be, m_bp, xi) * (T_D/T_DM-1)

            #brem heating + ion heating - recomb cooling
            ff_pi_pr = phys.dtdz(rs) * (
                    Gam_ff(x_be, T_DM, rs, alphaD, m_be, m_bp, xi) * eps
                    + Gam_pi(x_be, T_DM, rs, alphaD, m_be, m_bp, xi)
                    - Gam_pr(x_be, T_DM, rs, alphaD, m_be, m_bp, xi)
                    -0
                    )

            deriv = Rayl + comp + adia + ff_pi_pr

            #Baryon-DM energy exchange
            V_pec = 0
            fDM = 2 * x_be
            xsec = None #DM baryon scattering cross-section
            #baryon = phys.dtdz(rs) * DM_IGM_cooling_rate(
            #    m_be, m_bp, phys.TCMB(rs), T_DM, V_pec, x_be, rs,
            #    fDM, particle_type='DM', eps=eps
            #)/(3/2 * n_D * (1 + x_be))


            return deriv


        def dybe_dz(y_be, log_T, rs):
            #rs = np.exp(logrs)
            T = np.exp(log_T)
            x_be = get_x(y_be)
            xD = 1 - x_be

            if x_be > 0.999 and rs > 2000:
                # Use the Saha value. 
                dxdz = d_x_be_Saha_dz(rs, alphaD, m_be, m_bp, xi)
                return 2 * np.cosh(y_be)**2 * dxdz


            peeb_C = dark_peebles_C(x_be, rs, alphaD, m_be, m_bp, xi)
            alpha = dark_alpha_recomb(T, alphaD, m_be, m_bp, xi)
            beta = dark_beta_ion(T_D, alphaD, m_be, m_bp, xi)
            return 2 * np.cosh(y_be)**2 * phys.dtdz(rs) * (
                - peeb_C * (alpha * x_be**2 * n_D 
                    - 4 * beta * xD * np.exp(-Lya_D/T_D)
                    )
                )


        nH = phys.nH*rs**3
        def dlogTb_dz(yHII, log_Tb, rs):

            Tb = np.exp(log_Tb)

            xHII = get_x(yHII)
            xHI = 1 - xHII

            adia = 2/rs
            denom = 3/2 * Tb * nH * (1 + chi + xe)
            comp = phys.dtdz(rs) * compton_cooling_rate(xHII, Tb, rs)/denom

            return adia + comp


        log_TDM, y_be = var[0], var[1]

        if both_sectors:
            return 0
        else:
            return np.array([dlogTDM_dz(y_be, log_TDM, rs), 
                    dybe_dz(y_be, log_TDM, rs)])


    if init_cond is None:
        #rs_start = np.exp(logrs_vec[0])
        rs_start = rs_vec[0]
        x_Saha = x_be_Saha(rs_start, alphaD, m_be, m_bp, xi)
        _init_cond = [xi*phys.TCMB(rs_start), x_Saha]

    else:

        _init_cond = np.array(init_cond)

        if init_cond[1] == 1:
            _init_cond[1] = 1 - 1e-12

    _init_cond[0] = np.log(_init_cond[0])
    _init_cond[1] = np.arctanh(2*(_init_cond[1] - 0.5))
    _init_cond = np.array(_init_cond)

    # Note: no reionization model implemented.
    soln = odeint(
            tla_before_reion, _init_cond, rs_vec, 
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
