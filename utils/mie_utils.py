"""
Mie scattering calculations for perfect spheres JITTED!.

Extensive documentation is at <https://miepython.readthedocs.io>

`miepython` is a pure Python module to calculate light scattering of
a plane wave by non-np.absorbing, partially-np.absorbing, or perfectly conducting
spheres.

The extinction efficiency, scattering efficiency, backscattering, and
scattering asymmetry for a sphere with complex index of refraction m,
diameter d, and wavelength lambda can be found by::

    qext, qsca, qback, g = miepython.ez_mie(m, d, lambda0)

The normalized scattering values for angles mu=cos(theta) are::

    Ipar, Iper = miepython.ez_intensities(m, d, lambda0, mu)

If the size parameter is known, then use::

    miepython.mie(m, x)

Mie scattering amplitudes S1 and S2 (complex numbers):

    miepython.mie_S1_S2(m, x, mu)

Normalized Mie scattering intensities for angles mu=cos(theta)::

    miepython.i_per(m, x, mu)
    miepython.i_par(m, x, mu)
    miepython.i_unpolarized(m, x, mu)
"""

import numpy as np
from numba import njit, int32, float64, complex128

__all__ = ('ez_mie',
           'ez_intensities',
           'i_par',
           'i_per',
           'i_unpolarized',
           'mie',
           'mie_S1_S2',
           'mie_phase_matrix',
           'mie_cdf',
           'mie_mu_with_uniform_cdf',
           'generate_mie_costheta',
           )


@njit((complex128, int32), cache=True)
def _Lentz_Dn(z, N):
    """
    Compute the logarithmic derivative of the Ricatti-Bessel function.

    Args:
        z: function argument
        N: order of Ricatti-Bessel function

    Returns:
        This returns the Ricatti-Bessel function of order N with argument z
        using the continued fraction technique of Lentz, Appl. Opt., 15,
        668-671, (1976).
    """
    zinv = 2.0 / z
    alpha = (N + 0.5) * zinv
    aj = -(N + 1.5) * zinv
    alpha_j1 = aj + 1 / alpha
    alpha_j2 = aj
    ratio = alpha_j1 / alpha_j2
    runratio = alpha * ratio

    while np.abs(np.abs(ratio) - 1.0) > 1e-12:
        aj = zinv - aj
        alpha_j1 = 1.0 / alpha_j1 + aj
        alpha_j2 = 1.0 / alpha_j2 + aj
        ratio = alpha_j1 / alpha_j2
        zinv *= -1
        runratio = ratio * runratio

    return -N / z + runratio


@njit((complex128, int32, complex128[:]), cache=True)
def _D_downwards(z, N, D):
    """
    Compute the logarithmic derivative by downwards recurrence.

    Args:
        z: function argument
        N: order of Ricatti-Bessel function
        D: gets filled with the Ricatti-Bessel function values for orders
           from 0 to N for an argument z using the downwards recurrence relations.
    """
    last_D = _Lentz_Dn(z, N)
    for n in range(N, 0, -1):
        last_D = n / z - 1.0 / (last_D + n / z)
        D[n - 1] = last_D


@njit((complex128, int32, complex128[:]), cache=True)
def _D_upwards(z, N, D):
    """
    Compute the logarithmic derivative by upwards recurrence.

    Args:
        z: function argument
        N: order of Ricatti-Bessel function
        D: gets filled with the Ricatti-Bessel function values for orders
           from 0 to N for an argument z using the upwards recurrence relations.
    """
    exp = np.exp(-2j * z)
    D[1] = -1 / z + (1 - exp) / ((1 - exp) / z - 1j * (1 + exp))
    for n in range(2, N):
        D[n] = 1 / (n / z - D[n - 1]) - n / z


@njit((complex128, float64, int32), cache=True)
def _D_calc(m, x, N):
    """
    Compute the logarithmic derivative using best method.

    Args:
        m: the complex index of refraction of the sphere
        x: the size parameter of the sphere
        N: order of Ricatti-Bessel function

    Returns:
        The values of the Ricatti-Bessel function for orders from 0 to N.
    """
    n = m.real
    kappa = np.abs(m.imag)
    D = np.zeros(N, dtype=np.complex128)

    if n < 1 or n > 10 or kappa > 10 or x * kappa >= 3.9 - 10.8 * n + 13.78 * n**2:
        _D_downwards(m * x, N, D)
    else:
        _D_upwards(m * x, N, D)
    return D


@njit((complex128, float64), cache=True)
def _mie_An_Bn(m, x):
    """
    Compute arrays of Mie coefficients A and B for a sphere.

    This estimates the size of the arrays based on Wiscombe's formula. The length
    of the arrays is chosen so that the error when the series are summed is
    around 1e-6.

    Args:
        m: the complex index of refraction of the sphere
        x: the size parameter of the sphere

    Returns:
        a, b: arrays of Mie coefficents An and Bn
    """
    nstop = int(x + 4.05 * x**0.33333 + 2.0) + 1
    a = np.zeros(nstop - 1, dtype=np.complex128)
    b = np.zeros(nstop - 1, dtype=np.complex128)

    psi_nm1 = np.sin(x)                   # nm1 = n-1 = 0
    psi_n = psi_nm1 / x - np.cos(x)       # n = 1
    xi_nm1 = complex(psi_nm1, np.cos(x))
    xi_n = complex(psi_n, np.cos(x) / x + np.sin(x))

    if m.real > 0.0:
        D = _D_calc(m, x, nstop + 1)

        for n in range(1, nstop):
            temp = D[n] / m + n / x
            a[n - 1] = (temp * psi_n - psi_nm1) / (temp * xi_n - xi_nm1)
            temp = D[n] * m + n / x
            b[n - 1] = (temp * psi_n - psi_nm1) / (temp * xi_n - xi_nm1)
            xi = (2 * n + 1) * xi_n / x - xi_nm1
            xi_nm1 = xi_n
            xi_n = xi
            psi_nm1 = psi_n
            psi_n = xi_n.real

    else:
        for n in range(1, nstop):
            a[n - 1] = (n * psi_n / x - psi_nm1) / (n * xi_n / x - xi_nm1)
            b[n - 1] = psi_n / xi_n
            xi = (2 * n + 1) * xi_n / x - xi_nm1
            xi_nm1 = xi_n
            xi_n = xi
            psi_nm1 = psi_n
            psi_n = xi_n.real

    return a, b


@njit((complex128, float64), cache=True)
def _small_conducting_mie(_m, x):
    """
    Calculate the efficiencies for a small conducting spheres.

    Typically used for small conducting spheres where x < 0.1 and
    m.real == 0

    Args:
        _m: the complex index of refraction of the sphere (unused)
        x: the size parameter of the sphere

    Returns:
        qext: the total extinction efficiency
        qsca: the scattering efficiency
        qback: the backscatter efficiency
        g: the average cosine of the scattering phase function
    """
    ahat1 = complex(0, 2.0 / 3.0 * (1 - 0.2 * x**2))
    ahat1 /= complex(1 - 0.5 * x**2, 2.0 / 3.0 * x**3)

    bhat1 = complex(0.0, (x**2 - 10.0) / 30.0)
    bhat1 /= complex(1 + 0.5 * x**2, -x**3 / 3.0)
    ahat2 = complex(0.0, x**2 / 30.0)
    bhat2 = complex(0.0, -x**2 / 45.0)

    qsca = x**4 * (6 * np.abs(ahat1)**2
                   + 6 * np.abs(bhat1)**2
                   + 10 * np.abs(ahat2)**2
                   + 10 * np.abs(bhat2)**2)
    qext = qsca
    g = ahat1.imag * (ahat2.imag + bhat1.imag)
    g += bhat2.imag * (5.0 / 9.0 * ahat2.imag + bhat1.imag)
    g += ahat1.real * bhat1.real
    g *= 6 * x**4 / qsca

    qback = 9 * x**4 * np.abs(ahat1 - bhat1 - 5 / 3 * (ahat2 - bhat2))**2

    return [qext, qsca, qback, g]


@njit((complex128, float64), cache=True)
def _small_mie(m, x):
    """
    Calculate the efficiencies for a small sphere.

    Typically used for small spheres where x<0.1

    Args:
        m: the complex index of refraction of the sphere
        x: the size parameter of the sphere

    Returns:
        qext: the total extinction efficiency
        qsca: the scattering efficiency
        qback: the backscatter efficiency
        g: the average cosine of the scattering phase function
    """
    m2 = m * m
    x2 = x * x

    D = m2 + 2 + (1 - 0.7 * m2) * x2
    D -= (8 * m**4 - 385 * m2 + 350) * x**4 / 1400.0
    D += 2j * (m2 - 1) * x**3 * (1 - 0.1 * x2) / 3
    ahat1 = 2j * (m2 - 1) / 3 * (1 - 0.1 * x2 + (4 * m2 + 5) * x**4 / 1400) / D

    bhat1 = 1j * x2 * (m2 - 1) / 45 * (1 + (2 * m2 - 5) / 70 * x2)
    bhat1 /= 1 - (2 * m2 - 5) / 30 * x2

    ahat2 = 1j * x2 * (m2 - 1) / 15 * (1 - x2 / 14)
    ahat2 /= 2 * m2 + 3 - (2 * m2 - 7) / 14 * x2

    T = np.abs(ahat1)**2 + np.abs(bhat1)**2 + 5 / 3 * np.abs(ahat2)**2
    temp = ahat2 + bhat1
    g = (ahat1 * temp.conjugate()).real / T

    qsca = 6 * x**4 * T

    if m.imag == 0:
        qext = qsca
    else:
        qext = 6 * x * (ahat1 + bhat1 + 5 * ahat2 / 3).real

    sback = 1.5 * x**3 * (ahat1 - bhat1 - 5 * ahat2 / 3)
    qback = 4 * np.abs(sback)**2 / x2

    return [qext, qsca, qback, g]


@njit((complex128, float64), cache=True)
def _mie_scalar(m, x):
    """
    Calculate the efficiencies for a sphere when both m and x are scalars.

    Args:
        m: the complex index of refraction of the sphere
        x: the size parameter of the sphere

    Returns:
        qext: the total extinction efficiency
        qsca: the scattering efficiency
        qback: the backscatter efficiency
        g: the average cosine of the scattering phase function
    """
    if m.real == 0 and x < 0.1:
        qext, qsca, qback, g = _small_conducting_mie(m, x)

    elif m.real > 0.0 and np.abs(m) * x < 0.1:
        qext, qsca, qback, g = _small_mie(m, x)

    else:
        a, b = _mie_An_Bn(m, x)

        nmax = len(a)
        n = np.arange(1, nmax + 1)
        cn = 2.0 * n + 1.0

        qext = 2 * np.sum(cn * (a.real + b.real)) / x**2
        qsca = qext

        if m.imag != 0:
            qsca = 2 * np.sum(cn * (np.abs(a)**2 + np.abs(b)**2)) / x**2

        qback = np.abs(np.sum((-1)**n * cn * (a - b)))**2 / x**2

        c1n = n * (n + 2) / (n + 1)
        c2n = cn / n / (n + 1)
        g = 0
        for i in range(nmax - 1):
            asy1 = c1n[i] * (a[i] * a[i + 1].conjugate()
                             + b[i] * b[i + 1].conjugate()).real
            asy2 = c2n[i] * (a[i] * b[i].conjugate()).real
            g += 4 * (asy1 + asy2) / qsca / x**2

    return [qext, qsca, qback, g]


def mie(m, x):
    """
    Calculate the efficiencies for a sphere where m or x may be arrays.

    Args:
        m: the complex index of refraction of the sphere
        x: the size parameter of the sphere

    Returns:
        qext: the total extinction efficiency
        qsca: the scattering efficiency
        qback: the backscatter efficiency
        g: the average cosine of the scattering phase function
    """
    mm = m
    xx = x

    if np.isscalar(m) and np.isscalar(x):
        return _mie_scalar(mm, xx)

    if np.isscalar(m):
        mlen = 0
    else:
        mlen = len(m)

    if np.isscalar(x):
        xlen = 0
    else:
        xlen = len(x)

    if xlen > 0 and mlen > 0 and xlen != mlen:
        raise RuntimeError('m and x arrays to mie must be same length')

    thelen = max(xlen, mlen)
    qext = np.empty(thelen, dtype=np.float64)
    qsca = np.empty(thelen, dtype=np.float64)
    qback = np.empty(thelen, dtype=np.float64)
    g = np.empty(thelen, dtype=np.float64)

    for i in range(thelen):
        if mlen > 0:
            mm = m[i]

        if xlen > 0:
            xx = x[i]

        qext[i], qsca[i], qback[i], g[i] = _mie_scalar(mm, xx)

    return qext, qsca, qback, g


@njit((complex128, float64, int32), cache=True)
def normalization_factor(m, x, norm_int):
    """
    Figure out scattering function normalization.

    Args:
        m: the complex index of refraction of the sphere
        x: dimensionless sphere size
        norm_int: integer that specifies normalization

    Returns:
        scaling factor needed for scattering function
    """
    norm = None

    # Qsca normalization
    if norm_int == 3:
        norm = x * np.sqrt(np.pi)

    # Bohren Normalization
    elif norm_int == 5:
        norm = 0.5

    # Wiscombe Normalization
    elif norm_int == 6:
        norm = 1

    else:
        # calculate qsca and qext
        qext, qsca, _, _ = _mie_scalar(m, x)

        # albedo Normalization
        if norm_int == 0:
            norm = x * np.sqrt(np.pi * qext)

        # Unity normalization
        elif norm_int == 1:
            norm = x * np.sqrt(qsca * np.pi)

        # 4pi Normalization
        elif norm_int == 2:
            norm = x * np.sqrt(qsca / 4)

        # Qext Normalization
        elif norm_int == 4:  # 4pi
            norm = x * np.sqrt(qsca * np.pi / qext)

    if norm is None:
        raise ValueError("norm-int must be in the range 0..6")

    return norm


def norm_string_to_integer(s):
    """
    Encode normalization choice as an integer.

    This is needed because these string operations cannot be
    done in a jitted function under numba.  We cannot use enums for
    the same reason!

    Args:
        s: string describing normalization desired.

    Returns:
        integer used in _mie_S1_S2() determine normalization
    """
    ii = None
    norm = s.lower()

    if norm in ['a', 'albedo']:
        ii = 0

    elif norm in ['1', 'one', 'unity']:
        ii = 1

    elif norm in ['four_pi', '4pi']:
        ii = 2

    elif norm in ['qsca', 'scattering_efficiency']:
        ii = 3

    elif norm in ['qext', 'extinction_efficiency']:
        ii = 4

    elif norm in ['bohren']:
        ii = 5

    elif norm in ['wiscombe']:
        ii = 6

    if ii is None:
        raise ValueError("normalization must be one of 'albedo' (default), 'one'"
                         "'4pi', 'qext', 'qsca', 'bohren', or 'wiscombe'")
    return ii


@njit((complex128, float64, float64[:], int32), cache=True)
def _mie_S1_S2(m, x, mu, norm_int):
    """
    Calculate the scattering amplitude functions for spheres.

    The amplitude functions have been normalized so that when integrated
    over all 4*pi solid angles, the integral will be qext*pi*x**2.

    The units are weird, sr**(-0.5)

    Args:
        m: the complex index of refraction of the sphere
        x: the size parameter of the sphere
        mu: array of angles, cos(theta), to calculate scattering amplitudes
        norm_int: integer describing type of normalization

    Returns:
        S1, S2: the scattering amplitudes at each angle mu [sr**(-0.5)]
    """
    a, b = _mie_An_Bn(m, x)

    nangles = len(mu)
    S1 = np.zeros(nangles, dtype=np.complex128)
    S2 = np.zeros(nangles, dtype=np.complex128)

    nstop = len(a)
    for k in range(nangles):
        pi_nm2 = 0
        pi_nm1 = 1
        for n in range(1, nstop):
            tau_nm1 = n * mu[k] * pi_nm1 - (n + 1) * pi_nm2
            S1[k] += (2 * n + 1) * (pi_nm1 * a[n - 1] + tau_nm1 * b[n - 1]) / (n + 1) / n
            S2[k] += (2 * n + 1) * (tau_nm1 * a[n - 1] + pi_nm1 * b[n - 1]) / (n + 1) / n
            temp = pi_nm1
            pi_nm1 = ((2 * n + 1) * mu[k] * pi_nm1 - (n + 1) * pi_nm2) / n
            pi_nm2 = temp

    normalization = normalization_factor(m, x, norm_int)

    S1 /= normalization
    S2 /= normalization

    return [S1, S2]


def mie_S1_S2(m, x, mu, norm='albedo'):
    """
    Calculate the scattering amplitude functions for spheres.

    The normalization is controlled by `norm` and should be one of
    ['albedo', 'one', '4pi', 'qext', 'qsca', 'bohren', or 'wiscombe']
    The normalization describes the integral of the scattering phase
    function over all 4𝜋 steradians.

    The units are weird, sr**(-0.5)

    Args:
        m: the complex index of refraction of the sphere
        x: the size parameter of the sphere
        mu: cos(theta) or array of angles [cos(theta_i)]
        norm: (optional) string describing scattering function normalization

    Returns:
        S1, S2: the scattering amplitudes at each angle mu [sr**(-0.5)]
    """
    norm_int = norm_string_to_integer(norm)
    if np.isscalar(mu):
        mu_array = np.array([mu], dtype=float)
        s1, s2 = _mie_S1_S2(m, x, mu_array, norm_int)
        return s1[0], s2[0]

    return _mie_S1_S2(m, x, mu, norm_int)


def mie_phase_matrix(m, x, mu, norm='albedo'):
    """
    Calculate the scattering (Mueller) matrix.

    If mu has length N, then the returned matrix is 4x4xN.  If mu is a scalar
    then the matrix is 4x4

    The phase scattering matrix is computed from the scattering amplitude
    functions, according to equations 5.2.105-6 in K. N. Liou (**2002**) -
    *An Introduction to Atmospheric Radiation*, Second Edition.

    or

    Bohren and Huffman, *Absorption and Scattering of Light by Small Particles*,
    JOHN WILEY & SONS, page 112, (1983).

    The normalization is controlled by `norm` and should be one of
    ['albedo', 'one', '4pi', 'qext', 'qsca', 'bohren', or 'wiscombe']
    The normalization describes the integral of the scattering phase
    function over all 4𝜋 steradians.

    Args:
        m: the complex index of refraction of the sphere
        x: the size parameter of the sphere
        mu: the angles, cos(theta), for the phase scattering matrix
        norm: (optional) string describing scattering function normalization

    Returns:
        p: the phase scattering matrix [sr**(-1.0)]
    """
    s1, s2 = mie_S1_S2(m=m, x=x, mu=mu, norm=norm)

    mu = np.atleast_1d(mu)
    s1_star = np.conjugate(s1)
    s2_star = np.conjugate(s2)
    m1 = (s1 * s1_star).real
    m2 = (s2 * s2_star).real
    s21 = (0.5 * (s1 * s2_star + s2 * s1_star)).real
    d21 = (-0.5j * (s1 * s2_star - s2 * s1_star)).real
    phase = np.zeros(shape=(4, 4, mu.size))
    phase[0, 0] = 0.5 * (m2 + m1)
    phase[0, 1] = 0.5 * (m2 - m1)
    phase[1, 0] = phase[0, 1]
    phase[1, 1] = phase[0, 0]
    phase[2, 2] = s21
    phase[2, 3] = -d21
    phase[3, 2] = d21
    phase[3, 3] = s21

    return phase.squeeze()


def mie_cdf(m, x, num):
    """
    Create a CDF for unpolarized scattering uniformly spaced in cos(theta).

    The CDF covers scattered (exit) angles ranging from 180 to 0 degrees.
    (The cosines are uniformly distributed over -1 to 1.) Because the angles
    are uniformly distributed in cos(theta), the scattering function is not
    sampled uniformly and therefore huge array sizes are needed to adequately
    sample highly anisotropic phase functions.

    Since this is a cumulative distribution function, the maximum value
    should be 1.

    Args:
        m: the complex index of refraction of the sphere
        x: the size parameter of the sphere
        num: length of desired CDF array

    Returns:
        mu: array of cosines of angles
        cdf: array of cumulative distribution function values
    """
    mu = np.linspace(-1, 1, num)
    intensity_per_mu = i_unpolarized(m, x, mu, norm='4pi') / num
    cdf = np.cumsum(intensity_per_mu)
    return mu, cdf


def mie_mu_with_uniform_cdf(m, x, num):
    """
    Create a CDF for unpolarized scattering for uniform CDF.

    The CDF covers scattered (exit) angles ranging from 180 to 0 degrees.
    (The cosines are uniformly distributed over -1 to 1.) These angles mu
    correspond to uniform spacing of the cumulative distribution function
    for unpolarized Mie scattering where cdf[i] = i/(num-1).

    This is a brute force implementation that solves the problem by
    calculating the CDF at many points and then scanning to find the
    specific angles that correspond to uniform interval of the CDF.

    Since this is a cumulative distribution function, the maximum value
    should be 1.

    Args:
        m: the complex index of refraction of the sphere
        x: the size parameter of the sphere
        num: length of desired CDF array

    Returns:
        mu: array of cosines of angles (irregularly spaced)
        cdf: array of cumulative distribution function values
    """
    big_num = 2000                  # large to work with x up to 10
    big_mu, big_cdf = mie_cdf(m, x, big_num)
    mu = np.empty(num)
    cdf = np.empty(num)

    mu[0] = -1                       # cos[180 degrees] is -1
    cdf[0] = 0                       # initial cdf is zero

    big_k = 0                        # index into big_cdf
    for k in range(1, num - 1):

        target = k / (num - 1)
        while big_cdf[big_k] < target:
            big_k += 1

        delta = big_cdf[big_k] - target
        delta_cdf = big_cdf[big_k] - big_cdf[big_k - 1]
        delta_mu = big_mu[big_k] - big_mu[big_k - 1]

        mu[k] = big_mu[big_k] - delta / delta_cdf * delta_mu   # interpolate
        cdf[k] = target

#       print(' mu[',k,']=% .5f'%mu[k],' cdf[',k,']=% .5f'%cdf[k],
#       'cdf=',big_cdf[big_k], fraction)

    mu[num - 1] = 1                    # cos[0 degrees] is 1
    cdf[num - 1] = 1                   # last cdf is one

    return [mu, cdf]


def generate_mie_costheta(mu_cdf):
    """
    Generate a new scattering angle using a cdf.

    A uniformly spaced cumulative distribution function (CDF) is needed.
    New random angles are generated by selecting a random interval
    mu[i] to mu[i+1] and choosing an angle uniformly distributed over
    the interval.

    Args:
       mu_cdf: a cumulative distribution function

    Returns:
       The cosine of the scattering angle
    """
    # the following should be equivalent to these four lines
    # index = np.random.randint(0, high=len(mu_cdf))
    num = len(mu_cdf) - 1
    index = int(np.random.random() * num)
    if index >= num:
        index = num - 1

    x = mu_cdf[index]
    x += (mu_cdf[index + 1] - mu_cdf[index]) * np.random.random()

    return x


def i_per(m, x, mu, norm='albedo'):
    """
    Return the scattered intensity in a plane normal to the incident light.

    This is the scattered intensity in a plane that is perpendicular to the
    field of the incident plane wave. The intensity is normalized such
    that the integral of the unpolarized intensity over 4π steradians
    is equal to the single scattering albedo.

    The normalization is controlled by `norm` and should be one of
    ['albedo', 'one', '4pi', 'qext', 'qsca', 'bohren', or 'wiscombe']
    The normalization describes the integral of the scattering phase
    function over all 4𝜋 steradians.

    Args:
        m: the complex index of refraction of the sphere
        x: the size parameter of the sphere
        mu: the angles, cos(theta), to calculate intensities
        norm: (optional) string describing scattering function normalization

    Returns:
        The intensity at each angle in the array mu.  Units [1/sr]
    """
    s1, _ = mie_S1_S2(m, x, mu, norm)
    intensity = np.abs(s1)**2
    return intensity.astype('float')


def i_par(m, x, mu, norm='albedo'):
    """
    Return the scattered intensity in a plane parallel to the incident light.

    This is the scattered intensity in a plane that is parallel to the
    field of the incident plane wave. The intensity is normalized such
    that the integral of the unpolarized intensity over 4π steradians
    is equal to the single scattering albedo.

    The normalization is controlled by `norm` and should be one of
    ['albedo', 'one', '4pi', 'qext', 'qsca', 'bohren', or 'wiscombe']
    The normalization describes the integral of the scattering phase
    function over all 4𝜋 steradians.

    Args:
        m: the complex index of refraction of the sphere
        x: the size parameter of the sphere
        mu: the angles, cos(theta), to calculate intensities
        norm: (optional) string describing scattering function normalization

    Returns:
        The intensity at each angle in the array mu.  Units [1/sr]
    """
    _, s2 = mie_S1_S2(m, x, mu, norm)
    intensity = np.abs(s2)**2
    return intensity.astype('float')


def i_unpolarized(m, x, mu, norm='albedo'):
    """
    Return the unpolarized scattered intensity at specified angles.

    This is the average value for randomly polarized incident light.
    The intensity is normalized such that the integral of the unpolarized
    intensity over 4π steradians is equal to the single scattering albedo.

    The normalization is controlled by `norm` and should be one of
    ['albedo', 'one', '4pi', 'qext', 'qsca', 'bohren', or 'wiscombe']
    The normalization describes the integral of the scattering phase
    function over all 4𝜋 steradians.

    Args:
        m: the complex index of refraction of the sphere
        x: the size parameter of the sphere
        mu: the angles, cos(theta), to calculate intensities
        norm: (optional) string describing scattering function normalization

    Returns:
        The intensity at each angle in the array mu.  Units [1/sr]
    """
    s1, s2 = mie_S1_S2(m, x, mu, norm)
    intensity = (np.abs(s1)**2 + np.abs(s2)**2) / 2
    return intensity.astype('float')


def ez_mie(m, d, lambda0, n_env=1.0):
    """
    Calculate the efficiencies of a sphere.

    Args:
        m: the complex index of refraction of the sphere [-]
        d: the diameter of the sphere                    [same units as lambda0]
        lambda0: wavelength in a vacuum                  [same units as d]
        n_env: real index of medium around sphere, optional.

    Returns:
        qext: the total extinction efficiency                  [-].
        qsca: the scattering efficiency                        [-].
        qback: the backscatter efficiency                      [-].
        g: the average cosine of the scattering phase function [-].
    """
    m_env = m / n_env
    x_env = np.pi * d / (lambda0 / n_env)
    return mie(m_env, x_env)


def ez_intensities(m, d, lambda0, mu, n_env=1.0, norm='albedo'):
    """
    Return the scattered intensities from a sphere.

    These are the scattered intensities in a plane that is parallel (ipar) and
    perpendicular (iper) to the field of the incident plane wave.

    The scattered intensity is normalized such that the integral of the
    unpolarized intensity over 4𝜋 steradians is equal to the single scattering
    albedo.  The scattered intensity has units of inverse steradians [1/sr].

    The unpolarized scattering is the average of the two scattered intensities.

    The normalization is controlled by `norm` and should be one of
    ['albedo', 'one', '4pi', 'qext', 'qsca', 'bohren', or 'wiscombe']
    The normalization describes the integral of the scattering phase
    function over all 4𝜋 steradians.

    Args:
        m: the complex index of refraction of the sphere [-]
        d: the diameter of the sphere                    [same units as lambda0]
        lambda0: wavelength in a vacuum                  [same units as d]
        mu: the cos(theta) of each direction desired     [-]
        n_env: real index of medium around sphere, optional
        norm: (optional) string describing scattering function normalization

    Returns:
        ipar, iper: scattered intensity in parallel and perpendicular planes [1/sr]
    """
    m_env = m / n_env
    lambda_env = lambda0 / n_env
    x_env = np.pi * d / lambda_env
    s1, s2 = mie_S1_S2(m_env, x_env, mu, norm)
    ipar = np.abs(s2)**2
    iper = np.abs(s1)**2
    Ipar = ipar.astype('float')
    Iper = iper.astype('float')
    return Ipar, Iper