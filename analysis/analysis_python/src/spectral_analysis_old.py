"""
Original version of spectral_analysis.py before vectorization and Dask/xarray optimizations.
"""

import xarray as xr
import pyshtools as pysh
import numpy as np
from analysis.analysis_python.src.cospectra_old import compute_eddy_cospectra
import matplotlib.pyplot as plt
from utils import get_zavg, get_pderiv, get_phideriv


def spectral_analysis(ucomp, vcomp, theta, ps, phalf, r, omega, meanstats=None, apply_smoothing=False, smooth_width=3):
    """
    Compute comprehensive spectral diagnostics for FMS model output.
    
    Parameters
    ----------
    ucomp : xarray.DataArray
        Zonal wind component
    vcomp : xarray.DataArray
        Meridional wind component
    theta : xarray.DataArray
        Potential temperature
    ps : xarray.DataArray
        Surface pressure
    phalf : xarray.DataArray
        Pressure levels at half levels
    r : float
        Planet radius
    omega : float
        Planet rotation rate
    smooth : int, optional
        Smoothing parameter for cospectra, by default 0
    width : int, optional
        Width parameter for cospectra smoothing, by default 3
        
    Returns
    -------
    spectrum : xarray.Dataset
        Vertically averaged spherical harmonic spectra
        Contains variables:
        - eke_spectrum: Total eddy kinetic energy spectrum
        - eke_bt_spectrum: Barotropic EKE spectrum
        - eke_bc_spectrum: Baroclinic EKE spectrum
        - u_spectrum, v_spectrum: Velocity component spectra
        - theta_spectrum: Potential temperature spectrum
        (each with barotropic/baroclinic components)
    spectrum_sigma : xarray.Dataset
        Level-by-level spherical harmonic spectra
        Contains same variables as spectrum but on pressure levels
    xspectrum : xarray.Dataset
        Vertically averaged zonal wavenumber spectra
        Contains same variables as spectrum but for zonal decomposition
    xspectrum_sigma : xarray.Dataset
        Level-by-level zonal wavenumber spectra
    cospectra : xarray.Dataset
        Cospectra of eddy fluxes
        Contains:
        - cospectra_thetav: Temperature flux cospectra
        - cospectra_uv: Momentum flux cospectra
        - cospectra_qv: PV flux cospectra
    """
    # Define some constants
    phirad = np.radians(ucomp.lat)
    f = 2*omega*np.sin(phirad)

    # Calculate eddy terms by removing time and zonal means
    if meanstats is None:
        psmean = ps.mean(['lon', 'time'])
        umean = (ucomp * ps).mean(['lon', 'time']) / psmean
        vmean = (vcomp * ps).mean(['lon', 'time']) / psmean
        thetamean = (theta * ps).mean(['lon', 'time']) / psmean
    else:
        umean = meanstats.u
        vmean = meanstats.v
        thetamean = meanstats.theta
        psmean = meanstats.ps
    
    up = ucomp - umean
    vp = vcomp - vmean
    thetap = theta - thetamean

    # Calculate QG PV components
    dthetamean_dp = get_pderiv(thetamean, psmean)
    dup_dphi = get_phideriv(up, r)
    qp_stretch = f*get_pderiv(thetap/dthetamean_dp, psmean)
    qp = -dup_dphi + qp_stretch

    # Calculate vertical averages
    up_zavg = get_zavg(up, phalf)
    vp_zavg = get_zavg(vp, phalf)
    thetap_zavg = get_zavg(thetap, phalf)

    # Calculate spherical harmonic spectra
    # Full spectra on pressure levels
    eke_spectrum_full = spherical_spectrum_full(up, vp).rename("eke_spectrum_sigma")
    eke_barot_spectrum = spherical_energy_spectrum(up_zavg, vp_zavg).rename("eke_bt_spectrum")
    eke_baroc_spectrum_full = spherical_spectrum_full(up-up_zavg, vp-vp_zavg).rename("eke_bc_spectrum_sigma")

    u_spectrum_full = spherical_spectrum_full(up).rename("u_spectrum_sigma")
    v_spectrum_full = spherical_spectrum_full(vp).rename("v_spectrum_sigma")
    theta_spectrum_full = spherical_spectrum_full(thetap).rename("theta_spectrum_sigma")

    u_barot_spectrum = spherical_power_spectrum(up_zavg).rename("u_bt_spectrum")
    v_barot_spectrum = spherical_power_spectrum(vp_zavg).rename("v_bt_spectrum")
    theta_barot_spectrum = spherical_power_spectrum(thetap_zavg).rename("theta_bt_spectrum")

    u_baroc_spectrum_full = spherical_spectrum_full(up-up_zavg).rename("u_bc_spectrum_sigma")
    v_baroc_spectrum_full = spherical_spectrum_full(vp-vp_zavg).rename("v_bc_spectrum_sigma")
    theta_baroc_spectrum_full = spherical_spectrum_full(thetap-thetap_zavg).rename("theta_bc_spectrum_sigma")

    eke_spectrum = get_zavg(eke_spectrum_full, phalf).rename("eke_spectrum")
    eke_baroc_spectrum = get_zavg(eke_baroc_spectrum_full, phalf).rename("eke_bc_spectrum")
    u_spectrum = get_zavg(u_spectrum_full, phalf).rename("u_spectrum")
    u_baroc_spectrum = get_zavg(u_baroc_spectrum_full, phalf).rename("u_bc_spectrum")
    v_spectrum = get_zavg(v_spectrum_full, phalf).rename("v_spectrum")
    v_baroc_spectrum = get_zavg(v_baroc_spectrum_full, phalf).rename("v_bc_spectrum")
    theta_spectrum = get_zavg(theta_spectrum_full, phalf).rename("theta_spectrum")
    theta_baroc_spectrum = get_zavg(theta_baroc_spectrum_full, phalf).rename("theta_bc_spectrum")

    # Now do the same, but for zonal spectra at each latitude band
    eke_xspectrum_full = zonal_energy_spectrum(up, vp).rename("eke_xspectrum_sigma")
    eke_barot_xspectrum = zonal_energy_spectrum(up_zavg, vp_zavg).rename("eke_bt_xspectrum")
    eke_baroc_xspectrum_full = zonal_energy_spectrum(up - up_zavg, vp-vp_zavg).rename("eke_bc_xspectrum_sigma")
    eke_xspectrum = get_zavg(eke_xspectrum_full, phalf).rename("eke_xspectrum")
    eke_baroc_xspectrum = get_zavg(eke_baroc_xspectrum_full, phalf).rename("eke_bc_xspectrum")

    u_xspectrum_full = zonal_power_spectrum(up).rename("u_xspectrum_sigma")
    u_barot_xspectrum = zonal_power_spectrum(up_zavg).rename("u_bt_xspectrum")
    u_baroc_xspectrum_full = zonal_power_spectrum(up - up_zavg).rename("u_bc_xspectrum_sigma")
    u_xspectrum = get_zavg(u_xspectrum_full, phalf).rename("u_xspectrum")
    u_baroc_xspectrum = get_zavg(u_baroc_xspectrum_full, phalf).rename("u_bc_xspectrum")

    v_xspectrum_full = zonal_power_spectrum(vp).rename("v_xspectrum_sigma")
    v_barot_xspectrum = zonal_power_spectrum(vp_zavg).rename("v_bt_xspectrum")
    v_baroc_xspectrum_full = zonal_power_spectrum(vp - up_zavg).rename("v_bc_xspectrum_sigma")
    v_xspectrum = get_zavg(v_xspectrum_full, phalf).rename("v_xspectrum")
    v_baroc_xspectrum = get_zavg(v_baroc_xspectrum_full, phalf).rename("v_bc_xspectrum")

    theta_xspectrum_full = zonal_power_spectrum(vp).rename("theta_xspectrum_sigma")
    theta_barot_xspectrum = zonal_power_spectrum(vp_zavg).rename("theta_bt_xspectrum")
    theta_baroc_xspectrum_full = zonal_power_spectrum(vp - up_zavg).rename("theta_bc_xspectrum_sigma")
    theta_xspectrum = get_zavg(theta_xspectrum_full, phalf).rename("theta_xspectrum")
    theta_baroc_xspectrum = get_zavg(theta_baroc_xspectrum_full, phalf).rename("theta_bc_xspectrum")

    # Put dataarrays together into one dataset

    spectrum = xr.merge([eke_spectrum, eke_barot_spectrum, eke_baroc_spectrum,
                            u_spectrum, u_barot_spectrum, u_baroc_spectrum,
                            v_spectrum, v_barot_spectrum, v_baroc_spectrum,
                            theta_spectrum, theta_barot_spectrum, theta_baroc_spectrum,])
    spectrum_sigma = xr.merge([eke_spectrum_full, eke_baroc_spectrum_full,
                                  u_spectrum_full, u_baroc_spectrum_full,
                                  v_spectrum_full, v_baroc_spectrum_full,
                                  theta_spectrum_full, theta_baroc_spectrum_full])

    xspectrum = xr.merge([eke_xspectrum, eke_barot_xspectrum, eke_baroc_xspectrum,
                         u_xspectrum, u_barot_xspectrum, u_baroc_xspectrum,
                         v_xspectrum, v_barot_xspectrum, v_baroc_xspectrum,
                         theta_xspectrum, theta_barot_xspectrum, theta_baroc_xspectrum])

    xspectrum_sigma = xr.merge([eke_xspectrum_full, eke_baroc_xspectrum_full,
                             u_xspectrum_full, u_baroc_xspectrum_full,
                             v_xspectrum_full, v_baroc_xspectrum_full,
                             theta_xspectrum_full, theta_baroc_xspectrum_full])

    cospectra_thetav = cospectra_eddyflux(thetap, vp, r, apply_smoothing=apply_smoothing, smooth_width=smooth_width).rename("cospectra_thetav")
    del(thetap)
    cospectra_uv = cospectra_eddyflux(up, vp, r, apply_smoothing=apply_smoothing, smooth_width=smooth_width).rename("cospectra_uv")
    del(up)
    cospectra_qv = cospectra_eddyflux(qp, vp, r, apply_smoothing=apply_smoothing, smooth_width=smooth_width).rename("cospectra_qv")
    del(qp)
    cospectra = xr.merge([cospectra_thetav, cospectra_uv, cospectra_qv])

    # Add metadata to final datasets
    spectrum = xr.merge([eke_spectrum, eke_barot_spectrum, eke_baroc_spectrum,
                        u_spectrum, u_barot_spectrum, u_baroc_spectrum,
                        v_spectrum, v_barot_spectrum, v_baroc_spectrum,
                        theta_spectrum, theta_barot_spectrum, theta_baroc_spectrum])
    
    spectrum.attrs['description'] = 'Vertically averaged spherical harmonic spectra'
    spectrum.attrs['long_name'] = 'Spherical harmonic power spectra'
    
    spectrum_sigma = xr.merge([eke_spectrum_full, eke_baroc_spectrum_full,
                             u_spectrum_full, u_baroc_spectrum_full,
                             v_spectrum_full, v_baroc_spectrum_full,
                             theta_spectrum_full, theta_baroc_spectrum_full])
    
    spectrum_sigma.attrs['description'] = 'Level-by-level spherical harmonic spectra'
    spectrum_sigma.attrs['long_name'] = 'Pressure level spherical harmonic spectra'

    xspectrum = xr.merge([eke_xspectrum, eke_barot_xspectrum, eke_baroc_xspectrum,
                         u_xspectrum, u_barot_xspectrum, u_baroc_xspectrum,
                         v_xspectrum, v_barot_xspectrum, v_baroc_xspectrum,
                         theta_xspectrum, theta_barot_xspectrum, theta_baroc_xspectrum])
    
    xspectrum.attrs['description'] = 'Vertically averaged zonal wavenumber spectra'
    xspectrum.attrs['long_name'] = 'Zonal wavenumber power spectra'

    xspectrum_sigma = xr.merge([eke_xspectrum_full, eke_baroc_xspectrum_full,
                               u_xspectrum_full, u_baroc_xspectrum_full,
                               v_xspectrum_full, v_baroc_xspectrum_full,
                               theta_xspectrum_full, theta_baroc_xspectrum_full])
    
    xspectrum_sigma.attrs['description'] = 'Level-by-level zonal wavenumber spectra'
    xspectrum_sigma.attrs['long_name'] = 'Pressure level zonal wavenumber spectra'

    cospectra = xr.merge([cospectra_thetav, cospectra_uv, cospectra_qv])
    cospectra.attrs['description'] = 'Cospectra of meridional eddy fluxes'
    cospectra.attrs['long_name'] = 'Eddy flux cospectra'

    return spectrum, spectrum_sigma, xspectrum, xspectrum_sigma, cospectra   
   


# inputs are zona land meridional velocity anomalies from the time/zonal
# average mean
def spherical_spectrum_full(u, v=None):
    """
    Calculate spherical harmonic spectrum at each pressure level.
    
    Parameters
    ----------
    u : xarray.DataArray
        Primary field for spectrum calculation
    v : xarray.DataArray, optional
        Secondary field for energy spectrum calculation
        
    Returns
    -------
    xarray.DataArray
        Spectrum at each pressure level
    """
    spectra = []
    for p in u.sigma:
        if v is None:
            spectrum = spherical_power_spectrum(u.sel(sigma=p))
        else:
            spectrum = spherical_energy_spectrum(u.sel(sigma=p), v.sel(sigma=p))
        spectra.append(spectrum)
    spectrum = xr.concat(spectra, dim=u.sigma)
    return spectrum

def spherical_spectrum(x):
    """
    Calculate raw spherical harmonic spectrum.
    
    Parameters
    ----------
    x : xarray.DataArray
        Field to decompose
        
    Returns
    -------
    numpy.ndarray
        Power spectrum in terms of total wavenumber l
    """
    grid_x = pysh.SHGrid.from_xarray(x)
    clm_x = grid_x.expand(normalization='4pi')
    spectrum = clm_x.spectrum(unit='per_l')
    return spectrum



def spherical_energy_spectrum(u, v):
    """
    Calculate kinetic energy spectrum from velocity components.
    
    Parameters
    ----------
    u : xarray.DataArray
        Zonal velocity component
    v : xarray.DataArray
        Meridional velocity component
        
    Returns
    -------
    xarray.DataArray
        Kinetic energy spectrum
    """
    u = u.interp(lat=np.linspace(u.lat.min().item(), u.lat.max().item(), len(u.lon.values)))
    v = v.interp(lat=np.linspace(v.lat.min().item(), v.lat.max().item(), len(v.lon.values)))

    ke_spectra = []
    for t in u.time: 
        u_spectrum = spherical_spectrum(u.sel(time=t))
        v_spectrum = spherical_spectrum(v.sel(time=t))

        # Compute kinetic energy spectrum
        ke_spectrum = 0.5 * (u_spectrum + v_spectrum)
        ke_spectra.append(ke_spectrum)

    spectrum = np.mean(np.stack(ke_spectra, axis=0), axis=0)
    spectrum = xr.DataArray(spectrum, coords={"l":("l",range(len(spectrum)))})
    return spectrum

# spectrum for kinetic energy variables (e.g. potential temp., velocity variance, etc)
def spherical_power_spectrum(x):
    """
    Calculate power spectrum of a scalar field.
    
    Parameters
    ----------
    x : xarray.DataArray
        Field to analyze
        
    Returns
    -------
    xarray.DataArray
        Power spectrum
    """
    x = x.interp(lat=np.linspace(x.lat.min().item(), x.lat.max().item(), len(x.lon.values)))

    spectra = []
    for t in x.time: 
        spectrum = spherical_spectrum(x.sel(time=t))
        spectra.append(spectrum)

    spectrum = np.mean(np.stack(spectra, axis=0), axis=0)
    spectrum = xr.DataArray(spectrum, coords={"l":("l",range(len(spectrum)))})
    return spectrum

# Meridional eddy flux of either temperature or zonal momentum 
def cospectra_eddyflux(varp, vp, r, apply_smoothing=False, smooth_width=3):
    """
    Calculate cospectra of meridional eddy fluxes.
    
    Parameters
    ----------
    varp : xarray.DataArray
        Variable being fluxed (temperature, momentum, etc)
    vp : xarray.DataArray
        Meridional velocity
    r : float
        Planet radius
    apply_smoothing : bool, optional
        Whether to apply smoothing, by default False
    smooth_width : int, optional
        Smoothing width, by default 3
        
    Returns
    -------
    xarray.DataArray
        Cospectra of eddy flux
    """
    cospectras = []
    sigmas = varp.sigma
    for sigma in sigmas:
        varp0 = varp.sel(sigma=sigma, method='nearest')
        vp0 = vp.sel(sigma=sigma, method='nearest')
        varp0 = varp0.transpose("time", "lat", "lon")
        vp0 = vp0.transpose("time", "lat", "lon")
        dt = 6*60*60

        p_spec, ncps = compute_eddy_cospectra(varp0.values, vp0.values, r, vp0.lat.values, dt, apply_smoothing=apply_smoothing, smooth_width=smooth_width)
        k = range(p_spec.shape[0])
        p_spec = np.expand_dims(p_spec, axis=2)
        cospectra = xr.DataArray(p_spec, coords={'lat':('lat',vp.lat.values), 'c':('c', ncps), 'sigma':('sigma', [sigma.item()])})
        cospectras.append(cospectra)
    cospectras = xr.concat(cospectras, dim='sigma')

    return cospectras

def zonal_power_spectrum(x):
    """
    Calculate zonal wavenumber power spectrum.
    
    Parameters
    ----------
    x : xarray.DataArray
        Field to analyze
        
    Returns
    -------
    xarray.DataArray
        Zonal power spectrum
    """
    if 'sigma' not in x.dims:
        x = x.transpose("lat", "lon", "time")
    else:
        x = x.transpose("lat", "lon", "sigma", "time")

    nlon = len(x.lon.values)
    xf = np.fft.fft(x.values, axis=1) / nlon

    power = np.abs(xf)**2
    power_mean = np.mean(power, axis=-1)

    k = np.fft.fftfreq(nlon)  # degrees spacing -> wavenumber (cycles per 360 degrees)

    # Sort positive wavenumbers
    power = power_mean[:,:nlon//2]
    if 'sigma' in x.dims:
        power = xr.DataArray(power, coords={'lat':x.lat, 'k':('k', np.arange(len(power[0]))), 'sigma':x.sigma})
    else:
        power = xr.DataArray(power, coords={'lat':x.lat, 'k':('k', np.arange(len(power[0])))})
   
    return power

# Calculate zonal power spectra
# inputs are eddy velocities
def zonal_energy_spectrum(u,v):
    """
    Calculate zonal kinetic energy spectrum.
    
    Parameters
    ----------
    u : xarray.DataArray
        Zonal velocity component
    v : xarray.DataArray
        Meridional velocity component
        
    Returns
    -------
    xarray.DataArray
        Zonal kinetic energy spectrum
    """
    power_u = zonal_power_spectrum(u)
    power_v = zonal_power_spectrum(v)
    energy_spectrum = 0.5*(power_u + power_v)
    return energy_spectrum

