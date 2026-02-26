"""
Functions for computing Quasi-Geostrophic Potential Vorticity (QGPV) diagnostics from FMS model output.

This module provides tools to analyze QGPV and related quantities from atmospheric model data,
including Eliassen-Palm flux calculations and eddy statistics.
"""

from utils import get_pderiv, get_phiflux, get_phideriv
import numpy as np
import xarray as xr

# Full 4D zonal velocity field, potential temperature field
# surface pressure field, and planet radius
def get_qgpv(ucomp, vcomp, thetacomp, ps, radius, omega, meanstats=None):
    """
    Calculate Quasi-Geostrophic Potential Vorticity (QGPV) diagnostics from model output.
    
    Parameters
    ----------
    ucomp : xarray.DataArray
        Zonal velocity field
    vcomp : xarray.DataArray
        Meridional velocity field
    thetacomp : xarray.DataArray
        Potential temperature field
    ps : xarray.DataArray
        Surface pressure field
    radius : float
        Planet radius in meters
    omega : float
        Planet rotation rate in rad/s
    meanstats : xarray.Dataset, optional
        Dataset containing mean statistics of the flow
    Returns
    -------
    xarray.Dataset
        Dataset containing QGPV diagnostics including:
        - dqmean_dphi: Meridional gradient of mean QGPV
        - dqmean_stretch: Stretching term in mean QGPV gradient
        - qvcos_eddy: Meridional eddy QGPV flux
        - qvcos_stretch_eddy: Meridional eddy QGPV flux from stretching term only
        - dqvcos_eddy_dphi: Meridional divergence of eddy QGPV flux
        - q_var: QGPV variance
        - EP_p: Vertical component of E-P flux
        - EP_phi: Meridional component of E-P flux
        - EP_dp: Vertical divergence of E-P flux
        - EP_dphi: Meridional divergence of E-P flux
        - EP_flux: Total E-P flux divergence
    """
    # Convert latitude to radians and compute basic quantities
    phirad = np.radians(ucomp.lat)
    cosphi = np.cos(phirad)
    f = 2*omega*np.sin(phirad)  # Coriolis parameter
    beta = 2*omega/radius*np.cos(phirad)  # Beta parameter
    lat_d = np.pi*radius/180  # Latitude spacing in radians

    # Calculate zonal means and eddy components
    if meanstats is None:  
        psmean = ps.mean(['lon', 'time'])
        umean = (ucomp * ps).mean(['lon', 'time']) / psmean
        thetamean = (thetacomp * ps).mean(['lon', 'time']) / psmean
        vmean = (vcomp * ps).mean(['lon', 'time']) / psmean
    else:
        umean = meanstats.u
        thetamean = meanstats.theta
        psmean = meanstats.ps
        vmean = meanstats.v
        
    up = ucomp - umean  # Zonal velocity perturbation
    thetap = thetacomp - thetamean  # Potential temperature perturbation
    vp = vcomp - vmean  # Meridional velocity perturbation

    # Calculate derivatives
    dthetamean_dp = get_pderiv(thetamean, psmean)  # Vertical gradient of mean theta
    dumean_dphi = get_phideriv(umean, radius)  # Meridional gradient of mean u
    dup_dphi = get_phideriv(up, radius)  # Meridional gradient of u perturbation
    d2umean_dphi2 = get_phideriv(dumean_dphi, radius)  # Second meridional derivative of mean u
    dthetamean_dphi = get_phideriv(thetamean, radius)  # Meridional gradient of mean theta

    # Compute QGPV components
    dqmean_stretch = f*get_pderiv(dthetamean_dphi/dthetamean_dp, psmean)
    dqmean_dphi = beta - d2umean_dphi2 + dqmean_stretch
    qp_stretch = f*get_pderiv(thetap/dthetamean_dp, psmean)
    qp = -dup_dphi + qp_stretch

    # Calculate eddy statistics
    qvcos_eddy = ((qp*vp*cosphi) * ps).mean(['lon', 'time']) / psmean
    qvcos_stretch_eddy = ((qp_stretch*vp*cosphi) * ps).mean(['lon', 'time']) / psmean
    dqvcos_eddy_dphi = get_phiflux(qvcos_eddy, radius, psmean)
    q_var = ((qp**2) * ps).mean(['lon', 'time']) / psmean

    # Compute Eliassen-Palm flux components
    ehf = ((vp*thetap*cosphi) * ps).mean(['lon', 'time']) / psmean  # Eddy heat flux
    EP_p = f*ehf/dthetamean_dp  # Vertical E-P flux
    EP_phi = -((up*vp*cosphi) * ps).mean(['lon', 'time']) / psmean  # Meridional E-P flux
    EP_dp = get_pderiv(EP_p, psmean)  # Vertical E-P flux divergence
    EP_dphi = get_phiflux(EP_phi, radius, psmean)  # Meridional E-P flux divergence
    EP_flux = EP_dphi + EP_dp  # Total E-P flux divergence

    # Rename xarray dataarrays and add metadata
    dqmean_dphi = dqmean_dphi.rename("dqmean_dphi").assign_attrs({
        'long_name': 'Meridional Gradient of Mean QGPV',
        'units': 'm^-1 s^-1'
    })

    dqmean_stretch = dqmean_stretch.rename("dqmean_stretch").assign_attrs({
        'long_name': 'Stretching Term in Mean QGPV Gradient',
        'units': 'm^-1 s^-1'
    })

    qvcos_eddy = qvcos_eddy.rename("qvcos_eddy").assign_attrs({
        'long_name': 'Meridional Eddy QGPV Flux',
        'units': 'm s^-2'
    })

    qvcos_stretch_eddy = qvcos_stretch_eddy.rename("qvcos_stretch_eddy").assign_attrs({
        'long_name': 'Meridional Eddy QGPV Flux from Stretching Term',
        'units': 'm s^-2'
    })

    dqvcos_eddy_dphi = dqvcos_eddy_dphi.rename("dqvcos_eddy_dphi").assign_attrs({
        'long_name': 'Meridional Divergence of Eddy QGPV Flux',
        'units': 'm s^-2'
    })

    q_var = q_var.rename("q_var").assign_attrs({
        'long_name': 'QGPV Variance',
        'units': 's^-2'
    })

    EP_p = EP_p.rename("EP_p").assign_attrs({
        'long_name': 'Vertical Component of E-P Flux',
        'units': 'm^2 s^-2'
    })

    EP_phi = EP_phi.rename("EP_phi").assign_attrs({
        'long_name': 'Meridional Component of E-P Flux',
        'units': 'm^2 s^-2'
    })

    EP_dp = EP_dp.rename("EP_dp").assign_attrs({
        'long_name': 'Vertical Divergence of E-P Flux',
        'units': 'm s^-2'
    })

    EP_dphi = EP_dphi.rename("EP_dphi").assign_attrs({
        'long_name': 'Meridional Divergence of E-P Flux',
        'units': 'm s^-2'
    })

    EP_flux = EP_flux.rename("EP_flux").assign_attrs({
        'long_name': 'Total E-P Flux Divergence',
        'units': 'm s^-2'
    })

    # Merge into one dataset
    qgpv_data = xr.merge([dqmean_dphi, dqmean_stretch, qvcos_eddy, qvcos_stretch_eddy, dqvcos_eddy_dphi, q_var, EP_p, EP_phi, \
                   EP_dp, EP_dphi, EP_flux])

    # Add dataset-level metadata
    qgpv_data = qgpv_data.assign_attrs({
        'description': 'Quasi-Geostrophic Potential Vorticity (QGPV) diagnostics',
        'references': 'Based on standard QGPV theory (e.g., Holton & Hakim)'
    })

    qgpv_data = qgpv_data.transpose("sigma", "lat")

    return qgpv_data



def get_qp(thetap, thetamean, up, psmean, r, omega, dthetamean_dp=None):
    """
    Calculate the perturbation quasi-geostrophic potential vorticity (QGPV).
    
    Parameters
    ----------
    thetap : xarray.DataArray
        Perturbation potential temperature field
    thetamean : xarray.DataArray
        Mean potential temperature field
    up : xarray.DataArray
        Perturbation zonal velocity field
    psmean : xarray.DataArray
        Mean surface pressure field
    r : float
        Planet radius in meters
    dthetamean_dp : xarray.DataArray, optional
        Vertical gradient of mean potential temperature field
        
    Returns
    -------
    xarray.DataArray
        Perturbation QGPV field
    """
    if dthetamean_dp is None:
        dthetamean_dp = get_pderiv(thetamean, psmean)  # Vertical gradient of mean theta
    dup_dphi = get_phideriv(up, r)  # Meridional gradient of perturbation u
    phirad = np.radians(thetap.lat)
    f = 2*omega*np.sin(phirad)
    qp_stretch = f*get_pderiv(thetap/dthetamean_dp, psmean)  # Stretching term
    qp = -dup_dphi + qp_stretch  # Total perturbation QGPV
    return qp
