import xarray as xr
import numpy as np
from utils import get_phiflux, get_pderiv

def get_eddy_statistics(u, v, w, theta, ps, r, meanstats=None):
    """Calculate eddy statistics from atmospheric variables.
    
    Parameters
    ----------
    u : xarray.DataArray
        Zonal velocity
    v : xarray.DataArray
        Meridional velocity
    w : xarray.DataArray
        Vertical velocity
    theta : xarray.DataArray
        Potential temperature
    ps : xarray.DataArray
        Surface pressure
    r : float
        Planet radius (for calculating derivatives)
        
    Returns
    -------
    xarray.Dataset
        Dataset containing eddy statistics
    """
    # Calculate basic quantities
    coslat = np.cos(np.radians(u.lat))
    lat_d = r * np.pi / 180  # convert degrees latitude to meters

    # Calculate zonal means
    if meanstats is None:
        psmean = ps.mean(['time', 'lon'])
        umean = (u * ps).mean(['time', 'lon']) / psmean
        vmean = (v * ps).mean(['time', 'lon']) / psmean
        wmean = (w * ps).mean(['time', 'lon']) / psmean
        thetamean = (theta * ps).mean(['time', 'lon']) / psmean
    else:
        umean = meanstats.u
        vmean = meanstats.v
        wmean = meanstats.w
        thetamean = meanstats.theta
        psmean = meanstats.ps

    # Calculate perturbations
    up = u - umean
    vp = v - vmean
    wp = w - wmean 
    thetap = theta - thetamean 

    del(umean, vmean, thetamean, wmean)

    # Calculate variances
    u_var = ((up**2) * ps).mean(['time', 'lon']) / psmean
    v_var = ((vp**2) * ps).mean(['time', 'lon']) / psmean
    w_var = ((wp**2) * ps).mean(['time', 'lon']) / psmean
    theta_var = ((thetap**2) * ps).mean(['time', 'lon']) / psmean

    # Assign attributes
    u_var = u_var.assign_attrs({'long_name': 'Zonal velocity variance', 'units': 'm^2/s^2'})
    v_var = v_var.assign_attrs({'long_name': 'Meridional velocity variance', 'units': 'm^2/s^2'})
    w_var = w_var.assign_attrs({'long_name': 'Vertical velocity variance', 'units': 'm^2/s^2'})
    theta_var = theta_var.assign_attrs({'long_name': 'Potential temperature variance', 'units': 'K^2'})

    # Calculate eddy fluxes
    uvcos_eddy = ((up*vp*coslat) * ps).mean(['time', 'lon']) / psmean
    uw_eddy = ((up*wp) * ps).mean(['time', 'lon']) / psmean
    vthetacos_eddy = ((thetap*vp*coslat) * ps).mean(['time', 'lon']) / psmean
    wtheta_eddy = ((wp*thetap) * ps).mean(['time', 'lon']) / psmean

    # Assign attributes
    uvcos_eddy = uvcos_eddy.assign_attrs({'long_name': 'Meridional flux of zonal momentum', 'units': 'm^2/s^2'})
    uw_eddy = uw_eddy.assign_attrs({'long_name': 'Vertical flux of zonal momentum', 'units': 'm^2/s^2'})
    vthetacos_eddy = vthetacos_eddy.assign_attrs({'long_name': 'Meridional heat flux', 'units': 'K m/s'})
    wtheta_eddy = wtheta_eddy.assign_attrs({'long_name': 'Vertical heat flux', 'units': 'K m/s'})

    # Calculate eddy flux divergences
    emfd = get_phiflux(uvcos_eddy, r, psmean)
    emfd = emfd.assign_attrs({
        'long_name': 'Eddy momentum flux divergence',
        'units': 'm/s^2'
    })
    
    ehfd = get_phiflux(vthetacos_eddy, r, psmean)
    ehfd = ehfd.assign_attrs({
        'long_name': 'Eddy heat flux divergence',
        'units': 'K/s'
    })

    # Calculate vertical derivatives in pressure coordinates
    dz_uw_eddy = get_pderiv(uw_eddy, psmean)
    dz_uw_eddy = dz_uw_eddy.interp(sigma=u.sigma).assign_attrs({
        'long_name': 'Vertical derivative of zonal momentum flux',
        'units': 'm/s^2'
    })

    dz_wtheta_eddy = get_pderiv(wtheta_eddy, psmean)
    dz_wtheta_eddy = dz_wtheta_eddy.interp(sigma=u.sigma).assign_attrs({
        'long_name': 'Vertical derivative of heat flux',
        'units': 'K/s'
    })

    # Rename variables
    u_var = u_var.rename('u_var')
    v_var = v_var.rename('v_var')
    w_var = w_var.rename('w_var')
    theta_var = theta_var.rename('theta_var')
    uvcos_eddy = uvcos_eddy.rename('uvcos_eddy')
    uw_eddy = uw_eddy.rename('uw_eddy')
    vthetacos_eddy = vthetacos_eddy.rename('vthetacos_eddy')
    wtheta_eddy = wtheta_eddy.rename('wtheta_eddy')
    emfd = emfd.rename('emfd')
    ehfd = ehfd.rename('ehfd')
    dz_uw_eddy = dz_uw_eddy.rename('dz_uw_eddy')
    dz_wtheta_eddy = dz_wtheta_eddy.rename('dz_wtheta_eddy')

    # Merge all variables into a single dataset
    eddystats = xr.merge([
        u_var, v_var, w_var, theta_var, 
        uvcos_eddy, uw_eddy, vthetacos_eddy, wtheta_eddy,
        emfd, ehfd, dz_uw_eddy, dz_wtheta_eddy
    ])

    return eddystats







