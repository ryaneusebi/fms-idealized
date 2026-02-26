import xarray as xr
import numpy as np
from typing import Any
from utils import get_zcumint

def calc_streamfunction(
    v: xr.DataArray,
    ps: xr.DataArray,
    phalf: xr.DataArray,
    radius: float,
) -> xr.DataArray:
    """Calculate the mean meridional mass streamfunction in sigma coordinates.
    
    The streamfunction (ψ) represents the mass transport in the meridional plane,
    calculated by vertically integrating the meridional velocity weighted by surface pressure
    and taking the zonal and time mean.
    
    Parameters
    ----------
    v : xr.DataArray
        Meridional velocity field with dimensions (lat, sigma).
        The order of dimensions is flexible as the output will be transposed.
    ps : xr.DataArray
        Surface pressure field
    phalf : xr.DataArray
        Half-level pressure field for vertical integration
    radius : float
        Planetary radius in meters
        
    Returns
    -------
    xr.DataArray
        Mean meridional mass streamfunction with dimensions (sigma, lat)
        Units: kg s^-1
    """
    g = 9.81  # gravitational acceleration [m/s^2]
    coslat = np.cos(np.radians(v.lat))
    
    # Vertically integrate meridional velocity
    vpint = get_zcumint(v, ps, phalf)
    
    # Calculate streamfunction
    psi = 2 * np.pi * radius * coslat * vpint
    
    # Take zonal and time mean, transpose to standard coordinates
    psi = psi.transpose("sigma", "lat")
    
    # Add metadata
    psi = psi.rename("psi")
    psi.attrs['long_name'] = 'Mean Meridional Mass Streamfunction'
    psi.attrs['units'] = 'kg s^-1'
    
    return psi