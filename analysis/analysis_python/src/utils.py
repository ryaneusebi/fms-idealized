import xarray as xr
import numpy as np
from typing import Union

def get_zavg(data: xr.DataArray, phalf: xr.DataArray, pmin=0, pmax=1e3) -> xr.DataArray:
    """Calculate the vertical average of a field weighted by pressure thickness.
    
    Args:
        data: DataArray to be vertically averaged. Must have 'sigma' dimension.
        phalf: DataArray of pressure coordinates at vertical grid edges.
              Must have 'phalf' dimension.
        pmin: Minimum pressure to include in the average. (in hPa)
        pmax: Maximum pressure to include in the average. (in hPa)
    Returns:
        xr.DataArray: Vertically averaged data with 'sigma' dimension reduced.
    """

    # Convert pmin and pmax to sigma units
    pmin = pmin/1e3
    pmax = pmax/1e3

    # Calculate the weight for the vertical average
    sigma_weight = data.sigma * 0 + phalf.values[1:] - phalf.values[:-1]
    if hasattr(pmin, '__len__') and not isinstance(pmin, str):
        if len(pmin) > 1:
            sigma_weight = data*0 + sigma_weight
            data_sigma = data*0 + data.sigma 
    else:
        data_sigma = data.sigma
    sigma_weight = sigma_weight.where((data_sigma >= pmin) & (data_sigma <= pmax), 0)

    # Calculate the vertical average
    data_zavg = (data * sigma_weight).sum('sigma') / sigma_weight.sum('sigma')
    return data_zavg

def get_zint(data: xr.DataArray, phalf: xr.DataArray, ps: xr.DataArray, pmin=0, pmax=1e3) -> xr.DataArray:
    """Calculate the vertical integral of a field weighted by pressure thickness.
    
    Args:
        data: DataArray to be vertically integrated. Must have 'sigma' dimension.
        phalf: DataArray of pressure coordinates at vertical grid edges.
              Must have 'phalf' dimension.
        ps: Surface pressure DataArray.
        pmin: Minimum pressure to include in the integral. (in hPa)
        pmax: Maximum pressure to include in the integral. (in hPa)
    Returns:
        xr.DataArray: Vertically integrated data with 'sigma' dimension reduced.
    """

    # Convert pmin and pmax to sigma units
    pmin = pmin/1e3
    pmax = pmax/1e3

    # Calculate the weight for the vertical integral
    sigma_weight = data.sigma * 0 + (phalf.values[1:] - phalf.values[:-1]) / 1e3
    if hasattr(pmin, '__len__') and not isinstance(pmin, str):
        if len(pmin) > 1:
            sigma_weight = data*0 + sigma_weight
            data_sigma = data*0 + data.sigma 
    else:
        data_sigma = data.sigma
    sigma_weight = sigma_weight.where((data.sigma >= pmin) & (data.sigma <= pmax), 0)

    # Calculate the vertical integral
    data_zint = (data * sigma_weight).sum('sigma')*ps / 9.81
    return data_zint


def get_theta(temp, ps):
   p0 = 1e5
   kappa = 2./7.
   theta = temp*(p0/ps)**kappa
   return theta

def get_zcumint(data: xr.DataArray, ps: xr.DataArray, phalf: xr.DataArray) -> xr.DataArray:
    """Calculate the cumulative integral of a field in the vertical direction.
    
    Args:
        data: DataArray to integrate vertically. Must have 'sigma' dimension.
        ps: Surface pressure DataArray.
        phalf: DataArray of pressure coordinates at vertical grid edges.
              Must have 'phalf' dimension.
    
    Returns:
        xr.DataArray: Cumulative integral from top to bottom.
    """
    # Convert phalf to sigma by dividing by 1e3
    dp = data.sigma * 0 + (phalf.values[1:] - phalf.values[:-1]) / 1e3
    
    # Calculate streamfunction from bottom up and then flip back
    data_z_cumint = (data * dp * ps).isel({'sigma': slice(None, None, -1)}).cumsum('sigma') / 9.81
    return data_z_cumint.isel({'sigma': slice(None, None, -1)})

def get_pcumsum(data: xr.DataArray) -> xr.DataArray:
    """Calculate the cumulative sum of a field in the vertical direction.
    
    Args:
        data: DataArray to integrate vertically. Must have 'sigma' dimension.
    
    Returns:    
        xr.DataArray: Cumulative sum from bottom to top.
    """

    # Calculate streamfunction from bottom up and then flip back
    data_z_cumsum = data.isel({'sigma': slice(None, None, -1)}).cumsum('sigma')
    return data_z_cumsum.isel({'sigma': slice(None, None, -1)})

def get_pderiv(data: xr.DataArray, ps: xr.DataArray) -> xr.DataArray:
    """Calculate the derivative with respect to pressure (sigma coordinates).
    
    Args:
        data: DataArray to differentiate. Must have 'sigma' dimension.
        ps: Surface pressure DataArray.
    
    Returns:
        xr.DataArray: Pressure derivative of the input field.
    """
    
    return data.differentiate("sigma") / ps

def get_phiflux_nomass(data: xr.DataArray, radius: Union[float, int]) -> xr.DataArray:
    """Calculate the meridional flux term (includes cos(phi) factor).
    
    Args:
        data: DataArray to compute flux for. Must have 'lat' dimension.
        radius: Planet radius in meters.
    
    Returns:
        xr.DataArray: Meridional flux term.
    """
    cosphi = np.cos(np.radians(data.lat))
    lat_d = np.pi * radius / 180
    return (data * cosphi).differentiate("lat") / lat_d / cosphi

def get_phiflux(data: xr.DataArray, radius: Union[float, int], ps) -> xr.DataArray:
    """Calculate the meridional flux term (includes cos(phi) factor
      pressure weighting).
    
    Args:
        data: DataArray to compute flux for. Must have 'lat' dimension.
        radius: Planet radius in meters.
        ps: DataArray of surface pressure.
    
    Returns:
        xr.DataArray: Meridional flux term.
    """
    cosphi = np.cos(np.radians(data.lat))
    lat_d = np.pi * radius / 180
    return (data * cosphi * ps).differentiate("lat") / lat_d / cosphi / ps

def get_phideriv(data: xr.DataArray, radius: Union[float, int]) -> xr.DataArray:
    """Calculate the derivative with respect to latitude.
    
    Args:
        data: DataArray to differentiate. Must have 'lat' dimension.
        radius: Planet radius in meters.
    
    Returns:
        xr.DataArray: Latitudinal derivative.
    """
    lat_d = np.pi * radius / 180
    return data.differentiate("lat") / lat_d

def get_londeriv(data: xr.DataArray, radius: Union[float, int]) -> xr.DataArray:
    """Calculate the derivative with respect to longitude.
    
    Args:
        data: DataArray to differentiate. Must have 'lon' and 'lat' dimensions.
        radius: Planet radius in meters.
    
    Returns:
        xr.DataArray: Longitudinal derivative.
    """
    cosphi = np.cos(np.radians(data.lat))
    lon_d = np.pi * radius * cosphi / 180
    return data.differentiate("lon") / lon_d

