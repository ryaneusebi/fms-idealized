import numpy as np
import metpy.calc as mpcalc
import matplotlib.pyplot as plt
from metpy.units import units

def select_best_contour(contour_lines, desired_min_lat=-80, desired_max_lat=80):
    """
    Select the contour line that best covers the desired latitude range.
    
    Parameters
    ----------
    contour_lines : list
        List of contour lines from matplotlib.contour
    desired_min_lat : float, optional
        Minimum desired latitude coverage, by default -80
    desired_max_lat : float, optional
        Maximum desired latitude coverage, by default 80
        
    Returns
    -------
    numpy.ndarray
        The selected contour line coordinates
        
    Raises
    ------
    ValueError
        If no contour line with sufficient coverage is found
    """
    best_contour = None
    max_coverage = 0

    for path in contour_lines:
        latitudes_contour = path[:, 0]
        min_lat = np.min(latitudes_contour)
        max_lat = np.max(latitudes_contour)
        
        # Check if this contour line covers more of our desired range
        if min_lat <= desired_min_lat and max_lat >= desired_max_lat:
            # If we find a perfect match, use it immediately
            best_contour = path
            break
        else:
            # Calculate how much of the desired range this contour covers
            coverage = (min(max_lat, desired_max_lat) - max(min_lat, desired_min_lat))
            if coverage > max_coverage:
                max_coverage = coverage
                best_contour = path

    if best_contour is None:
        raise ValueError("No contour line found with sufficient latitude coverage")
        
    return best_contour

def trop_height(tempmean, psmean):
    """
    Calculate the tropopause height using the WMO lapse rate criterion.
    
    This function calculates the tropopause height by finding where the temperature
    lapse rate first decreases to 2 K/km or less. The calculation is performed on
    zonally and temporally averaged temperature data.
    
    Parameters
    ----------
    tempmean : xarray.DataArray
        Zonally and temporally averaged temperature field
    psmean : xarray.DataArray
        Zonally and temporally averaged surface pressure
        
    Returns
    -------
    xarray.DataArray
        Tropopause height in sigma coordinates
        
    Notes
    -----
    The function interpolates the temperature field to a higher resolution before
    calculating the lapse rate. It uses the WMO criterion of finding where the
    lapse rate decreases to 2 K/km.
    """
    # Interpolate temperature to higher resolution
    temp = tempmean.interp(sigma=np.linspace(tempmean.sigma.min().item(), 
                                            tempmean.sigma.max().item(), 1000), 
                          method='cubic')
    g = 9.81

    # Calculate pressure and density fields
    pfull = (temp*0 + psmean)*temp.sigma
    density = mpcalc.density(pfull*units.Pa, temp*units.kelvin, 0).metpy.dequantify()
    
    # Calculate temperature gradient with respect to pressure and height
    dTdp = temp.differentiate("sigma")/psmean
    dTdz = dTdp*(density*g)*1000  # Convert to units per km

    latitude = dTdz['lat']
    sigma = dTdz['sigma']
    data = dTdz.values

    # Create meshgrid and generate contour at 2 K/km
    LAT, SIGMA = np.meshgrid(latitude, sigma)
    fig, ax = plt.subplots()
    CS = ax.contour(LAT, SIGMA, data, levels=[2])

    # Select the best contour line for tropopause calculation
    best_contour = select_best_contour(CS.allsegs[0])
    
    tropopause_heights = []
    lats_ds = dTdz.lat.values

    # Extract coordinates from selected contour
    latitudes_contour = best_contour[:, 0]
    sigma_contour = best_contour[:, 1]

    # Default highest point - initialize to np.nan for start
    highest_point = np.nan
    # Calculate tropopause height for each latitude
    for k, latk in enumerate(lats_ds):
        if k == 0:
            mask = (latitudes_contour < dTdz.lat.values[k+1])
        elif k < len(dTdz.lat.values)-1:
            mask = (latitudes_contour >= latk) & \
                  (latitudes_contour < dTdz.lat.values[k+1])
        else:
            mask = (latitudes_contour >= latk)
            
        if np.any(mask):
            # Since sigma axis is inverted, smallest sigma is highest altitude
            highest_point = sigma_contour[mask].min()
            tropopause_heights.append((latk, highest_point))
        elif k > 0:
            # Use previous point if contour doesn't have point in this range
            tropopause_heights.append((latk, highest_point))
        else:
            # add NaN if no tropopause height found at latitude k
            tropopause_heights.append((latk, np.nan))

    # Convert to numpy array and sort by latitude
    tropopause_heights = np.array(tropopause_heights)
    tropopause_heights = tropopause_heights[tropopause_heights[:, 0].argsort()][:,1]

    # Create output DataArray with metadata
    tropo_p = tempmean.isel(sigma=0).drop('sigma')*0 + tropopause_heights
    tropo_p = tropo_p.rename("tropo_p")
    tropo_p.attrs['long_name'] = 'tropopause height'
    tropo_p.attrs['unit'] = 'sigma'

    density = density.rename("density").interp(sigma=tempmean.sigma)
    density.attrs['long_name'] = 'density'
    density.attrs['unit'] = 'kg/m^3'

    dTdz = dTdz.rename("dTdz").interp(sigma=tempmean.sigma)
    dTdz.attrs['long_name'] = 'temperature gradient'
    dTdz.attrs['unit'] = 'K/km'

    return tropo_p, density, dTdz

