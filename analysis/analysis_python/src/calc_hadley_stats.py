import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import sys
from utils import get_zint
import glob
import os
def get_max(x, lat):
   return (x**10 * lat).sum()/(x**10).sum()

def get_zero_crossing(x, l_idx):
   r_idx = l_idx + 1
   phi_l = x.lat.isel(lat=l_idx)
   phi_r = x.lat.isel(lat=r_idx)

   x_l = x.sel(lat=phi_l)
   x_r = x.sel(lat=phi_r)

   # Linear interpolation: lat_zero = lat1 + (lat2-lat1) * (0-psi1)/(psi2-psi1)
   phi_e = phi_l - (phi_r - phi_l) * x_l / (x_r - x_l)

   return phi_e.item()

# inputs: 500 hPa psi for a single hemisphere 
# and the sign of the streamfunction in the hemisphere
def get_hc_bound(psi, psi_sign):

   # Get first boundary coordinate
   if psi.lat.isel(lat=2) < 0:
      phi_idx_l = psi.lat.where(psi*psi_sign <=0).argmax()
   else:
      phi_idx_l = psi.lat.where(psi*psi_sign <=0).argmin()

   phi_e = get_zero_crossing(psi, phi_idx_l)
   return phi_e


# Calculate hadley cell bounds as 500 hPa streamfunction changing sign
def get_hc_bounds(ds):

   phi0 = ds.attrs['sim_phi0']
   psi500 = ds.psi.sel(sigma=0.5, method='nearest')
   lat = ds.lat

   # find out what sign the southern hemisphere psi has
   # since phi0 will be >=0, we know 5S will be southern HC
   psi_southern = np.sign(psi500.sel(lat=slice(-7,-3)).mean())

   # Get southern boundary
   cond = (psi500*psi_southern<=0)&(lat<0)
   if np.any(cond):
      phi_idx_l = psi500.lat.where(cond).argmax()
      phi_se = get_zero_crossing(psi500, phi_idx_l)
   else:
      phi_se = -90

   # Get southern HCnorthern boundary (phi_a)
   if phi0 > 0:
      # try zero-crossing method first. Might not work if no northern hadley cell
      cond = (psi500*psi_southern<=0)&(lat>0)
      if np.any(cond):
         phi_idx_l = psi500.lat.where(cond).argmin() - 1
         phi_a = get_zero_crossing(psi500, phi_idx_l)
      else:
         phi_a = 90
   else:
      phi_a = 0

   # Get northern boundary
   cond = (np.sign(psi500)==psi_southern)&(lat>phi_a)
   if np.any(cond):
      phi_idx_l = psi500.lat.where(cond).argmin()-1
      phi_ne = get_zero_crossing(psi500, phi_idx_l)
   else:
      phi_ne = 90

   return phi_se, phi_ne, phi_a
   

   
# Get 10 percent boundary magnitudes for southern hadley cell boundaries
def get_hc_bounds_10p(v, phi_se, phi_a, phi_psimax, thresh=0.1):
   v = v.interp(lat=np.linspace(phi_se, phi_a, 100))
   v = np.abs(v)
   lat = v.lat
   
   v -= v.max()*thresh

   cond = (v<=0)&(lat<phi_psimax)
   if np.any(cond):
      phi_idx_l = v.lat.where(cond).argmax()
      phi_se_10p = get_zero_crossing(v, phi_idx_l)
   else:
      phi_se_10p = phi_se


   phi_idx_l = v.lat.where((v>=0)).argmax()
   if phi_idx_l == v.lat.size-1:
      phi_a_10p = phi_a
   else:
      phi_a_10p = get_zero_crossing(v, phi_idx_l)

   return phi_se_10p, phi_a_10p





def calc_hadley_stats(fname):
   outpath = fname.replace('fms_analysis.nc', 'hadley_stats.nc')

   ds = xr.open_dataset(fname, decode_times=False)
   cosphi = np.cos(np.radians(ds.lat))
   radius = 6371e3*ds.attrs['sim_radius']
   omega = 7.292e-5*ds.attrs['sim_omega']
   cp = 1004

   # Get Hadley cell bounds
   phi_se, phi_ne, phi_a = get_hc_bounds(ds)
 
   # Find the location of maximum in psi
   sh_slice = slice(phi_se,phi_a)
   psi_max_idx = ds.psi.sel(lat=sh_slice).argmax(dim=['lat', 'sigma'])
   phi_psimax_sh = ds.sel(lat=sh_slice).lat[psi_max_idx['lat']].values
   sigma_psimax_sh = ds.sel(lat=sh_slice).sigma[psi_max_idx['sigma']].values
   psimax_sh = np.abs(ds.psi).sel(lat=sh_slice).max().item()

   heatflux = get_zint((ds.theta*ds.v), ds.phalf, ds.ps)*cosphi*2*np.pi*radius*cp
   sigma_psimax = np.abs(ds.psi).idxmax(dim='sigma')
   psimax = ds.psi.sel(sigma=sigma_psimax)

   gs = -heatflux/psimax/cp

   nh_slice = slice(phi_a, phi_ne)
   nh_data = ds.psi.sel(lat=nh_slice)

   if nh_data.lat.size > 0:
      psi_max_idx = nh_data.argmax(dim=['lat', 'sigma'])
      phi_psimax_nh = ds.sel(lat=nh_slice).lat[psi_max_idx['lat']].values
      sigma_psimax_nh = ds.sel(lat=nh_slice).sigma[psi_max_idx['sigma']].values
      psimax_nh = np.abs(nh_data).max().item()
   else:
      # No NH Hadley cell - set to NaN
      phi_psimax_nh = np.nan
      sigma_psimax_nh = np.nan
      psimax_nh = np.nan

   llc = -1/(radius*cosphi)*psimax.differentiate('lat')*180/np.pi
   llc_max = llc.sel(lat=slice(phi_se, phi_a)).where(llc>0).max()
   phi_itcz = get_max(llc.sel(lat=slice(phi_se, phi_a)).where(llc>0), ds.lat.sel(lat=slice(phi_se, phi_a)).where(llc>0))


   phi_se_10p, phi_a_10p = get_hc_bounds_10p(-heatflux, phi_se, phi_a, phi_psimax_sh)
   phi_se_15p, phi_a_15p = get_hc_bounds_10p(-heatflux, phi_se, phi_a, phi_psimax_sh, thresh=0.15)
   phi_se_20p, phi_a_20p = get_hc_bounds_10p(-heatflux, phi_se, phi_a, phi_psimax_sh, thresh=0.2)

   phi_psi_se_10p, phi_psi_a_10p = get_hc_bounds_10p(ds.psi.sel(sigma=0.7, method='nearest'), phi_se, phi_a, phi_psimax_sh)
   phi_psi_se_20p, phi_psi_a_20p = get_hc_bounds_10p(ds.psi.sel(sigma=0.7, method='nearest'), phi_se, phi_a, phi_psimax_sh, thresh=0.2)

   psimax = xr.DataArray(psimax.values, coords=[ds.lat], dims=['lat']).rename('psimax')
   heatflux = xr.DataArray(heatflux.values, coords=[ds.lat], dims=['lat']).rename('heatflux')
   llc = xr.DataArray(llc.values, coords=[ds.lat], dims=['lat']).rename('llc')
   gs = xr.DataArray(gs.values, coords=[ds.lat], dims=['lat']).rename('gs')
   phi_se = xr.DataArray(phi_se, coords=[], dims=[]).rename('phi_se') 
   phi_a = xr.DataArray(phi_a, coords=[], dims=[]).rename('phi_a')
   phi_ne = xr.DataArray(phi_ne, coords=[], dims=[]).rename('phi_ne')
   phi_itcz = xr.DataArray(phi_itcz, coords=[], dims=[]).rename('phi_itcz')
   phi_se_10p = xr.DataArray(phi_se_10p, coords=[], dims=[]).rename('phi_se_10p')
   phi_a_10p = xr.DataArray(phi_a_10p, coords=[], dims=[]).rename('phi_a_10p')
   phi_se_15p = xr.DataArray(phi_se_15p, coords=[], dims=[]).rename('phi_se_15p')
   phi_a_15p = xr.DataArray(phi_a_15p, coords=[], dims=[]).rename('phi_a_15p')
   phi_se_20p = xr.DataArray(phi_se_20p, coords=[], dims=[]).rename('phi_se_20p')
   phi_a_20p = xr.DataArray(phi_a_20p, coords=[], dims=[]).rename('phi_a_20p')
   phi_psi_se_10p = xr.DataArray(phi_psi_se_10p, coords=[], dims=[]).rename('phi_psi_se_10p')
   phi_psi_a_10p = xr.DataArray(phi_psi_a_10p, coords=[], dims=[]).rename('phi_psi_a_10p')
   phi_psi_se_20p = xr.DataArray(phi_psi_se_20p, coords=[], dims=[]).rename('phi_psi_se_20p')
   phi_psi_a_20p = xr.DataArray(phi_psi_a_20p, coords=[], dims=[]).rename('phi_psi_a_20p')
   psimax_sh = xr.DataArray(psimax_sh, coords=[], dims=[]).rename('psimax_sh')
   psimax_nh = xr.DataArray(psimax_nh, coords=[], dims=[]).rename('psimax_nh')
   phi_psimax_sh = xr.DataArray(phi_psimax_sh, coords=[], dims=[]).rename('phi_psimax_sh')
   phi_psimax_nh = xr.DataArray(phi_psimax_nh, coords=[], dims=[]).rename('phi_psimax_nh')
   sigma_psimax_sh = xr.DataArray(sigma_psimax_sh, coords=[], dims=[]).rename('sigma_psimax_sh')
   sigma_psimax_nh = xr.DataArray(sigma_psimax_nh, coords=[], dims=[]).rename('sigma_psimax_nh')
   llc_max = xr.DataArray(llc_max, coords=[], dims=[]).rename('llc_max')


   ds_out = xr.merge([psimax, heatflux, llc, gs, phi_se, phi_a, phi_ne, phi_itcz, phi_se_10p, phi_a_10p, phi_se_15p, phi_a_15p, phi_se_20p, phi_a_20p, phi_psi_se_10p, phi_psi_a_10p, phi_psi_se_20p, phi_psi_a_20p, psimax_sh, psimax_nh, phi_psimax_sh, phi_psimax_nh, sigma_psimax_sh, sigma_psimax_nh, llc_max])
   ds_out.to_netcdf(outpath)