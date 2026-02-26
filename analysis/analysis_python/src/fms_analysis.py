import numpy as np
import xarray as xr
from eddy_statistics import get_eddy_statistics
from spectral_analysis import spectral_analysis
from streamfunction import calc_streamfunction
from tropopause_height import trop_height
from qgpv import get_qgpv
from utils import get_theta
import matplotlib.pyplot as plt
import argparse
from vertical_animation import make_vertical_animation
from horizontal_animation import make_horizontal_animation
from calc_hadley_stats import calc_hadley_stats
import os
from time import time
from glob import glob
from plot_streamfunction import plot_streamfunction
from thvar_budget import get_thvar_budget
from eke_budget import get_eke_budget
from utils import get_pcumsum, get_zavg

def analyze(runname, raw_output_dir, start_analysis, runs_per_script, days):
   start_time = time()
   day_list = np.arange(start_analysis*days, (runs_per_script+1)*days, days)
   nt_file = days*4
   R = 287 # J/kg/K

   print(f'Analyzing {runname}')

   simulation = proc_runname(runname)
   user = os.environ['USER']
   savepath = f"/resnick/groups/esm/{user}/fms_analysis/{runname}"
   if not os.path.exists(savepath):
      os.makedirs(savepath)
   fnames = np.concatenate([glob(f"{raw_output_dir}/combine/day{day:04}h00/day{day:04}h00.4xday.nc*") for day in day_list])
   # Use open_mfdataset for efficient, lazy loading of multiple files
   if len(fnames) > len(day_list):
      drop_vars = ['latb']
   else:
      drop_vars = []
   if simulation['res'] == 'T42':
      chunks = {'time': 4, 'sigma': 5, 'lat': -1, 'lon': -1}
   elif simulation['res'] == 'T85':
      chunks = {'time': 1, 'sigma': 5, 'lat': -1, 'lon': -1}
   elif simulation['res'] == 'T127':
      chunks = {'time': 4, 'sigma': 5, 'lat': -1, 'lon': -1}
   ds = xr.open_mfdataset(fnames, combine='by_coords', decode_times=False, chunks=chunks, drop_variables=drop_vars)
   try:
      ds = ds[['ucomp', 'vcomp', 'omega', 'temp', 'ps', 'teq', 'phalf', 'pfull','dt_tg_diffusion','dt_tg_convection', 'dt_tg_radiation', 'dt_ug_diffusion', 'dt_vg_diffusion', 'diff_m', 'phalf']]
   except:
      ds = ds[['ucomp', 'vcomp', 'omega', 'temp', 'ps', 'teq', 'phalf', 'pfull','dt_tg_convection', 'dt_tg_radiation','phalf' ]]
      ds['dt_ug_diffusion'] = ds.ucomp*0
      ds['dt_vg_diffusion'] = ds.vcomp*0
      ds['diff_m'] = ds.ucomp*0
      ds['dt_tg_diffusion'] = ds.temp*0
   print(f'opened dataset in {time() - start_time} seconds')
   start_time = time()

   ds = ds.assign_coords({"pfull": ds['pfull']/1e3}).rename({"pfull":"sigma"})
   thetacomp = get_theta(ds.temp, ds.ps*ds.sigma)

   radius = 6371000*simulation['radius'] #m  
   omega = 7.292e-5*simulation['omega'] # radians per second

   # Get mean flow statistics on pressure coordinates
   meanstats = mean_flow_stats(ds).compute()

   # Get heating rate
   heating_rate = get_heating_rate(meanstats, simulation)

   print(f'got mean flow stats in {time() - start_time} seconds')
   start_time = time()

   # Get tropopause height pressure according to WMO definition (2 K/km)
   tropo_p, density, dTdz = trop_height(meanstats.temp, meanstats.ps)
   tropo_p = tropo_p.compute()
   density = density.compute()
   dTdz = dTdz.compute()

   print(f'got tropopause height in {time() - start_time} seconds')
   start_time = time()

   # Get eddy statistics
   eddystats = get_eddy_statistics(ds.ucomp, ds.vcomp, ds.omega, thetacomp, ds.ps, radius, meanstats=meanstats).compute()

   print(f'got eddy statistics in {time() - start_time} seconds')
   start_time = time()

   # Get streamfunction
   psi = calc_streamfunction(meanstats.v, meanstats.ps, ds.phalf, radius).compute()
   psi_max = psi.sel(lat=slice(None,20)).max().compute().item() # avoid contribution from Ferrel cell

   print(f'got streamfunction in {time() - start_time} seconds')
   start_time = time()

   # get psi std_err:
   nperiods = len(ds.time)//nt_file # number of 100 day periods
   psis = []
   print(len(ds.time))
   for k in range(0, len(ds.time), nt_file):
      print(k)
      dsi = ds.isel(time=range(k, k+nt_file))
      psmean = dsi.ps.mean(['lon', 'time'])
      vmean = (dsi.vcomp * dsi.ps).mean(['lon', 'time']) / psmean
      dsi_psi = calc_streamfunction(vmean, psmean, dsi.phalf, radius)
      dsi_psi_max = dsi_psi.sel(lat=slice(None,25)).max().compute().item() # avoid contribution from Ferrel cell
      psis.append(dsi_psi_max)
   psi_std = np.std(psis)
   psi_stderr = psi_std / np.sqrt(nperiods)

   print(f'got streamfunction std_err in {time() - start_time} seconds')
   start_time = time()

   eke_arr = (ds.ucomp - meanstats.u)**2 + (ds.vcomp - meanstats.v)**2
   eke_arr = (eke_arr*ds.ps).mean(['lon', 'time'])/psmean
   pslat = ds.ps.mean(['lon', 'time'])
   eke_arr = get_zavg((eke_arr*np.cos(np.radians(ds.lat))*pslat).mean('lat')/psmean, ds.phalf)
   eke_arr = np.squeeze(eke_arr.values)
   

   print('STR MAX', psi_max/1e9)
   print('STR STDERR', psi_stderr/1e9)
   print('PSI SERIES', np.array(psis)/1e9)
   print('EKE ARR', np.round(eke_arr[0], 2), np.round(eke_arr[-1], 2))


   if not simulation['axisymm']:
      # Get spectral information
      spectrum, spectrum_sigma, xspectrum, xspectrum_sigma, cospectra  \
               = spectral_analysis(ds.ucomp, ds.vcomp, thetacomp, ds.ps, ds.phalf, radius, omega, chunks, meanstats=meanstats)

   print(f'got spectral information in {time() - start_time} seconds')

   start_time = time()

   # Get GQPV information and fluxes
   qgpv  = get_qgpv(ds.ucomp, ds.vcomp, thetacomp, ds.ps, radius, omega, meanstats=meanstats).compute()

   print(f'got GQPV information in {time() - start_time} seconds')
   start_time = time()


   if not simulation['axisymm']:
      ds_analysis = xr.merge([meanstats, eddystats, spectrum, spectrum_sigma, xspectrum, \
             xspectrum_sigma, cospectra, psi, qgpv, tropo_p, density, dTdz, heating_rate])
      

   else:
      ds_analysis = xr.merge([meanstats, eddystats, \
            psi, qgpv, tropo_p, density, dTdz, heating_rate])
   
   # Add phalf as a coordinate to the dataset
   ds_analysis = ds_analysis.assign_coords(phalf=ds.phalf)
       
   ds_analysis.attrs['psi_max'] = psi_max
   ds_analysis.attrs['eke_arr'] = eke_arr
   ds_analysis.attrs['psi_stderr'] = psi_stderr
   ds_analysis.attrs['psi_series'] = psis
   ds_analysis.attrs['start_day'] = day_list[0]
   ds_analysis.attrs['end_day'] = day_list[-1]

   for k, v in simulation.items():
      if isinstance(v, bool):
         ds_analysis.attrs[f'sim_{k}'] = int(v)
      else: 
         ds_analysis.attrs[f'sim_{k}'] = v

   ds_analysis.to_netcdf(f"{savepath}/fms_analysis.nc")

   print(f'saved analysis in {time() - start_time} seconds')
   start_time = time()

   # Make animations of flow fields
   # animate_flow_fields(ds, savepath, radius, omega)

   # print(f'made animations in {time() - start_time} seconds')

   start_time = time()
   calc_hadley_stats(f"{savepath}/fms_analysis.nc")
   print(f'got hadley stats in {time() - start_time} seconds')

   plot_streamfunction(ds_analysis, f"{savepath}/streamfunction.png")

   if not simulation['axisymm']:
      start_time = time()
      thvar_budget = get_thvar_budget(ds, meanstats, thetacomp, ds.temp, radius)

      thvar_budget.to_netcdf(f"{savepath}/thvar_budget.nc")
      print(f'saved thvar_budget in {time() - start_time} seconds')

      del thvar_budget

      start_time = time()
      # Calculate pressure difference between half levels
      phalf = ds.phalf.values
      phalf[0] = 0.1
      p_upper = ds.sigma*0 + phalf[:-1] / 1e3
      p_lower = ds.sigma*0 + phalf[1:] / 1e3
      # Create DataArray with same dimensions as ds.sigma

      p_upper = ds.ps*(ds.ucomp*0 + p_upper) # Broadcast dp to match dimensions of 4D array
      p_lower = ds.ps*(ds.ucomp*0 + p_lower) # Broadcast dp to match dimensions of 4D array
      
      # Calculate geopotential difference
      del_geopot = R * ds.temp * (np.log(p_lower) - np.log(p_upper))
      geo_pot = get_pcumsum(del_geopot)
      # Get geopotential at bottom boundary (zero)
      geo_pot_bottom = geo_pot * 0
         # Shift geo_pot up by one level and fill bottom with zeros
      geo_pot_shifted = xr.concat([geo_pot.isel(sigma=np.arange(1, len(geo_pot.sigma))).assign_coords(sigma=geo_pot.sigma[:-1]), geo_pot_bottom.isel(sigma=-1)], dim='sigma')
      geo_pot_shifted = geo_pot_shifted.chunk(chunks)
      
      # Take average between upper and lower boundaries
      geo_pot = (geo_pot + geo_pot_shifted) / 2
      
      # Get EKE budget terms
      eke_budget = get_eke_budget(ds, meanstats, geo_pot, radius, simulation['sigmab'], simulation['drag'])

      eke_budget.to_netcdf(f"{savepath}/eke_budget.nc")
      print(f'saved eke_budget in {time() - start_time} seconds')
   

def getsplit(x, runname):
   return runname.split(x)[-1].split('_')[0]


def proc_runname(runname):
   delh = int(getsplit('delh', runname))
   gamma = float(getsplit('gamma', runname))
   radius = float(getsplit('radius', runname))
   omega = float(getsplit('omega', runname))
   phi0 = getsplit('phi0', runname)
   if phi0 == '0.5':
      phi0 = 0.5
   else:
      phi0 = int(phi0)
   drag = float(getsplit('drag', runname))
   res = getsplit('res', runname)
   axisymm = getsplit('axisymm', runname) == 'True'
   Tsfcavg = float(getsplit('Tsfcavg', runname))
   kadays = int(getsplit('kadays', runname))
   ksdays = int(getsplit('ksdays', runname))
   sigmab = float(getsplit('sigmab', runname))
   sigmalat = int(getsplit('sigmalat', runname))
   tauc = int(getsplit('tauc', runname))
   freediff = getsplit('freediff', runname) == 'True'
   diff_coef = float(getsplit('diffcoef', runname))
   if 'zlev' in runname:
      zlev = int(getsplit('zlev', runname))
   else:
      zlev = 30

   simulation = {
      'delh': delh,
      'gamma': gamma,
      'radius': radius,
      'omega': omega,
      'phi0': phi0,
      'drag': drag,
      'res': res,
      'axisymm': axisymm,
      'Tsfcavg': Tsfcavg,
      'kadays': kadays,
      'ksdays': ksdays,
      'sigmab': sigmab,
      'sigmalat': sigmalat,
      'tauc': tauc,
      'freediff': freediff,
      'diff_coef': diff_coef,
      'zlev': zlev
   }

   return simulation

def get_heating_rate(meanstats, simulation):
   tau_f = simulation['kadays']*24*60*60 # seconds
   tau_s = simulation['ksdays']*24*60*60 # seconds
   sigma_b = simulation['sigmab']
   sigma_lat = simulation['sigmalat']
   phi0 = simulation['phi0']

   LAT = meanstats.temp*0 + meanstats.lat
   SIGMA = meanstats.temp*0 + meanstats.sigma
   sigma_max = np.maximum((SIGMA - sigma_b) / (1.0 - sigma_b), 0)
   tau = tau_f + (tau_s - tau_f) * sigma_max * np.exp(-(LAT - phi0)**2 / (2*sigma_lat**2))
   heating_rate = (meanstats.tempeq - meanstats.temp) / tau
   heating_rate = heating_rate.rename("heating_rate")
   heating_rate.attrs['long_name'] = "Newtonian heating tendency"
   heating_rate.attrs['units'] = "K/s"
   return heating_rate

def mean_flow_stats(ds):

   psmean = ds.ps.mean(['lon', 'time'])
   u = (ds.ucomp * ds.ps).mean(['lon', 'time']) / psmean
   v = (ds.vcomp * ds.ps).mean(['lon', 'time']) / psmean
   u = u.rename("u")
   v = v.rename("v")
   w = (ds.omega * ds.ps).mean(['lon', 'time']) / psmean
   w = w.rename("w")
   thetacomp = get_theta(ds.temp, ds.ps*ds.sigma)
   theta = (thetacomp * ds.ps).mean(['lon', 'time']) / psmean
   theta = theta.rename("theta")
   temp = (ds.temp * ds.ps).mean(['lon', 'time']) / psmean
   temp = temp.rename("temp")
   tempeq = ds.teq.isel(lon=0, time=0).rename("tempeq")
   thetaeq = get_theta(tempeq, 1e5*ds.sigma)
   thetaeq = thetaeq.rename("thetaeq")
   ps = psmean.rename("ps")
   ps_var = (((ds.ps-psmean)**2) * ds.ps).mean(['lon', 'time']) / psmean
   ps_var = ps_var.rename("ps_var")

   dt_tg_diffusion = (ds.dt_tg_diffusion*ds.ps).mean(['lon', 'time'])/psmean 
   dt_tg_diffusion = dt_tg_diffusion.rename("dt_tg_diffusion")
   dt_tg_convection = (ds.dt_tg_convection*ds.ps).mean(['lon', 'time'])/psmean
   dt_tg_convection = dt_tg_convection.rename("dt_tg_convection")
   dt_tg_radiation = (ds.dt_tg_radiation*ds.ps).mean(['lon', 'time'])/psmean
   dt_tg_radiation = dt_tg_radiation.rename("dt_tg_radiation")
   dt_ug_diffusion = (ds.dt_ug_diffusion*ds.ps).mean(['lon', 'time'])/psmean
   dt_ug_diffusion = dt_ug_diffusion.rename("dt_ug_diffusion")
   dt_vg_diffusion = (ds.dt_vg_diffusion*ds.ps).mean(['lon', 'time'])/psmean
   dt_vg_diffusion = dt_vg_diffusion.rename("dt_vg_diffusion")
   diff_m = (ds.diff_m*ds.ps).mean(['lon', 'time'])/psmean
   diff_m = diff_m.rename("diff_m")
   

   dt_tg_convection.attrs['long_name'] = "Convective heating tendency"
   dt_tg_convection.attrs['units'] = "K/s"
   dt_tg_radiation.attrs['long_name'] = "Radiative heating tendency"
   dt_tg_radiation.attrs['units'] = "K/s"
   dt_ug_diffusion.attrs['long_name'] = "Diffusive momentum tendency"
   dt_ug_diffusion.attrs['units'] = "m/s^2"
   dt_vg_diffusion.attrs['long_name'] = "Diffusive momentum tendency"
   dt_vg_diffusion.attrs['units'] = "m/s^2"
   dt_tg_diffusion.attrs['long_name'] = "Diffusive heating tendency"
   dt_tg_diffusion.attrs['units'] = "K/s"
   diff_m.attrs['long_name'] = "Momentum Diffusion coefficient"
   diff_m.attrs['units'] = "m^2/s"

   # Merge all variables
   meanstats = xr.merge([u,v,w,theta,temp,tempeq,thetaeq,ps,ps_var,dt_tg_diffusion,dt_tg_convection,dt_tg_radiation,dt_ug_diffusion,dt_vg_diffusion,diff_m])
   
   # Standardize missing value handling
   for var in meanstats.variables:
       if 'missing_value' in meanstats[var].encoding:
           del meanstats[var].encoding['missing_value']
   
   return meanstats
   

def animate_flow_fields(ds, savepath, radius, omega):
   # Make animations of flow fields
   ds = ds.isel(time=range(-200,0))

   # Make vertical animations
   make_vertical_animation(ds, 0, 7, radius, omega, savepath=f'{savepath}/vertical_animation_0_7.mp4')
   make_vertical_animation(ds, -7, 0, radius, omega, savepath=f'{savepath}/vertical_animation_-7_0.mp4')
   make_vertical_animation(ds, -5, 5, radius, omega, savepath=f'{savepath}/vertical_animation_-5_5.mp4')
   make_vertical_animation(ds, -40, -30, radius, omega, savepath=f'{savepath}/vertical_animation_-40_-30.mp4')
   make_vertical_animation(ds, 30, 40, radius, omega, savepath=f'{savepath}/vertical_animation_30_40.mp4')

   plevs = [800, 500, 300, 100]
   # Make horizontal animations
   make_horizontal_animation(ds, radius, omega, plevs, savepath=f'{savepath}/horizontal_animation_level.mp4')


if __name__ == '__main__':
   parser = argparse.ArgumentParser()
   parser.add_argument('--runname', type=str, required=True)
   parser.add_argument('--raw_output_dir', type=str, required=True)
   parser.add_argument('--days', type=int, required=True)
   parser.add_argument('--runs_per_script', type=int, required=True)
   parser.add_argument('--start_analysis', type=int, required=True)
   args = parser.parse_args()
   runname = args.runname
   raw_output_dir = args.raw_output_dir
   analyze(runname, raw_output_dir,args.start_analysis, args.runs_per_script, args.days)

