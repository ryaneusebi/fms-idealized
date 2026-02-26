import sys
import os
import numpy as np
import xarray as xr
from time import time
from glob import glob
# from fms_analysis import proc_runname, mean_flow_stats
from utils import get_theta
from utils import get_phideriv, get_pderiv, get_phiflux
import argparse



def get_thvar_budget(ds, meanstats, theta, temp, radius):
    vmean = meanstats.v
    wmean = meanstats.w
    thetamean = meanstats.theta
    psmean = meanstats.ps

    Qrad_mean = meanstats.dt_tg_radiation
    Qconv_mean = meanstats.dt_tg_convection
    Qdiff_mean = meanstats.dt_tg_diffusion

    thetap = theta - thetamean
    vp = ds.vcomp - vmean
    wp = ds.omega - wmean
    thetap = theta - thetamean
    ps = ds.ps

    Qrad_p = ds.dt_tg_radiation - Qrad_mean
    Qconv_p = ds.dt_tg_convection - Qconv_mean
    Qdiff_p = ds.dt_tg_diffusion - Qdiff_mean

    conversion_factor = theta/temp
    

    th_var = (thetap**2 * ps).mean(['lon', 'time']) / psmean
    wtheta_eddy = (wp*thetap * ps).mean(['lon', 'time']) / psmean
    vtheta_eddy = (vp*thetap * ps).mean(['lon', 'time']) / psmean
    vtheta2 = (vp*thetap**2 * ps).mean(['lon', 'time']) / psmean
    wtheta2 = (wp*thetap**2 * ps).mean(['lon', 'time']) / psmean

    dtheta_dy = get_phideriv(thetamean, radius)
    dtheta_dp = get_pderiv(thetamean, psmean)

    trans_mean_phi = 1/2 * vmean * get_phideriv(th_var, radius)
    trans_mean_p = 1/2 * wmean * get_pderiv(th_var, psmean)
    trans_eddy_phi = 1/2 * get_phiflux(vtheta2, radius, psmean)
    trans_eddy_p = 1/2 * get_pderiv(wtheta2, psmean)

    baroc_prod_phi = -vtheta_eddy * dtheta_dy
    baroc_prod_p = -wtheta_eddy * dtheta_dp

    Qrad_source = (Qrad_p*thetap * ps * conversion_factor).mean(['lon', 'time']) / psmean
    Qconv_source = (Qconv_p*thetap * ps * conversion_factor).mean(['lon', 'time']) / psmean
    Qdiff_source = (Qdiff_p*thetap * ps * conversion_factor).mean(['lon', 'time']) / psmean


    trans_mean_phi = trans_mean_phi.rename("trans_mean_phi").assign_attrs(
        {'long_name': 'Mean meridional transport of potential temperature variance', 'units': 'K^2 / s'})
    trans_mean_p = trans_mean_p.rename("trans_mean_p").assign_attrs(
        {'long_name': 'Mean vertical transport of potential temperature variance', 'units': 'K^2 / s'})
    trans_eddy_phi = trans_eddy_phi.rename("trans_eddy_phi").assign_attrs(
        {'long_name': 'Eddy meridional transport of potential temperature variance', 'units': 'K^2 / s'})
    trans_eddy_p = trans_eddy_p.rename("trans_eddy_p").assign_attrs(
        {'long_name': 'Eddy vertical transport of potential temperature variance', 'units': 'K^2 / s'})
    baroc_prod_phi = baroc_prod_phi.rename("baroc_prod_phi").assign_attrs(
        {'long_name': 'Baroclinic production of potential temperature variance', 'units': 'K^2 / s'})
    baroc_prod_p = baroc_prod_p.rename("baroc_prod_p").assign_attrs(
        {'long_name': 'Baroclinic production of potential temperature variance', 'units': 'K^2 / s'})
    Qrad_source = Qrad_source.rename("Qrad_source").assign_attrs(
        {'long_name': 'Radiative source of potential temperature variance', 'units': 'K^2 / s'})
    Qconv_source = Qconv_source.rename("Qconv_source").assign_attrs(
        {'long_name': 'Convective source of potential temperature variance', 'units': 'K^2 / s'})
    Qdiff_source = Qdiff_source.rename("Qdiff_source").assign_attrs(
        {'long_name': 'Diffusive source of potential temperature variance', 'units': 'K^2 / s'})

    thvar_budget = xr.merge([trans_mean_phi, trans_mean_p, trans_eddy_phi, trans_eddy_p, baroc_prod_phi, baroc_prod_p, Qrad_source, Qconv_source, Qdiff_source])
    
    thvar_budget = thvar_budget.assign_attrs(
        {'long_name': 'Eddy potential temperature variance budget', 'units': 'K^2 / s'})
    
    return thvar_budget
    

def main(runname, raw_output_dir,days):
    start_time = time()

    print(f'Analyzing {runname}')

    simulation = proc_runname(runname)
    user = os.environ['USER']
    savepath = f"/resnick/groups/esm/{user}/fms_analysis/{runname}"
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    fnames = np.concatenate([glob(f"{raw_output_dir}/combine/day{day:04}h00/day{day:04}h00.4xday.nc*") for day in days])
    if len(fnames) > len(days):
        drop_vars = ['latb']
    else:
        drop_vars = []
    ds = xr.open_mfdataset(fnames, combine='by_coords', decode_times=False, chunks={'time': 4, 'sigma': 5, 'lat': -1, 'lon': -1}, drop_variables=drop_vars)
    ds = ds[['ucomp', 'vcomp', 'omega', 'temp', 'ps', 'teq', 'phalf', 'pfull','dt_tg_diffusion','dt_tg_convection', 'dt_tg_radiation', 'dt_ug_diffusion', 'dt_vg_diffusion', 'diff_m']]
    print(f'opened dataset in {time() - start_time} seconds')
    start_time = time()

    ds = ds.assign_coords({"pfull": ds['pfull']/1e3}).rename({"pfull":"sigma"})
    thetacomp = get_theta(ds.temp, ds.ps*ds.sigma)

    radius = 6371000*simulation['radius'] #m  

    # Get mean flow statistics on pressure coordinates
    meanstats = mean_flow_stats(ds).compute()

    thvar_budget = get_thvar_budget(ds, meanstats, thetacomp, ds.temp, radius)

    thvar_budget.to_netcdf(f"{savepath}/thvar_budget.nc")
    print(f'saved thvar_budget in {time() - start_time} seconds')


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
   days = np.arange(args.start_analysis, args.runs_per_script+1)*args.days
   main(runname, raw_output_dir,days)
