import sys
import os
import numpy as np
import xarray as xr
from time import time
from glob import glob
# from fms_analysis import proc_runname, mean_flow_stats
from utils import get_theta, get_zint
from utils import get_phideriv, get_pderiv, get_phiflux, get_pcumsum
import argparse

R = 287 # J/kg/K


def get_eke_budget(ds, meanstats, geo_pot, radius, sigma_b,Cdrag):
    vmean = meanstats.v
    wmean = meanstats.w
    umean = meanstats.u
    psmean = meanstats.ps
    geomean = (geo_pot*ds.ps).mean(['lon', 'time']) / psmean
    u_dissmean = meanstats.dt_ug_diffusion
    v_dissmean = meanstats.dt_vg_diffusion

    vp = ds.vcomp - vmean
    wp = ds.omega - wmean
    up = ds.ucomp - umean
    ps = ds.ps
    geop = geo_pot - geomean
    u_dissp = ds.dt_ug_diffusion - u_dissmean
    v_dissp = ds.dt_vg_diffusion - v_dissmean

    alpha = R*ds.temp/((ds.temp*0 + ds.sigma)*ps)
    alphap = alpha - (alpha * ps).mean(['lon', 'time']) / psmean

    walpha_eddy = (wp*alphap * ps).mean(['lon', 'time']) / psmean
    uv_eddy = (up*vp * ps).mean(['lon', 'time']) / psmean
    v2_eddy = (vp**2 * ps).mean(['lon', 'time']) / psmean
    uw_eddy = (up*wp * ps).mean(['lon', 'time']) / psmean
    vw_eddy = (vp*wp * ps).mean(['lon', 'time']) / psmean
    u_disseddy = (u_dissp*up * ps).mean(['lon', 'time']) / psmean
    v_disseddy = (v_dissp*vp * ps).mean(['lon', 'time']) / psmean

    sigma_b = 0.85
    drag_w = ds.ucomp*0 + np.maximum((ds.sigma - sigma_b)/(1-sigma_b), 0)
    V = np.sqrt(ds.ucomp**2 + ds.vcomp**2)
    bl_drag_u = drag_w * Cdrag * V * ds.ucomp
    bl_drag_v = drag_w * Cdrag * V * ds.vcomp
    bl_drag_up = bl_drag_u - (bl_drag_u * ps).mean(['lon', 'time']) / psmean
    bl_drag_vp = bl_drag_v - (bl_drag_v * ps).mean(['lon', 'time']) / psmean
    bl_drag = ((bl_drag_up * up + bl_drag_vp * vp) * ps).mean(['lon', 'time']) / psmean
    bl_drag_full = get_zint(bl_drag, ds.phalf, psmean)

    e = 1/2 * (vp**2 + up**2)
    emean = (e * ps).mean(['lon', 'time']) / psmean

    de_dphi = get_phideriv(emean, radius)
    de_dp = get_pderiv(emean, psmean)
    du_dphi = get_phideriv(umean, radius)
    du_dp = get_pderiv(umean, psmean)
    dv_dphi = get_phideriv(vmean, radius)
    dv_dp = get_pderiv(vmean, psmean)

    vep = (vp*(geop + e) * ps).mean(['lon', 'time']) / psmean
    wep = (wp*(geop + e) * ps).mean(['lon', 'time']) / psmean

    geophi_eddy = (vp*geop * ps).mean(['lon', 'time']) / psmean
    geop_eddy = (wp*geop * ps).mean(['lon', 'time']) / psmean
    vgeo_eddy = -get_phiflux(geophi_eddy, radius, psmean)
    wgeo_eddy = -get_pderiv(geop_eddy, psmean)

    trans_mean_phi = vmean * de_dphi
    trans_mean_p = wmean * de_dp
    trans_eddy_phi = get_phiflux(vep, radius, psmean)
    trans_eddy_p = get_pderiv(wep, psmean)

    ve_k = (vp * e * ps).mean(['lon', 'time']) / psmean
    we_k = (wp * e * ps).mean(['lon', 'time']) / psmean
    trans_eddy_k = get_phiflux(ve_k, radius, psmean) + get_pderiv(we_k, psmean)

    baroc_conv = -walpha_eddy

    shear_prod = -uv_eddy*du_dphi - uw_eddy*du_dp - v2_eddy*dv_dphi - vw_eddy*dv_dp

    dissipation = trans_mean_phi + trans_mean_p + trans_eddy_phi + trans_eddy_p - baroc_conv - shear_prod

    diss_other = u_disseddy + v_disseddy
    diss_other = diss_other.rename("diss_other").assign_attrs(
        {'long_name': 'Other dissipation', 'units': 'm^2 / s^3'})

    trans_mean_phi = trans_mean_phi.rename("trans_mean_phi").assign_attrs(
        {'long_name': 'Mean meridional transport of eddy kinetic energy', 'units': 'm^2 / s^3'})
    trans_mean_p = trans_mean_p.rename("trans_mean_p").assign_attrs(
        {'long_name': 'Mean vertical transport of eddy kinetic energy', 'units': 'm^2 / s^3'})
    trans_eddy_phi = trans_eddy_phi.rename("trans_eddy_phi").assign_attrs(
        {'long_name': 'Eddy meridional transport of eddy kinetic energy', 'units': 'm^2 / s^3'})
    trans_eddy_p = trans_eddy_p.rename("trans_eddy_p").assign_attrs(
        {'long_name': 'Eddy vertical transport of eddy kinetic energy', 'units': 'm^2 / s^3'})
    trans_eddy_k = trans_eddy_k.rename("trans_eddy_k").assign_attrs(
        {'long_name': 'Eddy kinetic energy transport by eddies', 'units': 'm^2 / s^3'})
    baroc_conv = baroc_conv.rename("baroc_conv").assign_attrs(
        {'long_name': 'Baroclinic conversion', 'units': 'm^2 / s^3'})
    pressure_work_phi = vgeo_eddy.rename("pressure_work_phi").assign_attrs(
        {'long_name': 'Pressure work', 'units': 'm^2 / s^3'})
    pressure_work_p = wgeo_eddy.rename("pressure_work_p").assign_attrs(
        {'long_name': 'Pressure work', 'units': 'm^2 / s^3'})
    shear_prod = shear_prod.rename("shear_prod").assign_attrs(
        {'long_name': 'Shear production', 'units': 'm^2 / s^3'})
    dissipation = dissipation.rename("dissipation").assign_attrs(
        {'long_name': 'Dissipation', 'units': 'm^2 / s^3'})
    bl_drag_full = bl_drag_full.rename("bl_drag_full").assign_attrs(
        {'long_name': 'Bottom level drag', 'units': 'm^2 / s^3'})
    bl_drag = bl_drag.rename("bl_drag").assign_attrs(
        {'long_name': 'Bottom level drag', 'units': 'm^2 / s^3'})
    eke_budget = xr.merge([trans_mean_phi, trans_mean_p, trans_eddy_phi, trans_eddy_p, baroc_conv, pressure_work_phi, pressure_work_p, shear_prod, dissipation, diss_other, trans_eddy_k, bl_drag_full, bl_drag])
    
    eke_budget = eke_budget.assign_attrs(
        {'long_name': 'Eddy kinetic energy budget', 'units': 'm^2 / s^3'})
    
    return eke_budget
    

def main(runname, raw_output_dir,days):
    start_time = time()

    print(f'Analyzing {runname}')

    simulation = proc_runname(runname)
    savepath = f"/resnick/groups/esm/reusebi/fms_analysis/{runname}"
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    fnames = np.concatenate([glob(f"{raw_output_dir}/combine/day{day:04}h00/day{day:04}h00.4xday.nc*") for day in days])
    if len(fnames) > len(days):
        drop_vars = ['latb']
    else:
        drop_vars = []

    chunks = {'time': 1, 'sigma': -1, 'lat': -1, 'lon': -1}
    ds = xr.open_mfdataset(fnames, combine='by_coords', decode_times=False, chunks=chunks, drop_variables=drop_vars)
    
    ds = ds[['ucomp', 'vcomp', 'omega', 'temp', 'ps', 'teq', 'phalf', 'pfull','dt_tg_diffusion','dt_tg_convection', 'dt_tg_radiation', 'dt_ug_diffusion', 'dt_vg_diffusion', 'diff_m']]
    print(f'opened dataset in {time() - start_time} seconds')
    start_time = time()

    ds = ds.assign_coords({"pfull": ds['pfull']/1e3}).rename({"pfull":"sigma"})
    radius = 6371000*simulation['radius'] #m  

    # Get mean flow statistics on pressure coordinates
    meanstats = mean_flow_stats(ds).compute()

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
