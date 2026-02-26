"""
Module to calculate lagrangian parcel backtrajectories from the 
850 hPa pressure level at all latitudes from a GCM FMS simulation.
"""



import numpy as np
import xarray as xr
from scipy.interpolate import RegularGridInterpolator
import os
from utils import get_theta
from fms_analysis import proc_runname
import argparse
from glob import glob
from multiprocessing import Pool
from functools import partial

# ---------------- CONFIG ----------------
RUNTIME_HOURS = 120        # 5 days backward
DT_SEC = -600              # 10-min step (negative = backward)
OUTPUT_EVERY_STEPS = 6     # save every 1 hour
DAY2S = 86400.0
a_earth = 6.371e6
SIGMA_INIT = 0.85
# START_DAYS = [320, 340, 360, 380, 400, 420, 440, 460, 480, 500]
START_DAYS = [200,210,220,230,240,250,260,270,280,290,300]
N_PER_TIME = 20            # 100 trajectories per start time
OUTFILE = "backtrajectories_sigma085.nc"
N_WORKERS = 10             # Number of parallel workers

# Global interpolators (set by worker_init)
u4d_global = None
v4d_global = None
omega4d_global = None
ps3d_global = None
theta4d_global = None

# ---------------- INTERPOLATORS ----------------
class TInterp4D:
    def __init__(self, da, sigma_name):
        da = da.transpose("time", "lon", "lat", sigma_name)
        self.t = da.time.values.astype(float) * DAY2S
        self.sig = da[sigma_name].values
        self.lat = da.lat.values; self.lon = da.lon.values
        self.data = da.data
    def interp(self, t_s, pos):
        i = np.searchsorted(self.t, t_s) - 1
        i = np.clip(i,0,len(self.t)-2)
        pos[:,0] = (pos[:,0] + 360) % 360
        a = (t_s - self.t[i])/(self.t[i+1]-self.t[i])
        f0 = RegularGridInterpolator((self.lon,self.lat,self.sig), np.asarray(self.data[i]),
                                     bounds_error=False, fill_value=np.nan)
        f1 = RegularGridInterpolator((self.lon,self.lat,self.sig), np.asarray(self.data[i+1]),
                                     bounds_error=False, fill_value=np.nan)

        return (1-a)*f0(pos) + a*f1(pos)

class TInterp3D:
    def __init__(self, da):
        da = da.transpose("time","lon","lat")
        self.t = da.time.values.astype(float)*DAY2S
        self.lat = da.lat.values; self.lon = da.lon.values
        self.data = da.data
    def interp(self, t_s, pos):
        i = np.searchsorted(self.t, t_s) - 1
        i = np.clip(i,0,len(self.t)-2)
        pos[:,0] = (pos[:,0] + 360) % 360
        a = (t_s - self.t[i])/(self.t[i+1]-self.t[i])
        f0 = RegularGridInterpolator((self.lon,self.lat), np.asarray(self.data[i]),
                                     bounds_error=False, fill_value=np.nan)
        f1 = RegularGridInterpolator((self.lon,self.lat), np.asarray(self.data[i+1]),
                                     bounds_error=False, fill_value=np.nan)
        return (1-a)*f0(pos) + a*f1(pos)

# ---------------- WORKER INITIALIZATION ----------------
def worker_init(u4d, v4d, omega4d, ps3d, theta4d):
    """Initialize global interpolators in each worker process."""
    global u4d_global, v4d_global, omega4d_global, ps3d_global, theta4d_global
    u4d_global = u4d
    v4d_global = v4d
    omega4d_global = omega4d
    ps3d_global = ps3d
    theta4d_global = theta4d

# ---------------- DYNAMICS (used by workers) ----------------
def rhs_lonlatp(t_s, y):
    """RHS for trajectory integration using global interpolators."""
    ps = ps3d_global.interp(t_s, y[:,:2])
    u = u4d_global.interp(t_s, y)
    v = v4d_global.interp(t_s, y)
    omg = omega4d_global.interp(t_s, y)/ps
    # Convert u from m/s to lon/s using a_earth * cos(lat)
    cosphi = np.cos(np.deg2rad(y[:,1]))  # latitude in degrees -> radians
    u_lon = u / (a_earth * cosphi) * 180 / np.pi # lon/s
    v_lat = v / a_earth * 180 / np.pi # lat/s
    return np.stack([u_lon, v_lat, omg], axis=-1)

def rk4_step(t_s, y, h):
    """RK4 integration step."""
    k1 = rhs_lonlatp(t_s, y)
    k2 = rhs_lonlatp(t_s+0.5*h, y+0.5*h*k1)
    k3 = rhs_lonlatp(t_s+0.5*h, y+0.5*h*k2)
    k4 = rhs_lonlatp(t_s+h, y+h*k3)
    y_new = y+(h/6.0)*(k1+2*k2+2*k3+k4)
    y_new[:,0] = (y_new[:,0] + 360) % 360
    return y_new

# ---------------- WORKER FUNCTION ----------------
def integrate_latitude(lat0, start_days, ntraj=100,
                    sigma0=0.85, runtime_hours=240,
                    dt_sec=-900, output_every=4):
    """
    Integrate (ntraj × len(start_days)) parcels for one latitude.
    This function runs in a worker process and uses global interpolators.
    
    Returns arrays with shape (ntime_saved, ntraj_total)
    """
    print(f"Processing latitude {lat0:.2f}°")
    
    nsteps = int(abs(runtime_hours * 3600 / dt_sec))
    ntime  = (nsteps // output_every) + 1
    nlaunch = len(start_days)
    ntraj_total = nlaunch * ntraj

    # Preallocate
    lon_all = np.full((ntime, ntraj_total), np.nan, dtype=np.float32)
    lat_all = np.full_like(lon_all, np.nan, dtype=np.float32)
    p_all   = np.full_like(lon_all, np.nan, dtype=np.float32)
    sigma_all = np.full_like(lon_all, np.nan, dtype=np.float32)
    th_all  = np.full_like(lon_all, np.nan, dtype=np.float32)
    u_all = np.full_like(lon_all, np.nan, dtype=np.float32)
    v_all = np.full_like(lon_all, np.nan, dtype=np.float32)
    omg_all = np.full_like(lon_all, np.nan, dtype=np.float32)

    # Loop over start times
    for it, t0_days in enumerate(start_days):
        j0, j1 = it * ntraj, (it + 1) * ntraj

        lon0 = np.linspace(0, 360, ntraj, endpoint=False)
        lat0_vec = np.full(ntraj, lat0)
        sigma_vec = np.full(ntraj, sigma0)

        t_s = float(t0_days) * DAY2S
        save_step = 0
        y = np.stack([lon0, lat0_vec, sigma_vec], axis=-1)
        
        for k in range(nsteps + 1):
            # Save every 'output_every' integration steps
            if k % output_every == 0:
                ps = ps3d_global.interp(t_s, y[:,:2])
                lon_all[save_step, j0:j1] = y[:,0]
                lat_all[save_step, j0:j1] = y[:,1]
                v = rhs_lonlatp(t_s, y)
                cosphi = np.cos(np.deg2rad(y[:,1]))
                u_all[save_step, j0:j1] = v[:,0] * (a_earth * cosphi) * np.pi / 180
                v_all[save_step, j0:j1] = v[:,1] * a_earth * np.pi / 180
                omg_all[save_step, j0:j1] = v[:,2] * ps
                sigma_all[save_step, j0:j1] = y[:,2]
                p_all[save_step, j0:j1] = y[:,2] * ps
                th_all[save_step, j0:j1] = theta4d_global.interp(t_s, y)
                save_step += 1
                if save_step == ntime:
                    break

            # advance all parcels
            y = rk4_step(t_s, y, dt_sec)
            t_s += dt_sec

    print(f"Completed latitude {lat0:.2f}°")
    return lon_all, lat_all, sigma_all, p_all, th_all, u_all, v_all, omg_all

def get_trajectories(ds, a_earth):
    """
    Main function to compute trajectories using multiprocessing.
    """
    sigma_name = 'sigma'

    # Create interpolators and LOAD DATA INTO MEMORY
    # This is critical - we must load data before multiprocessing to avoid
    # concurrent NetCDF file access issues
    print("Loading data into memory (this may take a minute)...")
    u_da = ds["ucomp"].load()  # Force load into memory
    v_da = (ds["comp"] if "comp" in ds else ds["vcomp"]).load()
    omega_da = ds["omega"].load()
    ps_da = ds["ps"].load()
    theta_da = get_theta(ds.temp, ds.ps*ds.sigma).load()
    
    print("Creating interpolators...")
    u4d = TInterp4D(u_da, sigma_name)
    v4d = TInterp4D(v_da, sigma_name)
    omega4d = TInterp4D(omega_da, sigma_name)
    theta4d = TInterp4D(theta_da, sigma_name)
    ps3d = TInterp3D(ps_da)

    # Get latitudes to process
    latitudes = np.unique(ds.lat.sel(lat=slice(-80,80)).values)
    
    sigma_init = ds.sigma.sel(sigma=SIGMA_INIT, method='nearest').item()
    # Prepare arguments for each latitude
    lat_args = [(lat0, START_DAYS, N_PER_TIME, sigma_init, 
                 RUNTIME_HOURS, DT_SEC, OUTPUT_EVERY_STEPS) 
                for lat0 in latitudes]

    # Use multiprocessing to parallelize across latitudes
    print(f"Processing {len(latitudes)} latitudes using {N_WORKERS} workers...")
    with Pool(processes=N_WORKERS, 
              initializer=worker_init, 
              initargs=(u4d, v4d, omega4d, ps3d, theta4d)) as pool:
        # Use starmap to unpack arguments
        results = pool.starmap(integrate_latitude, lat_args)

    # Collect results
    print("Collecting results...")
    nsteps = int(abs(RUNTIME_HOURS*3600 / DT_SEC))
    ntime  = (nsteps // OUTPUT_EVERY_STEPS) + 1
    times = np.arange(ntime) * abs(DT_SEC)*OUTPUT_EVERY_STEPS / DAY2S
    TRAJ_PER_LAT = N_PER_TIME * len(START_DAYS)
    L = len(latitudes)

    # Preallocate output arrays
    out_lon = np.full((ntime, TRAJ_PER_LAT, L), np.nan, dtype=np.float32)
    out_lat = np.full_like(out_lon, np.nan)
    out_p   = np.full_like(out_lon, np.nan)
    out_th  = np.full_like(out_lon, np.nan)
    out_sigma = np.full_like(out_lon, np.nan)
    out_u = np.full_like(out_lon, np.nan)
    out_v = np.full_like(out_lon, np.nan)
    out_omg = np.full_like(out_lon, np.nan)

    # Fill output arrays from results
    for j, (lon_all, lat_all, sigma_all, p_all, th_all, u_all, v_all, omg_all) in enumerate(results):
        out_lon[:, :, j] = lon_all
        out_lat[:, :, j] = lat_all
        out_p  [:, :, j] = p_all
        out_th [:, :, j] = th_all
        out_sigma[:, :, j] = sigma_all
        out_u[:, :, j] = u_all
        out_v[:, :, j] = v_all
        out_omg[:, :, j] = omg_all
    print("All latitudes processed!")
    
    # ---------------- BUILD OUTPUT NETCDF ----------------
    dsout = xr.Dataset(
        {
            "p_lon":   (("time", "traj", "lat"), out_lon),
            "p_lat":   (("time", "traj", "lat"), out_lat),
            "p_sigma": (("time", "traj", "lat"), out_sigma),
            "p_p":     (("time", "traj", "lat"), out_p),
            "p_theta": (("time", "traj", "lat"), out_th),
            "p_u":     (("time", "traj", "lat"), out_u),
            "p_v":     (("time", "traj", "lat"), out_v),
            "p_omega":   (("time", "traj", "lat"), out_omg),
        },
        coords={
            "time": ("time", times, {"units": "days since launch"}),
            "traj": ("traj", np.arange(TRAJ_PER_LAT, dtype=np.int32)),
            "lat":  ("lat", np.array(latitudes, dtype=np.float32)),
            "sigma": SIGMA_INIT,
        },
        attrs={
            "description": "Vectorized back trajectories at sigma=0.85 using omega (Dp/Dt)",
            "start_days": str(START_DAYS),
            "n_per_time": N_PER_TIME,
            "output_every_steps": OUTPUT_EVERY_STEPS,
            "runtime_hours": RUNTIME_HOURS,
            "dt_seconds": DT_SEC,
        },
    )
    return dsout


def main(runname, raw_output_dir, days):
    simulation = proc_runname(runname)
    savepath = f"/resnick/groups/esm/reusebi/fms_analysis/{runname}"
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    fnames = np.concatenate([glob(f"{raw_output_dir}/combine/day{day:04}h00/day{day:04}h00.4xday.nc*") for day in days])

    # Use open_mfdataset for efficient, lazy loading of multiple files
    # chunks = {'time': 800, 'lon': -1, 'lat': -1, 'sigma':10}
    if len(fnames) > len(days):
        drop_vars = ['latb']
    else:
        drop_vars = []
    if 'T85' in runname:
        chunks = {'time': 1, 'sigma': 5, 'lat': -1, 'lon': -1}
    else:
        chunks = {'time': 4, 'sigma': 5, 'lat': -1, 'lon': -1}
    ds = xr.open_mfdataset(fnames, combine='by_coords', decode_times=False, chunks=chunks, drop_variables=drop_vars)
    ds = ds[['ucomp', 'vcomp', 'omega', 'temp', 'ps']]

    ds = ds.assign_coords({"pfull": ds['pfull']/1e3}).rename({"pfull":"sigma"})
    
    # Add ghost point at lon=360 (copy of lon=0 for periodic boundary)
    print("Adding periodic boundary at lon=360...")
    ds_lon0 = ds.isel(lon=0)
    ds = xr.concat([ds, ds_lon0], dim='lon')
    new_lon = np.append(ds.lon.values[:-1], 360.0)
    ds = ds.assign_coords(lon=new_lon)
    
    radius = 6371000*simulation['radius'] #m  

    trajectories = get_trajectories(ds, radius)
    print(trajectories)
    trajectories.to_netcdf(f"{savepath}/parcel_trajectories.nc")
    print('Saved to', f"{savepath}/parcel_trajectories.nc")

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
