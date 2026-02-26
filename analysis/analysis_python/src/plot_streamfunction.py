import xarray as xr
import matplotlib.pyplot as plt
import numpy as np



def plot_streamfunction(ds, outpath):
   fig, ax = plt.subplots(figsize=(12,4))
   ds.psi.plot.contour(ax=ax, levels=20, colors='black', linewidths=1)
   cosphi = np.cos(np.radians(ds.lat))
   omega = ds.attrs['sim_omega']*7.292e-5 # radians per second
   radius = 6371000*ds.attrs['sim_radius'] #m  
   ang_mom = omega*radius**2*cosphi**2 + radius*cosphi*ds.u
   ds.emfd.plot(ax=ax)
   phi_levels = np.radians(np.concatenate([np.arange(0, 21, 2), np.arange(25, 50, 5)]))[::-1]
   ang_mom_levels = np.cos(phi_levels)**2

   ang_mom.plot.contour(ax=ax, levels=ang_mom_levels, colors='gray', linewidths=0.5, x='lat')


   # Find streamfunction maximum and its location
   psi_max = float(ds.psi.max())
   psi_max_coords = ds.psi.where(ds.psi == ds.psi.max(), drop=True)
   max_lat = float(psi_max_coords.lat)
   max_sigma = float(psi_max_coords.sigma)

   ax.invert_yaxis()
   print(psi_max)
   
   # Add arrow annotation
   ax.annotate(f'{round(psi_max/1e9)}', 
                xy=(max_lat, max_sigma),  # Arrow tip
                xytext=(20,0.2),  # Text position
                arrowprops=dict(facecolor='black', shrink=0.02, width=0.5, headwidth=6, headlength=8),
                fontsize=14)
   

   ax.set_ylabel('Sigma')
   ax.set_title('Streamfunction')
   ax.set_xlabel('Latitude')
   xticks = [-90,-60,-30,0,30,60,90]
   ax.set_xticks(xticks, labels=xticks)
   ax.set_yticks([0.2, 0.8], labels=[0.2, 0.8])
   ax.set_title('Streamfunction')
   plt.savefig(outpath)

if __name__ == '__main__':
   res = 'T85'
   delh = 120
   phi0 = 0
   omega = 1
   axisymm = 'False'
   radius = 1
   drag = 5e-6
   tauc = 86400
   freediff = 'False'
   diffcoef = 0.0
   gamma = 0.7
   sigmab = 0.85
   sigmalat = 20
   kadays = 50
   ksdays = 7
   fname = f'/resnick/groups/esm/reusebi/fms_analysis/delh{delh}_gamma{gamma}_phi0{phi0}_radius{radius}_omega{omega}_drag{drag}_res{res}_axisymm{axisymm}_Tsfcavg310_kadays{kadays}_ksdays{ksdays}_sigmab{sigmab}_sigmalat{sigmalat}_tauc{tauc}_freediff{freediff}_diffcoef{diffcoef}/fms_analysis.nc'
   ds = xr.open_dataset(fname, decode_times=False)
   outpath = fname.replace('fms_analysis.nc', 'streamfunction.png')
   plot_streamfunction(ds, outpath)