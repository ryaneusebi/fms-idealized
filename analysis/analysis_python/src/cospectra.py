"""
Xarray/Dask-compatible version of cospectra.py for eddy flux co-spectra using the Hayashi (1971) method.
This version uses xarray DataArray throughout, avoids .values, and leverages xarray.apply_ufunc for efficient, parallel computation with open_mfdataset.
"""

import numpy as np
import xarray as xr
import scipy.signal as ss
import scipy.interpolate as si
import time

def _compute_spacetime_spectrum(var1, var2, radius_at_lat, time_step=1.0, apply_smoothing=True, smooth_width=5.0):
    # var1, var2: (time, lon)
    time_steps, spatial_points = np.shape(var1)
    half_spatial = spatial_points // 2 
    nfft = 256
    fft1 = np.fft.fft(var1, axis=1) / spatial_points
    fft2 = np.fft.fft(var2, axis=1) / spatial_points
    wavenumbers = np.arange(spatial_points)[:half_spatial] / radius_at_lat
    cos_fft1 = fft1[:, :half_spatial].real
    sin_fft1 = fft1[:, :half_spatial].imag
    cos_fft2 = fft2[:, :half_spatial].real
    sin_fft2 = fft2[:, :half_spatial].imag
    time_freq = nfft // 2 + 1
    
    # Vectorized CSD - process all wavenumbers at once
    # Transpose to (wavenumber, time) for csd with axis=-1
    freqs, csd_cos = ss.csd(cos_fft1.T, cos_fft2.T, fs=1/time_step, nfft=nfft, window='hann', scaling='density', axis=-1)
    _, csd_sin = ss.csd(sin_fft1.T, sin_fft2.T, fs=1/time_step, nfft=nfft, window='hann', scaling='density', axis=-1)
    _, csd_cos_sin = ss.csd(cos_fft1.T, sin_fft2.T, fs=1/time_step, nfft=nfft, window='hann', scaling='density', axis=-1)
    _, csd_sin_cos = ss.csd(sin_fft1.T, cos_fft2.T, fs=1/time_step, nfft=nfft, window='hann', scaling='density', axis=-1)
    
    # csd_* have shape (half_spatial, time_freq) - transpose to (time_freq, half_spatial)
    k_pos = (csd_cos.real + csd_sin.real + csd_cos_sin.imag - csd_sin_cos.imag)[:, :time_freq].T
    k_neg = (csd_cos.real + csd_sin.real - csd_cos_sin.imag + csd_sin_cos.imag)[:, :time_freq].T
    
    k_combined = np.zeros((time_freq * 2, half_spatial))
    k_combined[:time_freq, :] = k_neg[::-1, :]
    k_combined[time_freq:, :] = k_pos
    if apply_smoothing:
        x = np.linspace(-time_freq // 2, time_freq // 2., time_freq)
        gauss_kernel = np.exp(-x ** 2 / (2. * smooth_width ** 2))
        gauss_kernel /= gauss_kernel.sum()
        # Vectorized smoothing using scipy.ndimage
        from scipy.ndimage import convolve1d
        k_combined = convolve1d(k_combined, gauss_kernel, axis=0, mode='reflect')
    k_neg = k_combined[:time_freq, :][::-1, :]
    k_pos = k_combined[time_freq:, :]
    return k_pos, k_neg, wavenumbers, freqs

def _compute_phase_speed_spectrum(power_pos, power_neg, wavenumbers, frequencies, max_phase_speed, n_speeds, min_wavenumber=1, max_wavenumber=50):
    n_wavenumbers = len(wavenumbers)
    phase_speeds = np.linspace(0., max_phase_speed, n_speeds)
    power_pos_c = np.zeros((n_speeds, n_wavenumbers))
    power_neg_c = np.zeros((n_speeds, n_wavenumbers))
    for i in range(min_wavenumber, min(max_wavenumber, n_wavenumbers)):
        interp_pos = si.interp1d(frequencies / wavenumbers[i], power_pos[:, i], 'linear', bounds_error=False, fill_value=0)
        interp_neg = si.interp1d(frequencies / wavenumbers[i], power_neg[:, i], 'linear', bounds_error=False, fill_value=0)
        max_valid_speed = max(frequencies) / wavenumbers[i]
        valid_speeds = phase_speeds <= max_valid_speed
        power_pos_c[valid_speeds, i] = interp_pos(phase_speeds[valid_speeds]) * wavenumbers[i]
        power_neg_c[valid_speeds, i] = interp_neg(phase_speeds[valid_speeds]) * wavenumbers[i]
    return np.sum(power_pos_c, axis=1), np.sum(power_neg_c, axis=1), phase_speeds

def compute_eddy_cospectra(field1: xr.DataArray, field2: xr.DataArray, planet_radius: float, time_step: float, max_phase_speed: float = 50, n_phase_speeds: int = 50, apply_smoothing: bool = True, smooth_width: float = 5) -> xr.DataArray:
    """
    Xarray/Dask-compatible version of compute_eddy_cospectra.
    field1, field2: DataArray with dims (time, lat, lon) or (sigma, lat, time, lon)
    Returns: DataArray with dims (sigma, lat, c) or (lat, c)
    """
    def _per_lat(field1_1d, field2_1d, lat):
        radius_at_lat = 2 * np.pi * planet_radius * np.cos(np.radians(lat))
        k_pos, k_neg, wavenums, freqs = _compute_spacetime_spectrum(
            field1_1d, field2_1d, radius_at_lat, time_step, apply_smoothing, smooth_width
        )
        power_pos, power_neg, speeds = _compute_phase_speed_spectrum(
            k_pos, k_neg, wavenums, freqs, max_phase_speed, n_phase_speeds
        )
        phase_spec = np.concatenate([power_neg[::-1], power_pos])
        full_speeds = np.concatenate([-speeds[::-1], speeds])
        return phase_spec, full_speeds

    # Determine if sigma is a dimension
    has_sigma = 'sigma' in field1.dims

    # Only do over final 300 days
    if len(field1.time) > 1200:
        field1 = field1.isel(time=np.arange(-1200,0,1))
        field2 = field2.isel(time=np.arange(-1200,0,1))

    start_time = time.time()
    # Rechunk for cospectra: need all time and lon in one chunk for FFT/CSD
    cospec_chunks = {'time': -1, 'lon': -1}
    if 'sigma' in field1.dims:
        cospec_chunks['sigma'] = 5
    if 'lat' in field1.dims:
        cospec_chunks['lat'] = 2
    field1 = field1.chunk(cospec_chunks)
    field2 = field2.chunk(cospec_chunks)
    print(f'chunked in {time.time() - start_time} seconds')
    start_time = time.time()

    # Prepare input_core_dims and transpose order
    if has_sigma:
        # (sigma, lat, time, lon)
        input_core_dims = [['time', 'lon'], ['time', 'lon'], []]
        sigma = field1['sigma']
        lat = field1['lat']
        # Vectorize over sigma and lat
        result, phase_speeds = xr.apply_ufunc(
            _per_lat,
            field1,
            field2,
            lat,
            input_core_dims=input_core_dims,
            output_core_dims=[['c'], ['c']],
            vectorize=True,
            dask='parallelized',
            output_dtypes=[float, float],
            dask_gufunc_kwargs={'output_sizes': {'c': 2 * n_phase_speeds}}
        )
        phase_speed_coord = phase_speeds.isel(sigma=0, lat=0).values
        result = xr.DataArray(result, dims=['sigma', 'lat', 'c'], coords={'sigma': sigma, 'lat': lat, 'c': phase_speed_coord})
    else:
        # (lat, time, lon)
        input_core_dims = [['time', 'lon'], ['time', 'lon'], []]
        lat = field1['lat']
        result, phase_speeds = xr.apply_ufunc(
            _per_lat,
            field1,
            field2,
            lat,
            input_core_dims=input_core_dims,
            output_core_dims=[['c'], ['c']],
            vectorize=True,
            dask='parallelized',
            output_dtypes=[float, float],
            dask_gufunc_kwargs={'output_sizes': {'c': 2 * n_phase_speeds}}
        )
        phase_speed_coord = phase_speeds.isel(lat=0).values
        result = xr.DataArray(result, dims=['lat', 'c'], coords={'lat': lat, 'c': phase_speed_coord})

    print(f'finished cospectra in {time.time() - start_time} seconds')

    del(field2)
    return result 