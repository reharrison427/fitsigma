import matplotlib.pyplot as plt
import pymcfost
from astropy.io import fits
from scipy import constants
import os
from scipy.ndimage import rotate
import scipy.interpolate
import glob
from astropy import units as u
from astropy.modeling import models
from astropy.constants import M_sun
import numpy as np
import shutil

def get_kappa(parafile, fitsfile):
    # returns kappa in units of cm^2/g at freq of fits file
    dir = os.getcwd()
    pymcfost.run(parafile, options='-dust_prop')
    dust = pymcfost.Dust_model(dir+'/data_dust')
    kappa = dust.kappa
    lam = dust.wl
    # read in parameters from fits file
    fh = fits.open(fitsfile)
    header = fh[0].header
    freq = header['CRVAL3']
    wl = constants.c/freq
    slambda = wl*1e6
    kappa_at_slambda = np.interp(slambda, lam, kappa) * u.cm**2 / u.g
    return kappa_at_slambda

def get_radii(parafile):
    # get disk structure
    dir = os.getcwd()
    pymcfost.run(parafile, options='-disk_struct')
    r_mcfost = pymcfost.Disc(dir+"/data_disk").r()[0][0]
    return r_mcfost


def I_profile_jyarcsec(parafile = None, fitsfile = None, distance = None, rmax = None, pa = 0.0):
    r_mcfost = get_radii(parafile)
    # get fits info
    fh = fits.open(fitsfile)
    header = fh[0].header
    bmaj = header['BMAJ'] * 3600.
    bmin = header['BMIN'] * 3600.
    beam_area = np.pi/(4.*np.log(2.))*bmaj*bmin
    pix_scale = header['CDELT2'] * 3600.
    pix_scale_AU = pix_scale * distance
    freq = header['CRVAL3'] * u.Hz
    beam_area = np.pi/(4.*np.log(2.))*bmaj*bmin

     # rotate image so majax is along x axis
    image = fh[0].data[0, 0]
    # convert nan to zero
    image = np.nan_to_num(image, nan=0.0)
    # rotate image so that major axis is along x-axis
    # scipy.ndimage.rotate rotates image clockwise
    # this rotates image so that N side of majax in the positive x direction
    image = rotate(image, pa+90.0)

    # get cut along major axis
    # get center of image
    center = (int(image.shape[0]/2), int(image.shape[1]/2))# use center pixel of image
    center_x = center[1]
    center_y = center[0]
    xmax = int((rmax/distance)/pix_scale)  # maximum x distance, in pixels
    x_profile = np.linspace(1, xmax, xmax) # distances along x axis in pixels, exclude center pixel
    x_profile_arcsec = x_profile * pix_scale
    x_profile_au = x_profile * pix_scale_AU 
    profile_pos = np.array([image[center_y][i] for i in range(center_x+1, center_x+xmax+1)]) # I values for positive x distances along majax
    profile_neg = np.array([image[center_y][i] for i in range(center_x-xmax, center_x)]) # I values for negative x distances along majax
    profile = np.mean(np.array([profile_pos, np.flip(profile_neg)]), axis=0)
    Cut = profile/beam_area # Convert to Jy/arcsec^2

    # interpolate onto mcfost grid
    spl_data = scipy.interpolate.CubicSpline(x_profile_au, Cut)
    Cut_data_interp = spl_data(r_mcfost)

    # convert to Jy/sr
    Cut_data_interp_Jyarcsec = Cut_data_interp * u.Jy / u.arcsec**2

    return Cut_data_interp_Jyarcsec

# get plots of residuals vs. radius
def make_residual_plot(radius, data, model, iteration, sigma=None):

    residual = model - data

    np.savetxt(
        f"residual_iter{iteration}.txt",
        np.column_stack([radius, residual]),
        header="radius_AU residual_Jy_arcsec2"
    )

    plt.figure()

    plt.plot(radius, residual)

    plt.axhline(0, color='k')

    if sigma is not None:

        plt.axhline(3*sigma, linestyle='--')

        plt.axhline(-3*sigma, linestyle='--')

    plt.xlabel("Radius (AU)")
    plt.ylabel("Model − Data (Jy/arcsec$^2$)")
    #plt.xscale('log')

    plt.savefig(f"residual_iter{iteration}.png")

    plt.close()

    return residual

# get plots of percent residual vs. radius
def make_percent_residual_plot(radius, data, model, iteration):

    percent_residual = (100.0 *(model - data)/ data)

    percent_residual = np.nan_to_num(percent_residual, nan=0.0, posinf=0.0, neginf=0.0)

    np.savetxt(f"percent_residual_iter{iteration}.txt", np.column_stack([radius,percent_residual]), header="radius_AU percent_residual")

    plt.figure()

    plt.plot(radius, percent_residual)

    plt.axhline(10, linestyle='--')

    plt.axhline(-10, linestyle='--')

    plt.axhline(0, color='k')

    plt.xlabel("Radius (AU)")
    plt.ylabel("Percent residual")
    #plt.xscale('log')

    plt.savefig(f"percent_residual_iter{iteration}.png")

    plt.close()

    return percent_residual

# get plots of percent change between model iterations
def make_percent_change_plot(radius, previous_model, current_model, iteration):

    percent_change = (100.0 *(current_model - previous_model)/ previous_model)

    percent_change = np.nan_to_num(percent_change, nan=0.0, posinf=0.0, neginf=0.0)

    np.savetxt(f"percent_change_iter{iteration}.txt", np.column_stack([radius,percent_change]), header="radius_AU percent_change")

    plt.figure()

    plt.plot(radius, percent_change)

    plt.axhline(5, linestyle='--')

    plt.axhline(-5, linestyle='--')

    plt.axhline(0, color='k')

    plt.xlabel("Radius (AU)")
    plt.ylabel("Percent change from previous model")

    #plt.xscale('log')

    plt.savefig(f"percent_change_iter{iteration}.png")

    plt.close()

    return percent_change

# beam-aware reduced chi-sq: calculate chi-sq along major axis, sampled at beam FWHM
def beam_reduced_chi2_1d(residual, rms, beam_fwhm_arcsec, pix_scale, n_params=1, r_in=0.0, r_out=250.0, distance=None):
    '''
    residual = I_model - I_data
    rms = image rms
    beam_fwhm_arcsec = beam FWHM in arcsec (sqrt(BMAJ*BMIN))
    pix_scale = pixel size in arcsec
    n_params = number of parameters
    '''
    residual = residual[np.isfinite(residual)]

    if len(residual) == 0:
        return np.nan, np.nan, np.nan
    
    # length of major axis in arcsec
    L_arcsec = (r_out-r_in)/distance
    # number of independent samples
    N_eff = L_arcsec / beam_fwhm_arcsec

    rms_eff = rms/np.sqrt(2) # data pos and neg sides of profile are averaged, reducing effective rms
    chi2 = np.sum((residual / rms_eff)**2)
    pix_per_beam_1d = beam_fwhm_arcsec / pix_scale
    N_pix = len(residual)
    N_eff = N_pix / max(pix_per_beam_1d, 1.0)
    # degrees of freedom
    dof = max(N_eff - n_params, 1)
    chi2_red = chi2 / dof
    return chi2_red, chi2, N_eff

# Function: initial Sigma and mass (inclination-corrected)
def initial_sigma_and_mass(parafile=None, fitsfile=None, distance=None, rmax=None,
                           pa=0.0, T_0=150.0, inclination_deg=0.0):
    """
    Compute initial surface density profile and disk mass using optically thin assumption,
    corrected for disk inclination.
    """
    # Get mcfost radial grid
    r_mcfost = get_radii(parafile)  # in au

    # Initial disk temperature profile
    T_slope = -0.5
    T_profile = (T_0 * (r_mcfost/10)**T_slope) * u.K

    # Load data from fits file
    fh = fits.open(fitsfile)
    header = fh[0].header
    freq = header['CRVAL3'] * u.Hz

    # Get kappa at wavelength
    kappa_at_slambda = get_kappa(parafile, fitsfile)

    # Interpolate brightness profile onto mcfost grid
    Cut_data_interp_Jyarcsec = I_profile_jyarcsec(parafile=parafile,
                                                  fitsfile=fitsfile,
                                                  distance=distance,
                                                  rmax=rmax,
                                                  pa=pa)
    Cut_data_interp_cgs = Cut_data_interp_Jyarcsec.to(u.erg / (u.s * u.cm**2 * u.Hz * u.sr))

    # Planck function
    bb = models.BlackBody(temperature=T_profile)
    Bnu = bb(freq)

    # Inclination correction
    cosi = np.cos(np.radians(inclination_deg))

    # Initial surface density
    Sigma_initial = (Cut_data_interp_cgs / (Bnu * kappa_at_slambda)) * cosi
    Sigma_initial = Sigma_initial.to(u.g / u.cm**2)

    # Disk area elements
    r_cm = (r_mcfost * u.au).to(u.cm)
    dr = np.gradient(r_cm)
    dA = 2 * np.pi * r_cm * dr

    # Initial disk mass
    M_initial = np.sum(Sigma_initial * dA)
    M_initial = M_initial.to(u.Msun)

    return Sigma_initial, M_initial

def fitsigma(parafile=None, fitsfile=None, distance=None, rmax=None,
             inc=0.0, pa=0.0, niter=40, r_in=0.0, r_out=250.0, data_sigma=None, restart=False, sat_damping=True):
    """
    Iteratively fit the surface density profile and total dust mass of a disk
    to reproduce the major-axis intensity profile and integrated flux, 
    accounting for optical depth, scattering, and inclination.

    inc: inclination in degrees (0 = face on, 90 = edge on)

    pa: position angle of disk in degrees E of N

    rmax: maximum radius to perform fit out to

    r_in and r_out: radii in au between which residuals and chi^2 will be calculated

    data_sigma: rms of data in Jy/beam

    sat_damping: damp corrections to surface density profile by e^(-tau)
    """
    dir = os.getcwd()
    para_file_prefix = parafile.split('1.para')[0]
    print(para_file_prefix)

    # Read image header and beam info 
    fh = fits.open(fitsfile)
    header = fh[0].header
    bmaj = header['BMAJ'] * 3600.  # arcsec
    bmin = header['BMIN'] * 3600.  # arcsec
    beam_area = np.pi/(4.*np.log(2.))*bmaj*bmin
    pix_scale = header['CDELT2'] * 3600.
    pix_scale_AU = pix_scale * distance

    freq = header['CRVAL3']
    wl = constants.c/freq
    slambda = str(wl*1e6)

    # convert rms to Jy/arcsec^2
    data_sigma = data_sigma/beam_area

    # Extract major-axis profile 
    image = np.nan_to_num(fh[0].data[0,0], nan=0.0)
    image = rotate(image, pa+90.0)
    center_y = 267
    center_x = 269
    xmax = int((rmax/distance)/pix_scale)
    x_profile = np.linspace(1, xmax, xmax)
    x_profile_au = x_profile * pix_scale_AU

    profile_pos = np.array([image[center_y][i] for i in range(center_x+1, center_x+xmax+1)])
    profile_neg = np.array([image[center_y][i] for i in range(center_x-xmax, center_x)])
    profile = np.mean(np.array([profile_pos, np.flip(profile_neg)]), axis=0)

    Cut = profile / beam_area  # Jy/arcsec^2
    x_profile_arcsec = x_profile * pix_scale
    Integ_data = np.trapz(Cut * x_profile_arcsec, x_profile_arcsec)

    # Disk radial grid for MCFOST
    pymcfost.run(parafile, options='-disk_struct')
    r_mcfost = pymcfost.Disc(dir+"/data_disk").r()[0][0]

    # Interpolate data onto MCFOST radii
    spl_data = scipy.interpolate.CubicSpline(x_profile_au, Cut)
    Cut_data_interp = spl_data(r_mcfost) * u.Jy / (u.arcsec**2)
    Cut_data_interp_cgs = Cut_data_interp.to(u.erg / (u.s * u.cm**2 * u.Hz * u.sr))

    # Initial Sigma and mass (assume opt. thin, correct for inclination)
    Sigma_initial, M_initial = initial_sigma_and_mass(
        parafile=parafile,
        fitsfile=fitsfile,
        distance=distance,
        rmax=rmax,
        pa=pa,
        T_0=150.0,
        inclination_deg=inc
    )

    Sigma = Sigma_initial.copy()
    logSigma = np.log(Sigma.value)
    dust_masses = [M_initial.value]
    dust_mass_correction_factors = [1.0]
    chi2_history = []

    iteration_numbers = []

    chi2_values = []
    chi2red_values = []

    avg_abs_percent_residuals = []
    avg_abs_residuals = []

    start_iter = 1

    if restart:
        # find last surface density file
        
        sigma_files = sorted(glob.glob("surface_density*.fits.gz"),key=lambda x: int(x.split("surface_density")[1].split(".fits")[0]))

        if len(sigma_files)==0:
            print("No previous surface density files found.")
            return
        
        last_sigma=sigma_files[-1]

        last_iter = int(last_sigma.split("surface_density")[1].split(".fits")[0])

        Sigma = (fits.getdata(last_sigma)* u.g / u.cm**2)

        logSigma = np.log(Sigma.value)

        start_iter = last_iter

        print(f"Loaded {last_sigma}")
        print(f"Restarting at iteration {start_iter}")

        if os.path.exists("dust_masses.txt"):

            dust_masses = list(np.loadtxt("dust_masses.txt"))
            current_mass = dust_masses[-1]
        else:
            print("dust_masses.txt not found")
            return

        if os.path.exists("dust_mass_correction_factors.txt"):
            dust_mass_correction_factors = list(np.loadtxt("dust_mass_correction_factors.txt"))   
        else:
            print("dust_mass_correction_factors.txt not found")
            return 

        if os.path.exists("chi2_history.txt"):

            chi_data = np.loadtxt("chi2_history.txt")

            if chi_data.ndim == 1:
                chi_data = chi_data.reshape(1, -1)

            iteration_numbers = chi_data[:,0].astype(int).tolist()
            chi2_values = chi_data[:,1].tolist()
            chi2red_values = chi_data[:,2].tolist()
            chi2_history = chi2red_values.copy()

        else:

            print("chi2_history.txt not found")

            iteration_numbers = []
            chi2_values = []
            chi2red_values = []

        if os.path.exists("residual_metrics.txt"):

            resid_data = np.loadtxt("residual_metrics.txt")

            if resid_data.ndim == 1:
                resid_data = resid_data.reshape(1, -1)

            avg_abs_percent_residuals = resid_data[:,1].tolist()
            avg_abs_residuals = resid_data[:,2].tolist()

        else:

            print("residual_metrics.txt not found")

            avg_abs_percent_residuals = []
            avg_abs_residuals = []
        print(f"Loaded {len(iteration_numbers)} previous iterations")
        print(f"Restart will begin with iteration {start_iter}")
        print(f"Using sigma file: surface_density{start_iter}.fits.gz")
        print(f"Using para file: {para_file_prefix}{start_iter}.para")
        
        
    else:

        start_iter = 1

        Sigma = Sigma_initial.copy()
        current_mass = M_initial.value

        base_para = pymcfost.parameters.Params(parafile)
        base_para.zones[0].dust_mass = current_mass
        base_para.writeto(parafile)

        # create initial surface density profile
        fits.writeto("surface_density1.fits.gz", Sigma.value, overwrite=True)

        print("Starting new run")

    if restart:
        final_iter = start_iter + niter - 1
    else:
        final_iter = niter

    for i in range(start_iter,final_iter+1):
        
        # Initial Sigma and mass
        print(f"\nIteration {i}")

        sigma_filename = f"surface_density{i}.fits.gz"

        print(f"Using {sigma_filename}")

        para_file = para_file_prefix + str(i) + ".para"

        print(f"Using {para_file}")

        para_obj = pymcfost.parameters.Params(para_file)

        print(f"Mass in para file = "f"{para_obj.zones[0].dust_mass:.3e} Msun")

        # Run MCFOST 
        if os.path.exists(f'rm -r data_th_iter{i}'):
            os.remove(f'rm -r data_th_iter{i}')
        if os.path.exists(f'data_{slambda}_iter{i}'):
            os.remove(f'data_{slambda}_iter{i}')
        pymcfost.run(para_file, options=f'-sigma {sigma_filename}', logfile=f'mcfost_iter{i}_th.log')
        pymcfost.run(para_file, options=f'-sigma {sigma_filename} -img {slambda}', logfile=f'mcfost_iter{i}_img.log')

        # Load model image and major-axis cut 
        mydisk = pymcfost.Image(f"data_{slambda}")
        mydisk.plot(type="I", bmaj=bmaj, bmin=bmin, bpa=header['BPA'], Jy=True, per_arcsec2 = True)
        plt.close()
        model_pixsize_arcsec = mydisk.pixelscale
        model_pixsize_au = model_pixsize_arcsec*distance
        pix_area = model_pixsize_arcsec**2
        model_Jy = mydisk.last_image
        model_center_x = int(1 + mydisk.last_image.shape[1] / 2)
        model_center_y = int(1 + mydisk.last_image.shape[0] / 2)

        xmax_model = int((rmax / distance) / mydisk.pixelscale)
        x_model_profile_au = np.linspace(1, xmax_model, xmax_model) * mydisk.pixelscale * distance
        Cut_model = [model_Jy[model_center_y][x] for x in range(model_center_x+1, model_center_x+xmax_model+1)]
        spl_model = scipy.interpolate.CubicSpline(x_model_profile_au, Cut_model)
        Cut_model_interp = spl_model(r_mcfost) * u.Jy / (u.arcsec**2)
        Cut_model_interp_cgs = Cut_model_interp.to(u.erg / (u.s * u.cm**2 * u.Hz * u.sr))

        Cut_model_interp_value = Cut_model_interp.value
        Cut_data_interp_value = Cut_data_interp.value

        x_model_profile_arcsec = (np.linspace(1, xmax_model, xmax_model) * mydisk.pixelscale)
        
        Integ_model = np.trapz(np.array(Cut_model) * x_model_profile_arcsec,x_model_profile_arcsec)

        flux_ratio = Integ_model / Integ_data

        make_residual_plot(r_mcfost, Cut_data_interp_value, Cut_model_interp_value, i, sigma=data_sigma)

        make_percent_residual_plot(r_mcfost, Cut_data_interp_value, Cut_model_interp_value, i)

        residual = Cut_data_interp.value - Cut_model_interp.value

        mask = ((r_mcfost >= r_in) & (r_mcfost <= r_out))

        percent_residual = (100.0 * (Cut_model_interp_value - Cut_data_interp_value) / Cut_data_interp_value)
        percent_residual = np.nan_to_num(percent_residual, nan=0.0, posinf=0.0, neginf=0.0)
        avg_percent = np.mean(np.abs(percent_residual[mask]))
        avg_abs_percent_residuals.append(avg_percent)

        abs_residual = np.abs(Cut_model_interp_value - Cut_data_interp_value)
        avg_abs = np.mean(abs_residual[mask])
        avg_abs_residuals.append(avg_abs)

        chi2_red, chi2, N_eff = beam_reduced_chi2_1d(
            residual=residual,
            rms=data_sigma,
            beam_fwhm_arcsec=np.sqrt(bmaj * bmin),
            pix_scale=pix_scale,
            r_in=r_in,
            r_out=r_out,
            distance=distance
        )

        iteration_numbers.append(i)

        chi2_values.append(chi2)
        chi2red_values.append(chi2_red)

        np.savetxt(
            "chi2_history.txt",
            np.column_stack([
                iteration_numbers,
                chi2_values,
                chi2red_values
            ]),
            header="iteration chi2 chi2_red"
        )

        np.savetxt(
            "residual_metrics.txt",
            np.column_stack([
                iteration_numbers,
                avg_abs_percent_residuals,
                avg_abs_residuals
            ]),
            header="iteration avg_abs_percent_residual avg_abs_residual"
        )

        with open("chisq.txt", "w") as f:
            f.write("# iteration chi2 chi2_red\n")

            for it, c, c_red in zip(iteration_numbers, chi2_values, chi2red_values):
                f.write(f"{it:d} "f"{c:.6e} "f"{c_red:.6e}\n")

        chi2_history.append(chi2_red)

        print(f"chi_sq_red = {chi2_red:.3f}")

        # Compute tau for data and model
        # Use midplane temperature from mcfost to get temperature profile
        T_file = './data_th/Temperature.fits.gz'
        T_data = fits.open(T_file)[0].data
        T_midplane = T_data[0, :] * u.K
        bb = models.BlackBody(temperature=T_midplane)
        Bnu = bb(freq)
        
        # Convert data and model cuts to cgs
        ratio_data = Cut_data_interp_cgs.value / Bnu.value
        ratio_data = np.clip(ratio_data, 0, 0.99)
        ratio_model = Cut_model_interp_cgs.value / Bnu.value
        ratio_model = np.clip(ratio_model, 0, 0.99)
        tau_data  = -np.log(1 - ratio_data)
        tau_model = -np.log(1 - ratio_model)
        
        # saturation factor: decrease when emission gets optically thick to avoid large changes to Sigma
        sens = np.exp(-tau_model)
        
        if sat_damping:
            delta_logSigma = np.log(tau_data /np.maximum(tau_model, 1e-6)) * sens
        
        else:
            delta_logSigma = np.log(tau_data /np.maximum(tau_model, 1e-6))
        
        if flux_ratio < 0.2:
            clip_lo = np.log(0.5)
            clip_hi = np.log(1.5)

        else:
            clip_lo = np.log(0.8)
            clip_hi = np.log(1.2)

        delta_logSigma = np.clip(delta_logSigma, clip_lo, clip_hi)
        
        delta_logSigma = np.clip(delta_logSigma, np.log(0.8), np.log(1.2))

        Sigma *= np.exp(delta_logSigma)
        
        # next iteration input file
        next_sigma_file = f"surface_density{i+1}.fits.gz"

        fits.writeto(next_sigma_file, Sigma.value, overwrite=True)

        print(f"Wrote {next_sigma_file}")
        r_cm = (r_mcfost * u.au).to(u.cm)
        dr = np.gradient(r_cm)
        dA = 2 * np.pi * r_cm * dr

        M_corr = np.sum(Sigma * dA)
        M_corr = M_corr.to(u.Msun).value
        '''
        if len(dust_mass_correction_factors) > 0 and dust_mass_correction_factors[-1] == 1.0:
            correction_dust_mass = 1.0
            print("Previous iteration converged; freezing future dust mass corrections.")
        '''
        # Update dust mass for next iteration
        dust_mass_corrected = M_corr
        current_mass = dust_mass_corrected

        if len(dust_masses) > 0:
            mass_corr_factor = dust_mass_corrected / dust_masses[-1]
        else:
            mass_corr_factor = 1.0

        dust_mass_correction_factors.append(mass_corr_factor)

        dust_masses.append(dust_mass_corrected)
        
        print(f"Iteration {i}: Corrected mass={dust_mass_corrected:.3e} Msun")

        # Save new para file for next iteration 
        next_para_file = (para_file_prefix + str(i+1) + ".para")
        para_obj = pymcfost.parameters.Params(para_file)
        para_obj.zones[0].dust_mass = dust_mass_corrected
        para_obj.writeto(next_para_file)
        print(f"Wrote {next_para_file} "f"with mass {dust_mass_corrected:.3e} Msun")

        # Diagnostic plots
        plt.plot(r_mcfost, Cut_data_interp, label='data')
        plt.plot(r_mcfost, Cut_model_interp, label=f'model iter {i}')
        plt.xlabel('r (au)')
        plt.ylabel('I (Jy/arcsec²)')
        plt.xscale('log')
        plt.yscale('log')
        plt.legend()
        plt.savefig(f'model_v_data_iteration{i}.png')
        plt.close()

        plt.plot(r_mcfost, np.exp(delta_logSigma))
        plt.xlabel('r (au)')
        plt.ylabel('Sigma correction factor')
        plt.xscale('log')
        plt.yscale('log')
        plt.axhline(y=1.0, color='k', linestyle='--')
        plt.savefig(f'correction_factor_iter{i}.png')
        plt.close()    

        os.rename('data_th', 'data_th_iter'+str(i))
        os.rename('data_'+slambda, 'data_'+slambda+'_iter'+str(i))

        np.savetxt('dust_mass_correction_factors.txt', dust_mass_correction_factors)
        plt.plot(dust_mass_correction_factors)
        plt.xlabel('iteration')
        plt.ylabel('dust mass correction factor')
        plt.savefig('dust_mass_correction_factors.png')
        plt.close()

        np.savetxt('dust_masses.txt', dust_masses)
        plt.plot(dust_masses)
        plt.xlabel('iteration')
        plt.ylabel('dust mass (Msun)')
        plt.savefig('dust_masses.png')
        plt.close()

        # convergence check
        if len(chi2_history) > 2:
            if abs(chi2_history[-1] - chi2_history[-2]) < 1e-2:
                print("Converged (chi sq stabilised)")
                break
                
        
    best_chi_iter = iteration_numbers[np.argmin(chi2red_values)]
    best_percent_iter = iteration_numbers[np.argmin(avg_abs_percent_residuals)]
    best_abs_iter = iteration_numbers[np.argmin(avg_abs_residuals)]

    with open("summary.txt","w") as f:
        f.write(f"Number of iterations: "f"{len(chi2red_values)}\n\n")
        f.write(f"Minimum reduced chi^2: "f"{np.min(chi2red_values):.6e}\n")
        f.write(f"Iteration: "f"{best_chi_iter}\n\n")
        f.write(f"Minimum average percent residual: "f"{np.min(avg_abs_percent_residuals):.3f}%\n")
        f.write(f"Iteration: "f"{best_percent_iter}\n\n")
        f.write(f"Minimum average absolute residual: "f"{np.min(avg_abs_residuals):.6e}\n")
        f.write(f"Iteration: "f"{best_abs_iter}\n")

    # save files for best fit iter
    best_iter = best_percent_iter
    cwd = os.path.basename(os.getcwd())
    best_dir = (f"{cwd}_iter{best_iter}_best_fit")
    os.makedirs(best_dir, exist_ok=True)
    best_sigma_file = (f"surface_density{best_iter}.fits.gz")

    if os.path.exists(best_sigma_file):
        shutil.copy2(best_sigma_file, best_dir)
    
    best_para_file = (para_file_prefix + str(best_iter) + ".para")

    if os.path.exists(best_para_file):
        shutil.copy2(best_para_file, best_dir)

    for dirname in [f"data_th_iter{best_iter}", f"data_{slambda}_iter{best_iter}"]:
        if os.path.exists(dirname):
            shutil.copytree(dirname, os.path.join(best_dir, os.path.basename(dirname)), dirs_exist_ok=True)
    
    patterns = [
        f"*iter{best_iter}*.txt",
        f"*iter{best_iter}*.png",
        f"*iteration{best_iter}*.png",
        f"*iteration{best_iter}*.txt"
    ]

    for pattern in patterns:
        for file in glob.glob(pattern):
            if os.path.isfile(file):
                shutil.copy2(file, best_dir)
    
    for file in [
        "dust_masses.txt",
        "dust_mass_correction_factors.txt",
        "chisq.txt",
        "summary.txt"
    ]:
        if os.path.exists(file):
            shutil.copy2(file, best_dir)

    print(f"Copied best fit model into {cwd}_iter{best_iter}_best_fit")
    
parafile = glob.glob('HD97048_*_1.para')[0]
fitsfile = '../HD97048_LB_contap2_robust-2_circbeam_new.fits'
distance = 184.4 
rmax = 350 
inc = 40.9
pa = 4.11

kappa = get_kappa(parafile, fitsfile)
print(f'kappa: {kappa}')

fitsigma(parafile = parafile, fitsfile =fitsfile, distance = distance, rmax = rmax, inc = inc, pa = pa, niter=40, r_in=5.0, r_out=250.0, data_sigma=9.6e-05, restart=False, sat_damping=True)

