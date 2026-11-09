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
import argparse
from pathlib import Path
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

def rotate_image(fitsfile, pa):
    """
    Rotate the image in the FITS file so that the major axis is aligned with the x-axis.
    """
    fh = fits.open(fitsfile)
    image = fh[0].data[0, 0]
    # Convert NaN to zero
    image = np.nan_to_num(image, nan=0.0)
    # Rotate image clockwise by (pa + 90) degrees
    rotated_image = rotate(image, pa + 90.0)
    return rotated_image

def I_profile_jyarcsec(parafile = None, fitsfile = None, distance = None, rmax = None, pa = 0.0, center_x = 269, center_y = 267):
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

    image = rotate_image(fitsfile, pa)

    # get cut along major axis
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


# --------------------------
# Function: initial Sigma and mass (inclination-corrected)
# --------------------------
def initial_sigma_and_mass(parafile=None, fitsfile=None, distance=None, rmax=None,
                           pa=0.0, T_0=150.0, inclination_deg=0.0, center_x=269, center_y=267):
    """
    Compute initial surface density profile and disk mass using optically thin assumption,
    corrected for disk inclination.
    """
    # --- Get radial grid ---
    r_mcfost = get_radii(parafile)  # in au

    # --- Disk temperature profile ---
    T_slope = -0.5
    T_profile = (T_0 * (r_mcfost/10)**T_slope) * u.K

    # --- Load image and frequency ---
    fh = fits.open(fitsfile)
    header = fh[0].header
    freq = header['CRVAL3'] * u.Hz

    # --- Opacity at observing wavelength ---
    kappa_at_slambda = get_kappa(parafile, fitsfile)

    # --- Interpolated brightness profile ---
    Cut_data_interp_Jyarcsec = I_profile_jyarcsec(parafile=parafile,
                                                  fitsfile=fitsfile,
                                                  distance=distance,
                                                  rmax=rmax,
                                                  pa=pa,
                                                  center_x=center_x,
                                                  center_y=center_y)
    Cut_data_interp_cgs = Cut_data_interp_Jyarcsec.to(u.erg / (u.s * u.cm**2 * u.Hz * u.sr))

    # --- Planck function ---
    bb = models.BlackBody(temperature=T_profile)
    Bnu = bb(freq)

    # --- Inclination correction ---
    cosi = np.cos(np.radians(inclination_deg))

    # --- Initial surface density ---
    Sigma_initial = (Cut_data_interp_cgs / (Bnu * kappa_at_slambda)) * cosi
    Sigma_initial = Sigma_initial.to(u.g / u.cm**2)

    # --- Disk area elements ---
    r_cm = (r_mcfost * u.au).to(u.cm)
    dr = np.gradient(r_cm)
    dA = 2 * np.pi * r_cm * dr

    # --- Initial disk mass ---
    M_initial = np.sum(Sigma_initial * dA)
    M_initial = M_initial.to(u.Msun)

    return Sigma_initial, M_initial

def fitsigma(args):
    """
    Iteratively fit the surface density profile and total dust mass of a disk
    to reproduce the major-axis intensity profile and integrated flux, 
    accounting for optical depth, scattering, and inclination.
    """
    dir = os.getcwd()
    para_file_prefix = args.parafile.split('1.para')[0]
    print(para_file_prefix)

    # Read image header and beam info 
    fh = fits.open(args.fitsfile)
    header = fh[0].header
    bmaj = header['BMAJ'] * 3600.  # arcsec
    bmin = header['BMIN'] * 3600.  # arcsec
    beam_area = np.pi/(4.*np.log(2.))*bmaj*bmin
    pix_scale = header['CDELT2'] * 3600.
    pix_scale_AU = pix_scale * args.distance

    freq = header['CRVAL3']
    wl = constants.c/freq
    slambda = str(wl*1e6)

    # convert rms to Jy/arcsec^2
    r_mcfost = get_radii(args.parafile)
    Cut_data_interp = I_profile_jyarcsec(
        parafile=args.parafile,
        fitsfile=args.fitsfile,
        distance=args.distance,
        rmax=args.rmax,
        pa=args.pa,
        center_x=args.center_x,
        center_y=args.center_y
    )

    Cut_data_interp_cgs = Cut_data_interp.to(
        u.erg / (u.s * u.cm**2 * u.Hz * u.sr)
    )

    r_arcsec = r_mcfost / args.distance

    #Integ_data = np.trapz(Cut_data_interp.value * x_profile_arcsec, x_profile_arcsec)
    Integ_data = np.trapz(
        Cut_data_interp.value * r_arcsec,
        r_arcsec
    )

    # Disk radial grid for MCFOST
    

    # Initial Sigma and mass (assume opt. thin, correct for inclination)
    Sigma_initial, M_initial = initial_sigma_and_mass(
        parafile=args.parafile,
        fitsfile=args.fitsfile,
        distance=args.distance,
        rmax=args.rmax,
        pa=args.pa,
        T_0=150.0,
        inclination_deg=args.inc,
        center_x=args.center_x,
        center_y=args.center_y
    )

    Sigma = Sigma_initial.copy()
    logSigma = np.log(Sigma.value)
    dust_masses=[]
    dust_mass_correction_factors=[]
    chi2_history = []

    chi2_values = []
    chi2red_values = []

    avg_abs_percent_residuals = []
    avg_abs_residuals = []

    start_iter = 1

    
    if args.restart:

        import re

        # Find completed-iteration checkpoints
        checkpoint_files = glob.glob("fit_state_iter*.npz")

        checkpoints = []

        for filename in checkpoint_files:
            match = re.fullmatch(
                r"fit_state_iter(\d+)\.npz",
                os.path.basename(filename)
            )

            if match:
                checkpoints.append((int(match.group(1)), filename))

        checkpoints.sort(reverse=True)

        if not checkpoints:
            raise FileNotFoundError(
                "No restart checkpoints found. "
                "A previous run must save fit_state_iter*.npz."
            )

        # Select the newest checkpoint with valid next-model inputs
        selected = None

        for completed_iter, checkpoint_file in checkpoints:

            candidate_iter = completed_iter + 1
            candidate_sigma = f"surface_density{candidate_iter}.fits.gz"
            candidate_para = (
                para_file_prefix + str(candidate_iter) + ".para"
            )

            if (
                os.path.isfile(candidate_sigma)
                and os.path.isfile(candidate_para)
            ):
                selected = (
                    completed_iter,
                    checkpoint_file,
                    candidate_iter,
                    candidate_sigma,
                    candidate_para
                )
                break

        if selected is None:
            raise FileNotFoundError(
                "No checkpoint has matching surface-density "
                "and MCFOST parameter files."
            )

        (
            completed_iter,
            checkpoint_file,
            start_iter,
            sigma_filename,
            para_filename
        ) = selected

        with np.load(checkpoint_file, allow_pickle=False) as state:

            if int(state["completed_iteration"]) != completed_iter:
                raise ValueError("Checkpoint iteration mismatch.")

            if int(state["next_iteration"]) != start_iter:
                raise ValueError("Checkpoint next iteration mismatch.")

            Sigma = fits.getdata(sigma_filename) * u.g / u.cm**2

            previous_model_cut = state["previous_model_cut"].copy()

            previous_model_sigma = (
                state["previous_model_sigma"].copy()
                * u.g / u.cm**2
            )

            previous_sensitivity = (
                state["previous_sensitivity"].copy()
            )

            low_sensitivity_count = (
                state["low_sensitivity_count"].copy()
            )

            saturated = state["saturated"].copy()

            dust_masses = state["dust_masses"].tolist()

            dust_mass_correction_factors = (
                state["dust_mass_correction_factors"].tolist()
            )

            chi2_values = state["chi2_values"].tolist()
            chi2red_values = state["chi2red_values"].tolist()

            avg_abs_percent_residuals = (
                state["avg_abs_percent_residuals"].tolist()
            )

            avg_abs_residuals = (
                state["avg_abs_residuals"].tolist()
            )

        # Restore the convergence history
        chi2_history = chi2red_values.copy()

        # Validate checkpoint dimensions
        for name, arr in [
            ("Sigma", Sigma.value),
            ("previous_model_cut", previous_model_cut),
            ("previous_model_sigma", previous_model_sigma.value),
            ("previous_sensitivity", previous_sensitivity),
            ("low_sensitivity_count", low_sensitivity_count),
            ("saturated", saturated)
        ]:
            if arr.shape != r_mcfost.shape:
                raise ValueError(
                    f"{name} shape {arr.shape} does not match "
                    f"MCFOST grid shape {r_mcfost.shape}"
                )

        current_mass = (
            pymcfost.parameters.Params(
                para_filename
            ).zones[0].dust_mass
        )

        print(f"Loaded checkpoint: {checkpoint_file}")
        print(f"Loaded surface density: {sigma_filename}")
        print(f"Loaded parameters: {para_filename}")
        print(f"Restarting at iteration {start_iter}")

    else:

        start_iter = 1

        Sigma = Sigma_initial.copy()
        current_mass = M_initial.value

        base_para = pymcfost.parameters.Params(args.parafile)
        base_para.zones[0].dust_mass = current_mass
        base_para.writeto(args.parafile)

        fits.writeto(
            "surface_density1.fits.gz",
            Sigma.value,
            overwrite=True
        )

        previous_model_cut = None
        previous_model_sigma = None

        previous_sensitivity = np.ones_like(r_mcfost)

        low_sensitivity_count = np.zeros_like(
            r_mcfost, dtype=int
        )

        saturated = np.zeros_like(
            r_mcfost, dtype=bool
        )

        print("Starting new run")


    # Empirical sensitivity threshold for stopping Sigma increases 
    S_min = 0.03
    '''
    previous_model_cut = None
    previous_model_sigma = None

    # Start with optically thin sensitivity
    previous_sensitivity = np.ones_like(r_mcfost)

    # Number of consecutive iterations below S_min
    low_sensitivity_count = np.zeros_like(r_mcfost, dtype=int)

    # Once confirmed, prohibit further Sigma increases
    saturated = np.zeros_like(r_mcfost, dtype=bool)
    '''
    eps_I = 1e-12
    eps_sigma = 1e-12   

    for i in range(start_iter,args.niter+1):
        
        # Initial Sigma and mass
        print(f"\nIteration {i}")

        sigma_filename = f"surface_density{i}.fits.gz"
        # Sigma that produces the current model image
        Sigma_model = Sigma.copy()

        print(f"Using {sigma_filename}")

        para_file = para_file_prefix + str(i) + ".para"

        print(f"Using {para_file}")

        para_obj = pymcfost.parameters.Params(para_file)

        print(f"Mass in para file = "f"{para_obj.zones[0].dust_mass:.3e} Msun")

        # Run MCFOST 
        os.system(f'rm -r data_th_iter{i} data_{slambda}_iter{i}')
        pymcfost.run(para_file, options=f'-sigma {sigma_filename}', logfile=f'mcfost_iter{i}_th.log')
        pymcfost.run(para_file, options=f'-sigma {sigma_filename} -img {slambda}', logfile=f'mcfost_iter{i}_img.log')

        # Load model image and major-axis cut 
        mydisk = pymcfost.Image(f"data_{slambda}")
        mydisk.plot(type="I", bmaj=bmaj, bmin=bmin, bpa=header['BPA'], Jy=True, per_arcsec2 = True)
        plt.close()
        model_pixsize_arcsec = mydisk.pixelscale
        model_pixsize_au = model_pixsize_arcsec*args.distance
        pix_area = model_pixsize_arcsec**2
        model_Jy = mydisk.last_image
        model_center_x = (model_Jy.shape[1] - 1) // 2
        model_center_y = (model_Jy.shape[0] - 1) // 2

        xmax_model = int((args.rmax / args.distance) / mydisk.pixelscale)
        x_model_profile_au = np.linspace(1, xmax_model, xmax_model) * mydisk.pixelscale * args.distance
        Cut_model = [model_Jy[model_center_y][x] for x in range(model_center_x+1, model_center_x+xmax_model+1)]
        spl_model = scipy.interpolate.CubicSpline(x_model_profile_au, Cut_model)
        Cut_model_interp = spl_model(r_mcfost) * u.Jy / (u.arcsec**2)
        Cut_model_interp_cgs = Cut_model_interp.to(u.erg / (u.s * u.cm**2 * u.Hz * u.sr))

        Cut_model_interp_value = Cut_model_interp.value
        Cut_data_interp_value = Cut_data_interp.value

        x_model_profile_arcsec = (np.linspace(1, xmax_model, xmax_model) * mydisk.pixelscale)
        
        #Integ_model = np.trapz(np.array(Cut_model) * x_model_profile_arcsec,x_model_profile_arcsec)
        Integ_model = np.trapz(
            Cut_model_interp.value * r_arcsec,
            r_arcsec
        )

        flux_ratio = Integ_model / Integ_data

        make_residual_plot(r_mcfost, Cut_data_interp_value, Cut_model_interp_value, i, sigma=args.residual_sigma)

        make_percent_residual_plot(r_mcfost, Cut_data_interp_value, Cut_model_interp_value, i)
        

        residual = Cut_data_interp.value - Cut_model_interp.value

        mask = ((r_mcfost >= args.r_in) & (r_mcfost <= args.r_out))

        percent_residual = (100.0 * (Cut_model_interp_value - Cut_data_interp_value) / Cut_data_interp_value)
        percent_residual = np.nan_to_num(percent_residual, nan=0.0, posinf=0.0, neginf=0.0)
        avg_percent = np.mean(np.abs(percent_residual[mask]))
        avg_abs_percent_residuals.append(avg_percent)

        abs_residual = np.abs(Cut_model_interp_value - Cut_data_interp_value)
        avg_abs = np.mean(abs_residual[mask])
        avg_abs_residuals.append(avg_abs)

        chi2_red, chi2, N_eff = beam_reduced_chi2_1d(
            residual=residual,
            rms=args.residual_sigma,
            beam_fwhm_arcsec=np.sqrt(bmaj * bmin),
            pix_scale=abs(pix_scale),
            r_in=args.r_in,
            r_out=args.r_out,
            distance=args.distance
        )

        chi2_values.append(chi2)
        chi2red_values.append(chi2_red)

        with open("chisq.txt", "w") as f:
            f.write("# iteration chi2 chi2_red\n")

            for j,(c,c_red) in enumerate(zip(chi2_values,chi2red_values),start=1):
                f.write(f"{j:d} "f"{c:.6e} "f"{c_red:.6e}\n")

        chi2_history.append(chi2_red)

        print(f"chi_sq_red = {chi2_red:.3f}")
        
        # Empirical sensitivity of brightness to surface density measured from prev models

        I_model = Cut_model_interp_value
        I_data = Cut_data_interp_value

        # Calculate sensitivity from consecutive MCFOST models
        if previous_model_cut is None:

            # First iteration: assume optically thin response
            sensitivity = np.ones_like(r_mcfost)

        else:

            # Fractional change in model brightness
            dlnI = np.log(
                np.maximum(I_model, eps_I) /
                np.maximum(previous_model_cut, eps_I)
            )

            # Fractional change in surface density
            dlnSigma = np.log(
                np.maximum(Sigma_model.value, eps_sigma) /
                np.maximum(previous_model_sigma.value, eps_sigma)
            )

            # Calculate sensitivity where the previous update was nonzero
            valid = (
                np.isfinite(dlnI)
                & np.isfinite(dlnSigma)
                & (np.abs(dlnSigma) > 1e-8)
                & (I_model > 0)
                & (previous_model_cut > 0)
            )

            # Carry forward previous sensitivity where undefined
            sensitivity = previous_sensitivity.copy()

            sensitivity[valid] = (
                np.abs(dlnI[valid]) /
                np.abs(dlnSigma[valid])
            )

            signed_sensitivity = np.full_like(r_mcfost, np.nan)

            signed_sensitivity[valid] = (
                dlnI[valid] / dlnSigma[valid]
            )

            # Cap at 1 for damping purposes
            sensitivity = np.clip(sensitivity, 0.0, 1.0)


        # Identify radii with low sensitivity to surface density changes over 2 consecutive iterations. 

        # Only count iterations with a valid sensitivity measurement
        if previous_model_cut is not None:

            low_sensitivity = valid & (sensitivity < S_min)

            # Increment consecutive low-sensitivity count
            low_sensitivity_count[low_sensitivity] += 1
            low_sensitivity_count[~low_sensitivity] = 0

            # Reset count where sensitivity was reliably measured
            # and is above the threshold
            recovered = valid & (sensitivity >= S_min)
            low_sensitivity_count[recovered] = 0

            # Mark radii that have fallen below the threshold
            # for at least two consecutive valid measurements
            saturated |= (low_sensitivity_count >= 2)

        # Calculate brightness residual along major axis. 

        delta_logI = np.zeros_like(I_model)

        valid_intensity = (
            np.isfinite(I_data)
            & np.isfinite(I_model)
            & (I_data > 0)
            & (I_model > 0)
        )

        delta_logI[valid_intensity] = np.log(
            I_data[valid_intensity] /
            I_model[valid_intensity]
        )

        # Sensitivity-dependent Sigma correction
        delta_logSigma = sensitivity * delta_logI

        # Prevent increases in Sigma at saturated radii
        # but continue allowing decreases
        delta_logSigma[
            saturated & (delta_logSigma > 0)
        ] = 0.0

        # Limit maximum change per iteration
        # 50 pct if data flux is >5x model flux, 20 pct otherwise

        if flux_ratio < 0.2:
            clip_lo = np.log(0.5)
            clip_hi = np.log(1.5)
        else:
            clip_lo = np.log(0.8)
            clip_hi = np.log(1.2)

        delta_logSigma = np.clip(
            delta_logSigma,
            clip_lo,
            clip_hi
        )


        # Apply the surface density correction
        Sigma *= np.exp(delta_logSigma)


        # Save txt file and plot of sensitivity vs. radius, with saturated radii marked
        np.savetxt(
            f"sensitivity_iter{i}.txt",
            np.column_stack([
                r_mcfost,
                sensitivity,
                low_sensitivity_count,
                saturated.astype(int),
                delta_logI,
                delta_logSigma
            ]),
            header=(
                "radius_AU sensitivity "
                "low_sensitivity_count saturated "
                "delta_logI delta_logSigma"
            )
        )

        fig, ax = plt.subplots(figsize=(8, 5))

        ax.plot(r_mcfost, sensitivity, color="black",
                label="Empirical sensitivity")

        ax.axhline(S_min, color="red", linestyle="--",
                label=f"S_min = {S_min}")

        ax.scatter(
            r_mcfost[saturated],
            sensitivity[saturated],
            color="red",
            s=15,
            label="Saturated"
        )

        ax.set_xlabel("Radius (AU)")
        ax.set_ylabel(r"$|d\ln I/d\ln\Sigma|$")
        ax.set_xscale("log")
        ax.set_ylim(-0.05, 1.05)
        ax.legend()

        fig.savefig(f"sensitivity_iter{i}.png")
        plt.close(fig)


        # Save current model state and sensitivity for next iteration
        previous_model_cut = I_model.copy()
        previous_model_sigma = Sigma_model.copy()
        previous_sensitivity = sensitivity.copy()

        print(
            f"Iteration {i}: "
            f"{np.sum(saturated)} / {len(saturated)} "
            "radial bins saturation-limited"
        )


        # next iteration input file
        next_sigma_file = f"surface_density{i+1}.fits.gz"

        fits.writeto(next_sigma_file, Sigma.value, overwrite=True)

        print(f"Wrote {next_sigma_file}")
        r_cm = (r_mcfost * u.au).to(u.cm)
        dr = np.gradient(r_cm)
        dA = 2 * np.pi * r_cm * dr

        M_corr = np.sum(Sigma * dA)
        M_corr = M_corr.to(u.Msun).value
        
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


        # Save restart state after preparing the next model
        checkpoint_file = f"fit_state_iter{i}.npz"

        np.savez(
            checkpoint_file,
            completed_iteration=i,
            next_iteration=i + 1,
            previous_model_cut=previous_model_cut,
            previous_model_sigma=previous_model_sigma.value,
            previous_sensitivity=previous_sensitivity,
            low_sensitivity_count=low_sensitivity_count,
            saturated=saturated,
            dust_masses=np.asarray(dust_masses),
            dust_mass_correction_factors=np.asarray(
                dust_mass_correction_factors
            ),
            chi2_values=np.asarray(chi2_values),
            chi2red_values=np.asarray(chi2red_values),
            avg_abs_percent_residuals=np.asarray(
                avg_abs_percent_residuals
            ),
            avg_abs_residuals=np.asarray(avg_abs_residuals)
        )

        print(f"Saved restart checkpoint: {checkpoint_file}")


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

        np.savetxt('dust_masses.txt', dust_masses)
        plt.plot(dust_masses)
        plt.xlabel('iteration')
        plt.ylabel('dust mass (Msun)')
        plt.savefig('dust_masses.png')

        # convergence check
        if len(chi2_history) > 2:
            if abs(chi2_history[-1] - chi2_history[-2]) < 1e-2:
                print("Converged (chi sq stabilised)")
                break
    
    best_chi_iter = (np.argmin(chi2red_values) + 1)
    best_percent_iter = (np.argmin(avg_abs_percent_residuals) + 1)
    best_abs_iter = (np.argmin(avg_abs_residuals) + 1)

    best_dir_path = Path(
    f"{para_file_prefix}{best_percent_iter}_best_fit"
    )
    best_dir_path.mkdir(exist_ok=True)

    best_para = Path(
        f"{para_file_prefix}{best_percent_iter}.para"
    )
    best_sigma = Path(
        f"surface_density{best_percent_iter}.fits.gz"
    )

    shutil.copy2(best_para, best_dir_path / best_para.name)
    shutil.copy2(best_sigma, best_dir_path / best_sigma.name)

    dat_files = [
        'ice_warren_brandt.dat',
        'henning_troilite.dat',
        'henning_organics.dat',
    ]

    for file_name in dat_files:
        file_path = Path(file_name)

        if file_path.is_file():
            shutil.copy2(file_path, best_dir_path / file_path.name)


    with open("summary.txt","w") as f:
        f.write(f"Number of iterations: "f"{len(chi2red_values)}\n\n")
        f.write(f"Minimum reduced chi^2: "f"{np.min(chi2red_values):.6e}\n")
        f.write(f"Iteration: "f"{best_chi_iter}\n\n")
        f.write(f"Minimum average percent residual: "f"{np.min(avg_abs_percent_residuals):.3f}%\n")
        f.write(f"Iteration: "f"{best_percent_iter}\n\n")
        f.write(f"Minimum average absolute residual: "f"{np.min(avg_abs_residuals):.6e}\n")
        f.write(f"Iteration: "f"{best_abs_iter}\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parafile', type=str, required=True, help='MCFOST parameter file')
    p.add_argument('--fitsfile', type=str, required=True, help='FITS image file')
    p.add_argument('--distance', type=float, required=True, help='Distance to the source in parsecs')
    p.add_argument('--rmax', type=float, required=True, help='Maximum radius for the profile in AU')
    p.add_argument('--inc', type=float, required=True, help='Inclination in degrees')
    p.add_argument('--pa', type=float, required=True, help='Position angle in degrees')
    p.add_argument('--center_x', type=int, default=269, help='X coordinate of the center of the image after rotation')
    p.add_argument('--center_y', type=int, default=267, help='Y coordinate of the center of the image after rotation')
    p.add_argument('--niter', type=int, default=40, help='Number of iterations for fitting')
    p.add_argument('--r_in', type=float, default=5.0, help='Inner radius for fitting in AU')
    p.add_argument('--r_out', type=float, default=250.0, help='Outer radius for fitting in AU')
    p.add_argument('--residual_sigma', type=float, default=9.6e-05, help='Residual sigma for chi-squared calculation')
    p.add_argument('--restart', action='store_true', help='Restart from the last checkpoint if available')
    args = p.parse_args()

    args.parafile = str(Path(args.parafile).resolve())
    args.fitsfile = str(Path(args.fitsfile).resolve())

    if not Path(args.parafile).is_file():
        p.error(f"Parameter file not found: {args.parafile}")

    if not Path(args.fitsfile).is_file():
        p.error(f"FITS file not found: {args.fitsfile}")

    fitsigma(args)

if __name__=='__main__': main()
'''
parafile = 'HD97048_DSHARP_fAC0.3_amax500_porosity0.8_1.para'
fitsfile = '/Users/rh128/Documents/HD97048/HD97048_LB_contap2_robust-2_circbeam_new.fits'
distance = 184.4 
rmax = 350 
inc = 40.9
pa = 4.11

# x and y coordinates of the center of the image after rotation
x_center_rotated = 269
y_center_rotated = 267

kappa = get_kappa(parafile, fitsfile)
print(f'kappa: {kappa}')

fitsigma(parafile = parafile, fitsfile =fitsfile, distance = distance, rmax = rmax, inc = inc, pa = pa, center_x = x_center_rotated, center_y= y_center_rotated, niter=40, r_in=5.0, r_out=250.0, residual_sigma=9.6e-05, restart=False)
'''
# Condition to stop increasing surface density profile: sensitivity falls below threshold for 2 iterations in a row
# Still allow surface density to decrease after sensitivity falls below threshold, but not increase