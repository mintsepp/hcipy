from hcipy import *
import numpy as np
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

def plot_reconstruction_phase(phase_in, phase_out, telescope_pupil):
    '''Plot the incoming aberrated phase pattern and the reconstructed phase pattern

    Parameters
    ---------
    phase_in : Field
        The phase of the aberrated wavefront coming in
    phase_out : Field
        The phase of the aberrated wavefront as reconstructed by the ZWFS
    '''

    # Calculating the difference of the reconstructed phase and input phase
    diff = phase_out - phase_in
    diff -= np.mean(diff[telescope_pupil >= 0.5])

    # Plotting the phase pattern and the PSF
    fig = plt.figure()
    ax1 = fig.add_subplot(131)
    im1 = imshow_field(phase_in, cmap='RdBu', vmin=-0.2, vmax=0.2, mask=telescope_pupil)
    ax1.set_title('Input phase')

    divider = make_axes_locatable(ax1)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im1, cax=cax, orientation='vertical')

    ax2 = fig.add_subplot(132)
    im2 = imshow_field(phase_out, cmap='RdBu', vmin=-0.2, vmax=0.2, mask=telescope_pupil)
    ax2.set_title('Reconstructed phase')

    divider = make_axes_locatable(ax2)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im2, cax=cax, orientation='vertical')

    ax3 = fig.add_subplot(133)
    im3 = imshow_field(diff, cmap='RdBu', vmin=-0.02, vmax=0.02, mask=telescope_pupil)
    ax3.set_title('Difference')

    divider = make_axes_locatable(ax3)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im3, cax=cax, orientation='vertical')
    plt.show()

def plot_reconstruction_amplitude(amplitude_in, amplitude_out, telescope_pupil):
    '''Plot the incoming aberrated amplitude pattern and the reconstructed amplitude pattern

    Parameters
    ---------
    amplitude_in : Field
        The phase of the aberrated wavefront coming in
    amplitude_out : Field
        The amplitude of the aberrated wavefront as reconstructed by the vZWFS
    '''

    amplitude_in = amplitude_in - 1
    amplitude_out = amplitude_out - 1

    # Plotting the phase pattern and the PSF
    fig = plt.figure()
    ax1 = fig.add_subplot(131)
    im1 = imshow_field(amplitude_in, cmap='gray', vmin=-0.05, vmax=0.05, mask=telescope_pupil)
    ax1.set_title('Input amplitude')

    divider = make_axes_locatable(ax1)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im1, cax=cax, orientation='vertical')

    ax2 = fig.add_subplot(132)
    im2 = imshow_field(amplitude_out, cmap='gray', vmin=-0.05, vmax=0.05, mask=telescope_pupil)
    ax2.set_title('Reconstructed amplitude')

    divider = make_axes_locatable(ax2)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im2, cax=cax, orientation='vertical')

    ax3 = fig.add_subplot(133)
    im3 = imshow_field(amplitude_out - amplitude_in, cmap='gray', vmin=-0.01, vmax=0.01, mask=telescope_pupil)
    ax3.set_title('Difference')

    divider = make_axes_locatable(ax3)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im3, cax=cax, orientation='vertical')
    plt.show()

# Simulation parameters
wavelength = 1550e-9

# Make pupil
receiver_diameter = 0.01
pupil_grid = make_pupil_grid(1820/4, receiver_diameter)
focal_grid = make_focal_grid_from_pupil_grid(pupil_grid)

# Make phase screens for atmosphere
outer_scale = 25
#layers = make_HV57_atmospheric_layers(pupil_grid, outer_scale)
#atmosphere = MultiLayerAtmosphere(layers, True)
one_layer = InfiniteAtmosphericLayer(pupil_grid, 1.6226e-12, 25, 8.7925, 0)
atmosphere = one_layer
prop = FraunhoferPropagator(pupil_grid, focal_grid.scaled(wavelength))

# Aperture and wavefront
aperture = make_circular_aperture(receiver_diameter)(pupil_grid)
wf = Wavefront(aperture, wavelength)
wf.total_power = 1
wf2 = atmosphere.forward(wf)

# Shack-Hartmann parameters
sh_f_length = 18.8e-3
sh_pitch = 0.3e-3
sh_diameter = 10e-3
f_number = sh_f_length / sh_pitch
num_lenslets = int(sh_diameter/sh_pitch/4)

# These don't really do anything at the moment since sh_diameter = receiver_diameter
magnification = sh_diameter / receiver_diameter
magnifier = Magnifier(magnification)

# Create SHWFS and estimator
shwfs = SquareShackHartmannWavefrontSensorOptics(pupil_grid.scaled(magnification), f_number, \
                                                 num_lenslets, sh_diameter)
shwfse = ShackHartmannWavefrontSensorEstimator(shwfs.mla_grid, shwfs.micro_lens_array.mla_index)

# Calibration reference image
camera = NoiselessDetector(focal_grid)
camera.integrate(shwfs(wf),1)
image_ref = camera.read_out()
slopes_calib = shwfse.estimate([image_ref])

# Remove points with less than 50% flux (like edges)
fluxes = ndimage.measurements.sum(image_ref, shwfse.mla_index, shwfse.estimation_subapertures)
flux_limit = fluxes.max() * 0.5
estimation_subapertures = shwfs.mla_grid.zeros(dtype='bool')
estimation_subapertures[shwfse.estimation_subapertures[fluxes > flux_limit]] = True

# Estimate without those low flux points
shwfse = ShackHartmannWavefrontSensorEstimator(shwfs.mla_grid, shwfs.micro_lens_array.mla_index, estimation_subapertures)

camera.integrate(shwfs(wf),1)
image_ref = camera.read_out()
slopes_calib = shwfse.estimate([image_ref])

camera.integrate(shwfs(wf2),1)
image_sens = camera.read_out()

# Plot SHWFS
imshow_field(image_sens, cmap='inferno')
plt.colorbar()
plt.show()

# Calculate sensed slopes
slopes_sens = shwfse.estimate([image_sens]) - slopes_calib

phases = wf2.phase

plt.subplot(1,2,1)
imshow_field(phases, cmap='RdBu')
plt.subplot(1,2,2)
im2 = imshow_field(image_sens)
plt.quiver(shwfs.mla_grid.subset(shwfse.estimation_subapertures).x,
           shwfs.mla_grid.subset(shwfse.estimation_subapertures).y,
           slopes_sens[0, :], slopes_sens[1, :],
           color='white')
#plt.subplot(1,3,3)
#imshow_field(shifts,grid=shwfse.estimation_grid, cmap='RdBu')

plt.draw()
plt.show()

wf2 = atmosphere.forward(wf)

# Reconstruction while atmosphere evolves?
'''
# Plot
for t in np.linspace(0, 10, 501):
    atmosphere.evolve_until(t)

    wf2 = atmosphere.forward(wf)
    wf2.electric_field *= aperture
    img = Field(prop(wf2).intensity, focal_grid)

    plt.clf()
    plt.subplot(1,3,1)
    imshow_field(wf2.phase, cmap='RdBu')
    plt.subplot(1,3,2)
    imshow_field(np.log10(img / img.max()), vmin=-6)
    plt.subplot(1,3,3)
    imshow_field(wf2.amplitude, cmap='viridis')
    plt.draw()
    plt.pause(0.001)
'''