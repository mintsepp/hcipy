import random
from hcipy import *
import numpy as np
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from hcipy.fourier import fourier_transform


# File to read from
filename = "turb_hv57_1.pickle"

grid_size = 1024
wavelength = 1550e-9
screen_size = 1

pupil_grid = make_pupil_grid(grid_size, screen_size)
aperture = make_circular_aperture(screen_size)(pupil_grid)

# Read saved wavefront from file
r_field = read_field(filename)
wf_r = Wavefront(r_field,wavelength)

# Use only middle portion of the read wavefront
elec_r = wf_r.electric_field.shaped
elec_t = elec_r[512:1536,512:1536]
wf_t = Wavefront(Field(np.ones(pupil_grid.size), pupil_grid), wavelength)
wf_t.electric_field = Field(elec_t.ravel(),pupil_grid)*aperture

# Make pupil equal to the sensor size
receiver_diameter = 0.01
#pupil_grid = make_pupil_grid(2048, receiver_diameter)

# Wavefront sensor parameters
sh_f_length = 18.8e-3
sh_pitch = 0.3e-3
sh_diameter = 10e-3
f_number = sh_f_length / sh_pitch
num_lenslets = int(sh_diameter/sh_pitch)

magnification = sh_diameter / screen_size
magnifier = Magnifier(magnification)

# Create wavefront sensors and camera
shwfs = SquareShackHartmannWavefrontSensorOptics(pupil_grid.scaled(magnification), f_number, num_lenslets, sh_diameter)

sh_f_length = 18.8e-3
sh_pitch = 0.3e-3
sh_diameter = 10e-3
f_number = sh_f_length / sh_pitch
num_lenslets = int(sh_diameter/sh_pitch)

magnification = sh_diameter / screen_size
magnifier = Magnifier(magnification)

fpwfs = SquareFocusedPlenopticWavefrontSensorOptics(pupil_grid.scaled(magnification),f_number,f_number,num_lenslets,sh_diameter)
camera = NoiselessDetector(pupil_grid)

wf2 = wf_t.copy()

# Exposure on camera
exposure_time = 1
camera.integrate(shwfs(magnifier(wf2)),exposure_time)
image_sens = camera.read_out()
camera.integrate(fpwfs(magnifier(wf2)),exposure_time)
image2_sens = camera.read_out()


true_phase = (wf_t.phase*aperture).flatten()
true_amp = (wf_t.amplitude*aperture).flatten()
true_int = (wf_t.intensity*aperture).flatten()
res_amp = []
res_phase = []
res_int = []
n_samples = sum(aperture)

for t,a,ints in zip(true_phase,true_amp,true_int):
    res = t
    if res > 0 and abs(res) < np.pi:
        res_phase.append(res)
        res_amp.append(a)
        res_int.append(ints)

phs_var = np.var(res_phase)
amp_var = np.var(res_amp)
ints_var = np.var(res_int)

print("PHS var", phs_var, amp_var, amp_var*4,ints_var)


major_ticks = np.arange(-0.5, 0.5, 4)
focal_grid = make_focal_grid_from_pupil_grid(pupil_grid,wavelength=1550e-9,q=2,num_airy=20)
prop = FraunhoferPropagator(pupil_grid, focal_grid)

show_ticks = True
show_grids = True

plt.rcParams.update({'font.size': 22})
plt.subplot(121)
plt.gca().set_title('SHWFS with HV57 wavefront 1')
ax = plt.gca();
if show_grids:
    ax.set_xticks(np.arange(-0.005, 0.005,0.000303));
    ax.set_yticks(np.arange(-0.005, 0.005,0.000303));
    ax.set_xticklabels([]);
    ax.set_yticklabels([]);
    if show_ticks:
        ax.set_xticklabels(np.arange(0, 34,1));
        ax.set_yticklabels(np.arange(0, 34,1));
    ax.grid(color='white', linestyle='-', linewidth=2)
imshow_field(image_sens, cmap='inferno')
plt.colorbar(fraction=0.046, pad=0.04)
plt.subplot(122)
plt.gca().set_title('FPWFS with HV57 wavefront 1')
ax = plt.gca();
if show_grids:
    ax.set_xticks(np.arange(-0.005, 0.005,0.000303));
    ax.set_yticks(np.arange(-0.005, 0.005,0.000303));
    ax.set_xticklabels([]);
    ax.set_yticklabels([]);
    if show_ticks:
        ax.set_xticklabels(np.arange(0, 34,1));
        ax.set_yticklabels(np.arange(0, 34,1));
    ax.grid(color='white', linestyle='-', linewidth=2)
imshow_field(image2_sens, cmap='inferno')
plt.colorbar(fraction=0.046, pad=0.04)
'''
plt.subplot(143)
imshow_field(wf2.phase, cmap='twilight',vmin=-np.pi,vmax=np.pi)
plt.colorbar(fraction=0.046, pad=0.04)
plt.subplot(144)
imshow_field(wf2.amplitude, cmap='viridis')
plt.colorbar(fraction=0.046, pad=0.04)
'''
plt.draw()
plt.show()

quit()


# Plot cameras
plt.subplot(141)
imshow_field(image_sens, cmap='inferno')
plt.colorbar(fraction=0.046, pad=0.04)
plt.subplot(142)
imshow_field(image2_sens, cmap='inferno')
plt.colorbar(fraction=0.046, pad=0.04)
plt.subplot(143)
imshow_field(wf2.phase, cmap='twilight',vmin=-np.pi,vmax=np.pi)
plt.colorbar(fraction=0.046, pad=0.04)
plt.subplot(144)
imshow_field(wf2.amplitude, cmap='viridis')
plt.colorbar(fraction=0.046, pad=0.04)
plt.draw()
plt.show()