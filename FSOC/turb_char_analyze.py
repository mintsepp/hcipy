from hcipy import *
import numpy as np
import matplotlib.pyplot as plt
import time

# Characterize phase and intensity fluctuations

# File to write in
filename = "turb_front_1.pickle"

grid_size = 2048
wavelength = 1550e-9
screen_size = 2

pupil_grid = make_pupil_grid(grid_size, screen_size)
aperture = make_circular_aperture(screen_size)(pupil_grid)

r_field = read_field(filename)

wf_t = Wavefront(r_field,wavelength)
wf_ap = Wavefront(Field(np.ones(pupil_grid.size), pupil_grid), wavelength)
wf_ap.electric_field = wf_t.electric_field * make_circular_aperture(1)(pupil_grid)

plot_it = False
lims = 0.5

flat_phases = wf_ap.phase.flatten()

arr = []
for p in flat_phases:
    if p != 0 and p != np.pi:
        arr.append(p)

'''
fig, ax = plt.subplots()
ax.set_ylim([1e-4, 1])
ax.set_xlim([-np.pi*1.01, np.pi*1.01])
ax.hist(arr, density=True, bins=1000, histtype=u'step', label="Weak 1")
ax.set_title("Phase distribution")
ax.set_ylabel("probability density")
ax.set_xlabel("phase [rad]")
plt.legend()
plt.show()
'''

phases = wf_ap.phase
grady = np.gradient(phases)
gradx = np.gradient(phases)

plt.subplot(1,1,1)
im2 = imshow_field(phases)
plt.quiver(phases.grid.x[0::2],
           phases.grid.y[0::2],
           gradx[0::2], grady[0::2],
           color='white')
plt.draw()
plt.show()

if plot_it:
    # Plotting
    plt.clf()
    plt.subplot(1, 4, 1)
    imshow_field(wf_t.phase, cmap='twilight')
    plt.colorbar(fraction=0.046, pad=0.04)

    plt.subplot(1, 4, 2)
    imshow_field(wf_t.amplitude, cmap='viridis')
    plt.colorbar(fraction=0.046, pad=0.04)

    plt.subplot(1, 4, 3)
    imshow_field(wf_ap.phase, cmap='twilight')
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.xlim(-lims, lims)
    plt.ylim(-lims, lims)

    plt.subplot(1, 4, 4)
    imshow_field(wf_ap.amplitude, cmap='viridis')
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.xlim(-lims, lims)
    plt.ylim(-lims, lims)

    plt.suptitle('File: %s' % filename, fontsize='x-large')
    plt.draw()
    plt.pause(1000)



