from hcipy import *
import numpy as np
import matplotlib.pyplot as plt
import time

# Characterize phase and intensity fluctuations

# File to write in
filename = "turb_test.txt"

grid_size = 256/2
samples = 2
wavelength = 1550e-9
screen_size = 4

pupil_grid = make_pupil_grid(grid_size, screen_size)
aperture = make_circular_aperture(screen_size)(pupil_grid)

atmos = []

for i in range(samples):
    t0 = time.perf_counter()
    HV57_layers = make_HV57_atmospheric_layers(pupil_grid, 25)
    atmos.append(MultiLayerAtmosphere(HV57_layers, True))
    t1 = time.perf_counter()
    print("Generated", i + 1, "/", samples, f"in {t1-t0:0.4f} seconds")

wf = Wavefront(Field(np.ones(pupil_grid.size), pupil_grid), wavelength)
wf.electric_field *= aperture
wf_diffrac = FresnelPropagator(pupil_grid, 20889.81772516)(wf)
wf_diffrac.electric_field = wf.electric_field-wf_diffrac.electric_field*aperture

plt.subplot(121)
imshow_field(wf_diffrac.amplitude)
plt.subplot(122)
imshow_field(wf_diffrac.phase)
plt.show()

phases = []
amplitudes = []

residual = 999

previous = 100
current = 200

saturated = False

plot_it = True
lims = 0.5

for i in range(samples):
    print("Sample:", int(i+1))

    atmosphere = atmos[i]  # New atmosphere sample

    print("Cn2", atmosphere.Cn_squared)

    wf_t = atmosphere.forward(wf)
    wf_t.electric_field*aperture
    #wf_t.electric_field -= wf_diffrac.electric_field

    phases.append(wf_t.phase.flatten())
    amplitudes.append(wf_t.amplitude.flatten())

    wf_ap = Wavefront(Field(np.ones(pupil_grid.size), pupil_grid), wavelength)
    wf_ap.electric_field = wf_t.electric_field * make_circular_aperture(1)(pupil_grid)

    current = np.var(phases)
    residual = np.abs(current - previous)

    print("Phase var: ", np.var(phases))
    print("Ampli var: ", np.var(amplitudes))
    print("Change: ", residual)
    previous = np.var(phases)

    if residual < 0.001:
        if saturated:
            break
        saturated = True
    else:
        saturated = False

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

        plt.suptitle('Sample: %i' % i, fontsize='x-large')
        plt.draw()
        plt.pause(1000)


    '''
    data = np.ndarray(0)
    for j in range(5):
        print("run:",j, "\n")
        sim.run()
        for d in sim.coupled_flux():
            file = open(filename, "a")
            file.write(str(i) + " " + str(d) + "\n")
            file.close()
        data = numpy.append(data,sim.coupled_flux())
    '''



