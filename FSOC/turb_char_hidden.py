from hcipy import *
import numpy as np
import matplotlib.pyplot as plt
import time

last_prop_dist = 616.19462052
modulo = last_prop_dist % 1550e-9
phase_shift = (modulo/1550e-9)*2*np.pi
phase_shift = 2.5
print(phase_shift)

# Characterize phase and intensity fluctuations

# File to write in
filename = "turb_test.txt"

grid_size = 256/2
samples = 2
wavelength = 1550e-9
screen_size = 2

pupil_grid = make_pupil_grid(grid_size, screen_size)
aperture = make_circular_aperture(screen_size)(pupil_grid)

atmos = []
atmos2 = []

for i in range(samples):
    t0 = time.perf_counter()
    HV57_layers = make_HV57_atmospheric_layers(pupil_grid, 25)
    atmos.append(MultiLayerAtmosphere(HV57_layers, True))
    atmos2.append(MultiLayerAtmosphere(HV57_layers))
    t1 = time.perf_counter()
    print("Generated", i + 1, "/", samples, f"in {t1-t0:0.4f} seconds")

wf = Wavefront(Field(np.ones(pupil_grid.size), pupil_grid), wavelength)
wf.electric_field *= aperture

plot_it = True
lims = 0.5

for i in range(samples):
    print("Sample:", int(i+1))

    atmosphere = atmos[i]  # New atmosphere sample
    atmosphere2 = atmos2[i]

    print("Cn2", atmosphere.Cn_squared)

    wf_t = atmosphere.forward(wf)
    wf_t.electric_field *= aperture
    #wf_t.electric_field -= wf_diffrac.electric_field

    wf_t2 = atmosphere2.forward(wf)
    wf_t2.electric_field *= aperture

    wf_t3 = atmosphere2.forward(wf)
    wf_t3.electric_field *= aperture
    wf_t3.electric_field -= wf_t.electric_field

    phases = Wavefront(Field(np.ones(pupil_grid.size), pupil_grid), wavelength).phase

    for l in atmosphere2.layers:
        wf_p = l.forward(wf)
        phases += wf_p.phase

    phases *= aperture

    wf_p = Wavefront(Field(np.ones(pupil_grid.size), pupil_grid), wavelength)
    wf_p.electric_field *= np.exp(1j*phases)

    if plot_it:
        # Plotting
        plt.clf()
        plt.subplot(1, 5, 1)
        imshow_field(wf_t.phase, cmap='twilight')
        plt.colorbar(fraction=0.046, pad=0.04)

        plt.subplot(1, 5, 2)
        imshow_field(wf_t.amplitude, cmap='viridis')
        plt.colorbar(fraction=0.046, pad=0.04)

        plt.subplot(1, 5, 3)
        imshow_field(wf_t2.phase, cmap='twilight')
        plt.colorbar(fraction=0.046, pad=0.04)

        plt.subplot(1, 5, 4)
        imshow_field(phases, cmap='RdBu')
        plt.colorbar(fraction=0.046, pad=0.04)

        plt.subplot(1, 5, 5)
        imshow_field(wf_p.phase, cmap='twilight')

        plt.colorbar(fraction=0.046, pad=0.04)

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



