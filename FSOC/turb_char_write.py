from hcipy import *
import numpy as np
import matplotlib.pyplot as plt
import time
import datetime

# Characterize phase and intensity fluctuations

# File to write in
filename = "turb_hv57_turbo.pickle"

grid_size = 2048/8
wavelength = 1550e-9
screen_size = 2

pupil_grid = make_pupil_grid(grid_size, screen_size)
aperture = make_circular_aperture(screen_size)(pupil_grid)

wf = Wavefront(Field(np.ones(pupil_grid.size), pupil_grid), wavelength)
wf.electric_field *= aperture

p = np.exp(-(pupil_grid.as_('polar').r/0.9)**20)
wf = Wavefront(Field(p, pupil_grid), wavelength)

print("Started:", datetime.datetime.now().time(), "and done in ~", 9e-5*grid_size**2, "seconds.")
t0 = time.perf_counter()
HV57_layers = make_HV57_atmospheric_layers(pupil_grid, 1)
# Increase turbulence strength at high altitude
#HV57_layers[0].Cn_squared *= 10
'''
HV57_layers[0].Cn_squared *= 100000
HV57_layers[1].Cn_squared *= 100000
HV57_layers[2].Cn_squared *= 100000
HV57_layers[3].Cn_squared *= 100000
HV57_layers[4].Cn_squared *= 100000
'''

atmosphere = MultiLayerAtmosphere(HV57_layers, True)
t1 = time.perf_counter()
print(f"Generated in {t1 - t0:0.4f} seconds.")

print("Atmosphere r0:", fried_parameter_from_Cn_squared(atmosphere.Cn_squared))
print("Atmosphere Cn2:",atmosphere.Cn_squared)

layers2 = make_las_campanas_atmospheric_layers(pupil_grid,25)
atmos2 = MultiLayerAtmosphere(layers2,True)

print("Atmosphere2 r0:", fried_parameter_from_Cn_squared(atmosphere.Cn_squared))
print("Atmosphere2 Cn2:",atmosphere.Cn_squared)

quit()


plot_it = True
lims = 0.5

wf_t = atmosphere.forward(wf)
wf_t.electric_field*=aperture

wf_ap = Wavefront(Field(np.ones(pupil_grid.size), pupil_grid), wavelength)
wf_ap.electric_field = wf_t.electric_field * make_circular_aperture(1)(pupil_grid)

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

    write_field(wf_t.electric_field,filename)

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



