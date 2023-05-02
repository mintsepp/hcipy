from hcipy import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from hcipy.atmosphere import *

nn = 1
size = 1
grid_size = 256*nn
screen_scale = 4

screen_grid = make_pupil_grid(grid_size, size*screen_scale)
pupil_grid = make_pupil_grid(grid_size, size)

wavelength = 1550e-9

L0 = 25

heights = np.linspace(250,25000,10)
velocities = np.array([8.7925, 13.0704, 24.8266, 36.9636, 33.8588, 19.9747, 10.8762, 8.3583, 8.0233, 8.0008])
Cn_squared = np.array([1.6226e-12, 4.2760e-13, 5.0330e-14, 3.3898e-14, 4.4593e-14, 3.3403e-14, 1.5743e-14, 5.3596e-15, 1.4406e-15, 3.239e-16])
print(heights)
layers = []

#heights = np.array([0,0,0,0,0,0,0,0,0,0])
#heights[0] = 10

strong_layer = InfiniteAtmosphericLayer(screen_grid, Cn_squared[4]*10, L0, velocities[4], heights[4])

aperture = make_circular_aperture(0.5)(pupil_grid)
wf = Wavefront(Field(np.ones(screen_grid.size), screen_grid), wavelength)
#wf.electric_field *= aperture

#beam = GaussianBeam(0.5,0,1550e-9)
#wf = beam.evaluate(pupil_grid)

lims = 1

zernike_basis = make_zernike_basis(500,1,screen_grid,2)
strong_layer2 = InfiniteAtmosphericLayer(screen_grid, Cn_squared[4]*10, L0, velocities[4], heights[4])

for d in np.linspace(0, 1550000, 10000):

	wf0 = strong_layer.forward(wf)
	#wf0.electric_field = wf.electric_field * np.exp(1j*zernike_basis[1])

	wf1 = FresnelPropagator(screen_grid, d)(wf0)

	wf2 = FresnelPropagator(screen_grid,strong_layer.height)(wf0)
	wf02 = strong_layer2.forward(wf)
	wf2 = strong_layer.forward(wf2)
	wf2 = FresnelPropagator(screen_grid,1550)(wf2)
	wf2 = strong_layer2.forward(wf2)
	wf2 = FresnelPropagator(screen_grid,d-1550)(wf2)

	wf0.electric_field *= aperture
	wf1.electric_field *= aperture
	wf2.electric_field *= aperture
	wf02.electric_field *= aperture

	plt.clf()
	plt.subplot(1,6,1)
	imshow_field(wf0.phase, cmap='RdBu')
	plt.colorbar()
	plt.xlim(-lims, lims)
	plt.ylim(-lims, lims)

	plt.subplot(1,6,2)
	#imshow_field(wf0.amplitude, cmap='viridis')
	imshow_field(wf02.phase, cmap='RdBu')
	plt.colorbar()
	plt.xlim(-lims, lims)
	plt.ylim(-lims, lims)

	plt.subplot(1,6,3)
	imshow_field(wf1.phase, cmap='RdBu')
	plt.colorbar()
	plt.xlim(-lims, lims)
	plt.ylim(-lims, lims)

	plt.subplot(1,6,4)
	imshow_field(wf1.amplitude, cmap='viridis',vmax=2.5)
	plt.colorbar()
	plt.xlim(-lims, lims)
	plt.ylim(-lims, lims)

	plt.subplot(1,6,5)
	imshow_field(wf2.phase, cmap='RdBu')
	plt.colorbar()
	plt.xlim(-lims, lims)
	plt.ylim(-lims, lims)

	plt.subplot(1,6,6)
	imshow_field(wf2.amplitude, cmap='viridis',vmax=2.5)
	plt.colorbar()
	plt.xlim(-lims, lims)
	plt.ylim(-lims, lims)

	plt.suptitle('Fresnel propagation distance %d meters' % (d), fontsize='x-large')
	plt.draw()
	plt.pause(0.005)

'''
for t in np.linspace(0, 100, 5001):
	atmosphere.evolve_until(t)

	wf2 = atmosphere.forward(wf)
	wf2.electric_field *= aperture
	img = Field(prop(wf2).intensity, focal_grid)

	fresprop = FresnelPropagator(pupil_grid,layer9.height-layer0.height)
	wf3 = layer0.forward(wf)
	wf3 = fresprop(wf3)
	wf3 = layer9.forward(wf3)
	wf3.electric_field *= aperture

	plt.clf()
	plt.subplot(1,4,1)
	imshow_field(wf2.phase, cmap='RdBu')
	plt.colorbar()
	plt.subplot(1,4,2)
	#imshow_field(np.log10(img / img.max()), vmin=-6)
	imshow_field(wf2.amplitude, cmap='viridis')
	plt.colorbar()
	plt.subplot(1,4,3)
	imshow_field(wf3.amplitude, cmap='viridis')
	plt.colorbar()
	plt.subplot(1, 4, 4)
	imshow_field(wf3.phase, cmap='RdBu')
	plt.colorbar()
	plt.draw()
	plt.pause(5)
'''