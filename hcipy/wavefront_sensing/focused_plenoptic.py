from .wavefront_sensor import WavefrontSensorOptics, WavefrontSensorEstimator
from ..optics import OpticalSystem, MicroLensArray, ThinLens
from ..field import CartesianGrid, Field, SeparatedCoords
from ..propagation import FresnelPropagator, FraunhoferPropagator

import numpy as np
from scipy import ndimage

class FocusedPlenopticWavefrontSensorOptics(WavefrontSensorOptics):
	def __init__(self, input_grid, objective_lens, micro_lens_array):
		# Make propagators
		fp_prop_obj = FresnelPropagator(input_grid,micro_lens_array.focal_length+objective_lens.focal_length)
		fp_prop_mla = FresnelPropagator(input_grid,micro_lens_array.focal_length)

		# Make optical system
		OpticalSystem.__init__(self, (objective_lens, fp_prop_obj, micro_lens_array,fp_prop_mla))
		self.mla_index = micro_lens_array.mla_index
		self.mla_grid = micro_lens_array.mla_grid
		self.micro_lens_array = micro_lens_array
		self.objective_lens = objective_lens

class SquareFocusedPlenopticWavefrontSensorOptics(FocusedPlenopticWavefrontSensorOptics):
	## Helper class to create a focused plenoptic WFS with square microlens array
	def __init__(self, input_grid, f_num_obj, f_num_mla, num_lenslets, pupil_diameter):
		lenslet_diameter = float(pupil_diameter) / num_lenslets
		x = np.arange(-pupil_diameter, pupil_diameter, lenslet_diameter)
		self.mla_grid = CartesianGrid(SeparatedCoords((x,x)))

		f_len_obj = f_num_obj * pupil_diameter
		f_len_mla = f_num_mla * lenslet_diameter

		print("obj", f_len_obj)
		print("mla", f_len_mla)

		# Objective lens as single (micro) lens
		x = np.arange(-pupil_diameter, pupil_diameter, pupil_diameter)
		self.obj_grid = CartesianGrid(SeparatedCoords((x, x)))
		self.objective_lens = MicroLensArray(input_grid, self.obj_grid, f_len_obj)
		self.micro_lens_array = MicroLensArray(input_grid, self.mla_grid, f_len_mla)

		FocusedPlenopticWavefrontSensorOptics.__init__(self,input_grid,self.objective_lens,self.micro_lens_array)

class FocusedPlenopticWavefrontSensorEstimator(WavefrontSensorEstimator):
	#TODO: Implement for FPWFS later, for now keep this
	def __init__(self, mla_grid, mla_index, estimation_subapertures=None):
		self.mla_grid = mla_grid
		self.mla_index = mla_index
		if estimation_subapertures is None:
			self.estimation_subapertures = np.unique(self.mla_index)
		else:
			self.estimation_subapertures = np.flatnonzero(np.array(estimation_subapertures))
		self.estimation_grid = self.mla_grid.subset(estimation_subapertures)

	def estimate(self, images):
		image = images[0]

		fluxes = ndimage.measurements.sum(image, self.mla_index, self.estimation_subapertures)
		sum_x = ndimage.measurements.sum(image * image.grid.x, self.mla_index, self.estimation_subapertures)
		sum_y = ndimage.measurements.sum(image * image.grid.y, self.mla_index, self.estimation_subapertures)

		centroid_x = sum_x / fluxes
		centroid_y = sum_y / fluxes

		centroids = np.array((centroid_x, centroid_y)) - np.array(self.mla_grid.points[self.estimation_subapertures, :]).T
		return Field(centroids, self.estimation_grid)
