from .wavefront_sensor import WavefrontSensorOptics, WavefrontSensorEstimator
from ..optics import OpticalSystem, MicroLensArray
from ..field import CartesianGrid, Field, SeparatedCoords
from ..propagation import FresnelPropagator

import numpy as np
from scipy import ndimage

class FocusedPlenopticWavefrontSensorOptics(WavefrontSensorOptics):
	'''The optical elements for a focused plenoptic wavefront sensor.

	This class uses an optical system described in [Wu2015]_, which uses an objective lens
	and a microlens array for wavefront sensing.

	.. [Wu2015] Wu et al. 2015, "Determining the phase and amplitude
		distortion of a wavefront using a plenoptic sensor"

	Parameters
	----------
	input_grid : Grid
		The grid on which the input wavefront is defined.
	objective_lens : MicroLensArray
		The objective lens as a single (micro) lens, through which the wavefront propagates.
	micro_lens_array : MicroLensArray
		The microlens array through which the wavefront propagates after the objective lens.
	'''
	def __init__(self, input_grid, objective_lens, micro_lens_array):
		# Make propagators
		fp_prop_obj = FresnelPropagator(input_grid,objective_lens.focal_length+micro_lens_array.focal_length)
		fp_prop_mla = FresnelPropagator(input_grid,micro_lens_array.focal_length)

		# Make optical system
		OpticalSystem.__init__(self, (objective_lens, fp_prop_obj, micro_lens_array, fp_prop_mla))
		self.mla_index = micro_lens_array.mla_index
		self.mla_grid = micro_lens_array.mla_grid
		self.micro_lens_array = micro_lens_array
		self.objective_lens = objective_lens

class SquareFocusedPlenopticWavefrontSensorOptics(FocusedPlenopticWavefrontSensorOptics):
	'''Helper class to create a focused plenoptic WFS with a square microlens array.

	This class creates an optical system described in [Wu2015]_ with a square microlens array.

	.. [Wu2015] Wu et al. 2015, "Determining the phase and amplitude
		distortion of a wavefront using a plenoptic sensor"

	Parameters
		----------
		input_grid : Grid
			The grid on which the input wavefront is defined.
		f_num_obj : scalar
			Focal number of the objective lens.
		f_num_mla : scalar
			Focal number of the microlens array.
		num_lenslets : int
			Number of lenslets along one diameter.
		pupil_diameter : scalar
			Pupil (and objective lens) diameter in meters.
	'''
	def __init__(self, input_grid, f_num_obj, f_num_mla, num_lenslets, pupil_diameter):
		lenslet_diameter = float(pupil_diameter) / num_lenslets
		x = np.arange(-pupil_diameter, pupil_diameter, lenslet_diameter)
		self.mla_grid = CartesianGrid(SeparatedCoords((x,x)))

		f_len_obj = f_num_obj * pupil_diameter
		f_len_mla = f_num_mla * lenslet_diameter

		# Objective lens as single (micro) lens
		x = np.arange(-pupil_diameter, pupil_diameter, pupil_diameter)
		self.obj_grid = CartesianGrid(SeparatedCoords((x, x)))
		self.objective_lens = MicroLensArray(input_grid, self.obj_grid, f_len_obj)
		self.micro_lens_array = MicroLensArray(input_grid, self.mla_grid, f_len_mla)

		FocusedPlenopticWavefrontSensorOptics.__init__(self,input_grid,self.objective_lens,self.micro_lens_array)
