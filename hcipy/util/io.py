import datetime
import pickle

import numpy as np
import asdf
import asdf.fits_embed
from astropy import wcs
from astropy.io import fits

from ..field import Grid, Field
from ..mode_basis import ModeBasis

try:
	from .._version import version as _version
except ImportError:
	_version = ''

def read_fits(filename, extension=0):
	'''Read an array from a fits file.

	Parameters
	----------
	filename : string
		The filename of the file to read. This can include a path.
	extension : integer
		The extension of the fits file that is being read.

	Returns
	-------
	ndarray
		The ndarray read from the fits file.
	'''
	return fits.getdata(filename, extension).copy()

def write_fits(data, filename, shape=None, overwrite=True):
	'''Write the data to a fits-file.

	If the data is a Field with SeparatedCoords, the shaped field will be written to the
	fits-file, to allow for easier viewing with external tools.

	Parameters
	----------
	data : ndarray or Field
		The ndarray or Field to write to the fits file.
	filename : string
		The filename of the newly created file. This can include a path.
	shape : ndarray or None
		The shape to which to reshape the data. If this is given, it will override a potential
		shape from the grid accompaning the field.
	overwrite : boolean
		Whether to overwrite the fits-file if it already exists.
	'''
	hdu = None

	if shape is not None:
		hdu = fits.PrimaryHDU(data.reshape(shape))
	elif hasattr(data, 'grid'):
		if data.grid.is_separated:
			hdu = fits.PrimaryHDU(data.shaped)

	if hdu is None:
		hdu = fits.PrimaryHDU(data)

	hdu.writeto(filename, overwrite=True)

def _guess_file_format(filename):
	'''Guess the file format from the filename.

	Parameters
	----------
	filename : string
		The filename of the file.

	Returns
	-------
	string
		The file type. This is one of ["asdf", "fits"].
	'''
	if filename.endswith('asdf'):
		return 'asdf'
	elif filename.endswith('fits') or filename.endswith('fits.gz'):
		return 'fits'
	elif filename.endswith('pkl') or filename.endswith('pickle'):
		return 'pickle'
	else:
		return None

def _make_metadata(file_type):
	'''Make a metadata section for file output.

	This contains the version of HCIPy used to write the file, the current time and
	date and the file type.

	Parameters
	----------
	file_type : string
		The identifier string for what is going to be stored in the file.

	Returns
	-------
	dictionary
		The metadata.
	'''
	tree = {
		'meta': {
			'author': 'HCIPy %s' % _version,
			'date_utc': datetime.datetime.utcnow().isoformat(),
			'file_type': file_type
		}
	}

	return tree

def read_grid(filename, fmt=None):
	'''Read a grid from a file.

	Parameters
	----------
	filename : string
		The path of the file you want to read the grid from.
	fmt : string
		The file format. If it is not given, the file format will be guessed from the file extension.

	Returns
	-------
	Grid
		The read grid.

	Raises
	------
	ValueError
		If the file format could not be guessed from the file extension.
	NotImplementedError
		If the file format was not yet implemented.
	'''
	if fmt is None:
		fmt = _guess_file_format(filename)

		if fmt is None:
			raise ValueError('Format not given and could not be guessed based on the file extension.')

	if fmt in ['asdf', 'fits']:
		if fmt == 'fits':
			f = asdf.fits_embed.AsdfInFits.open(filename)
		else:
			f = asdf.open(filename)
		grid = Grid.from_dict(f.tree['grid'])
		f.close()

		return grid
	elif fmt == 'pickle':
		with open(filename, 'rb') as f:
			return pickle.load(f)
	else:
		raise NotImplementedError('The "%s" file format has not been implemented.' % fmt)

def write_grid(grid, filename, fmt=None, overwrite=True):
	'''Write a grid to a file.

	Parameters
	----------
	grid : Grid
		The grid to write to the file.
	filename : string
		The path of the file you want to write to.
	fmt : string
		The file format. If it is not given, the file format will be guessed based on the file extension.
	overwrite : boolean
		Whether to overwrite the file, if it already exists.

	Raises
	------
	NotImplementedError
		If the file format was not yet implemented.
	'''
	if fmt is None:
		fmt = _guess_file_format(filename)

		if fmt is None:
			raise ValueError('Format not given and could not be guessed based on the file extension.')

	tree = _make_metadata('grid')
	tree['grid'] = grid.to_dict()

	if fmt == 'asdf':
		target = asdf.AsdfFile(tree)
		target.write_to(filename, all_array_compression='zlib')
	elif fmt == 'fits':
		hdulist = fits.HDUList()

		ff = asdf.fits_embed.AsdfInFits(hdulist, tree)
		ff.write_to(filename, all_array_compression='zlib', overwrite=overwrite)
	elif fmt == 'pickle':
		with open(filename, 'wb') as f:
			pickle.dump(grid, f)
	else:
		raise NotImplementedError('The "%s" file format has not been implemented.' % fmt)

def read_field(filename, fmt=None):
	'''Read a field from a file.

	Parameters
	----------
	filename : string
		The path of the file you want to read the field from.
	fmt : string
		The file format. If it is not given, the file format will be guessed from the file extension.

	Returns
	-------
	Field
		The read field.

	Raises
	------
	ValueError
		If the file format could not be guessed from the file extension.
	NotImplementedError
		If the file format was not yet implemented.
	'''
	if fmt is None:
		fmt = _guess_file_format(filename)

		if fmt is None:
			raise ValueError('Format not given and could not be guessed based on the file extension.')

	if fmt == 'asdf':
		f = asdf.open(filename)
		field = Field.from_dict(f.tree['field'])
		f.close()

		return field
	elif fmt == 'fits':
		f = asdf.fits_embed.AsdfInFits.open(filename)
		tree = f.tree['field']

		if 'grid' in tree:
			grid = Grid.from_dict(tree['grid'])
			new_shape = np.concatenate((tree['values'].shape[:-grid.ndim], [grid.size])).astype('int')
			tree['values'] = tree['values'].reshape(new_shape)

		field = Field.from_dict(tree).reshape(new_shape)
		f.close()

		return field
	elif fmt == 'pickle':
		with open(filename, 'rb') as f:
			return pickle.load(f)
	else:
		raise NotImplementedError('The "%s" file format has not been implemented.' % fmt)

def write_field(field, filename, fmt=None, overwrite=True):
	'''Write a field to a file.

	Parameters
	----------
	field : Field
		The field to write to the file.
	filename : string
		The path of the file you want to write to.
	fmt : string
		The file format. If it is not given, the file format will be guessed based on the file extension.
	overwrite : boolean
		Whether to overwrite the file, if it already exists.

	Raises
	------
	NotImplementedError
		If the file format was not yet implemented.
	'''
	if fmt is None:
		fmt = _guess_file_format(filename)

		if fmt is None:
			raise ValueError('Format not given and could not be guessed based on the file extension.')

	tree = _make_metadata('field')
	tree['field'] = field.to_dict()

	if fmt == 'asdf':
		target = asdf.AsdfFile(tree)
		target.write_to(filename, all_array_compression='zlib')
	elif fmt == 'fits':
		hdulist = fits.HDUList()

		if field.grid.is_separated:
			hdulist.append(fits.ImageHDU(np.ascontiguousarray(field.shaped)))
			tree['field']['values'] = hdulist[0].data

		if field.grid.is_regular:
			w = wcs.WCS(naxis=field.grid.ndim)
			w.wcs.crpix = np.ones(field.grid.ndim)
			w.wcs.cdelt = field.grid.delta
			w.wcs.crval = field.grid.zero
			w.wcs.ctype = ['X', 'Y', 'Z', 'W'][:field.grid.ndim]

			hdulist[0].header.update(w.to_header())

		ff = asdf.fits_embed.AsdfInFits(hdulist, tree)
		ff.write_to(filename, all_array_compression='zlib', overwrite=overwrite)
	elif fmt == 'pickle':
		with open(filename, 'wb') as f:
			pickle.dump(field, f)
	else:
		raise NotImplementedError('The "%s" file format has not been implemented.' % fmt)

def read_mode_basis(filename, fmt=None):
	'''Read a mode basis from a file.

	Parameters
	----------
	filename : string
		The path of the file you want to read the mode basis from.
	fmt : string
		The file format. If it is not given, the file format will be guessed from the file extension.

	Returns
	-------
	ModeBasis
		The read mode basis.

	Raises
	------
	ValueError
		If the file format could not be guessed from the file extension.
	NotImplementedError
		If the file format was not yet implemented.
	'''
	if fmt is None:
		fmt = _guess_file_format(filename)

		if fmt is None:
			raise ValueError('Format not given and could not be guessed based on the file extension.')

	if fmt == 'asdf':
		f = asdf.open(filename)
		mode_basis = ModeBasis.from_dict(f.tree['mode_basis'])
		f.close()

		return mode_basis
	elif fmt == 'fits':
		f = asdf.fits_embed.AsdfInFits.open(filename)
		tree = f.tree['mode_basis']

		if 'modes' in tree:
			grid = Grid.from_dict(tree['grid'])
			modes = tree['modes']

			old_shape = np.concatenate((modes.shape[:-grid.ndim], [grid.size]))
			tree['transformation_matrix'] = modes.reshape(old_shape).T
			del tree['modes']

		mode_basis = ModeBasis.from_dict(tree)
		f.close()

		return mode_basis
	elif fmt == 'pickle':
		with open(filename, 'rb') as f:
			return pickle.load(f)
	else:
		raise NotImplementedError('The "%s" file format has not been implemented.' % fmt)

def write_mode_basis(mode_basis, filename, fmt=None, overwrite=True):
	'''Write a mode basis to a file.

	Parameters
	----------
	mode_basis : ModeBasis
		The mode basis to write to the file.
	filename : string
		The path of the file you want to write to.
	fmt : string
		The file format. If it is not given, the file format will be guessed based on the file extension.
	overwrite : boolean
		Whether to overwrite the file, if it already exists.

	Raises
	------
	NotImplementedError
		If the file format was not yet implemented.
	'''
	if fmt is None:
		fmt = _guess_file_format(filename)

		if fmt is None:
			raise ValueError('Format not given and could not be guessed based on the file extension.')

	tree = _make_metadata('mode_basis')
	tree['mode_basis'] = mode_basis.to_dict()

	if fmt == 'asdf':
		target = asdf.AsdfFile(tree)
		target.write_to(filename, all_array_compression='zlib')
	elif fmt == 'fits':
		hdulist = fits.HDUList()

		if mode_basis.grid and mode_basis.grid.is_separated:
			new_shape = np.concatenate(([-1], mode_basis.grid.shape))
			modes = np.ascontiguousarray(mode_basis.to_dense().transformation_matrix.T.reshape(new_shape))

			hdulist.append(fits.ImageHDU(np.asarray(modes)))

			w = wcs.WCS(naxis=mode_basis.grid.ndim)
			w.wcs.crpix = np.ones(mode_basis.grid.ndim)
			w.wcs.cdelt = mode_basis.grid.delta
			w.wcs.crval = mode_basis.grid.zero
			w.wcs.ctype = ['X', 'Y', 'Z', 'W'][:mode_basis.grid.ndim]

			hdulist[0].header.update(w.to_header())

			del tree['mode_basis']['transformation_matrix']
			tree['mode_basis']['modes'] = hdulist[0].data

		ff = asdf.fits_embed.AsdfInFits(hdulist, tree)
		ff.write_to(filename, all_array_compression='zlib', overwrite=overwrite)
	elif fmt == 'pickle':
		with open(filename, 'wb') as f:
			pickle.dump(mode_basis, f)
	else:
		raise NotImplementedError('The "%s" file format has not been implemented.' % fmt)
