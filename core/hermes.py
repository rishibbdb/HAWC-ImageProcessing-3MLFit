from __future__ import division

import hashlib
from astropy import wcs
import astropy.units as u
import numpy as np
from astropy.coordinates import ICRS, BaseCoordinateFrame, SkyCoord, Angle
from astropy.io import fits
from past.utils import old_div
from scipy.interpolate import RegularGridInterpolator
import scipy
from astromodels.functions.function import Function3D, FunctionMeta
from astromodels.utils.angular_distance import angular_distance_fast
from astropy.coordinates.angles.utils import angular_separation, position_angle
from astromodels.utils.angular_distance import angular_distance

#############Created from Hugo's Galprop Template Analysis#########

class Hermes(Function3D, metaclass=FunctionMeta):
    r"""
        description :

            Use a 3D template that has morphology and flux information.
            GalProp, DRAGON or a similar model in fits format would work. 
            Only parameter is a normalization factor. 

        latex : $ N $

        parameters :

            N :

                desc : normalization
                initial value : 1
                fix : yes

            hash :

                desc : hash of model map [needed for memoization]
                initial value : 1
                fix : yes

            ihdu:
                desc: header unit index of fits file
                initial value: 0
                fix: True
                min: 0

        properties:
            fits_file:
                desc: fits file to load
                defer: True
                function: _load_file
            frame:
                desc: coordinate frame
                initial value: icrs
                allowed values:
                    - icrs
                    - galactic
                    - fk5
                    - fk4
                    - fk4_no_e
    """

    def _set_units(self, x_unit, y_unit, z_unit, w_unit):
        #self.N.unit = (u.keV * u.cm**2 * u.s * u.sr) ** (-1)
        # The spectrum and morphology are embedded in the template.
        # The normalization N is for scaling up or down the whole template to fit the data.
        self.N.unit = w_unit

    def _setup(self):
        self._frame = "icrs"
        self._intmap = None

    def set_frame(self, new_frame):
        """
        Set a new frame for the coordinates (the default is ICRS J2000)

        :param new_frame: a coordinate frame from astropy
        :return: (none)
        """
        assert isinstance(new_frame, BaseCoordinateFrame)

        self._frame = new_frame

    def _load_file(self):

        if self.fits_file is None:
            raise RuntimeError(
                "Need to specify a fits file with a template map.")

        self._fitsfile=self.fits_file.value

        with fits.open(self._fitsfile) as f:

            self._delLon = f[int(self.ihdu.value)].header['CDELT1']
            self._delLat = f[int(self.ihdu.value)].header['CDELT2']
            self._delEn = f[int(self.ihdu.value)].header['CDELT3']
            self._refLon = f[int(self.ihdu.value)].header['CRVAL1']
            self._refLat = f[int(self.ihdu.value)].header['CRVAL2']
            self._refEn = f[int(self.ihdu.value)].header['CRVAL3']  # values in log10
            self._map = f[int(self.ihdu.value)].data
            self._wcs = wcs.WCS(header = f[int(self.ihdu.value)].header)
            if len(self._map.shape) == 4:
                self._map = self._map[0]
            self._nl = f[int(self.ihdu.value)].header['NAXIS1']  # longitude
            self._nb = f[int(self.ihdu.value)].header['NAXIS2']  # latitude
            self._ne = f[int(self.ihdu.value)].header['NAXIS3']  # energy

            # Create the function for the interpolation
            self._L = np.linspace(
                self._refLon, self._refLon-(self._nl-1)*self._delLon, self._nl)
            self._B = np.linspace(
                self._refLat, self._refLat+(self._nb-1)*self._delLat, self._nb)
            self._E = np.linspace(
                self._refEn, self._refEn+(self._ne-1)*self._delEn, self._ne)
            for i in range(len(self._E)):
                self._map[i] = np.fliplr(self._map[i])
            self._F = RegularGridInterpolator(
                (self._E, self._B, self._L), self._map, bounds_error=False)

            h = hashlib.sha224()
            h.update(self._map)
            h.update(repr(self._wcs).encode("utf-8"))
            self.hash = int(h.hexdigest(), 16)


    def evaluate(self, x, y, z, N, hash, ihdu):

        if self._map is None:
            self._load_file(self._fitsfile)

        if self._intmap is None:
            _coord = SkyCoord(ra=x, dec=y, frame=self._frame, unit="deg")
            b = _coord.transform_to('galactic').b.value
            l = _coord.transform_to('galactic').l.value
            lon = l
            lat = b
            energy = np.log10(z)

            if lon.size != lat.size:
                raise AttributeError("Lon and Lat should be the same size")
            f = np.zeros([lon.size, energy.size])
            E0 = self._refEn
            Ef = self._refEn + (self._ne-1)*self._delEn

            #shift = np.where(lon > 180.)
            #lon[shift] = 180 - lon[shift]

            for i in range(energy.size):
                e = np.repeat(energy[i], len(lon))
                f[:, i] = self._F(np.array([e, lat, lon]).T)

            bad_idx = np.isnan(f)
            f[bad_idx] = 0
            bad_idx = np.isinf(f)
            f[bad_idx] = 0
            bad_idx = np.isneginf(f)
            f[bad_idx] = 0
            assert np.all(np.isfinite(f)), "some interpolated values are wrong"
            self._intmap = f

        A = np.multiply(N, self._intmap)
        return A

    def get_boundaries(self):
        # Taken from SpatialTemplate_2D
        Xcorners = np.array( [0, 0,        self._nl, self._nl] )
        Ycorners = np.array( [0, self._nb, 0,        self._nb] )

        corners = SkyCoord.from_pixel( Xcorners, Ycorners, wcs=self._wcs, origin = 0).transform_to(self._frame)

        min_lon = min(corners.ra.degree)
        max_lon = max(corners.ra.degree)

        min_lat = min(corners.dec.degree)
        max_lat = max(corners.dec.degree)

        return (min_lon, max_lon), (min_lat, max_lat)


    def get_total_spatial_integral(self, z=None):
        if isinstance( z, u.Quantity):
            z = z.value
        return np.multiply(self.N.value, np.ones_like( z ))

