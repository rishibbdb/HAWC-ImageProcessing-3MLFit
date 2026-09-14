import itertools
import os
import sys
import warnings

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy
from astropy.coordinates import SkyCoord
for prefix in ("OMP", "MKL", "NUMEXPR"):
    os.environ[f"{prefix}_NUM_THREADS"] = "4"

import ROOT
import yaml
ROOT.PyConfig.IgnoreCommandLineOptions = True

import astromodels
import astromodels.functions.priors as priors
import threeML
import threeML.minimizer.minimization
from threeML import *
try:
    from hawc_hal import HAL, HealpixConeROI, HealpixMapROI

    have_hal = True
except:
    have_hal = False

try:
    import pygmo

    have_pagmo = True
except:
    have_pagmo = False

class PipelineConfig:
    """Load and manage configuration from YAML file"""
    def __init__(self, config_file: str):
        with open(config_file, 'r') as f:
            self.config = yaml.safe_load(f)
        self.config_file = config_file
    
    def get(self, key: str, default=None):
        """Get config value using dot notation: 'section.subsection.key'"""
        keys = key.split('.')
        value = self.config
        for k in keys:
            value = value.get(k, default) if isinstance(value, dict) else default
        return value
    
    def __getitem__(self, key):
        return self.config[key]
    
    def __repr__(self):
        return yaml.dump(self.config, default_flow_style=False)
    
class threeMLFit:

    def __init__(self, config_path: str, model, save_dir: str, roiTemplate: str = None, logger: str = None):
        self.config = PipelineConfig(config_file=config_path)
        self.logger = logger  
        self.map_tree = str(self.config.get('fitting.map_tree', None))
        self.det_res = str(self.config.get('fitting.detector_response', None))
        self.logger.info(f"Map tree: {self.map_tree}")
        self.logger.info(f"Detector response: {self.det_res}")
        self.roi_ra = self.config.get('coordinates.ra', 0.0)
        self.roi_dec = self.config.get('coordinates.dec', 0.0)

        if self.roi_ra is None or self.roi_dec is None:
            l = self.config.get('coordinates.l', 0.0)
            b = self.config.get('coordinates.b', 0.0)
            skycoord = SkyCoord(l, b, frame='galactic', unit='deg')
            self.roi_ra = skycoord.icrs.ra.deg
            self.roi_dec = skycoord.icrs.dec.deg
            self.logger.info(f"Converted galactic coordinates (l={l}, b={b}) to equatorial (RA={self.roi_ra}, Dec={self.roi_dec})")
        self.roi_radius = 15
        self.roi_radius_model = 20
        self.roiThreshold = 0.5
        self.error_samples = 5000
        self.logger.info(f"Model file: {model}")

        namespace = {"threeML": threeML}
        exec(open(model).read(), namespace)
        if "model" not in namespace:
            raise ValueError("Model file did not define 'model'")
        self.model_obj = namespace["model"]

        self.errors = str(self.config.get('fitting.errors', 'all'))
        self.TS = self.config.get('fitting.TS', False)
        number_of_sources, RAs, Decs, source_name = self.get_source_centers(self.model_obj)
        self.save_dir = save_dir

        if roiTemplate is not None:
            self.logger.info(f"Using ROI template: {roiTemplate}")
            self.roi = HealpixMapROI(data_radius=self.roi_radius, model_radius=self.roi_radius_model, ra=self.roi_ra, dec=self.roi_dec, roifile=roiTemplate, threshold=self.roiThreshold)
        else:
            self.logger.info("No ROI template provided, computing ROI from provided ra and dec")
            self.logger.info(f'ROI center: {self.roi_ra}, {self.roi_dec}')
            self.test_roi(self.roi_ra, self.roi_dec, self.roi_radius_model, number_of_sources, RAs, Decs, source_name)
            self.roi = HealpixConeROI(data_radius=self.roi_radius,model_radius=self.roi_radius_model,ra=self.roi_ra,dec=self.roi_dec,)

        self.bin_list=self.config.get('fitting.bins', [])
        self.logger.info(f"Running threeML fit pipeline with bins: {self.bin_list}")
        self.hawc = HAL("HAWC",self.map_tree,self.det_res,self.roi, n_workers=4)#, bin_list=self.bin_list)
        self.hawc.set_active_measurements(bin_list=self.bin_list)
        self.datalist = DataList(self.hawc)
        self.jl = JointLikelihood(self.model_obj, self.datalist, verbose=True)
        # self.jl.set_minimizer("ROOT")
        self.jl.set_minimizer("minuit")

    def circleDist(self, RA1, DEC1, RA2, DEC2):
        c1 = SkyCoord(ra=RA1, dec=DEC1, unit="degree")
        c2 = SkyCoord(ra=RA2, dec=DEC2, unit="degree")
        return c1.separation(c2).to(threeML.u.degree).value

    def get_source_centers(self, model):
        RAs = []
        Decs = []
        source_name = []

        for name, source in model.sources.items():
            try:
                pos_ra = source.position.ra.value
                pos_dec = source.position.dec.value
            except:
                try:
                    pos_ra = source.spatial_shape.lon0.value
                    pos_dec = source.spatial_shape.lat0.value

                except:
                    continue

            RAs.append(pos_ra)
            Decs.append(pos_dec)
            source_name.append(name)

        number_of_sources = len(RAs)

        return number_of_sources, RAs, Decs, source_name

    def get_roi_from_sources(self, number_of_sources, RAs, Decs, options):
        if number_of_sources == 0:
            return None, None

        # if there is only 1 source in the model, use that position for the roi
        if number_of_sources == 1:
            roi_ra = RAs[0]
            roi_dec = Decs[0]
        else:
            c = SkyCoord(ra=RAs, dec=Decs, unit="degree", frame="icrs")
            r = np.array(c.cartesian)
            m = np.mean(r)
            center = SkyCoord(x=m.x, y=m.y, z=m.z, representation_type="cartesian")
            roi_ra = center.spherical.lon.to(threeML.u.degree).value
            roi_dec = center.spherical.lat.to(threeML.u.degree).value

        if not options.roiCenter:
            options.roiCenter = (roi_ra, roi_dec)
        return roi_ra, roi_dec

    def test_roi(self, roi_ra, roi_dec, roi_radius, number_of_sources, RAs, Decs, source_name):
        # Now we have the center compute distances to center

        print(source_name)
        print(roi_ra, roi_dec, roi_radius)

        diffs = np.zeros(number_of_sources)
        dist_to_center = np.zeros(number_of_sources)
        print(
            "Center of ROI  RA: {0:0.4f} Dec: {1:0.4f} Radius: {2:0.2f}".format(
                roi_ra, roi_dec, roi_radius
            )
        )
        print("Source         RA:          Dec:           Distance to center")
        for i in range(number_of_sources):
            dist_to_center[i] = self.circleDist(RAs[i], Decs[i], roi_ra, roi_dec)
            diffs[i] = dist_to_center[i]
            print(
                "{0}          {2:0.4f}    {3:0.4f}   {1:0.3f} degrees".format(
                    source_name[i], diffs[i], RAs[i], Decs[i]
                )
            )

        # if all sources are more than 1 degree away from the ROI edge, otherwise yell at the user
        if any(diffs) > roi_radius - 1.0:
            print("")
            warnings.warn(
                "There is a source less than a degree away from the ROI edge, make ROI radius larger",
                UserWarning,
            )
            print("")

        if any(diffs) > roi_radius:
            print("")
            warnings.warn("THERE ARE SOURCES OUTSIDE THE ROI", UserWarning)
            warnings.warn("THERE ARE SOURCES OUTSIDE THE ROI", UserWarning)
            print("")

    def hal_fit(self):
        silence_logs()
        self.logger.info("Running MLE without error estimation")
        self.params, self.statistics = self.jl.fit(compute_covariance=False, n_samples=self.error_samples, quiet=False)
        self.jl.results.display()
        self.jl.results.write_to(self.save_dir / "likelihoodResults.fits", overwrite=True)
        self.logger.info(f"Fit results saved to {self.save_dir / 'likelihoodResults.fits'}")

    def hal_fit_with_covariance(self):
        silence_logs()
        self.logger.info("Running MLE with error estimation")
        self.params, self.statistics = self.jl.fit(compute_covariance=True, n_samples=self.error_samples)
        self.jl.results.display()
        try:
            self.errAll = self.jl.get_errors()
        except threeML.minimizer.minimization.MINOSFailed as e:
            self.logger.warning(
                f"MINOS error estimation failed ({e}); keeping the already-converged fit "
                f"without asymmetric errors."
            )
            self.errAll = None
        self.jl.results.write_to("{0}/likelihoodResults.fits".format(self.save_dir), overwrite=True)
        self.logger.info(f"Fit results saved to {self.save_dir / 'likelihoodResults.fits'}")

    def get_TS(self):
        self.logger.info("Calculating TS for all sources in the model")
        ts_by_source = {}
        for source in self.model_obj.sources:
            self.logger.info(f"Computing TS for source: {source}")
            try:
                ts_val = self.jl.compute_TS(source, self.statistics)
                ts_scalar = float(ts_val.TS.iloc[0]) if hasattr(ts_val.TS, 'iloc') else float(ts_val.TS)
                self.logger.info(f"TS for source {source}: {ts_scalar}")
                ts_by_source[source] = ts_scalar
            except Exception as e:
                self.logger.info(f"Error computing TS for source {source}: {e}")
                ts_by_source[source] = 9999
        return ts_by_source

        
    def make_maps(self):
        self.logger.info("Saving HAL output maps")
        likelihood_object=self.hawc
        # To save in the correct format
        self.logger.info("SAVE A BIG MAP")
        new_ROI=HealpixConeROI(data_radius=self.roi_radius_model, model_radius=self.roi_radius_model+5, ra=self.roi_ra, dec=self.roi_dec)
        large_like = HAL("AD_HAWC", self.map_tree, self.det_res, new_ROI) #,n_transits)
        large_like.set_active_measurements(bin_list=self.bin_list)

        large_jl = threeML.JointLikelihood( self.model_obj, self.datalist)
        large_jl.set_minimizer("minuit")

        large_like.set_model(self.model_obj)
        large_like.get_log_like()

        likelihood_object=large_like

        self.logger.info("Writing model map...")
        likelihood_object.write_model_map(self.save_dir/  "model_fit.hd5")
        likelihood_object.write_residual_map(self.save_dir /  "residual_fit.hd5")

    def run(self):
        self.logger.info("Running threeML fit pipeline with bins:", self.bin_list)
        self.hal_fit()


