"""
Model Generation Utilities

Extracted from: pipeline_helpers.py, pipeline_sourcedetector.py
Generates threeML model files from detected sources.
"""

from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
import copy
import astromodels
from astromodels.core.my_yaml import my_yaml  # or whatever astromodels uses internally
import pandas as pd
import numpy as np
import astropy.units as u
import re
import yaml
try:
    import threeML
    from threeML import (
        PointSource,
        ExtendedSource,
        Powerlaw,
        Gaussian_on_sphere,
        Model,
    )
except ImportError:
    raise ImportError("threeML package required for model generation")

class ModelGenerator:
    """Generate threeML model files from source catalogs"""
    
    # Constants for morphology classification
    POINT_SOURCE_SIGMA_LIMIT = 0.12  # degrees
    
    @staticmethod
    def is_point_source(sigma_radius: float) -> bool:
        """Determine if source is point-like based on sigma radius
        
        Parameters:
        -----------
        sigma_radius : float
            Spatial extension (sigma) in degrees
        
        Returns:
        --------
        bool
            True if source is point-like
        """
        return sigma_radius < ModelGenerator.POINT_SOURCE_SIGMA_LIMIT
    
    @staticmethod
    def create_point_source(
        source_index: int,
        ra: float,
        dec: float,
        ra_bounds: Optional[Tuple[float, float]] = None,
        dec_bounds: Optional[Tuple[float, float]] = None,
        logger: Optional[object] = None
    ) -> Tuple[PointSource, Powerlaw]:
        """Create point source with spectral model
        
        Parameters:
        -----------
        source_index : int
            Source index (for naming)
        ra : float
            Right Ascension (degrees)
        dec : float
            Declination (degrees)
        ra_bounds : tuple, optional
            (ra_min, ra_max) bounds in degrees
        dec_bounds : tuple, optional
            (dec_min, dec_max) bounds in degrees
        logger : object, optional
            Logger instance
        
        Returns:
        --------
        tuple
            (source_object, spectrum_object)
        """
        # Create Powerlaw spectrum
        spectrum = Powerlaw()
        
        # Set spectral parameters
        flux_unit = 1.0 / (u.keV * u.cm**2 * u.s)
        spectrum.K = 1e-22 * flux_unit
        spectrum.K.free = True
        spectrum.K.bounds = (1e-26, 1e-20) * flux_unit
        
        spectrum.index = -2.5
        spectrum.index.free = True
        spectrum.index.bounds = (-3.0, -1.0)
        
        spectrum.piv = 2.0 * u.TeV
        spectrum.piv.free = False
        
        # Create point source
        source = PointSource(
            f"Source{source_index}",
            ra=ra,
            dec=dec,
            spectral_shape=spectrum
        )
        
        # Set position bounds
        if ra_bounds is None:
            ra_bounds = (ra - 1.0, ra + 1.0)
        if dec_bounds is None:
            dec_bounds = (dec - 1.0, dec + 1.0)
        
        source.position.ra.free = True
        source.position.ra.bounds = tuple(b * u.degree for b in ra_bounds)
        source.position.dec.free = True
        source.position.dec.bounds = tuple(b * u.degree for b in dec_bounds)
        
        if logger:
            logger.debug(f"Created PointSource{source_index} at RA={ra}, Dec={dec}")
        
        return source, spectrum
    
    @staticmethod
    def create_extended_source(
        source_index: int,
        ra: float,
        dec: float,
        sigma_radius: float,
        ra_bounds: Optional[Tuple[float, float]] = None,
        dec_bounds: Optional[Tuple[float, float]] = None,
        sigma_bounds: Optional[Tuple[float, float]] = None,
        logger: Optional[object] = None
    ) -> Tuple[ExtendedSource, Powerlaw]:
        """Create extended source with Gaussian morphology
        
        Parameters:
        -----------
        source_index : int
            Source index (for naming)
        ra : float
            Right Ascension (degrees)
        dec : float
            Declination (degrees)
        sigma_radius : float
            Spatial extension (sigma, degrees)
        ra_bounds : tuple, optional
            (ra_min, ra_max) bounds in degrees
        dec_bounds : tuple, optional
            (dec_min, dec_max) bounds in degrees
        sigma_bounds : tuple, optional
            (sigma_min, sigma_max) bounds in degrees
        logger : object, optional
            Logger instance
        
        Returns:
        --------
        tuple
            (source_object, spectrum_object)
        """
        # Create Powerlaw spectrum
        spectrum = Powerlaw()
        
        # Set spectral parameters
        flux_unit = 1.0 / (u.keV * u.cm**2 * u.s)
        spectrum.K = 1e-22 * flux_unit
        spectrum.K.free = True
        spectrum.K.bounds = (1e-26, 1e-20) * flux_unit
        
        spectrum.index = -2.5
        spectrum.index.free = True
        spectrum.index.bounds = (-3.0, -1.0)
        
        spectrum.piv = 2.0 * u.TeV
        spectrum.piv.free = False
        
        # Create Gaussian spatial shape
        morphology = Gaussian_on_sphere()
        
        morphology.lon0 = ra * u.degree
        morphology.lon0.free = True
        if ra_bounds is None:
            ra_bounds = (ra - 1.0, ra + 1.0)
        morphology.lon0.bounds = tuple(b * u.degree for b in ra_bounds)
        
        morphology.lat0 = dec * u.degree
        morphology.lat0.free = True
        if dec_bounds is None:
            dec_bounds = (dec - 1.0, dec + 1.0)
        morphology.lat0.bounds = tuple(b * u.degree for b in dec_bounds)
        
        morphology.sigma = sigma_radius * u.degree
        morphology.sigma.free = True
        if sigma_bounds is None:
            sigma_bounds = (0.01, 4.0)
        morphology.sigma.bounds = tuple(b * u.degree for b in sigma_bounds)
        
        # Create extended source
        source = ExtendedSource(
            f"Source{source_index}",
            spatial_shape=morphology,
            spectral_shape=spectrum
        )
        
        if logger:
            logger.debug(
                f"Created ExtendedSource{source_index} at RA={ra}, Dec={dec}, "
                f"sigma={sigma_radius} deg"
            )
        
        return source, spectrum
    
    @staticmethod
    def create_model_from_sources(
        source_dataframe: pd.DataFrame,
        logger: Optional[object] = None
    ) -> Tuple[Model, Dict]:
        """Create threeML model from source DataFrame
        
        Parameters:
        -----------
        source_dataframe : pd.DataFrame
            DataFrame with columns: 'ra', 'dec', 'Sigma Radius', etc.
        logger : object, optional
            Logger instance
        
        Returns:
        --------
        tuple
            (model, source_dict)
            - model: threeML Model object
            - source_dict: dictionary of created sources
        
        Examples:
        ---------
        >>> df = pd.DataFrame({
        ...     'ra': [83.5, 88.0],
        ...     'dec': [22.0, 25.0],
        ...     'Sigma Radius': [0.1, 0.15]
        ... })
        >>> model, sources = ModelGenerator.create_model_from_sources(df)
        """
        sources = {}
        
        for i, row in source_dataframe.iterrows():
            ra = row['ra']
            dec = row['dec']
            sigma_radius = row.get('Sigma Radius', 0.0)
            
            # Determine morphology
            if ModelGenerator.is_point_source(sigma_radius):
                source, spectrum = ModelGenerator.create_point_source(
                    i, ra, dec, logger=logger
                )
            else:
                source, spectrum = ModelGenerator.create_extended_source(
                    i, ra, dec, sigma_radius, logger=logger
                )
            
            sources[f"source{i}"] = source
        
        if not sources:
            raise ValueError("No sources to create model")
        
        # Create model from all sources
        model = Model(*sources.values())
        
        if logger:
            logger.info(f"Created threeML model with {len(sources)} sources")
        
        return model, sources
    
    @staticmethod
    def write_model_file(
        model: Model,
        output_file: str,
        logger: Optional[object] = None
    ) -> Optional[Path]:
        """Write threeML model to Python file
        
        Parameters:
        -----------
        model : Model
            threeML Model object
        output_file : str
            Output Python filename
        logger : object, optional
            Logger instance
        
        Returns:
        --------
        Path or None
            Path to written file, or None if failed
        """
        try:
            output_path = Path(output_file)
            
            with open(output_path, 'w') as f:
                f.write("#!/usr/bin/env python\n")
                f.write("# Generated threeML model file\n")
                f.write("# Do not edit manually\n\n")
                f.write("from threeML import *\n\n")
                f.write(f"model = {repr(model)}\n")
            
            if logger:
                logger.info(f"Wrote model file: {output_file}")
            
            return output_path
        
        except Exception as e:
            if logger:
                logger.error(f"Failed to write model file: {e}")
            return None
    
    @staticmethod
    def write_model_string(
        model: Model,
        logger: Optional[object] = None
    ) -> str:
        """Get model as Python code string
        
        Parameters:
        -----------
        model : Model
            threeML Model object
        logger : object, optional
            Logger instance
        
        Returns:
        --------
        str
            Python code string defining the model
        """
        code = (
            "#!/usr/bin/env python\n"
            "# Generated threeML model file\n"
            "# Do not edit manually\n\n"
            "from threeML import *\n\n"
            f"model = {repr(model)}\n"
        )

        return code

    DEFAULT_SPATIAL_PARAMS = {
        'Gaussian_on_sphere': {'sigma': (0.3, 0.01, 4.0)},
        'Disk_on_sphere': {'radius': (0.3, 0.01, 4.0)},
        'Ellipse_on_sphere': {'a': (0.3, 0.01, 4.0), 'e': (0.5, 0.0, 0.99), 'theta': (0.0, -90.0, 90.0)},
    }
    DEFAULT_SPECTRUM_PARAMS = {
        'Log_parabola': {'K': (1e-23,1e-29,1e-19), 'alpha': (-2.5, -4.0, -1.0), 'beta': (0.1, -1.0, 1.0)},
        'Cutoff_powerlaw': {'K': (1e-23,1e-29,1e-19), 'xc': (10000000000.0 , 1000000000.0 , 200000000000.0 )},
    }

    @staticmethod
    def _param_lines(var_expr: str, param_name: str, param) -> List[str]:
        """Emit `var.param = ...` / `.free = ...` / `.bounds = ...` source lines
        for one live astromodels Parameter, reconstructing its unit from
        `str(param.unit)` (round-trippable via `u.Unit(...)`)."""
        unit = getattr(param, 'unit', None)
        unit_str = str(unit) if unit is not None else ''
        has_unit = bool(unit_str) and unit_str not in ('', 'dimensionless')

        def fmt(v):
            v = float(v)
            return f'{v!r} * u.Unit("{unit_str}")' if has_unit else f'{v!r}'

        lines = [
            f'{var_expr}.{param_name} = {fmt(param.value)}',
            f'{var_expr}.{param_name}.free = {bool(param.free)}',
        ]
        if param.min_value is not None and param.max_value is not None:
            lo, hi = param.min_value, param.max_value
            if has_unit:
                lines.append(f'{var_expr}.{param_name}.bounds = ({lo!r}, {hi!r}) * u.Unit("{unit_str}")')
            else:
                lines.append(f'{var_expr}.{param_name}.bounds = ({lo!r}, {hi!r})')
        return lines

    @staticmethod
    def write_model_from_live(
        model: Model,
        output_path: str,
        logger: Optional[object] = None
    ) -> Path:
        """Serialize a live, fitted threeML Model to an executable .model file.

        Generic over source/shape type (walks each source's live Parameter
        objects) rather than templated per shape, so it survives extension/
        spectrum-test swaps that create shapes create_point_source /
        create_extended_source don't know about. Compatible with
        threeMLFit's loader (pipeline_fitmodel.py exec()s this file expecting
        a `threeML` name and a top-level `model` assignment).
        """
        lines = [
            "#!/usr/bin/env python",
            "# Generated threeML model file (live-model writer)",
            "# Do not edit manually",
            "",
            "import threeML",
            "import astropy.units as u",
            "",
        ]
        source_vars = []
        for i, (name, source) in enumerate(model.sources.items()):
            var = f"src{i}"
            source_vars.append(var)
            spectrum_shape = source.spectrum.main.shape
            spec_class = type(spectrum_shape).__name__
            spec_var = f"{var}_spec"
            lines.append(f'{spec_var} = threeML.{spec_class}()')

            if hasattr(source, 'position'):
                ra_val = float(source.position.ra.value)
                dec_val = float(source.position.dec.value)
                lines.append(
                    f'{var} = threeML.PointSource("{name}", ra={ra_val!r}, dec={dec_val!r}, '
                    f'spectral_shape={spec_var})'
                )
                for pname, param in spectrum_shape.parameters.items():
                    lines += ModelGenerator._param_lines(spec_var, pname, param)
                lines += ModelGenerator._param_lines(f'{var}.position', 'ra', source.position.ra)
                lines += ModelGenerator._param_lines(f'{var}.position', 'dec', source.position.dec)
            else:
                spatial_shape = source.spatial_shape
                spat_class = type(spatial_shape).__name__
                spat_var = f"{var}_shape"
                lines.append(f'{spat_var} = threeML.{spat_class}()')
                lines.append(
                    f'{var} = threeML.ExtendedSource("{name}", spatial_shape={spat_var}, '
                    f'spectral_shape={spec_var})'
                )
                for pname, param in spectrum_shape.parameters.items():
                    lines += ModelGenerator._param_lines(spec_var, pname, param)
                for pname, param in spatial_shape.parameters.items():
                    lines += ModelGenerator._param_lines(spat_var, pname, param)
            lines.append("")

        lines.append(f"model = threeML.Model({', '.join(source_vars)})")
        lines.append("")

        output_path = Path(output_path)
        output_path.write_text("\n".join(lines))
        if logger:
            logger.info(f"Wrote live model file: {output_path}")
        return output_path

    @staticmethod
    def write_model_file_from_yaml(yaml_file_loc: str, model_file_loc: str, logger: Optional[object] = None) -> None:
        _name_type_re = re.compile(r'^(\S+)\s*\((\w+)\s*source\)$')
        with open(yaml_file_loc) as f:
            model_dict = yaml.safe_load(f)
        begin_source_marker = '#----------BEGIN_SOURCE----------#\n'
        end_source_marker   = '#-----------END_SOURCE-----------#\n'

        source_names = []
        with open(model_file_loc, 'w') as model_file:
            for key, body in model_dict.items():
                m = _name_type_re.match(key.strip())
                logger.debug(f"Parsing source header: {key!r} -> {m.groups() if m else None}")
                if key.strip() == 'HAWC_bkg_renorm (Parameter)':
                    logger.info(f"Skipping background source {key!r}")
                    continue
                if not m:
                    raise ValueError(f"Can't parse source header: {key!r}")
                source, src_kind = m.group(1), m.group(2)  # 'point' or 'extended'
                source_names.append(source)

                model_file.write(f'{begin_source_marker}\n')
                model_file.write(f"source_name = '{source}'\n\n")

                spec_block = body['spectrum']['main']
                spec_type = next(k for k in spec_block if k != 'polarization')
                spec_params = spec_block[spec_type]

                model_file.write(f'spectrum = threeML.{spec_type}()\n')

                if src_kind == 'point':
                    pos = body['position']
                    lon_key, lat_key = ('ra', 'dec') if 'ra' in pos else ('lon0', 'lat0')
                    lon_val, lat_val = pos[lon_key]['value'], pos[lat_key]['value']
                    model_file.write(
                        f'{source} = threeML.PointSource(source_name, ra={lon_val}, dec={lat_val}, '
                        f'spectral_shape=spectrum)\n'
                    )
                else:
                    shape_type = next(k for k in body if k != 'spectrum')
                    shape_params = body[shape_type]

                    if shape_type == 'Hermes':
                        fits_file = shape_params['fits_file']['value']
                        ihdu = shape_params['ihdu']['value']
                        model_file.write(f"shape = threeML.Hermes(fits_file='{fits_file}', ihdu={ihdu})\n")
                    else:
                        model_file.write(f'shape = threeML.{shape_type}()\n')

                    model_file.write(
                        f'{source} = threeML.ExtendedSource(source_name, spatial_shape=shape, '
                        f'spectral_shape=spectrum)\n'
                    )

                model_file.write('fluxUnit = 1. / (threeML.u.keV * threeML.u.cm ** 2 * threeML.u.s)\n')

                for pname, p in spec_params.items():
                    unit_mult = '* fluxUnit' if pname == 'K' else ''
                    model_file.writelines([
                        '\n',
                        f'spectrum.{pname} = {p["value"]} {unit_mult}\n',
                        f'spectrum.{pname}.fix = {not p["free"]}\n',
                        f'spectrum.{pname}.bounds = ({p["min_value"]}, {p["max_value"]}) {unit_mult}\n',
                    ])

                if src_kind == 'point':
                    for pname, p in pos.items():
                        if pname == 'equinox':
                            continue
                        # logger.info(f"Writing position parameter {pname} and param {p} for source {source}")
                        unit_mult = '* threeML.u.degree' if pname in (lon_key, lat_key) else ''
                        model_file.writelines([
                            '\n',
                            f'{source}.position.{pname}.bounds = ({p["min_value"]}, {p["max_value"]}) {unit_mult}\n',
                            f'{source}.position.{pname}.free = {p["free"]}\n',
                        ])
                else:
                    for pname, p in shape_params.items():
                        if pname in ('hash', 'ihdu', 'fits_file', 'frame'):
                            continue
                        unit_mult = '* threeML.u.degree' if pname in ('ra', 'dec', 'lon0', 'lat0') else ''
                        model_file.writelines([
                            '\n',
                            f'shape.{pname} = {p["value"]} {unit_mult}\n',
                            f'shape.{pname}.fix = {not p["free"]}\n',
                            f'shape.{pname}.bounds = ({p["min_value"]}, {p["max_value"]}) {unit_mult}\n',
                        ])

                model_file.write(f'\n{end_source_marker}\n')

            model_file.write('model = threeML.Model(' + ', '.join(source_names) + ')\n')

    @staticmethod
    def set_free(model: Model, source_names: List[str], kind: str, free: bool,
                free_diffuse: bool, param_names: Optional[List[str]] = None,
                logger: Optional[object] = None) -> None:
        for name in source_names:
            source = model.sources[name]
            logger.debug(f"Checking source {source.name} for extension test")

            if kind == 'spectral':
                logger.info("TESTING SPECTRAL")
                if name == 'URM':
                    logger.debug(f"Skipping spectral params for {name}: URM spectrum is always fixed")
                    continue
                params = source.spectrum.main.shape.parameters
            # else:
            #     params = source.spectrum.main.shape.parameters
                
            elif kind == 'spatial':
                if name == 'URM':
                    n_param = source.spatial_shape.parameters['N']
                    logger.debug(f"Setting diffuse source {name} N.free={free_diffuse}")
                    n_param.free = free_diffuse
                    continue 
                if hasattr(source, 'position'):
                    params = {'ra': source.position.ra, 'dec': source.position.dec}
                else:
                    params = source.spatial_shape.parameters
            else:
                raise ValueError(f"Unknown kind: {kind}")

            for pname, param in params.items():
                if param_names is None or pname in param_names:
                    logger.info(f"SOURCE {name} : param {pname} -> free={free}")
                    param.free = free
                if param_names is not None and pname not in param_names:
                    logger.info(f"SOURCE {name} : param {pname} -> FIXED (not in param_names)")
                    param.free = False

    from astromodels.functions.function import FunctionInstanceError
    @staticmethod
    def _clone_shape(shape):
        """Reconstruct a spatial/spectral shape from its parameter values,
        rather than copy.deepcopy(shape). Some shapes (e.g. those bound to a
        fitted HAL model) hold live PSF/convolution state that does not survive
        pickle-based deepcopy -- astromodels' __deepcopy__ round-trips via
        cPickle, and unpickled PSFWrapper objects can come back marked invalid
        (hawc_hal.psf_fast.psf_wrapper.InvalidPSFError). Reconstructing by value
        sidesteps that entirely.
        """
        shape_cls = type(shape)

        if shape_cls.__name__ == 'Hermes':
            new_shape = shape_cls(fits_file=shape.fits_file.value, ihdu=shape.ihdu.value)
            for pname, param in shape.parameters.items():
                target = getattr(new_shape, pname, None)
                if target is None:
                    continue
                target.value = param.value
                target.free = param.free
                if param.min_value is not None and param.max_value is not None:
                    target.bounds = (param.min_value, param.max_value)
            return new_shape

        try:
            new_shape = shape_cls()
        except (TypeError, FunctionInstanceError):
            return copy.deepcopy(shape)

        for pname, param in shape.parameters.items():
            target = getattr(new_shape, pname, None)
            if target is None:
                continue
            target.value = param.value
            target.free = param.free
            if param.min_value is not None and param.max_value is not None:
                target.bounds = (param.min_value, param.max_value)
        return new_shape

    @staticmethod
    def remove_sources(model: Model, exclude_names: List[str], logger: Optional[object] = None) -> Model:
        """Return a new Model with `exclude_names` dropped; every remaining
        source is cloned unchanged."""
        kept = [ModelGenerator._clone_source(s, logger) for n, s in model.sources.items() if n not in exclude_names]
        if not kept:
            raise ValueError("Removing these sources would leave an empty model")
        return Model(*kept)

    @staticmethod
    def swap_spatial_shape(
        model: Model,
        source_name: str,
        new_shape_name: str,
        coord_range: float = 1.0,
        logger: Optional[object] = None
    ) -> Model:
        """Return a new Model with `source_name` rebuilt as an ExtendedSource
        using `new_shape_name` (e.g. 'Disk_on_sphere'), carrying over its
        current fitted position and spectrum. Other sources are cloned
        unchanged. Defaults for the new shape's own parameters (radius,
        sigma, ...) come from DEFAULT_SPATIAL_PARAMS.
        """
        # print(model.sources.keys())
        source = model.sources[source_name]
        current_spatial_model = list(source._children.keys())[0]
        if current_spatial_model == "position":
            current_spatial_model = "Point_Source"
        if current_spatial_model == new_shape_name:
            logger.info(f"Source {source_name} already has spatial shape {new_shape_name}, skipping swap")
            return model
        logger.info(f"Spatial shape of source {current_spatial_model}, Alternate model {(new_shape_name)}")
        if hasattr(source, 'position'):
            ra, dec = source.position.ra.value, source.position.dec.value
        else:
            ra, dec = source.spatial_shape.lon0.value, source.spatial_shape.lat0.value

        logger.debug(f"Swapping {source_name} spatial shape from {current_spatial_model} to {new_shape_name}")
        spectrum = ModelGenerator._clone_shape(source.spectrum.main.shape)
        shape_cls = getattr(threeML, new_shape_name)
        new_shape = shape_cls()
        new_shape.lon0 = ra
        new_shape.lon0.free = False
        new_shape.lon0.bounds = (ra - coord_range, ra + coord_range)
        # new_shape.lat0 = dec * u.degree
        new_shape.lat0 = dec
        new_shape.lat0.free = False
        new_shape.lat0.bounds = (dec - coord_range, dec + coord_range)
        for pname, (val, lo, hi) in ModelGenerator.DEFAULT_SPATIAL_PARAMS.get(new_shape_name, {}).items():
            if not hasattr(new_shape, pname):
                continue
            getattr(new_shape, pname).value = val
            getattr(new_shape, pname).free = True
            getattr(new_shape, pname).bounds = (lo, hi)
        new_source = ExtendedSource(source_name, spatial_shape=new_shape, spectral_shape=spectrum)
        other_sources = [ModelGenerator._clone_source(s, logger) for n, s in model.sources.items() if n != source_name]
        
        if logger:
            logger.debug(f"Swapped {source_name} spatial shape -> {new_shape_name}")
        return Model(*other_sources, new_source)


    @staticmethod
    def swap_spectral_shape(
        model: Model,
        source_name: str,
        new_shape_name: str,
        logger: Optional[object] = None
    ) -> Model:
        """Return a new Model with `source_name`'s spectrum rebuilt using
        `new_shape_name` (e.g. 'LogParabola'), carrying over its current
        fitted position/morphology. Other sources are cloned unchanged.
        """
        source = model.sources[source_name]
        spectrum_cls = getattr(threeML, new_shape_name)
        new_spectrum = spectrum_cls()
        logger.info(f"Swapping {source_name} spectral shape from {type(source.spectrum.main.shape).__name__} to {new_shape_name}")
        logger.info(f"New spectrum class: {new_spectrum._children.keys()}")
        new_spectrum.piv = 2000000000.0  # 2 TeV in eV
        
        if hasattr(source, 'position'):
            new_source = PointSource(
                source_name, ra=source.position.ra.value, dec=source.position.dec.value,
                spectral_shape=new_spectrum,
            )
            new_source.position.ra.free = source.position.ra.free
            new_source.position.ra.bounds = (source.position.ra.min_value, source.position.ra.max_value)
            new_source.position.dec.free = source.position.dec.free
            new_source.position.dec.bounds = (source.position.dec.min_value, source.position.dec.max_value)
        else:
            new_spatial = ModelGenerator._clone_shape(source.spatial_shape)
            new_source = ExtendedSource(source_name, spatial_shape=new_spatial, spectral_shape=new_spectrum)

        other_sources = [ModelGenerator._clone_source(s, logger) for n, s in model.sources.items() if n != source_name]
        for pname, (val, lo, hi) in ModelGenerator.DEFAULT_SPECTRUM_PARAMS.get(new_shape_name, {}).items():
            if not hasattr(new_spectrum, pname):
                continue
            getattr(new_spectrum, pname).value = val
            getattr(new_spectrum, pname).free = True
            getattr(new_spectrum, pname).bounds = (lo, hi)
        if logger:
            logger.info(f"Swapped {source_name} spectral shape -> {new_shape_name}")
            logger.debug(f"New source: {new_source}")
        return Model(*other_sources, new_source)

    @staticmethod
    def _clone_source(source, logger: Optional[object] = None, new_name: Optional[str] = None):
        """Deep-copy a live Source (point or extended) by value, for reuse
        in a rebuilt Model alongside a swapped sibling source. Pass new_name
        to reconstruct under a different name (e.g. to avoid a collision when
        merging sources from a separate model)."""
        name = new_name if new_name is not None else source.name
        spectrum = ModelGenerator._clone_shape(source.spectrum.main.shape)
        if hasattr(source, 'position'):
            logger.debug(f"Source {source.name} is a PointSource with position RA={source.position.ra.value}, Dec={source.position.dec.value}")
            new_source = PointSource(
                name, ra=source.position.ra.value, dec=source.position.dec.value,
                spectral_shape=spectrum,
            )
            new_source.position.ra.free = source.position.ra.free
            new_source.position.dec.free = source.position.dec.free
            if source.position.ra.min_value is not None:
                new_source.position.ra.bounds = (source.position.ra.min_value, source.position.ra.max_value)
            if source.position.dec.min_value is not None:
                new_source.position.dec.bounds = (source.position.dec.min_value, source.position.dec.max_value)
        elif hasattr(source, 'spatial_shape'):
            logger.debug(f"Source {source.name} is an ExtendedSource with spatial shape {source.spatial_shape.__class__.__name__}")
            spatial = ModelGenerator._clone_shape(source.spatial_shape)
            new_source = ExtendedSource(name, spatial_shape=spatial, spectral_shape=spectrum)
        return new_source


    @staticmethod
    def merge_new_sources(base_model: Model, new_sources_model_path: str,
                        dedup_radius_deg: float = 0.1, logger: Optional[object] = None) -> Model:
        """Load new_sources_model_path (e.g. from a DRIPS reseed run on a
        residual map) and add any sources it defines that aren't already
        present in base_model, by position (both models reuse generic names
        like 'Source0', so name comparison isn't meaningful). New sources are
        renamed 'ReseedSource{n}' to avoid colliding with base_model's names.
        Existing base_model sources are cloned unchanged. URM is never
        compared/duplicated -- there is at most one diffuse template per model.
        """
        namespace = {'threeML': threeML}
        exec(open(new_sources_model_path).read(), namespace)
        if 'model' not in namespace:
            raise ValueError(f"{new_sources_model_path} did not define 'model'")
        candidate_model = namespace['model']

        def _position(source):
            if hasattr(source, 'position'):
                return source.position.ra.value, source.position.dec.value
            return source.spatial_shape.lon0.value, source.spatial_shape.lat0.value

        existing_positions = [
            _position(s) for name, s in base_model.sources.items() if name != 'URM'
        ]
        kept = [ModelGenerator._clone_source(s, logger) for s in base_model.sources.values()]
        existing_names = set(base_model.sources.keys())
        next_index = 0
        added = 0

        for cname, csource in candidate_model.sources.items():
            if cname == 'URM':
                logger.info('Skipping URM in candidate model (diffuse template is never merged as a new source)')
                continue

            cra, cdec = _position(csource)

            is_duplicate = any(
                ((cra - era) ** 2 + (cdec - edec) ** 2) ** 0.5 < dedup_radius_deg
                for era, edec in existing_positions
            )
            if is_duplicate:
                logger.info(f'{cname} at ({cra:.3f}, {cdec:.3f}) matches an existing source within {dedup_radius_deg} deg; skipping as duplicate')
                continue

            new_name = f'ReseedSource{next_index}'
            while new_name in existing_names:
                next_index += 1
                new_name = f'ReseedSource{next_index}'
            existing_names.add(new_name)
            next_index += 1

            kept.append(ModelGenerator._clone_source(csource, logger, new_name=new_name))
            existing_positions.append((cra, cdec))
            added += 1
            logger.info(f'Added new source {new_name} (from {cname}) at ({cra:.3f}, {cdec:.3f})')

        logger.info(f'merge_new_sources: {added} new source(s) added, {len(candidate_model.sources) - added - (1 if "URM" in candidate_model.sources else 0)} skipped as duplicates')
        return Model(*kept)