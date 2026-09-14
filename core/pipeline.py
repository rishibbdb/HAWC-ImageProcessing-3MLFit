"""
HAWC Analysis Pipeline Orchestrator (TASKS.md Task 5)

Ties core/ and seeding/ together, config-driven off the single unified
config.yaml (see RULES.md / CLAUDE.md for the config-unification override):

- Builds the significance map if requested (coordinates.create_sig_map).
- Runs the seeding method selected by `fitting_procedure` ('Drips' or 'Alps').
- If DRIPS and not coordinates.generate_seed_only: hands DRIPS's seed model to
  seeding.source_fitter.run() for one in-process joint fit, then (per config)
  extension test / spectrum test / final refit -- see
  seeding/source_fitter.py. This replaces ALPS's iterative hotspot-search loop
  for the DRIPS path; ALPS's own native hotspot search (`fitting_procedure:
  'Alps'`) is unchanged, see _run_alps().
- Persists results into the DirectoryManager tree and checkpoints each phase.

Reference: main.py::SourceSearchPipeline shows the origin step/dir structure
and the seed->fit->residual->model-map flow; this reuses that shape with
core/seeding equivalents swapped in for the subprocess-based origin pieces.
"""

from pathlib import Path
from typing import Optional, Union
from astropy.coordinates import SkyCoord
from core.config import ConfigManager
from core.logger import PipelineLogger
from core.checkpoint import CheckpointManager
from core.directory_manager import DirectoryManager
from core.map_tools import MapGenerator
from seeding.base import SeedingOutput
from seeding.drips_seeder import DRIPSSeeder
from seeding.alps_seeder import ALPSSeeder
import core.source_fitter as source_fitter
from core.fit_runner import FitResult  # add this import

class HAWCAnalysisPipeline:
    """Config-driven orchestrator: builds sig map (if needed), seeds, fits.

    Parameters:
    -----------
    config : str or ConfigManager
        Path to the unified pipeline config YAML, or an already-constructed
        ConfigManager.
    resume : bool
        If True, attempt to resume from a previous run if possible.
    """

    def __init__(self, config: Union[str, ConfigManager], resume: bool = False):
        self.config = config if isinstance(config, ConfigManager) else ConfigManager(str(config))
        self.resume = resume

        self.method = self.config.get('fitting_procedure', 'Drips')
        if self.method not in ('Drips', 'Alps'):
            raise ValueError(f"fitting_procedure must be 'Drips' or 'Alps', got: {self.method}")

        fit_name = self.config.get('fitting.fit_name')
        output_dir = self.config.get('fitting.output_dir')
        if not fit_name or not output_dir:
            raise ValueError("Config must set fitting.fit_name and fitting.output_dir")

        self.directory_manager = DirectoryManager(output_dir, fit_name)
        self.directory_manager.create_structure()
        
        self.logger = PipelineLogger(
            str(self.directory_manager.get_logs_dir()),
            self.config.get('alps.logging_level', 'INFO'),
        )
        self.checkpoint = CheckpointManager(str(self.directory_manager.get_root_dir() / 'checkpoints'))

        self.ra = self.config.get('coordinates.ra')
        self.dec = self.config.get('coordinates.dec')
        if self.ra is None or self.dec is None:
            l = self.config.get('coordinates.l')
            b = self.config.get('coordinates.b')
            if l is None or b is None:
                raise ValueError("Config must set either (ra, dec) or (l, b) for the ROI center")
            c = SkyCoord(l, b, frame='galactic', unit='deg')
            self.ra = float(c.icrs.ra.deg)
            self.dec = float(c.icrs.dec.deg)
            self.logger.info(f"Converted galactic coordinates (l={l}, b={b}) to equatorial (RA={self.ra}, Dec={self.dec})")
            self.config.set('coordinates.ra', self.ra)
            self.config.set('coordinates.dec', self.dec)
            # threeMLFit/FitRunner re-read the config from disk by path (they
            # don't share this in-memory ConfigManager), so the resolved
            # ra/dec need to land on disk for them to see it. Write a
            # resolved copy rather than overwriting config.yaml in place --
            # ConfigManager.set(..., save=True) does a plain yaml.safe_dump,
            # which would strip config.yaml's inline comments.
            resolved_path = self.directory_manager.get_root_dir() / 'resolved_config.yaml'
            self.config.config_file = resolved_path
            self.config.set('coordinates.ra', self.ra, save=True)
            self.config.set('coordinates.dec', self.dec, save=True)


    def _build_significance_map(self) -> Optional[Path]:
        """Build the significance map from count maps if coordinates.create_sig_map."""
        if not self.config.get('coordinates.create_sig_map', False):
            return None
        sig_map_path_cfe = self.config.get('coordinates.sig_map_path')
        if sig_map_path_cfe:
            sig_map_path = Path(sig_map_path_cfe)
        else:
            sig_map_path = self.directory_manager.get_datamap_dir() / "significance_map.fits"
            self.logger.info(f"Significance map already exists at {sig_map_path}, skipping generation")
            self.config.set("coordinates.sig_map_path", str(sig_map_path))
        if sig_map_path.exists():
            self.logger.info(f"Significance map already exists at {sig_map_path}, skipping generation")
            return sig_map_path
        
        self.checkpoint.save_step('build_significance_map', 0, 'running', {})
        count_map_dir = self.config.get('coordinates.count_map_dir')
        image_bins = self.config.get('coordinates.image_bins')
        detector_response = self.config.get('coordinates.detector_response')

        fits_mapping = MapGenerator.find_fits_files_by_bins(count_map_dir, image_bins, logger=self.logger)
        if not fits_mapping:
            self.checkpoint.save_step('build_significance_map', 0, 'failed', {'error': 'no count-map FITS files found'})
            raise RuntimeError(f"No count-map FITS files found in {count_map_dir} for bins {image_bins}")
        ra = self.config.get('coordinates.ra')
        dec = self.config.get('coordinates.dec')
        if ra is None or dec is None:
            l = self.config.get('coordinates.l')
            b = self.config.get('coordinates.b')
            skycoord = MapGenerator.convert_galactic_to_equatorial(l, b)
            ra = skycoord.ra.deg
            dec = skycoord.dec.deg
            self.logger.info(f"Converted galactic coordinates (l={l}, b={b}) to equatorial (RA={ra}, Dec={dec})")
            self.config.set('coordinates.ra', ra)
            self.config.set('coordinates.dec', dec) 
        
        output_path = MapGenerator.create_healpix_map(
            input_fits_files=list(fits_mapping.values()),
            energy_bins=list(fits_mapping.keys()),
            detector_response=detector_response,
            ra_center=float(self.config.get('coordinates.ra')),
            dec_center=float(self.config.get('coordinates.dec')),
            roi_x=float(self.config.get('coordinates.roi_y', 4.0)*3),
            roi_y=float(self.config.get('coordinates.roi_y', 4.0)*3),
            output_file=str(sig_map_path),
            logger=self.logger,
            pixi_manifest_path=self.config.get('alps.pixi_aerie_folder'),
        )
        if output_path is None:
            self.checkpoint.save_step('build_significance_map', 0, 'failed', {'error': 'create_healpix_map returned None'})
            raise RuntimeError("Significance map generation failed (see log)")

        self.checkpoint.save_step('build_significance_map', 0, 'completed', {'sig_map_path': str(output_path)})
        return output_path

    def _run_alps(self) -> SeedingOutput:
        """Run ALPS's own native hotspot-driven point-source search + fitting."""
        self.checkpoint.save_step('alps_seeding', 0, 'running', {})
        seeder = ALPSSeeder(config=self.config)
        output = seeder.run()
        self.checkpoint.save_step(
            'alps_seeding', 0, 'completed', output.to_dict(),
            metadata={'num_sources': output.num_sources},
        )
        return output

    def _run_drips(self) -> Union[SeedingOutput, FitResult]:
        """Run DRIPS detection, then (unless seed-only) hand off to
        source_fitter for the in-process joint fit + test/refit phases.
        Returns the DRIPS model file Path if generate_seed_only is set,
        else the final FitResult from source_fitter.run()."""
        step_dir = self.directory_manager.get_step_results_dir('Step0-Allpoint-sources')
        # self.checkpoint.save_step('drips_seeding', 0, 'running', {})
        # seeder = DRIPSSeeder(self.config, self.logger, self.directory_manager, step_path=str(step_dir))
        # drips_output = seeder.run()  # Path to the generated .model file
        # self.logger.info(f"DRIPS seeding completed: model written to {drips_output}")
        # self.checkpoint.save_step(
        #     'drips_seeding', 0, 'completed', {'model_path': str(drips_output)},
        # )
        if self.resume:
            self.logger.info('resume=True: skipping DRIPS re-seeding, using existing model file')
            drips_output = step_dir / 'curModel.model'
            if not drips_output.exists():
                raise FileNotFoundError(
                    f'--resume was passed but no existing seed model found at {drips_output}; '
                    f'cannot resume without a prior drips_seeding run'
                )
        else:
            self.checkpoint.save_step('drips_seeding', 0, 'running', {})
            seeder = DRIPSSeeder(self.config, self.logger, self.directory_manager, step_path=str(step_dir))
            drips_output = seeder.run()
            self.logger.info(f"DRIPS seeding completed: model written to {drips_output}")
            self.checkpoint.save_step(
                'drips_seeding', 0, 'completed', {'model_path': str(drips_output)},
            )

        if self.config.get('coordinates.generate_seed_only', False):
            self.logger.info("coordinates.generate_seed_only is True; returning DRIPS detection output without fitting")
            return drips_output

        self.checkpoint.save_step('drips_fit', 0, 'running', {})

        # fit_output = source_fitter.run(drips_output, self.config, self.logger, self.directory_manager, self.checkpoint)
        # num_sources = len(fit_output.model.sources)
        # self.checkpoint.save_step(
        #     'drips_fit', 0, 'completed',
        #     {'log_like': fit_output.log_like, 'aic': fit_output.aic, 'num_sources': num_sources},
        #     metadata={'num_sources': num_sources},
        # )
        # return fit_output
        fit_output = source_fitter.run(
            drips_output, self.config, self.logger, self.directory_manager, self.checkpoint,
            resume=self.resume,
        )
        num_sources = len(fit_output.model.sources)
        self.checkpoint.save_step(
            'drips_fit', 0, 'completed',
            {'log_like': fit_output.log_like, 'aic': fit_output.aic, 'num_sources': num_sources},
            metadata={'num_sources': num_sources},
        )
        return fit_output

    def run(self) -> Union[SeedingOutput, FitResult, Path]:
        """Run the full pipeline per `fitting_procedure` and return the
        result: a Path to the DRIPS model file (seed-only Drips run), a
        FitResult (full Drips run), or a SeedingOutput (Alps)."""
        self.logger.info(f"Starting HAWCAnalysisPipeline (method={self.method})")
        self._build_significance_map()

        if self.method == 'Drips':
            output = self._run_drips()
        else:
            output = self._run_alps()

        if isinstance(output, FitResult):
            num_sources = len(output.model.sources)
            self.logger.info(f"Pipeline completed: {num_sources} sources, -logL={output.log_like:.3f}")
        elif isinstance(output, Path):
            self.logger.info(f"Pipeline completed (seed-only): model at {output}")
        else:
            self.logger.info(f"Pipeline completed: {output.num_sources} sources, method={output.method}")
        return output