"""
In-process fit runner.

Generic wrapper around threeMLFit (pipeline_fitmodel.py). This is the fitting
transport shared by the joint fit, extension test, spectrum test, and final
refit phases in seeding/source_fitter.py -- none of it is ALPS-specific.
"""

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import astromodels
from core.pipeline_fitmodel import threeMLFit


@dataclass
class FitResult:
    """Result of a single in-process fit.

    Attributes:
    -----------
    model : object
        The fitted, live threeML Model object (fitter.model_obj). Parameter
        values reflect the fit outcome; free/frozen state reflects whatever
        was set on the input model file.
    log_like : float
        Total -log(likelihood) at the minimum.
    aic : float
        Akaike information criterion.
    model_map_path : Path or None
        Path to written model map .hd5 (if make_maps ran).
    residual_map_path : Path or None
        Path to written residual map .hd5 (if make_maps ran).
    step_dir : Path
        Directory holding this step's outputs.
    fitter : object
        The threeMLFit instance (for params/TS access if needed).
    """
    model: object
    log_like: float
    aic: float
    ts : float
    model_map_path: Optional[Path]
    residual_map_path: Optional[Path]
    step_dir: Path
    fitter: object


class FitRunner:
    """Runs threeMLFit in-process for one model file at a time.

    Parameters:
    -----------
    config_path : str
        Path to the pipeline config YAML (dot-notation schema; same file
        threeMLFit itself reads for map_tree/detector_response/bins/ROI).
    logger : object
        Logger passed through to threeMLFit.
    roi_template : str, optional
        ROI template path forwarded to threeMLFit.
    max_retries : int
        Retry attempts on fit failure, perturbing free position parameters
        between attempts.
    """

    def __init__(self, config_path: str, logger: object, roi_template: str = None, max_retries: int = 1):
        self.config_path = config_path
        self.logger = logger
        self.roi_template = roi_template
        self.max_retries = max_retries

    def fit(
        self,
        model_file: str,
        step_dir: str,
        compute_err: bool = True,
        compute_TS: bool = True,
        make_maps: bool = True,
    ) -> FitResult:
        step_dir = Path(step_dir)

        start = time.perf_counter()
        last_error = None
        fitter = None
        ts = None
        self.logger.info(f'Fitting model {model_file} in {step_dir}  with compute_err={compute_err}, compute_TS={compute_TS}, make_maps={make_maps}')

        fitter = threeMLFit(
            config_path=self.config_path,
            model=str(model_file),
            save_dir=step_dir,
            roiTemplate=self.roi_template,
            logger=self.logger,
        )
        if compute_err:
            fitter.hal_fit_with_covariance()
        else:
            fitter.hal_fit()
        if compute_TS:
            ts = fitter.get_TS()
        if fitter is None:
            raise RuntimeError(f'Fit at {model_file} failed after {self.max_retries} attempts: {last_error}')

        log_like = float(fitter.statistics.loc['total', '-log(likelihood)'])
        aic = self._extract_aic(fitter)

        model_map_path = None
        residual_map_path = None
        if make_maps:
            fitter.make_maps()
            mm = step_dir / 'model_fit.hd5'
            rm = step_dir / 'residual_fit.hd5'
            model_map_path = mm if mm.exists() else None
            residual_map_path = rm if rm.exists() else None

        elapsed = (time.perf_counter() - start) / 60.0
        self.logger.info(f'Fit {step_dir.name}: -logL={log_like:.3f}, AIC={aic:.3f} ({elapsed:.2f} min)')

        return FitResult(
            model=fitter.model_obj,
            log_like=log_like,
            aic=aic,
            ts=ts,
            model_map_path=model_map_path,
            residual_map_path=residual_map_path,
            step_dir=step_dir,
            fitter=fitter,
        )
    def _extract_aic(self, fitter) -> float:
        """Pull AIC from the threeML MLEResults.

        Confirmed accessor (see seeding/alps_fit_adapter.py, TASKS.md Task 2):
        `jl.results.statistical_measures['AIC']`.
        """
        results = getattr(fitter.jl, 'results', None)
        if results is None:
            return float('nan')
        sm = getattr(results, 'statistical_measures', None)
        if sm is not None:
            try:
                return float(sm['AIC'])
            except Exception:
                pass
        self.logger.warning('Could not extract AIC from fit results; returning NaN')
        return float('nan')
