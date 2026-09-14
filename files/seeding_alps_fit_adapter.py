"""
ALPS Fit Adapter (Part 2a)

Bridges ALPS's file-based fit loop onto the in-process threeMLFit fitter.

The original ALPS shelled out to fitModel.py and read results back off disk
(STAT0/MV0 from *_likeResults.fits, the refit model from *_modelFit.yml,
residual/model maps from *.hd5). This adapter runs threeMLFit in-process and
exposes the same three things ALPS's control flow needs:

  - total log-likelihood  (ALPS STAT0)  -> statistics.loc['total','-log(likelihood)']
  - AIC                   (ALPS MV0)    -> jl.results.get_statistic_measure('AIC')
  - fitted source_info_db (ALPS reload) -> rebuilt from the fitted model_obj
  - residual/model .hd5 maps            -> threeMLFit.make_maps()

Nothing here is a verbatim ALPS extraction — this is the new integration layer
whose behaviour was specified from the threeMLFit interface.
"""

import os
import glob
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

from core.pipeline_fitmodel import threeMLFit


@dataclass
class FitStepResult:
    """Result of a single in-process fit step (mirrors what ALPS read off disk).

    Attributes:
    -----------
    log_like : float
        Total -log(likelihood) at the minimum. Equivalent to ALPS STAT0.
    aic : float
        Akaike information criterion. Equivalent to ALPS MV0.
    fitted_db : pd.DataFrame
        source_info_db rebuilt from the fitted model (fitted parameter values).
    model_map_hd5 : Path or None
        Path to written model map .hd5 (if make_maps ran).
    residual_map_hd5 : Path or None
        Path to written residual map .hd5 (if make_maps ran).
    step_dir : Path
        Directory holding this step's outputs.
    fitter : object
        The threeMLFit instance (for params/TS access if needed).
    """
    log_like: float
    aic: float
    fitted_db: pd.DataFrame
    model_map_hd5: Optional[Path]
    residual_map_hd5: Optional[Path]
    step_dir: Path
    fitter: object


class ALPSFitAdapter:
    """Runs threeMLFit in-process and returns ALPS-shaped results.

    Parameters:
    -----------
    config_path : str
        Path to the threeMLFit/pipeline config YAML (dot-notation schema).
    logger : object
        Logger passed through to threeMLFit.
    db_from_model_fn : callable
        Function (model_obj) -> source_info_db. Supplied by the ALPS seeder so
        the fitted model is re-parsed with the exact same logic as
        _load_from_model_file (kept as the single source of truth).
    roi_template : str, optional
        ROI template path forwarded to threeMLFit.
    """

    def __init__(self, config_path: str, logger: object, db_from_model_fn, roi_template: str = None):
        self.config_path = config_path
        self.logger = logger
        self.db_from_model_fn = db_from_model_fn
        self.roi_template = roi_template

    def fit(
        self,
        model_file_loc: str,
        step_dir: str,
        compute_err: bool = True,
        make_maps: bool = True,
    ) -> FitStepResult:
        """Run one fit in-process and package results the way ALPS expects.

        Parameters:
        -----------
        model_file_loc : str
            Path to the .model file to fit (already written by
            initialize_model_file).
        step_dir : str
            Output directory for this step. threeMLFit writes maps under
            step_dir/results/.
        compute_err : bool
            If True use hal_fit_with_covariance(), else hal_fit().
        make_maps : bool
            If True, write model_fit.hd5 and residual_fit.hd5 via make_maps().

        Returns:
        --------
        FitStepResult
        """
        step_dir = Path(step_dir)
        (step_dir / 'results').mkdir(parents=True, exist_ok=True)

        fitter = threeMLFit(
            config_path=self.config_path,
            model=str(model_file_loc),
            save_dir=step_dir,
            roiTemplate=self.roi_template,
            logger=self.logger,
        )

        if compute_err:
            fitter.hal_fit_with_covariance()
        else:
            fitter.hal_fit()

        # --- total log-likelihood (ALPS STAT0): stored as -log(likelihood) ---
        log_like = float(fitter.statistics.loc['total', '-log(likelihood)'])

        # --- AIC (ALPS MV0) from the MLEResults object ---
        aic = self._extract_aic(fitter)

        # --- rebuild fitted source_info_db from the fitted model object ---
        fitted_db = self.db_from_model_fn(fitter.model_obj)

        model_map_hd5 = None
        residual_map_hd5 = None
        if make_maps:
            fitter.make_maps()
            mm = step_dir / 'results' / 'model_fit.hd5'
            rm = step_dir / 'results' / 'residual_fit.hd5'
            model_map_hd5 = mm if mm.exists() else None
            residual_map_hd5 = rm if rm.exists() else None

        self.logger.info(f'Fit step {step_dir.name}: -logL={log_like:.3f}, AIC={aic:.3f}')

        return FitStepResult(
            log_like=log_like,
            aic=aic,
            fitted_db=fitted_db,
            model_map_hd5=model_map_hd5,
            residual_map_hd5=residual_map_hd5,
            step_dir=step_dir,
            fitter=fitter,
        )

    def _extract_aic(self, fitter) -> float:
        """Pull AIC from the threeML MLEResults, with fallbacks.

        threeML's MLEResults exposes statistical measures (AIC/BIC). The exact
        accessor differs slightly across versions, so try the common ones.
        """
        results = getattr(fitter.jl, 'results', None)
        if results is None:
            return float('nan')
        # Preferred: explicit statistic-measure accessor
        for accessor in ('get_statistic_measure',):
            fn = getattr(results, accessor, None)
            if callable(fn):
                try:
                    return float(fn('AIC'))
                except Exception:
                    pass
        # Fallback: some versions expose an 'optimal_statistic_values' or a
        # statistical_measures dict/series.
        for attr in ('statistical_measures',):
            sm = getattr(results, attr, None)
            if sm is not None:
                try:
                    return float(sm['AIC'])
                except Exception:
                    pass
        self.logger.warning('Could not extract AIC from fit results; returning NaN')
        return float('nan')
