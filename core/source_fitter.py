"""
Source fitter: DRIPS-seeded joint fit + extension/spectrum testing + refit.

"""

from pathlib import Path
from typing import List
import os
from astropy.coordinates import SkyCoord
import pandas as pd
import astromodels
from astropy.io import fits
from core.fit_runner import FitRunner, FitResult
from seeding.model_generator import ModelGenerator
from seeding.base import SeedingOutput
from core.hdf5_handler import HDF5Handler
from core.map_tools import MapGenerator
from core.pipeline_helpers import load_hawc_data, find_peak, make_plots
from threeML.minimizer.minimization import FitFailed
from datetime import datetime

import json
import numpy as np
from astropy.io import fits


def save_fit_summary(fit_result: FitResult, logger: object, extra: dict = None) -> Path:
    """Read likelihoodResults.fits from fit_result.step_dir, combine its
    per-parameter table with fit_result.ts (if computed), and write both to
    fit_result.step_dir/fit_summary.json (and a flat fit_summary.csv for
    quick inspection). Call this once after each fit, wherever the caller
    has a finished FitResult in hand.

    Returns the path to the written JSON summary, or None if the FITS
    results file wasn't found (e.g. a fit that failed before writing it).
    """
    step_dir = Path(fit_result.step_dir)
    results_path = step_dir / 'likelihoodResults.fits'

    if not results_path.exists():
        logger.warning(f'save_fit_summary: no likelihoodResults.fits at {results_path}; skipping')
        return None

    with fits.open(results_path) as hdul:
        data = hdul[1].data

    params = {}
    for row in data:
        name = row['NAME'].decode() if isinstance(row['NAME'], bytes) else str(row['NAME'])
        unit = row['UNIT'].decode() if isinstance(row['UNIT'], bytes) else str(row['UNIT'])
        params[name] = {
            'value': float(row['VALUE']),
            'negative_error': float(row['NEGATIVE_ERROR']),
            'positive_error': float(row['POSITIVE_ERROR']),
            'error': float(row['ERROR']),
            'unit': unit,
        }

    ts_by_source = fit_result.ts if isinstance(fit_result.ts, dict) else None

    summary = {
        'step_dir': str(step_dir),
        'log_like': fit_result.log_like,
        'aic': fit_result.aic,
        'ts': ts_by_source,
        'parameters': params,
    }
    if extra:
        summary.update(extra)

    json_path = step_dir / 'fit_summary.json'
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2, default=_json_safe)
    logger.info(f'Wrote fit summary: {json_path}')

    csv_path = step_dir / 'fit_summary.csv'
    with open(csv_path, 'w') as f:
        f.write('name,value,negative_error,positive_error,error,unit\n')
        for name, p in params.items():
            f.write(f"{name},{p['value']},{p['negative_error']},{p['positive_error']},{p['error']},{p['unit']}\n")
        if ts_by_source:
            f.write('\nsource,ts\n')
            for source, ts in ts_by_source.items():
                f.write(f'{source},{ts}\n')
    logger.info(f'Wrote fit summary CSV: {csv_path}')

    return json_path


def _json_safe(obj):
    """json.dump default= handler for numpy scalars/NaN that plain json
    chokes on."""
    if isinstance(obj, (np.floating, np.integer)):
        val = float(obj)
        return None if val != val else val  # NaN -> null
    if isinstance(obj, float) and obj != obj:
        return None
    raise TypeError(f'Object of type {type(obj)} is not JSON serializable')

def _as_list(value) -> List[str]:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]

def check_hotspots(path, fit_output, config, logger) -> bool:
    """Returns True if a residual excess above threshold (max_value > 5) was
    found, else False. Still writes the diagnostic plot either way."""
    mapname = path
    l = config.get('coordinates.l')
    b = config.get('coordinates.b')
    x_length = config.get('coordinates.roi_x')
    y_length = config.get('coordinates.roi_y')
    coord_sys = config.get('coordinates.coord_sys')
    array, _, wcs, _, _, pixel_size = load_hawc_data(mapname, l, b, x_length, y_length, coord_sys)
    max_value = find_peak(array, wcs)
    logger.info(f"Max value in residual map: {max_value}")

    has_excess = max_value > 5
    if has_excess:
        name, ra, dec, ext = [], [], [], []
        logger.info(f"Sources in model: {list(fit_output.model.sources.keys())}")
        for source in fit_output.model.sources:
            if source == 'URM':
                continue
            src_obj = fit_output.model[source]
            try:
                if hasattr(src_obj, 'position'):
                    src_ra = src_obj.position.ra.value
                    src_dec = src_obj.position.dec.value
                    src_ext = 0.01
                else:
                    src_ra = src_obj.spatial_shape.lon0.value
                    src_dec = src_obj.spatial_shape.lat0.value
                    src_ext = src_obj.spatial_shape.sigma.value
            except Exception as e:
                logger.warning(f'Skipping source {source} in hotspot plot: could not extract position/extent ({e})')
                continue
            name.append(source)
            ra.append(src_ra)
            dec.append(src_dec)
            ext.append(src_ext)
        df = {'Name': name, 'ra': ra, 'dec': dec, 'ext': ext}
        logger.info(f"Hotspots found: {df}")
        make_plots(array, wcs, pixel_size, coord_sys, save_dir=str(path.parent), title="Map", cmap='ult', hotspots=df)
    else:
        logger.info(f"No hotspots found in {path}. Max value: {max_value}")
        make_plots(array, wcs, pixel_size, coord_sys, save_dir=str(path.parent), title="Map", cmap='ult', hotspots=None)

    return has_excess

def _is_valid_healpix_fits(path, logger) -> bool:
    """Sanity-check a cached healpix FITS file before trusting it as a
    valid regeneration-skip target. A file that exists but is truncated,
    corrupted, or from an incompatible prior run should be regenerated,
    not silently reused."""
    try:
        with fits.open(path) as hdul:
            if len(hdul) < 2:
                logger.warning(f'{path} has only {len(hdul)} HDU(s), expected >=2; treating as invalid')
                return False
            _ = hdul[1].data['significance']
        return True
    except Exception as e:
        logger.warning(f'Cached FITS file {path} failed validation ({e}); will regenerate')
        return False
    
def _build_fit_maps(config, logger, directory_manager, path, name, checkpoint=None):
    """Create significance maps."""
    existing_path = path / 'fits' / f'{name}.fits'
    if existing_path.exists():
        if _is_valid_healpix_fits(existing_path, logger):
            logger.info(f"Output FITS file {existing_path} already exists and is valid, skipping map generation entirely")
            return existing_path
        else:
            logger.warning(f"Existing FITS file {existing_path} failed validation; regenerating")

    ra = config.get('coordinates.ra')
    dec = config.get('coordinates.dec')
    if ra is None or dec is None:
        l = config.get('coordinates.l')
        b = config.get('coordinates.b')
        skycoord = SkyCoord(l, b, frame='galactic', unit='deg')
        ra = skycoord.icrs.ra.deg
        dec = skycoord.icrs.dec.deg
        logger.info(f"Converted galactic coordinates (l={l}, b={b}) to equatorial (RA={ra}, Dec={dec})")
        config.set('coordinates.ra', ra)
        config.set('coordinates.dec', dec)

    created_files = HDF5Handler.convert_hd5_to_fits(
        input_dir=str(path),
        hd5_filename='residual_fit.hd5',
        output_basename='residual',
        logger=logger
    )
    print(f"Created FITS files: {created_files}")
    bins = config.get('fitting.bins')
    detector_response = config.get('coordinates.detector_response')
    print(f"Path: {path}")

    output_path = MapGenerator.create_healpix_map(
        input_fits_files=list(created_files),
        energy_bins=list(bins),
        detector_response=detector_response,
        ra_center=float(config.get('coordinates.ra')),
        dec_center=float(config.get('coordinates.dec')),
        roi_x=float(config.get('coordinates.roi_x', 4.0)*2.5),
        roi_y=float(config.get('coordinates.roi_y', 4.0)*2),
        output_file=str(existing_path),
        logger=logger,
        pixi_manifest_path=config.get('alps.pixi_aerie_folder'),
    )
    return output_path

def _model_summary_df(model) -> pd.DataFrame:
    """Descriptive (not round-trippable) summary of a live model's sources,
    for the SeedingOutput.source_info_db contract."""
    rows = []
    for name, source in model.sources.items():
        if hasattr(source, 'position'):
            ra, dec = source.position.ra.value, source.position.dec.value
        else:
            ra, dec = source.spatial_shape.lon0.value, source.spatial_shape.lat0.value
        rows.append({
            'source': name,
            'ra': ra,
            'dec': dec,
            'spatial_model': type(source.spatial_shape).__name__ if hasattr(source, 'spatial_shape') else 'PointSource',
            'spectral_model': type(source.spectrum.main.shape).__name__,
        })
    return pd.DataFrame(rows, columns=['source', 'ra', 'dec', 'spatial_model', 'spectral_model'])

def _check_and_remove_low_ts(trial_result: FitResult, source_names_to_protect: List[str],
                              ts_threshold: float, step_label: str, config, logger,
                              directory_manager) -> FitResult:
    ts_by_source = trial_result.ts
    if not isinstance(ts_by_source, dict):
        logger.warning('trial_result.ts is not a dict; skipping low-TS check')
        return None

    low_ts_sources = [
        n for n, ts in ts_by_source.items()
        if n != 'URM' and n not in source_names_to_protect and ts < ts_threshold
    ]
    if not low_ts_sources:
        return None

    logger.info(f'Sources dropped below TS threshold {ts_threshold} during {step_label}: {low_ts_sources}; removing and refitting')
    pruned_model = ModelGenerator.remove_sources(trial_result.model, low_ts_sources, logger=logger)
    remaining_names = list(pruned_model.sources.keys())

    ModelGenerator.set_free(pruned_model, remaining_names, kind='spatial', free=True, free_diffuse=True, logger=logger)
    ModelGenerator.set_free(pruned_model, remaining_names, kind='spectral', free=True, free_diffuse=True, logger=logger)

    prune_step_name = f'{step_label}-Pruned_{"_".join(low_ts_sources)}'
    prune_step_dir = directory_manager.get_step_results_dir(prune_step_name)
    yml_path = f'{prune_step_dir}/curModel.yml'
    model_path = f'{prune_step_dir}/curModel.model'
    pruned_model.save(yml_path, overwrite=True)
    ModelGenerator.write_model_file_from_yaml(yml_path, model_path, logger=logger)

    runner = FitRunner(
        config_path=str(config.config_file), logger=logger,
        roi_template=config.get('roi.roi_template_path'),
    )
    refit_result = runner.fit(
        model_file=str(model_path),
        step_dir=str(prune_step_dir),
        compute_err=config.get('error_and_TS.error_point', True),
        compute_TS=True,
        make_maps=True,
    )
    save_ts_values(refit_result.ts, prune_step_name, directory_manager, logger)
    save_fit_summary(refit_result, logger)
    _record_manifest(directory_manager, logger, prune_step_dir, f'Removed low-TS source(s) {low_ts_sources} during {step_label}; refit -- {len(refit_result.model.sources)} sources remain')
    return refit_result

def run_extension_test(fit_result: FitResult, config, logger, directory_manager, skip_sources: List[str] = None) -> FitResult:
    """Test alternate spatial models per source; accept if TS improvement
    exceeds likelihood_thresholds.extension_test. Other sources are frozen
    during each trial to isolate the tested source's effect; nothing is
    permanently frozen going into the next phase -- run_final_refit unfreezes
    everything.
    """
    alt_models = _as_list(config.get('fitting.alternate_spatial_models'))
    if not alt_models:
        logger.info('No fitting.alternate_spatial_models configured; skipping extension test')
        return fit_result
    free_dbe = config.get('diffuse.free_diffuse_norm', False)
    logger.info(f"Diffuse background normalization status during spectrum test: {free_dbe}")
    source_ts_threshold = config.get('likelihood_thresholds.point_source_detection', 16)
    extension_ts_threshold = config.get('likelihood_thresholds.extension_test', 16)
    coord_range = config.get('fitting.extended_source_coord_range', 1.0)

    force_low_ts_source = config.get('testing.force_low_ts_source', None)
    # force_low_ts_source = 'Source3' 
    skip_sources = skip_sources or []
    if force_low_ts_source:
        logger.info(
            f'TESTING OVERRIDE ACTIVE: forcing TS for source {force_low_ts_source!r} '
            f'below threshold {source_ts_threshold} after every trial fit. '
            f'Unset testing.force_low_ts_source for production runs.'
        )
    runner = FitRunner(
        config_path=str(config.config_file),
        logger=logger,
        roi_template=config.get('roi.roi_template_path'),
    )

    model = fit_result.model
    baseline_log_like = fit_result.log_like
    source_names = list(model.sources.keys())
    for source_name in source_names:
        if source_name == 'URM':
            logger.info(f'Skipping extension test for {source_name} (URM source)')
            continue
        if source_name not in model.sources:
            logger.info(f'{source_name} was removed by a prior low-TS check; skipping its extension test')
            continue
        if source_name in skip_sources:
            logger.info(f'{source_name} already tested in a prior run; skipping (resume)')
            continue
        other_sources = [n for n in model.sources.keys() if n != source_name]
        best_log_like = baseline_log_like
        logger.info(f'Current best log-likelihood: {best_log_like:.3f}')
        best_model = model

        for alt_shape in alt_models:

            source = model.sources[source_name]
            current_spatial_model = list(source._children.keys())[0]
            if current_spatial_model == alt_shape:
                logger.info(f"Source {source_name} already has spatial shape {alt_shape}, skipping swap")
                continue

            trial_model = ModelGenerator.swap_spatial_shape(
                model, source_name, alt_shape, coord_range=coord_range, logger=logger,
            )
            ModelGenerator.set_free(trial_model, other_sources, kind='spatial', free=True, free_diffuse=free_dbe, param_names=['sigma', 'e', 'theta'], logger=logger)
            ModelGenerator.set_free(trial_model, other_sources, kind='spectral', free=True, free_diffuse=free_dbe, param_names=['K', 'index', 'alpha', 'beta'], logger=logger)

            step_name = f'Step2-{source_name}-Extension-{alt_shape}'
            step_dir = directory_manager.get_step_results_dir(step_name)
            trial_model.save("{1}/{0}.yml".format('curModel', step_dir), overwrite=True)
            ModelGenerator.write_model_file_from_yaml("{1}/{0}.yml".format('curModel', step_dir), "{1}/{0}.model".format('curModel', step_dir), logger=logger)
            model_file = "{1}/{0}.model".format('curModel', step_dir)
            trial_result = runner.fit(
                model_file=str(model_file),
                step_dir=str(step_dir),
                compute_err=config.get('error_and_TS.error_extension', True),
                compute_TS=False,
                make_maps=True,
            )
            save_fit_summary(trial_result, logger)
            delta_ts = 2 * (best_log_like - trial_result.log_like)
            logger.info(f'Extension test {source_name} -> {alt_shape}: delta_TS={delta_ts:.2f} (threshold {extension_ts_threshold})')

            if delta_ts <= extension_ts_threshold:
                logger.info(f'Rejected alternate spatial model {alt_shape} for {source_name}; skipping TS computation')
                continue

            logger.info(f'delta_TS above threshold; computing per-source TS for {step_name}')
            trial_result.ts = trial_result.fitter.get_TS()
            save_ts_values(trial_result.ts, step_name, directory_manager, logger)

            if force_low_ts_source and isinstance(trial_result.ts, dict) and force_low_ts_source in trial_result.ts:
                real_ts = trial_result.ts[force_low_ts_source]
                trial_result.ts[force_low_ts_source] = source_ts_threshold - 1.0
                logger.warning(
                    f'TESTING OVERRIDE: {force_low_ts_source} real TS={real_ts:.2f} '
                    f'-> forced to {trial_result.ts[force_low_ts_source]:.2f}'
                )

            pruned_result = _check_and_remove_low_ts(
                trial_result, source_names_to_protect=[source_name],
                ts_threshold=source_ts_threshold, step_label=f'{step_name}',
                config=config, logger=logger, directory_manager=directory_manager,
            )
            if pruned_result is not None:
                trial_result = pruned_result
                best_log_like = trial_result.log_like
                best_model = trial_result.model
                fit_result = trial_result
                model = best_model
                baseline_log_like = best_log_like
                low_ts_dropped = set(source_names) - set(model.sources.keys())
                source_names = [n for n in source_names if n not in low_ts_dropped]
                other_sources = [n for n in other_sources if n not in low_ts_dropped]

                if force_low_ts_source in low_ts_dropped:
                    force_low_ts_source = None
                continue

            best_log_like = trial_result.log_like
            best_model = trial_result.model
            logger.info(f'Accepted alternate spatial model {alt_shape} for {source_name}')
            fit_result = trial_result
            _record_manifest(directory_manager, logger, step_dir, f'Extension test: accepted {alt_shape} for {source_name} (delta_TS={delta_ts:.2f})')
        model = best_model
        baseline_log_like = best_log_like

    results = FitResult(
        model=model, log_like=baseline_log_like, aic=fit_result.aic, ts = fit_result.ts,
        model_map_path=fit_result.model_map_path, residual_map_path=fit_result.residual_map_path,
        step_dir=fit_result.step_dir, fitter=fit_result.fitter,
    )
    return results


def run_spectrum_test(fit_result: FitResult, config, logger, directory_manager, skip_sources: List[str] = None) -> FitResult:
    """Test alternate spectral models per source. The current model (C) is
    nested within every alternate model in fitting.alternate_spectral_models
    (e.g. Powerlaw is a restricted case of Log_parabola and of
    Cutoff_powerlaw), so each alternate is first screened against C via a
    likelihood-ratio test (delta_TS vs likelihood_thresholds.spectrum_test) --
    a valid comparison only because each alternate nests C. The alternates
    themselves are NOT nested in each other, so once more than one clears
    the LRT screen against C, the winner among them is picked by AIC (the
    correct non-nested comparison), not by comparing their delta_TS values
    to each other. TS is computed only for the eventual winner.
    """
    alt_models = _as_list(config.get('fitting.alternate_spectral_models'))
    if not alt_models:
        logger.info('No fitting.alternate_spectral_models configured; skipping spectrum test')
        return fit_result

    threshold = config.get('likelihood_thresholds.spectrum_test', 16)
    source_ts_threshold = config.get('likelihood_thresholds.point_source_detection', 16)
    runner = FitRunner(
        config_path=str(config.config_file),
        logger=logger,
        roi_template=config.get('roi.roi_template_path'),
    )
    free_dbe = config.get('diffuse.free_diffuse_norm', False)
    logger.info(f"Diffuse background normalization status during spectrum test: {free_dbe}")
    model = fit_result.model
    baseline_log_like = fit_result.log_like
    source_names = list(model.sources.keys())
    skip_sources = skip_sources or []

    for source_name in source_names:
        if source_name == 'URM':
            logger.info(f'Skipping spectrum test for {source_name} (URM source)')
            continue
        if source_name not in model.sources:
            logger.info(f'{source_name} was removed by a prior low-TS check; skipping its spectrum test')
            continue
        if source_name in skip_sources:
            logger.info(f'{source_name} already tested in a prior run; skipping (resume)')
            continue
        
        other_sources = [n for n in model.sources.keys() if n != source_name]
        # C's own log_like/AIC -- the nested-parent baseline every alt_spectrum
        # trial is screened against. Stays fixed for the whole inner loop:
        # each alt is compared to C, never to a sibling alt (not nested).
        c_log_like = baseline_log_like
        c_aic = fit_result.aic if model is fit_result.model else None

        candidates = []  # trial_results that passed the LRT screen against C

        for alt_spectrum in alt_models:
            trial_model = ModelGenerator.swap_spectral_shape(
                model, source_name, alt_spectrum, logger=logger,
            )
            ModelGenerator.set_free(trial_model, other_sources, kind='spatial', free=True, free_diffuse=free_dbe, param_names=['sigma', 'e', 'theta'], logger=logger)
            ModelGenerator.set_free(trial_model, other_sources, kind='spectral', free=True, free_diffuse=free_dbe, param_names=['K', 'index', 'alpha', 'beta', 'xc'], logger=logger)

            step_name = f'Step3-{source_name}-Spectrum-{alt_spectrum}'
            step_dir = directory_manager.get_step_results_dir(step_name)
            trial_model.save("{1}/{0}.yml".format('curModel', step_dir), overwrite=True)
            ModelGenerator.write_model_file_from_yaml("{1}/{0}.yml".format('curModel', step_dir), "{1}/{0}.model".format('curModel', step_dir), logger=logger)
            model_file = "{1}/{0}.model".format('curModel', step_dir)
            try:
                trial_result = runner.fit(
                    model_file=str(model_file),
                    step_dir=str(step_dir),
                    compute_err=config.get('error_and_TS.error_spectrum', True),
                    compute_TS=False,
                    make_maps=False,
                )
            except FitFailed as e:
                logger.warning(f'Spectrum test {source_name} -> {alt_spectrum}: fit did not converge ({e}); skipping this candidate')
                continue
            save_fit_summary(trial_result, logger)
            delta_ts = 2 * (c_log_like - trial_result.log_like)
            logger.info(
                f'Spectrum test {source_name} -> {alt_spectrum} vs baseline C: '
                f'delta_TS={delta_ts:.2f} (threshold {threshold}), AIC={trial_result.aic:.3f}'
            )

            if delta_ts <= threshold:
                logger.info(f'Rejected alternate spectral model {alt_spectrum} for {source_name} (LRT vs C failed)')
                continue

            logger.info(f'{alt_spectrum} passed LRT screen against C; candidate for AIC ranking')
            candidates.append((alt_spectrum, trial_result))

        if not candidates:
            logger.info(f'No alternate spectral model beat baseline C for {source_name}; keeping current model')
            continue

        # Non-nested comparison among survivors: pick lowest AIC.
        best_alt_spectrum, winner = min(candidates, key=lambda pair: pair[1].aic)
        logger.info(
            f'{source_name}: {len(candidates)} candidate(s) passed LRT vs C; '
            f'selected {best_alt_spectrum} by AIC ({winner.aic:.3f})'
        )

        logger.info(f'Computing per-source TS for selected winner: {source_name} -> {best_alt_spectrum}')
        _record_manifest(directory_manager, logger, winner.step_dir, f'Spectrum test: selected {best_alt_spectrum} for {source_name} (AIC={winner.aic:.3f}, {len(candidates)} candidate(s) considered)')
        winner.ts = winner.fitter.get_TS()

        pruned_result = _check_and_remove_low_ts(
            winner, source_names_to_protect=[source_name],
            ts_threshold=source_ts_threshold, step_label=f'Step3-{source_name}-Spectrum-{best_alt_spectrum}',
            config=config, logger=logger, directory_manager=directory_manager,
        )
        if pruned_result is not None:
            winner = pruned_result
            low_ts_dropped = set(source_names) - set(winner.model.sources.keys())
            source_names = [n for n in source_names if n not in low_ts_dropped]

        model = winner.model
        baseline_log_like = winner.log_like
        fit_result = winner

    return FitResult(
        model=model, log_like=baseline_log_like, aic=fit_result.aic, ts=fit_result.ts,
        model_map_path=fit_result.model_map_path, residual_map_path=fit_result.residual_map_path,
        step_dir=fit_result.step_dir, fitter=fit_result.fitter,
    )

from threeML.minimizer.minuit_minimizer import MINOSFailed

def run_final_refit(fit_result: FitResult, config, logger, directory_manager) -> FitResult:
    """Unfreeze every source's parameters and do one more joint fit.
    ...
    """
    model = fit_result.model
    all_sources = list(model.sources.keys())
    ModelGenerator.set_free(model, all_sources, kind='spatial', free=True, free_diffuse=True, param_names=['ra', 'dec', 'lon0', 'lat0', 'sigma', 'e', 'theta'], logger=logger)
    ModelGenerator.set_free(model, all_sources, kind='spectral', free=True, free_diffuse=True, param_names=['K', 'index', 'alpha', 'beta', 'xc'], logger=logger)

    step_name = 'Step4-FinalRefit'
    step_dir = directory_manager.get_step_results_dir(step_name)
    yml_path = "{1}/{0}.yml".format('curModel', step_dir)
    model_file = "{1}/{0}.model".format('curModel', step_dir)
    model.save(yml_path, overwrite=True)
    ModelGenerator.write_model_file_from_yaml(yml_path, model_file, logger=logger)

    runner = FitRunner(
        config_path=str(config.config_file),
        logger=logger,
        roi_template=config.get('roi.roi_template_path'),
    )
    logger.info('Running final joint refit with all parameters free')
    try:
        result = runner.fit(
            model_file=str(model_file),
            step_dir=str(step_dir),
            compute_err=True,
            compute_TS=True,
            make_maps=True,
        )
    except MINOSFailed as e:
        logger.warning(
            f'MINOS error estimation failed on final refit ({e}); the fit itself converged '
            f'(best-fit values are valid) but asymmetric errors could not be computed -- '
            f'retrying without error estimation (compute_err=False)'
        )
        result = runner.fit(
            model_file=str(model_file),
            step_dir=str(step_dir),
            compute_err=False,
            compute_TS=True,
            make_maps=True,
        )

    resmap = _build_fit_maps(config, logger, directory_manager, result.step_dir, 'residual')
    check_hotspots(resmap, result, config, logger)
    save_ts_values(result.ts, step_name, directory_manager, logger)
    _record_manifest(directory_manager, logger, step_dir, f'Final refit, all parameters free -- {len(result.model.sources)} sources, -logL={result.log_like:.3f}, AIC={result.aic:.3f}')
    save_fit_summary(result, logger)
    return result

def run(drip_model_path, config, logger, directory_manager, checkpoint=None, resume: bool = False) -> FitResult:
    """Run joint fit -> extension test -> spectrum test -> final refit (each
    gated by config) and return the final FitResult.

    resume : bool
        If True, scan directory_manager's Results tree for the
        furthest-completed step and resume from there instead of starting
        at run_joint_fit, skipping any per-source extension/spectrum tests
        already completed in a prior (interrupted) run. Falls back to a
        fresh run if nothing completed is found.
    """
    logger.info('Starting source_fitter (DRIPS-seeded in-process fit)')

    skip_ext, skip_spec = [], []
    resumed_past_joint_fit = False
    if resume:
        resume_step_dir, skip_ext, skip_spec = find_resume_point(directory_manager, logger)
        if resume_step_dir is not None:
            result = load_fit_result_from_step_dir(resume_step_dir, logger)
            resumed_past_joint_fit = resume_step_dir.name != 'Step1-JointFit-Iter0' and not resume_step_dir.name.startswith('Step1-JointFit-Iter')
        else:
            logger.info('resume=True but nothing to resume from; starting fresh')
            result = run_joint_fit(drip_model_path, config, logger, directory_manager)
            save_fit_summary(result, logger, extra={'step': 'Step1-JointFit'})
    else:
        result = run_joint_fit(drip_model_path, config, logger, directory_manager)
        save_fit_summary(result, logger, extra={'step': 'Step1-JointFit'})

    if checkpoint is not None:
        checkpoint.save_step(
            'drips_joint_fit', 0, 'completed',
            {'log_like': result.log_like, 'aic': result.aic, 'ts': result.ts, 'step_dir': str(result.step_dir)},
            metadata={'num_sources': len(result.model.sources)},
        )

    if not resumed_past_joint_fit:
        _build_fit_maps(config, logger, directory_manager, result.step_dir, 'residual')

    if config.get('fitting.run_extension_test', True):
        result = run_extension_test(result, config, logger, directory_manager, skip_sources=skip_ext)
        save_fit_summary(result, logger, extra={'step': 'Step2-ExtensionTest'})

    if config.get('fitting.run_spectrum_test', True):
        result = run_spectrum_test(result, config, logger, directory_manager, skip_sources=skip_spec)
        save_fit_summary(result, logger, extra={'step': 'Step3-SpectrumTest'})

    if config.get('fitting.run_final_refit', True):
        result = run_final_refit(result, config, logger, directory_manager)
        # run_final_refit already calls save_fit_summary internally.

    num_sources = len(result.model.sources)
    model_path = directory_manager.get_model_file_path('Final')
    yml_path = str(model_path).replace('.model', '.yml') if str(model_path).endswith('.model') else f'{model_path}.yml'
    result.model.save(yml_path, overwrite=True)
    ModelGenerator.write_model_file_from_yaml(yml_path, str(model_path), logger=logger)

    save_fit_summary(result, logger, extra={'step': 'Final'})

    if checkpoint is not None:
        checkpoint.save_step(
            'final_refit', 1, 'completed',
            {'log_like': result.log_like, 'aic': result.aic, 'ts': result.ts, 'step_dir': str(result.step_dir)},
            metadata={'num_sources': num_sources},
        )

    logger.info(f'source_fitter complete: {num_sources} sources, -logL={result.log_like:.3f}')
    return result


def load_fit_result_from_step_dir(step_dir, logger) -> FitResult:
    """Reconstruct a FitResult from a previously-completed step's on-disk
    outputs, for resuming a run without refitting. Requires curModel.model
    and fit_summary.json to already exist in step_dir.
    """
    step_dir = Path(step_dir)
    model_file = step_dir / 'curModel.model'
    summary_file = step_dir / 'fit_summary.json'

    if not model_file.exists():
        raise FileNotFoundError(f'Cannot resume: {model_file} not found')
    if not summary_file.exists():
        raise FileNotFoundError(f'Cannot resume: {summary_file} not found')

    import threeML
    namespace = {'threeML': threeML}
    exec(open(model_file).read(), namespace)
    if 'model' not in namespace:
        raise ValueError(f"{model_file} did not define 'model'")
    model = namespace['model']

    with open(summary_file) as f:
        summary = json.load(f)

    model_map_path = step_dir / 'model_fit.hd5'
    residual_map_path = step_dir / 'residual_fit.hd5'

    result = FitResult(
        model=model,
        log_like=summary['log_like'],
        aic=summary['aic'],
        ts=summary.get('ts'),
        model_map_path=model_map_path if model_map_path.exists() else None,
        residual_map_path=residual_map_path if residual_map_path.exists() else None,
        step_dir=step_dir,
        fitter=None,
    )
    logger.info(f'Resumed FitResult from {step_dir}: -logL={result.log_like:.3f}, AIC={result.aic:.3f}, {len(model.sources)} sources')
    return result


def find_resume_point(directory_manager, logger):
    """Scan the Results tree for the most recently completed step with a
    fit_summary.json, and derive which sources' extension/spectrum tests
    already ran, for resuming an interrupted run.

    Returns (resume_step_dir, completed_extension_sources, completed_spectrum_sources).
    """
    import re
    results_root = directory_manager.get_step_results_dir('Step1-JointFit').parent

    ext_pattern = re.compile(r'^Step2-(?P<source>.+?)-Extension-(?P<shape>.+?)(?:-Pruned_.*)?$')
    spec_pattern = re.compile(r'^Step3-(?P<source>.+?)-Spectrum-(?P<spectrum>.+?)(?:-Pruned_.*)?$')

    completed = []
    for step_dir in results_root.iterdir():
        if not step_dir.is_dir():
            continue
        summary_file = step_dir / 'fit_summary.json'
        if not summary_file.exists():
            continue
        mtime = summary_file.stat().st_mtime

        m = ext_pattern.match(step_dir.name)
        if m:
            completed.append((mtime, step_dir, 'extension', m.group('source')))
            continue
        m = spec_pattern.match(step_dir.name)
        if m:
            completed.append((mtime, step_dir, 'spectrum', m.group('source')))
            continue
        if step_dir.name.startswith('Step1-JointFit-Iter') or step_dir.name == 'Step4-FinalRefit':
            completed.append((mtime, step_dir, step_dir.name, None))

    if not completed:
        logger.info('No completed steps found; nothing to resume from')
        return None, [], []

    completed.sort(key=lambda t: t[0])
    for mtime, step_dir, kind, source in completed:
        logger.info(f'Found completed step: {step_dir.name} (kind={kind}, source={source})')

    resume_step_dir = completed[-1][1]
    completed_extension_sources = sorted({s for _, _, k, s in completed if k == 'extension' and s})
    completed_spectrum_sources = sorted({s for _, _, k, s in completed if k == 'spectrum' and s})

    logger.info(f'Resume point: {resume_step_dir}')
    logger.info(f'Extension test already done for: {completed_extension_sources}')
    logger.info(f'Spectrum test already done for: {completed_spectrum_sources}')
    return resume_step_dir, completed_extension_sources, completed_spectrum_sources

from seeding.drips_seeder import DRIPSSeeder
def run_joint_fit(drip_model_path: Path, config, logger, directory_manager, max_reseed_iterations: int = 2) -> FitResult:
    """One in-process joint fit of DRIPS's full seed model. If the residual
    map still shows excess after fitting, re-run DRIPS on the residual to
    find additional sources, merge them in, and refit -- up to
    max_reseed_iterations times. Set fitting.disable_reseed: true in config
    to skip this reseed loop entirely and return after the first fit
    regardless of residual excess.
    """
    runner = FitRunner(
        config_path=str(config.config_file),
        logger=logger,
        roi_template=config.get('roi.roi_template_path'),
    )
    disable_reseed = config.get('fitting.disable_reseed', False)
    if disable_reseed:
        logger.info('fitting.disable_reseed is True; joint fit will not reseed even if residual excess is found')

    model_file = drip_model_path
    for iteration in range(max_reseed_iterations + 1):
        step_name = f'Step1-JointFit-Iter{iteration}'
        step_dir = directory_manager.get_step_results_dir(step_name)
        logger.info(f'Running joint fit ({model_file}) in {step_dir} [reseed iteration {iteration}]')
        result = runner.fit(
            model_file=str(model_file),
            step_dir=str(step_dir),
            compute_err=False,
            compute_TS=True,
            make_maps=True,
        )
        save_ts_values(result.ts, step_name, directory_manager, logger)
        _record_manifest(
            directory_manager, logger, step_dir,
            f'Joint fit, iteration {iteration} ({"initial DRIPS seed" if iteration == 0 else f"after reseed {iteration}"}) '
            f'-- {len(result.model.sources)} sources, -logL={result.log_like:.3f}',
        )
        resmap = _build_fit_maps(config, logger, directory_manager, result.step_dir, 'residual')
        has_excess = check_hotspots(resmap, result, config, logger)

        if disable_reseed:
            if has_excess:
                logger.info('Residual excess found, but fitting.disable_reseed is True; skipping reseed and returning current fit')
            else:
                logger.info('No residual excess found')
            return result

        if not has_excess:
            logger.info(f'No residual excess after iteration {iteration}; stopping reseed loop')
            return result
        if iteration == max_reseed_iterations:
            logger.info(f'Residual excess remains after {max_reseed_iterations} reseed iteration(s); giving up and returning current fit')
            return result

        logger.info(f'Residual excess found; re-running DRIPS on the residual map (iteration {iteration + 1})')
        reseed_step_dir = directory_manager.get_step_results_dir(f'Step0-Reseed{iteration + 1}')
        reseeder = DRIPSSeeder(
            config, logger, directory_manager, step_path=str(reseed_step_dir),
            sig_map_path_override=str(resmap),
        )
        new_model_path = reseeder.run()
        _record_manifest(
            directory_manager, logger, reseed_step_dir,
            f'DRIPS re-seeding on the residual of iteration {iteration}; candidate model at {new_model_path}',
        )

        merged_model = ModelGenerator.merge_new_sources(
            base_model=result.model, new_sources_model_path=str(new_model_path), logger=logger,
        )

        yml_path = f'{reseed_step_dir}/merged_model.yml'
        model_path = f'{reseed_step_dir}/merged_model.model'
        merged_model.save(yml_path, overwrite=True)
        ModelGenerator.write_model_file_from_yaml(yml_path, model_path, logger=logger)
        model_file = model_path

    return result  # unreachable, loop always returns


def _record_manifest(directory_manager, logger, step_dir, description):
    """Append a line to Results/RUN_MANIFEST.md describing what a step
    directory represents, in the order steps actually ran. Best-effort --
    never raises, since a manifest write failure shouldn't fail a fit."""
    try:
        results_root = Path(step_dir).parent
        manifest_path = results_root / 'RUN_MANIFEST.md'
        first_write = not manifest_path.exists()
        with open(manifest_path, 'a') as f:
            if first_write:
                f.write('# Pipeline run manifest\n\nChronological log of what each Results/ subdirectory represents.\n\n')
            ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            f.write(f'- `{Path(step_dir).name}` -- {description} ({ts})\n')
    except Exception as e:
        logger.warning(f'Could not write to RUN_MANIFEST.md: {e}')


def save_ts_values(ts_by_source: dict, step_label: str, directory_manager, logger) -> None:
    """Append TS values from one fit to Results/ts_history.csv (created on
    first call) and Results/ts_history.json (a running dict of
    step_label -> {source: ts}, rewritten each call). Call this anywhere
    trial_result.ts / winner.ts / refit_result.ts is computed, right after
    it's populated.

    Best-effort -- never raises, since a logging failure shouldn't fail a fit.
    """
    if not isinstance(ts_by_source, dict):
        return
    try:
        results_root = directory_manager.get_step_results_dir('Step1-JointFit-Iter0').parent
        csv_path = results_root / 'ts_history.csv'
        json_path = results_root / 'ts_history.json'

        first_write = not csv_path.exists()
        with open(csv_path, 'a') as f:
            if first_write:
                f.write('step,source,ts\n')
            for source, ts in ts_by_source.items():
                f.write(f'{step_label},{source},{ts}\n')

        history = {}
        if json_path.exists():
            with open(json_path) as f:
                history = json.load(f)
        history[step_label] = ts_by_source
        with open(json_path, 'w') as f:
            json.dump(history, f, indent=2, default=_json_safe)

        logger.info(f'Recorded TS values for {step_label}: {ts_by_source}')
    except Exception as e:
        logger.warning(f'Could not write to ts_history: {e}')