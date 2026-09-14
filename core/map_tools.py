"""
Map Generation Tools

Extracted from: pipeline_map_maker.py
Creates HealpixSigFluxMap maps from FITS files using aerie-apps.
"""

from pathlib import Path
from typing import List, Optional, Dict
import subprocess
import logging


class MapGenerator:
    """Generate HEALPix significance and flux maps from FITS data"""
    
    @staticmethod
    def create_healpix_map(
        input_fits_files: List[str],
        energy_bins: List[str],
        detector_response: str,
        ra_center: float,
        dec_center: float,
        roi_x: float,
        roi_y: float,
        output_file: str,
        spectral_index: float = 2.6,
        pivot_energy: int = 7,
        logger: Optional[object] = None,
        pixi_manifest_path: Optional[str] = None,
    ) -> Optional[Path]:
        """Create HEALPix map using aerie-apps-HealpixSigFluxMap

        Parameters:
        -----------
        input_fits_files : list of str
            List of input FITS file paths
        energy_bins : list of str
            List of energy bin labels (e.g., ['B2C0', 'B3C0', ...])
        detector_response : str
            Path to detector response file
        ra_center : float
            Right Ascension center (degrees)
        dec_center : float
            Declination center (degrees)
        roi_x : float
            Region of Interest X extent (degrees)
        roi_y : float
            Region of Interest Y extent (degrees)
        output_file : str
            Output HEALPix map filename
        spectral_index : float, optional
            Spectral index assumption (default: 2.6)
        pivot_energy : int, optional
            Pivot energy (default: 7)
        logger : object, optional
            Logger instance
        pixi_manifest_path : str, optional
            Directory (or file) containing the pixi project that provides
            aerie-apps-HealpixSigFluxMap (e.g. the aerie repo root). Without
            this, `pixi run` only finds the tool if the calling process's cwd
            happens to be inside that project. Matches the pattern already
            used by ALPSSeeder._find_next_hotpsot (alps.pixi_aerie_folder).

        Returns:
        --------
        Path or None
            Path to created map file, or None if failed
        """
        input_fits_files = [str(f) for f in input_fits_files]
        output_file = str(output_file)
        detector_response = str(detector_response)
        
        # Validate inputs
        if not input_fits_files:
            if logger:
                logger.error("No input FITS files provided")
            return None
        
        if len(input_fits_files) != len(energy_bins):
            if logger:
                logger.error(
                    f"Number of FITS files ({len(input_fits_files)}) "
                    f"doesn't match number of energy bins ({len(energy_bins)})"
                )
            return None
        
        # Check file existence
        for fits_file in input_fits_files:
            if not Path(fits_file).exists():
                if logger:
                    logger.error(f"FITS file not found: {fits_file}")
                return None
        
        if not Path(detector_response).exists():
            if logger:
                logger.error(f"Detector response file not found: {detector_response}")
            return None
        
        if logger:
            logger.info(f"Creating HEALPix map from {len(input_fits_files)} FITS files")
            logger.info(f"Energy bins: {energy_bins}")
        
        # Adjust declination if negative (HAWC convention)
        dec_adjusted = dec_center if dec_center >= 0 else 360 + dec_center
        # Build aerie-apps-HealpixSigFluxMap command
        pixi_prefix = ["pixi", "run"]
        if pixi_manifest_path is not None:
            pixi_prefix += ["-e", "threeml", "--manifest-path", str(pixi_manifest_path)]
        cmd = (
            pixi_prefix + ["aerie-apps-HealpixSigFluxMap"]
            + ["-i"] + input_fits_files
            + ["-b"] + energy_bins
            + ["-d", str(detector_response)]
            + ["--index", str(spectral_index)]
            + ["--pivot", str(pivot_energy)]
            + ["--window", str(ra_center), str(dec_adjusted), str(roi_x + 1), str(roi_y + 1)]
            + ["--negFlux", "--negSignif"]
            + ["-o", str(output_file)]
        )
        
        if logger:
            logger.info("Executing HealpixSigFluxMap command:")
            logger.debug(" ".join(cmd))
        

        try:
            result = subprocess.run(cmd, check=True)
            output_path = Path(output_file)
            if output_path.exists():
                if logger:
                    logger.info(f"HEALPix map created successfully: {output_file}")
                return output_path
            else:
                if logger:
                    logger.error(f"HealpixSigFluxMap did not create output file")
                return None
        
        except subprocess.CalledProcessError as e:
            if logger:
                logger.error(f"HealpixSigFluxMap command failed: {e}")
                logger.error(f"STDOUT: {e.stdout}")
                logger.error(f"STDERR: {e.stderr}")
            return None
        
        except Exception as e:
            if logger:
                logger.error(f"Error running HealpixSigFluxMap: {e}")
            return None
    
    @staticmethod
    def find_fits_files_by_bins(
        data_directory: str,
        energy_bins: List[str],
        logger: Optional[object] = None
    ) -> Optional[Dict[str, str]]:
        """Find FITS files matching energy bin names
        
        Parameters:
        -----------
        data_directory : str
            Directory containing FITS files
        energy_bins : list of str
            Energy bin labels to search for
        logger : object, optional
            Logger instance
        
        Returns:
        --------
        dict or None
            Dictionary mapping bin name to FITS file path, or None if not found
        
        Examples:
        ---------
        >>> files = MapGenerator.find_fits_files_by_bins(
        ...     './data/fits_files',
        ...     ['B2C0', 'B3C0', 'B4C0']
        ... )
        >>> for bin_name, file_path in files.items():
        ...     print(f"{bin_name}: {file_path}")
        B2C0: ./data/fits_files/model_binB2C0.fits.gz
        """
        data_path = Path(data_directory)
        found_files = {}
        
        if logger:
            logger.info(f"Searching for FITS files in {data_directory}")
            logger.info(f"Looking for bins: {energy_bins}")
        
        for bin_name in energy_bins:
            # Search for files matching this bin
            search_pattern = f"*{bin_name}*.fits*"
            matches = sorted(data_path.glob(search_pattern))
            
            if matches:
                file_path = str(matches[0])  # Use first match
                found_files[bin_name] = file_path
                if logger:
                    logger.debug(f"Found bin {bin_name}: {file_path}")
            else:
                if logger:
                    logger.warning(f"No FITS file found for bin {bin_name}")
        
        if not found_files:
            if logger:
                logger.error(f"No FITS files found for any of the specified bins")
            return None
        
        if logger:
            logger.info(f"Found {len(found_files)}/{len(energy_bins)} energy bins")
        
        return found_files
    
    @staticmethod
    def organize_fits_files_by_bins(
        file_bin_mapping: Dict[str, str]
    ) -> List[str]:
        """Organize FITS file paths in order of bin labels
        
        Parameters:
        -----------
        file_bin_mapping : dict
            Dictionary mapping bin name to file path
        
        Returns:
        --------
        list of str
            Ordered list of FITS file paths
        
        Examples:
        ---------
        >>> mapping = {'B2C0': 'path/to/binB2C0.fits', 'B3C0': 'path/to/binB3C0.fits'}
        >>> files = MapGenerator.organize_fits_files_by_bins(mapping)
        """
        # Extract bin numbers to sort
        def extract_bin_number(bin_name):
            # Extract integer from bin name like "B2C0" -> 2
            import re
            match = re.search(r'B(\d+)', bin_name)
            return int(match.group(1)) if match else 0
        
        # Sort by bin number
        sorted_bins = sorted(file_bin_mapping.keys(), key=extract_bin_number)
        
        # Return ordered file paths
        return [file_bin_mapping[bin_name] for bin_name in sorted_bins]
