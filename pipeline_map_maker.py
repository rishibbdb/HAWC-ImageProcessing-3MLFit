#!/usr/bin/env python
"""
Map Maker - Simplified standalone script for HAWC map generation.

Takes HDF5 files, converts to FITS, and generates Healpix maps.
Or directly generates maps from existing FITS files using manual bin list.
"""

from __future__ import print_function
from builtins import range
from hawc_hal.maptree import map_tree_factory
from astropy.io import fits
from pathlib import Path
from datetime import datetime

import healpy as hp
import numpy as np
import argparse
import os
import subprocess
import logging
from typing import List, Optional, Dict


class MapLogger:
    """Simple logging system for map processing"""
    
    def __init__(self, log_file: Optional[str] = None, verbose: bool = False):
        self.logger = logging.getLogger('MapMaker')
        level = logging.DEBUG if verbose else logging.INFO
        self.logger.setLevel(level)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(level)
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)
        
        # File handler if specified
        if log_file:
            fh = logging.FileHandler(log_file)
            fh.setLevel(level)
            fh.setFormatter(formatter)
            self.logger.addHandler(fh)
    
    def info(self, msg):
        self.logger.info(msg)
    
    def debug(self, msg):
        self.logger.debug(msg)
    
    def warning(self, msg):
        self.logger.warning(msg)
    
    def error(self, msg):
        self.logger.error(msg)


def convert_hd5_to_fits(dir, filename, outfile, logger: MapLogger) -> List[str]:
    """
    Convert HDF5 map tree to FITS files.
    
    Parameters:
    -----------
    dir : str
        Directory containing the HDF5 file
    filename : str
        Name of the HDF5 file
    outfile : str
        Prefix for output FITS files
    logger : MapLogger
        Logger instance
    
    Returns:
    --------
    list : Paths to created FITS files
    """
    input_filepath = os.path.join(dir, filename)
    if not os.path.isfile(input_filepath):
        logger.error(f"HDF5 file not found: {input_filepath}")
        raise FileNotFoundError(f"File not found: {input_filepath}")

    logger.info(f"Converting HDF5 to FITS: {input_filepath}")
    
    # Export the entire map tree (full sky)
    maptree = map_tree_factory(input_filepath, None)

    now = datetime.now()
    startMJD = 56987.9286332

    # FITS header configuration
    FITS_COMMENT = "FITS (Flexible Image Transport System) format is defined in 'Astronomy and Astrophysics', volume 376, page 359; bibcode: 2001A&A...376..359H"

    primary_keys = [
        'COMMENT', 'COMMENT', 'DATE', 'STARTMJD', 'STOPMJD',
        'NEVENTS', 'TOTDUR', 'DURATION', 'MAPTYPE', 'MAXDUR',
        'MINDUR', 'EPOCH', 'MAPFILETYPE'
    ]

    primary_values = [
        FITS_COMMENT, FITS_COMMENT, "{0}".format(now),
        56987.9286332451, 58107.2396848326,
        -1.0, 24412.9020670185, 1.9943578604616, 'duration', -1.0,
        -1.0, 'unknown', 'duration'
    ]

    primary_comments = [
        "file does conform to FITS standard",
        "number of bits per data pixel",
        "number of data axes",
        "FITS dataset may contain extension",
        "MJD of first event",
        "MJD of last event",
        "Number of events in map",
        "Total integration time [hours]",
        "Avg integration time [hours]",
        "e.g. Skymap, Moonmap",
        "Max integration time [hours]",
        "Min integration time [hours]",
        "e.g. J2000, current, J2016, B1950, etc.",
        "e.g. standard, duration, integration"
    ]

    labels = ['data map', 'background map', 'exposure map']
    label_format = [np.float64 for _ in range(len(labels))]
    label_units = ['unknown' for _ in range(len(labels))]

    output_files = []
    os.makedirs(dir, exist_ok=True)

    # Process each analysis bin
    for analysis_bin in maptree.analysis_bins_labels:
        map_bin = maptree[analysis_bin]
        
        # Get map properties
        nside = map_bin.nside
        npix = map_bin.npix
        scheme = map_bin.scheme
        transits = map_bin.n_transits

        nest_scheme = False
        if scheme.lower() == 'nested':
            nest_scheme = True

        # Extract map data
        data = map_bin.observation_map.as_dense()
        bkg = map_bin.background_map.as_dense()
        zeros = np.full(npix, 9e9)

        # Create output filename
        outFileName = os.path.join(dir, "{0}_bin{1}.fits.gz".format(outfile, analysis_bin))
        
        # Write FITS file
        hp.fitsfunc.write_map(
            outFileName, (data, bkg, zeros),
            column_names=labels, column_units=label_units, dtype=label_format,
            partial=False, fits_IDL=True, overwrite=True, nest=nest_scheme
        )

        # Add header cards
        with fits.open(outFileName, 'update') as hdu1:
            hdr = hdu1[0].header

            for i, key in enumerate(primary_keys):
                val = primary_values[i]
                comment = primary_comments[i]

                if key == 'TOTDUR':
                    val = 24.0 * transits
                elif key == 'STOPMJD':
                    val = startMJD + transits

                hdr[key] = (val, comment)

        logger.info(f"FITS file written: {outFileName}")
        output_files.append(outFileName)
    
    return output_files


class HealpixMapMaker:
    """Generate Healpix maps from FITS files using aerie-apps-HealpixSigFluxMap"""
    
    def __init__(self, logger: MapLogger):
        self.logger = logger
    
    def make_maps(self, input_files: List[str], bins: List[str], 
                  det_res: str, ra: float, dec: float, 
                  roi_x: float, roi_y: float, output_file: str) -> Optional[str]:
        """
        Create maps using HealpixSigFluxMap.
        
        Parameters:
        -----------
        input_files : list
            List of input FITS files
        bins : list
            List of bin names corresponding to input files
        det_res : str
            Path to detector response file
        ra : float
            Right Ascension (degrees)
        dec : float
            Declination (degrees)
        roi_x : float
            Region of Interest X radius
        roi_y : float
            Region of Interest Y radius
        output_file : str
            Output map filename
        
        Returns:
        --------
        str : Path to created map file, or None if failed
        """
        if not input_files:
            self.logger.error("No input files provided")
            return None
        
        if len(input_files) != len(bins):
            self.logger.error(f"Number of input files ({len(input_files)}) doesn't match number of bins ({len(bins)})")
            return None
        
        if not os.path.isfile(det_res):
            self.logger.error(f"Detector response file not found: {det_res}")
            return None
        
        self.logger.info(f"Creating map from {len(input_files)} input files")
        self.logger.info(f"Bins: {bins}")
        
        # Adjust declination if negative
        if dec < 0:
            dec = 360 + dec
        
        # Build the command
        cmd = (
            ["pixi", "run", "aerie-apps-HealpixSigFluxMap"]
            + ["-i"] + input_files
            + ["-b"] + bins
            + ["-d", str(det_res)]
            + ["--index", "2.7"]
            # + ["--extension", "0.5"]
            + ["--pivot", "7"]
            + ["--window", str(ra), str(dec), str(roi_x + 1), str(roi_y + 1)]
            # + ["--window", str(ra), str(dec), str(3), str(3)]
            + ["--negFlux", "--negSignif"]
            + ["-o", str(output_file)]
        )
        
        self.logger.info("Executing HealpixSigFluxMap command:")
        self.logger.debug(" ".join(cmd))
        
        try:
            subprocess.run(cmd, check=True)
            self.logger.info(f"Map created successfully: {output_file}")
            return output_file
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to create map: {e}")
            return None
    
    def find_fits_files_by_bins(self, data_dir: str, bins: List[str], 
                                pattern: str = "*bin*") -> Optional[Dict[str, str]]:
        """
        Find FITS files matching bin names in a directory.
        
        Parameters:
        -----------
        data_dir : str
            Directory to search for FITS files
        bins : list
            List of bin names to match
        pattern : str
            Search pattern (default: "*bin*")
        
        Returns:
        --------
        dict : Mapping of bin name to file path, or None if no files found
        """
        data_path = Path(data_dir)
        found_files = {}
        
        self.logger.info(f"Searching for FITS files in {data_dir}")
        self.logger.info(f"Looking for bins: {bins}")
        
        for bin_name in bins:
            # Search for files matching this bin
            search_pattern = f"*{bin_name}*.fits*"
            matched = list(data_path.glob(search_pattern))
            
            if matched:
                file_path = str(matched[0])
                found_files[bin_name] = file_path
                self.logger.info(f"Found bin {bin_name}: {file_path}")
            else:
                self.logger.warning(f"No FITS file found for bin {bin_name}")
        
        if not found_files:
            self.logger.error(f"No FITS files found for any of the specified bins")
            return None
        
        self.logger.info(f"Found {len(found_files)}/{len(bins)} bins")
        return found_files


def create_argument_parser() -> argparse.ArgumentParser:
    """Create and configure argument parser"""
    parser = argparse.ArgumentParser(
        description="HAWC Map Maker: Convert HDF5 to FITS and generate maps, or use existing FITS files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:

  # Convert residual HDF5 to FITS, then make maps
  %(prog)s hdf5 -d data/ -f residual_fit.hd5 -o output/ -p residual \
    --ra 45.0 --dec -15.0 --roi-x 5 --roi-y 5 \
    --det-res response.fits

  # Use existing FITS files with manual bin list
  %(prog)s fits -d data/ -b B7C0Ej B8C0Ej B9C0Ej \
    --ra 45.0 --dec -15.0 --roi-x 5 --roi-y 5 \
    --det-res response.fits
        """
    )
    
    subparsers = parser.add_subparsers(dest='mode', help='Operation mode', required=True)
    
    # Mode 1: HDF5 to FITS then maps
    hdf5_parser = subparsers.add_parser('hdf5', help='Convert HDF5 to FITS then make maps')
    
    hdf5_parser.add_argument(
        '-d', '--data-dir',
        type=str,
        required=True,
        help='Directory containing HDF5 file'
    )
    
    hdf5_parser.add_argument(
        '-f', '--filename',
        type=str,
        required=True,
        help='HDF5 filename (e.g., residual_fit.hd5)'
    )
    
    hdf5_parser.add_argument(
        '-o', '--output-dir',
        type=str,
        required=True,
        help='Output directory for FITS files and maps'
    )
    
    hdf5_parser.add_argument(
        '-p', '--prefix',
        type=str,
        default='map',
        help='Prefix for output FITS files (default: map)'
    )
    
    hdf5_parser.add_argument(
        '-M', '--map-output',
        type=str,
        default='map.fits',
        help='Output map filename (default: map.fits)'
    )
    
    # Mode 2: Direct maps from existing FITS
    fits_parser = subparsers.add_parser('fits', help='Create maps from existing FITS files')
    
    fits_parser.add_argument(
        '-d', '--data-dir',
        type=str,
        required=True,
        help='Directory containing FITS files'
    )
    
    fits_parser.add_argument(
        '-b', '--bins',
        type=str,
        nargs='+',
        required=True,
        help='Bin names to search for (space-separated, e.g., B7C0Ej B8C0Ej B9C0Ej)'
    )
    
    fits_parser.add_argument(
        '-M', '--map-output',
        type=str,
        default='map.fits',
        help='Output map filename (default: map.fits)'
    )
    
    fits_parser.add_argument(
        '-o', '--output-dir',
        type=str,
        help='Output directory for map file (if different from data directory)'
    )
    
    # Common map generation parameters for both modes
    for p in [hdf5_parser, fits_parser]:
        p.add_argument(
            '--ra',
            type=float,
            required=True,
            help='Right Ascension (degrees)'
        )
        
        p.add_argument(
            '--dec',
            type=float,
            required=True,
            help='Declination (degrees, negative for south)'
        )
        
        p.add_argument(
            '--roi-x',
            type=float,
            required=True,
            help='Region of Interest X radius (degrees)'
        )
        
        p.add_argument(
            '--roi-y',
            type=float,
            required=True,
            help='Region of Interest Y radius (degrees)'
        )
        
        p.add_argument(
            '--det-res',
            type=str,
            required=True,
            help='Path to detector response file'
        )
        
        p.add_argument(
            '-v', '--verbose',
            action='store_true',
            help='Enable verbose logging'
        )
        
        p.add_argument(
            '--log-file',
            type=str,
            help='Save logs to file'
        )
    
    return parser


def main():
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Setup logging
    log_file = args.log_file if hasattr(args, 'log_file') else None
    logger = MapLogger(log_file=log_file, verbose=args.verbose)
    
    try:
        if args.mode == 'hdf5':
            logger.info("="*80)
            logger.info("MODE 1: HDF5 TO FITS CONVERSION AND MAP GENERATION")
            logger.info("="*80)
            
            # Step 1: Convert HDF5 to FITS
            logger.info("Step 1: Converting HDF5 to FITS")
            os.makedirs(args.output_dir, exist_ok=True)
            
            fits_files = convert_hd5_to_fits(
                dir=args.data_dir,
                filename=args.filename,
                outfile=args.prefix,
                logger=logger
            )
            
            if not fits_files:
                logger.error("No FITS files were created")
                return 1
            
            logger.info(f"Successfully created {len(fits_files)} FITS files")
            
            # Step 2: Extract bin names from FITS files
            bin_names = []
            for fits_file in fits_files:
                basename = os.path.basename(fits_file)
                bin_name = basename.split('_bin')[-1].split('.fits')[0]
                bin_names.append(bin_name)
            
            logger.info(f"Extracted bin names: {bin_names}")
            
            # Step 3: Make maps
            logger.info("Step 2: Creating Healpix maps")
            map_maker = HealpixMapMaker(logger)
            output_path = os.path.join(args.output_dir, args.map_output)
            
            result = map_maker.make_maps(
                input_files=fits_files,
                bins=bin_names,
                det_res=args.det_res,
                ra=args.ra,
                dec=args.dec,
                roi_x=args.roi_x,
                roi_y=args.roi_y,
                output_file=output_path
            )
            
            if result:
                logger.info("="*80)
                logger.info("HDF5 TO FITS CONVERSION AND MAP GENERATION COMPLETED SUCCESSFULLY")
                logger.info(f"Output map: {result}")
                logger.info("="*80)
                return 0
            else:
                logger.error("Failed to create map")
                return 1
        
        elif args.mode == 'fits':
            logger.info("="*80)
            logger.info("MODE 2: DIRECT MAP GENERATION FROM EXISTING FITS FILES")
            logger.info("="*80)
            
            # Find FITS files for the specified bins
            map_maker = HealpixMapMaker(logger)
            found_files = map_maker.find_fits_files_by_bins(
                data_dir=args.data_dir,
                bins=args.bins
            )
            
            if not found_files:
                logger.error("Could not find FITS files for the specified bins")
                return 1
            
            # Order files to match bin order
            input_files = [found_files[bin_name] for bin_name in args.bins if bin_name in found_files]
            matched_bins = [bin_name for bin_name in args.bins if bin_name in found_files]
            
            if len(input_files) == 0:
                logger.error("No valid FITS files found")
                return 1
            
            logger.info(f"Using {len(input_files)} FITS files for map generation")
            
            # Create maps - use output-dir if specified, otherwise use data-dir
            output_directory = args.output_dir if args.output_dir else args.data_dir
            os.makedirs(output_directory, exist_ok=True)
            output_path = os.path.join(output_directory, args.map_output)
            result = map_maker.make_maps(
                input_files=input_files,
                bins=matched_bins,
                det_res=args.det_res,
                ra=args.ra,
                dec=args.dec,
                roi_x=args.roi_x,
                roi_y=args.roi_y,
                output_file=output_path
            )
            
            if result:
                logger.info("="*80)
                logger.info("MAP GENERATION COMPLETED SUCCESSFULLY")
                logger.info(f"Output map: {result}")
                logger.info("="*80)
                return 0
            else:
                logger.error("Failed to create map")
                return 1
    
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())