import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
import astropy.wcs.utils as astropy_utils

from scipy.ndimage import gaussian_filter
from skimage.filters import difference_of_gaussians
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import yaml


from core.pipeline_helpers import (
    load_hawc_data,
    make_plots,
    soft_floor,
    estimate_background_sigma,
    run_ps,
    run_ext,
    blob_filter_intensity,
    combine_blobs,
    remove_overlapping_blobs,
    compute_bright_frac,
    calculate_separation,
    circle_overlap,
    deduplicate_ps_group,
    ultimet,
    find_peak,
    threeML_model_from_sources,
)


class DRIPSSeeder():
    """A class to manage the seeding of Drips."""
    def __init__(
        self,
        config: object,
        logger: object,
        directory_manager: object,
        step_path,
        sig_map_path_override: Optional[str] = None,
    ):
        self.config = config
        self.logger = logger
        self.directory_manager = directory_manager
        self.step_path = step_path

        if sig_map_path_override is not None:
            self.initialmap = sig_map_path_override
            logger.info(f"Using overridden significance map: {self.initialmap}")
        else:
            try:
                self.initialmap = self.config.get('coordinates.sig_map_path')
                if self.initialmap is None:
                    raise ValueError("sig_map_path is not set in the config")
                logger.info(f"Using significance map from config: {self.initialmap}")
            except Exception as e:
                self.initialmap = directory_manager.get_datamap_dir() / 'significance_map.fits'
                logger.info(f"Using significance map created: {self.initialmap}")

        self.coord_sys = self.config.get('coordinates.coord_sys')
        if self.coord_sys not in ['C', 'G']:
            raise ValueError(f"Invalid coordinate system '{self.coord_sys}' in config. Must be 'C' or 'G'.")
        if self.coord_sys == 'C':
            if self.config.get('coordinates.ra') is not None:
                self.ra = float(self.config.get('coordinates.ra'))
                self.dec = float(self.config.get('coordinates.dec'))
                central_coord = SkyCoord(ra=self.ra*u.degree, dec=self.dec*u.degree, frame='icrs')
                self.l = central_coord.galactic.l.deg
                self.b = central_coord.galactic.b.deg
            else:
                raise ValueError("Coordinates not provided.")
        else:
            if self.config.get('coordinates.l') is not None or self.config.get('coordinates.b') is not None:
                self.l = float(self.config.get('coordinates.l'))
                self.b = float(self.config.get('coordinates.b'))
                
            else:
                raise ValueError("Coordinates not provided.")
        self.seed_coord_sys = 'G'
        self.x_length = self.config.get('coordinates.roi_x', 1.0)
        self.y_length = self.config.get('coordinates.roi_y', 1.0)
        if self.x_length < 2:
            self.x_length += 3
        if self.y_length < 2:
            self.y_length += 3
        
        self.out_dir = step_path
        self.logger.info(f"Using galactic coordinates from config: (l={self.l}, b={self.b})")
        self.logger.info(f"ROI size: {self.x_length}° x {self.y_length}°")
        self.logger.info(f"Output directory for seeding: {self.out_dir}")

        self.use_dbe = self.config.get('diffuse.use_diffuse_background')
        if self.use_dbe is True:
            self.dbe_template = self.config.get('diffuse.diffuse_template_path')
        else:
            self.dbe_template = None
        self.logger.info(f"Running seed model search in {self.out_dir}")
        os.makedirs(self.out_dir, exist_ok=True)
        self.SIG_THRESHOLD = 5.0
        self.SMEAR_RADII = [0.25, 0.3, 0.4, 0.5]

    def load_hawc_data(self) -> None:
        """Load significance map and set array, wcs, dims, pixel size."""
        self.logger.info(f"Loading HAWC data from {self.initialmap}")
        self.array, self.header, self.wcs, self.xnum, self.ynum, self.pixel_size = load_hawc_data( self.initialmap, self.l, self.b, self.x_length, self.y_length, self.seed_coord_sys )

    def plot_maps(
        self,
        array: np.ndarray,
        wcs: WCS,
        pixel_size: float,
        coord_sys: str,
        max_signif: float,
        vmin: float,
        vmax: float,
        threshold: float,
        contour: bool = True,
        title: str = 'Sky Map',
        hotspots: list = None,
        blobs: dict = None,
        labels: list = None,
    ) -> None:
        """Render a significance/normalised map to {out_dir}/{title}.png."""
        fig, _ = make_plots(
                array, wcs, pixel_size, coordsys=coord_sys,
                threshold=threshold, vmin=vmin, vmax=vmax, contour=contour, blobs=blobs,
                title=f'{title} (Max Sig {max_signif})',
                hotspots=hotspots, save_dir=None, pdf=False,
                cmap='ult', figsize=(10, 6),
                labels=labels,
            )
        plt.savefig(f"{self.out_dir}/{title}.png", dpi=300)
        plt.close()

    def normalise_image(self, array: np.ndarray) -> None:
        """Soft-floor and min-max normalise the significance map to [0,1]."""
        if np.max(self.array) < self.SIG_THRESHOLD:
            self.logger(f"    Below threshold ({np.max(self.array):.2f} < {self.SIG_THRESHOLD}sigma) — skipping.")
            fig_blank, ax_blank = plt.subplots(figsize=(8.5, 4))
            ax_blank.axis('off')
            ax_blank.text(0.5, 0.5,
                            f"No data > {self.SIG_THRESHOLD}sigma in file\n"
                            f"(Max = {np.max(self.array):.2f}sigma)",
                            fontsize=12, ha='center', va='center')
            raise ValueError(f"No excess found in the region")
        else:
            print(f"  Max significance {np.max(self.array):.2f}sigma exceeds threshold {self.SIG_THRESHOLD}sigma — proceeding with analysis.")
        if np.min(self.array) < -5:
            print("  Image softly floored to -5sigma")
            self.array = soft_floor(self.array, floor_min=-6, scale=1.0)

        self.norm_image = (self.array - self.array.min()) / (self.array.max() - self.array.min())

    def blob_detection(self, array: np.ndarray) -> None:
        """Run DoG blob detection across SMEAR_RADII; set final PS/EXT blobs."""
        self.all_ps_blobs,  self.all_ps_coords,  self.all_ps_radii  = [], [], []
        self.all_ext_blobs, self.all_ext_coords, self.all_ext_radii = [], [], []
        pbar = tqdm(self.SMEAR_RADII, desc='Processing Radii', leave=False)
        for radius in pbar:
            pbar.set_description(f"Processing radius {radius:.2f}°")
            smear_radius=radius
            pixel_smear = smear_radius/self.pixel_size

            print(f"Number of pixels corresponding to {smear_radius:.2f} smear radius = {pixel_smear:.2f}")

            dog_image = difference_of_gaussians(self.norm_image, 1, pixel_smear)
            sigma_resid = estimate_background_sigma(dog_image)
            print("Estimated background RMS:", sigma_resid)

            dog_final = np.where(dog_image > 2 * sigma_resid, dog_image, 0)
            dog_norm = (dog_final-np.min(dog_final))/(np.max(dog_final)-np.min(dog_final))
            extmap=gaussian_filter(self.norm_image-dog_norm, sigma=0.3/self.pixel_size)
            threshold_val = sigma_resid

            with ThreadPoolExecutor(max_workers=2) as ex:
                f_ps  = ex.submit(run_ps,  dog_final, self.pixel_size, threshold_val, self.border_pixels)
                f_ext = ex.submit(run_ext, extmap,    self.pixel_size, threshold_val, self.border_pixels)

            ps_blobs, ext_blobs = f_ps.result(), f_ext.result()

            print(f"Raw blobs — point source: {len(ps_blobs)}, extended: {len(ext_blobs)}")

            ps_filt, ps_coords, ps_radii = (
                    blob_filter_intensity(ps_blobs,  self.array, 5, self.wcs, self.pixel_size)
                    if len(ps_blobs)  > 0 else (np.empty((0,3)), [], [])
                )
            ext_filt, ext_coords, ext_radii = (
                blob_filter_intensity(ext_blobs, self.array, 5, self.wcs, self.pixel_size)
                if len(ext_blobs) > 0 else (np.empty((0,3)), [], [])
            )

            print(f"Sources after 5$\\sigma$ filtering: {len(ps_filt) + len(ext_filt)}")

            if len(ps_filt)  > 0:
                self.all_ps_blobs.append(ps_filt);   self.all_ps_coords.append(ps_coords);   self.all_ps_radii.append(ps_radii)
            if len(ext_filt) > 0:
                self.all_ext_blobs.append(ext_filt); self.all_ext_coords.append(ext_coords); self.all_ext_radii.append(ext_radii)


        combined_ps_blobs,  combined_ps_coords,  combined_ps_radii  = combine_blobs(self.all_ps_blobs,  self.all_ps_coords,  self.all_ps_radii)
        combined_ext_blobs, combined_ext_coords, combined_ext_radii = combine_blobs(self.all_ext_blobs, self.all_ext_coords, self.all_ext_radii)

        final_ps_blobs,  _,  _,  *_ = remove_overlapping_blobs(combined_ps_blobs,  combined_ps_coords,  combined_ps_radii)
        final_ext_blobs, _, _, *_ = remove_overlapping_blobs(combined_ext_blobs, combined_ext_coords, combined_ext_radii)

        self.final_ps_blobs,  self.final_ps_coords,  self.final_ps_radii  = blob_filter_intensity(final_ps_blobs,  self.array, 5.1, self.wcs, self.pixel_size)
        self.final_ext_blobs, self.final_ext_coords, self.final_ext_radii = blob_filter_intensity(final_ext_blobs, self.array, 5.1, self.wcs, self.pixel_size)

        print(f"  PS  after all cuts: {len(self.final_ps_blobs):>4}")
        print(f"  EXT after all cuts: {len(self.final_ext_blobs):>4}")

    def group_blobs(self) -> None:
        """Group overlapping PS blobs under their parent EXT blob."""
        self.groups = []
        matched_ps = set()

        for lb in self.final_ext_blobs:
            ly, lx, lr = lb
            matched = []
            for i, sb in enumerate(self.final_ps_blobs):
                sy, sx, sr = sb
                frac = self.overlap_fraction(ly, lx, lr, sy, sx, sr)
                if frac > 0.1:
                    matched.append((sb, frac))
                    matched_ps.add(i)
            self.groups.append((lb, matched))

        for i, sb in enumerate(self.final_ps_blobs):
            if i not in matched_ps:
                self.groups.append((None, [(sb, 0.0)]))

    def blob_filters(self) -> Tuple[list, list, list, list]:
        """Classify grouped blobs into kept/removed PS and EXT sources."""
        self.ext_removed_group, self.ps_removed_group, self.ext_filtered_group, self.ps_filtered_group = [], [], [], []
        for lb, sbs in self.groups:
            tag_ps     = 0
            tag_ex     = 0
            ps_flagged = []
            if lb is None:
                for sb, _ in sbs:
                    self.ps_filtered_group.append(sb)
                continue

            ly, lx, lr = lb
            ly, lx     = int(ly), int(lx)

            bright_frac = compute_bright_frac(self.array, ly, lx, lr)
            print(f"Intensity Fraction of pixels greater than 5 sigma detection threshold = {100*bright_frac:.1f}%")
            if bright_frac < 0.5:
                if len(sbs) > 1:
                    print("Larger blob is artifact of blob detection on subtracted map")
                    tag_ex += 1
                    for sb, _ in sbs:
                        ps_flagged.append(sb)
            else:
                coord_lb   = astropy_utils.pixel_to_skycoord(lx, ly, wcs=self.wcs).galactic
                if len(sbs) == 0:
                    self.ext_filtered_group.append(lb)
                    continue
                print(f"Larger blob coord = {coord_lb.l.deg, coord_lb.b.deg}")
                if len(sbs) > 1:
                    lb_ts = float(self.array[ly, lx])
                    sbs_ts = [float(self.array[int(sb[0]), int(sb[1])]) for sb, _ in sbs]

                    print(f"  Found {len(sbs)} smaller blobs overlapping larger blob")
                    print(f"  LB TS={lb_ts:.2f}, SB TSs={[f'{t:.2f}' for t in sbs_ts]}")

                    # Split small blobs into brighter-than-LB and dimmer-than-LB
                    brighter_sbs = [(sb, ov) for (sb, ov), ts in zip(sbs, sbs_ts) if ts >= lb_ts]
                    dimmer_sbs   = [(sb, ov) for (sb, ov), ts in zip(sbs, sbs_ts) if ts <  lb_ts]

                    if len(brighter_sbs) == len(sbs):
                        # ALL small blobs brighter than LB -> remove LB, keep all SBs
                        self.ps_filtered_group.extend([sb for sb, _ in sbs])
                        self.ext_removed_group.append(lb)
                        print("  All SBs brighter than LB -> keeping all SBs, removing LB")
                        continue
                    else:
                        # Some SBs are dimmer: keep LB + any SBs brighter than LB, drop dimmer SBs
                        self.ext_filtered_group.append(lb)
                        self.ps_filtered_group.extend([sb for sb, _ in brighter_sbs])
                        self.ps_removed_group.extend([sb for sb, _ in dimmer_sbs])
                        print(f"  LB is brightest or mixed -> keeping LB + {len(brighter_sbs)} brighter SBs, "
                            f"removing {len(dimmer_sbs)} dimmer SBs")
                        continue
                print(f"  No smaller blobs overlapping larger blob — tagging as EXT")
                for sb, _ in sbs:
                    sy, sx, sr  = sb
                    sy, sx      = int(sy), int(sx)
                    coord_sb    = astropy_utils.pixel_to_skycoord(sx, sy, wcs=self.wcs).galactic
                    sep_deg     = calculate_separation(coord_lb, coord_sb)
                    ovl         = circle_overlap(coord_lb, lr, coord_sb, sr, self.pixel_size)
                    delta_ts    = float(self.array[ly, lx]) - float(self.array[sy, sx])

                    if sep_deg < 0.1:
                        tag_ex +=1
                        ps_flagged.append(sb)
                        print(f"  Very close PS blob at (x={sx}, y={sy}, r={sr:.2f} pixels) with sep={sep_deg:.3f}° — tagging as EXT")
                        continue
                    if ovl is None:
                        if sep_deg > 0.3:
                            tag_ps += 1
                            ps_flagged.append(sb)
                            print(f"  PS blob at (x={sx}, y={sy}, r={sr:.2f} pixels) with sep={sep_deg:.3f}° and no overlap — tagging as PS")
                        else:
                            if delta_ts < 5:
                                tag_ex+=1
                                ps_flagged.append(sb)
                                print(f"  PS blob at (x={sx}, y={sy}, r={sr:.2f} pixels) with sep={sep_deg:.3f}° and no overlap but low TS difference ({delta_ts:.1f}) — tagging as EXT")
                            else:
                                self.ps_removed_group.append(sb)
                                print(f"  PS blob at (x={sx}, y={sy}, r={sr:.2f} pixels) with sep={sep_deg:.3f}° and no overlap but high TS difference ({delta_ts:.1f}) — tagging as PS")
                    elif ovl == 1.0:
                        if sep_deg > 0.3 and delta_ts >= 9:
                            tag_ps += 1;  tag_ex += 1;  ps_flagged.append(sb)
                            print(f"  PS blob at (x={sx}, y={sy}, r={sr:.2f} pixels) with sep={sep_deg:.3f}° and no overlap but high TS difference ({delta_ts:.1f}) — tagging as PS")
                        elif delta_ts >= 9:
                            tag_ps += 1;  ps_flagged.append(sb)
                            print(f"  PS blob at (x={sx}, y={sy}, r={sr:.2f} pixels) with sep={sep_deg:.3f}° and no overlap but high TS difference ({delta_ts:.1f}) — tagging as PS")
                        else:
                            self.ps_removed_group.append(sb)
                            print(f"  PS blob at (x={sx}, y={sy}, r={sr:.2f} pixels) with sep={sep_deg:.3f}° and no overlap but high TS difference ({delta_ts:.1f}) — tagging as PS")

            if tag_ex > 0:
                self.ext_removed_group.append(lb)
            else:
                self.ext_filtered_group.append(lb)
            print(f"PS flagged: {len(ps_flagged)}")
            self.ps_filtered_group.extend(ps_flagged)

        print(f"  Kept   — PS: {len(self.ps_filtered_group)}  EXT: {len(self.ext_filtered_group)}")
        print(f"  Removed— PS: {len(self.ps_removed_group)}  EXT: {len(self.ext_removed_group)}")

        self.ps_filtered_group, ps_dedup_removed = deduplicate_ps_group(
        self.ps_filtered_group, self.wcs, sep_threshold_deg=0.3
        )
        self.ps_removed_group.extend(ps_dedup_removed)

        print(f"Final — PS kept: {len(self.ps_filtered_group)}  EXT kept: {len(self.ext_filtered_group)}")
        print(f"        PS removed: {len(self.ps_removed_group)}  EXT removed: {len(self.ext_removed_group)}")
        return self.ps_filtered_group, self.ext_filtered_group, self.ps_removed_group, self.ext_removed_group

    def overlap_fraction(self, ly, lx, lr, sy, sx, sr) -> float:
        """Fraction of the smaller blob's area that overlaps with the larger blob."""
        dist = np.sqrt((sy - ly)**2 + (sx - lx)**2)
        if dist >= lr + sr:
            return 0.0
        if dist + sr <= lr:
            return 1.0

        r, R = sr, lr   # small, large radii
        d    = dist

        alpha = np.arccos((d**2 + r**2 - R**2) / (2 * d * r))  # half-angle at small circle center
        beta  = np.arccos((d**2 + R**2 - r**2) / (2 * d * R))  # half-angle at large circle center

        intersection = r**2 * alpha + R**2 * beta - 0.1 * (r**2 * np.sin(2*alpha) + R**2 * np.sin(2*beta))
        small_area   = np.pi * r**2

        return intersection / small_area

    def plot_filtering_results(self, title: str = 'FilteredBlobs') -> None:
        """Overlay kept PS/EXT blobs on the significance map and save PNG."""
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection=self.wcs)
        ult = ultimet(-5, 9, 4)
        im = ax.imshow(self.array, cmap=ult, vmin=-5, vmax=9)
        plt.colorbar(im, ax=ax, label=r'Significance($\sigma$)', fraction=0.046, pad=0.04)
        for lb in self.ext_filtered_group:
            ly, lx, lr = lb
            ly, lx = int(ly), int(lx)
            center_intensity_lb = self.array[ly, lx]

            circle_l = plt.Circle((lx, ly), lr, fill=False, edgecolor='black', linewidth=2.0, linestyle='--')
            ax.add_patch(circle_l)
            ax.scatter(lx, ly, s=80, c='black', marker='*', zorder=5)
            ax.text(lx, ly + lr + 10, f'TS={center_intensity_lb:.1f}',
                    color='black', fontsize=10, ha='center')


        for sb in self.ps_filtered_group:
            try:
                sy, sx, sr = sb
            except:
                sy, sx, sr = sb[0][0]
            sy, sx = int(sy), int(sx)
            center_intensity_sb = self.array[sy, sx]
            circle_s = plt.Circle((sx, sy), sr, fill=False, edgecolor='white', linewidth=2, linestyle='-')
            ax.add_patch(circle_s)
            ax.scatter(sx, sy, s=60, c='white', marker='*', zorder=5)
            ax.text(sx + sr + 20, sy + sr + 10, f'TS={center_intensity_sb:.1f}',
                    color='white', fontsize=10, ha='center')

        legend_handles = [
            Line2D([0], [0], color='black',  linewidth=2,   linestyle='--', label='larger blob (kept)'),
            Line2D([0], [0], color='cyan',   linewidth=1.5, linestyle='--', label='smaller blob (grouped)'),
            Line2D([0], [0], color='blue',   linewidth=1.2, linestyle=':',  label='smaller blob (ps pool)'),
        ]

        ax.set_xlim(0, self.xnum)
        ax.set_ylim(0, self.ynum)
        ax.legend(handles=legend_handles, fontsize=8, loc='upper right')
        ax.set_title('All filtered blobs — larger (white), grouped smaller (cyan), ps pool (blue)', fontsize=9)
        ax.set_xlabel('X (px)')
        ax.set_ylabel('Y (px)')
        plt.tight_layout()
        plt.savefig(f"{self.out_dir}/{title}.png", dpi=300)
        plt.close()

    def radius_to_sigma(self, R, fraction=0.6827):
        """Convert a blob circle radius to a Gaussian sigma for a containment fraction."""
        return R / np.sqrt(2 * np.log(1 / (1 - fraction)))

    def convert_to_coord(self, input_blobs, wcs, npix):
        """Convert pixel blobs to (blob, SkyCoord, angular-radius) triples."""
        blobs = []
        coords = []
        radius = []

        for y, x, r in input_blobs:
            coord = astropy_utils.pixel_to_skycoord(x, y, wcs=wcs).icrs

            blobs.append((y, x, r))
            coords.append(coord)
            radius.append(r * npix)
            print(f"Converting blob at (x={x}, y={y}, r={r:.2f} pixels) to coord (RA={coord.ra.deg:.3f}°, Dec={coord.dec.deg:.3f}°) with radius {r * npix:.2f}°")
        return blobs, coords, radius

    def save_model(self) -> None:
        """Convert blobs to coordinates and save results to YAML file, filtering to original ROI."""
        _, ext_coords, ext_radius = self.convert_to_coord(self.ext_filtered_group, self.wcs, self.pixel_size)
        _, ps_coords, ps_radius = self.convert_to_coord(self.ps_filtered_group, self.wcs, self.pixel_size)

        total_coords = ps_coords + ext_coords
        total_coords_ra = [i.ra.value for i in ps_coords] + [i.ra.value for i in ext_coords]
        total_coords_dec = [i.dec.value for i in ps_coords] + [i.dec.value for i in ext_coords]
        total_radius = ps_radius + ext_radius

        names = [f'Drip{i}' for i in range(len(total_coords))]

        # Create DataFrame with all sources
        df_all = pd.DataFrame({
            'Name': names,
            'ra': total_coords_ra,
            'dec': total_coords_dec,
            'ext': total_radius,
            'Circle Radius': total_radius,
            'Sigma Radius': [self.radius_to_sigma(R) for R in total_radius]
        })

        # Get original ROI (without the +3 extension)
        original_roi_x = self.config.get('coordinates.roi_x', 1.0)
        original_roi_y = self.config.get('coordinates.roi_y', 1.0)

        # Filter sources within original ROI bounds
        filtered_rows = []
        for _, row in df_all.iterrows():
            coord = SkyCoord(ra=row['ra']*u.degree, dec=row['dec']*u.degree, frame='icrs').galactic
            l_diff = abs(coord.l.deg - self.l)
            b_diff = abs(coord.b.deg - self.b)

            # Keep if within original ROI
            if l_diff <= original_roi_x and b_diff <= original_roi_y:
                filtered_rows.append(row)

        # Create filtered DataFrame
        if filtered_rows:
            self.filtered_df = pd.DataFrame(filtered_rows).reset_index(drop=True)
            self.filtered_df['Name'] = [f'Drip{i}' for i in range(len(self.filtered_df))]
        else:
            self.filtered_df = pd.DataFrame(columns=['Name', 'ra', 'dec', 'ext', 'Circle Radius', 'Sigma Radius'])

        # Save to YAML
        if len(self.filtered_df) > 0:
            sources_data = {
                'roi_info': {
                    'center_l': float(self.l),
                    'center_b': float(self.b),
                    'roi_x': float(original_roi_x),
                    'roi_y': float(original_roi_y),
                    'total_sources_detected': len(df_all),
                    'sources_in_roi': len(self.filtered_df)
                },
                'sources': [
                    {
                        'name': row['Name'],
                        'ra': float(row['ra']),
                        'dec': float(row['dec']),
                        'ext': float(row['ext']),
                        'circle_radius_deg': float(row['Circle Radius']),
                        'sigma_radius': float(row['Sigma Radius'])
                    }
                    for _, row in self.filtered_df.iterrows()
                ]
            }
            self.logger.info(f"Found {len(df_all)} sources, {len(self.filtered_df)} within original ROI")
        else:
            sources_data = {
                'roi_info': {
                    'center_l': float(self.l),
                    'center_b': float(self.b),
                    'roi_x': float(original_roi_x),
                    'roi_y': float(original_roi_y),
                    'total_sources_detected': len(df_all),
                    'sources_in_roi': 0
                },
                'sources': [],
                'note': 'No sources found within original ROI'
            }
            self.logger.info(f"Found {len(df_all)} sources, but none within original ROI")

        with open(f"{self.out_dir}/filtered_sources.yaml", 'w') as f:
            yaml.dump(sources_data, f, default_flow_style=False, sort_keys=False)

        self.logger.info(f"Results saved to: {self.out_dir}/filtered_sources.yaml")

        self.logger.info(f"Results saved to: {self.out_dir}/filtered_sources.yaml")

    def save_model_to_file(self, sources_dict, filtered_df, output_path="mymodel.model",
                       hermes_present=None, hermes_path=None) -> None:
        """
        Saves a threeML Model to a .model file matching the BEGINSOURCE/ENDSOURCE format.

        Parameters
        ----------
        model          : the threeML Model object
        sources_dict   : the `sources` dict you built (after removing spectrum keys)
        filtered_df    : your filtered DataFrame
        output_path    : path to write the .model file
        hermes_present : if True, prepend a URM/Hermes extended source block
        hermes_path    : path to the Hermes FITS file (required if hermes_present=True)
        """
        lines = []
        if hermes_present:
            if hermes_path is None:
                raise ValueError("hermes_path must be provided when hermes_present=True")

            lines += [
                "import astromodels",
                "##################################BEGINSOURCE##################################",
                'source_name = "URM"',
                "",
                "spectrum1 = threeML.Powerlaw()",
                f'shape1 = astromodels.Hermes(fits_file="{hermes_path}", ihdu=0)',
                "",
                "URM = threeML.ExtendedSource(source_name, spatial_shape=shape1, spectral_shape=spectrum1)",
                "",
                "fluxUnit = 1. / (threeML.u.TeV * threeML.u.cm ** 2 * threeML.u.s)",
                "",
                "shape1.N = 1",
                "shape1.N.fix = False",
                "shape1.N.bounds = (0.01, 100)",
                "",
                "spectrum1.K = 1",
                "spectrum1.K.fix = True",
                "",
                "spectrum1.index = 0",
                "spectrum1.index.fix = True",
                "###################################ENDSOURCE###################################",
                "",
            ]

        source_keys = sorted(sources_dict.keys(), key=lambda k: int(k.replace("source", "")))

        for key in source_keys:
            i = int(key.replace("source", ""))
            src       = sources_dict[key]
            src_name  = src.name
            is_extended = hasattr(src, 'spatial_shape')
            ra_val    = filtered_df['ra'].iloc[i]
            dec_val   = filtered_df['dec'].iloc[i]
            
            lines.append("##################################BEGINSOURCE##################################")
            lines.append(f'source_name = "{src_name}"')

            if not is_extended:
                sp = src.spectrum.main.shape

                lines += [
                    f"source_ra  = {ra_val}",
                    f"source_dec = {dec_val}",
                    "",
                    "spectrum = threeML.Powerlaw()",
                    f"{key} = threeML.PointSource(source_name, ra=source_ra, dec=source_dec, spectral_shape=spectrum)",
                    "fluxUnit = 1. / (threeML.u.keV * threeML.u.cm ** 2 * threeML.u.s)",
                    "",
                    f"spectrum.K = {sp.K.value:.3e} * fluxUnit",
                    "spectrum.K.fix = False",
                    "spectrum.K.bounds = (1e-26 * fluxUnit, 1e-20 * fluxUnit)",
                    "",
                    f"spectrum.piv = {10} * threeML.u.TeV",
                    "spectrum.piv.fix = True",
                    "",
                    f"spectrum.index = {sp.index.value}",
                    "spectrum.index.fix = False",
                    "spectrum.index.bounds = (-3., -1.)",
                    "",
                    f"{key}.position.ra.free = True",
                    f"{key}.position.ra.bounds = (({ra_val} -1.0), ({ra_val} +1.0)) * threeML.u.degree",
                    f"{key}.position.dec.free = True",
                    f"{key}.position.dec.bounds = (({dec_val} -1.0), ({dec_val} +1.0)) * threeML.u.degree",
                ]
            else:
                sp    = src.spectrum.main.shape
                morph = src.spatial_shape

                lines += [
                    "",
                    "spectrum = threeML.Powerlaw()",
                    "shape = threeML.Gaussian_on_sphere()",
                    f"{key} = threeML.ExtendedSource(source_name, spatial_shape=shape, spectral_shape=spectrum)",
                    "fluxUnit = 1. / (threeML.u.keV * threeML.u.cm ** 2 * threeML.u.s)",
                    "",
                    f"shape.lon0 = {morph.lon0.value} * threeML.u.degree",
                    "shape.lon0.fix = False",
                    f"shape.lon0.bounds = (({morph.lon0.value} -1.0), ({morph.lon0.value} +1.0)) * threeML.u.degree",
                    "",
                    f"shape.lat0 = {morph.lat0.value} * threeML.u.degree",
                    "shape.lat0.fix = False",
                    f"shape.lat0.bounds = (({morph.lat0.value} -1.0), ({morph.lat0.value} +1.0)) * threeML.u.degree",
                    "",
                    f"shape.sigma = {morph.sigma.value}",
                    "shape.sigma.fix = False",
                    "shape.sigma.bounds = (0.01, 3.0)",
                    "",
                    f"spectrum.K = {sp.K.value:.3e} * fluxUnit",
                    "spectrum.K.fix = False",
                    "spectrum.K.bounds = (1e-26 * fluxUnit, 1e-20 * fluxUnit)",
                    "",
                    f"spectrum.piv = {10} * threeML.u.TeV",
                    "spectrum.piv.fix = True",
                    "",
                    f"spectrum.index = {sp.index.value}",
                    "spectrum.index.fix = False",
                    "spectrum.index.bounds = (-3., -1.)",
                ]

            lines.append("###################################ENDSOURCE###################################")
            lines.append("")


        all_vars = ("URM, " if hermes_present else "") + ", ".join(source_keys)
        lines.append(f"model = threeML.Model({all_vars})")
        lines.append("")

        with open(output_path, "w") as f:
            f.write("\n".join(lines))

        print(f"Model saved to: {output_path}")

    def run_blob_detection(self) -> None:
        """Load data, normalise, detect blobs, and group them."""
        self.load_hawc_data()
        self.border_pixels = int(0.5 / self.pixel_size)
        print(f"  Map shape  : {self.ynum} × {self.xnum}  |  pixel size: {self.pixel_size:.4f}°")
        print(f"  Sig range  : [{np.min(self.array):.2f}, {np.max(self.array):.2f}]")
        self.max_signif = find_peak(self.array, self.wcs)

        self.plot_maps(self.array, self.wcs, self.pixel_size, self.seed_coord_sys, self.max_signif, -5, 15, 5, contour=True, title='SkyImage', labels=['4hwc'])

        self.normalise_image(self.array)

        self.plot_maps(self.norm_image, self.wcs, self.pixel_size, self.seed_coord_sys, self.max_signif, 0, 1, 0.4, contour=False, title='NormalisedImage', labels=['4hwc'])

        self.blob_detection(self.norm_image)

        self.blobs_dict = {'psblobs': self.final_ps_blobs, 'extblobs': self.final_ext_blobs}

        self.plot_maps(self.array, self.wcs, self.pixel_size, self.seed_coord_sys, self.max_signif, -5, 15, 5, contour=True, title='SkyImage_Blobs', blobs=self.blobs_dict, labels=['4hwc'])

        self.group_blobs()

    def run_filtering(self) -> None:
        """Run blob classification/filtering step."""
        self.blob_filters()
        print(f"  Kept   — PS: {len(self.ps_filtered_group)}  EXT: {len(self.ext_filtered_group)}")
        print(f"  Removed— PS: {len(self.ps_removed_group)}  EXT: {len(self.ext_removed_group)}")

    def run_source_detection(self) -> None:
        """Full detection: blobs -> filtering -> plots -> seed YAML."""
        self.run_blob_detection()
        self.run_filtering()
        self.plot_filtering_results(title='FilteredBlobs_Labels')
        self.save_model()
        self.plot_maps(self.array, self.wcs, self.pixel_size, self.seed_coord_sys, self.max_signif, -5, 15, 5, contour=True, title='Source Seeds', hotspots=self.filtered_df, labels=['4haaawc'])

    def run(self) -> None:
        """Original SourceSeedDetector.run(): detection + threeML model file.

        Extracted verbatim; produces filtered_df, allmodel, sources, and the
        curModel.model file.
        """
        if 'curModel.model' in os.listdir(self.out_dir):
            self.logger.info(f"Found existing model file in {self.out_dir}, skipping seeding.")
            model_path = Path(self.out_dir) / 'curModel.model'
            return model_path
        self.run_blob_detection()
        self.run_filtering()
        self.plot_filtering_results(title='FilteredBlobs_Labels')
        self.save_model()
        self.plot_maps(self.array, self.wcs, self.pixel_size, self.seed_coord_sys, self.max_signif, -5, 15, 5, contour=True, title='Source Seeds', hotspots=self.filtered_df, labels=['4haaawc'])
        self.allmodel, self.sources = threeML_model_from_sources(self.filtered_df)
        self.save_model_to_file(
            self.sources, self.filtered_df,
            output_path        = f"{self.out_dir}/curModel.model",
            hermes_present     = self.use_dbe,
            hermes_path        = self.dbe_template,
        )
        model_path = Path(self.out_dir) / 'curModel.model'
        return model_path