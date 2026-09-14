from astropy.io import fits
from matplotlib.colors import LogNorm
from functools import reduce
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import LinearSegmentedColormap
from reproject import reproject_from_healpix
from astropy.wcs import WCS
from threeML import *
from astropy.coordinates import Angle, SkyCoord
import astropy.units as u
import astropy.wcs.utils as astropy_utils
from astropy.io.fits import Header
import astropy
from astropy.wcs.utils import pixel_to_skycoord
import math
import tempfile
import urllib
import matplotlib.cm as cm
from datetime import datetime
import healpy as hp
import yaml
import re
from astropy.table import Table
from matplotlib.patches import Ellipse
from astropy.coordinates import ICRS
from astroquery.simbad import Simbad
import sys
import warnings
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
import sys
import warnings
warnings.filterwarnings("ignore")
sys.path.append("imagecatalog/")
import numpy as np
import sys, os, re
sys.path.append(os.path.abspath(".."))

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
import matplotlib.backends.backend_pdf as mpdf

from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.stats import sigma_clip
import astropy.wcs.utils as astropy_utils

from scipy.ndimage import gaussian_filter
from skimage.filters import difference_of_gaussians
from skimage.feature import blob_dog
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import yaml
import pandas as pd
import time


#Import HESS catalog
hess_catalog = Table.read("datasets/hgps_catalog_v1.fits.gz")
hess_sourcename = hess_catalog['Source_Name']
hess_coords = SkyCoord(l=hess_catalog["GLON"], b=hess_catalog["GLAT"], frame="galactic")

#Import HAWC Catalog
hawcv4_table = pd.read_csv('datasets/4hwc.txt', delimiter=',')
hwc4_name = hawcv4_table['Name'].tolist()
hwc4_ra = hawcv4_table['Ra'].tolist()
hwc4_dec = hawcv4_table['Dec'].tolist()
hwc4_ext = hawcv4_table['Ext'].tolist()
hawcv4_coords = SkyCoord(ra=hwc4_ra, dec=hwc4_dec, unit='deg', frame='icrs')

#Import Fermi-LAT Catalog
fermi_fits = fits.open('datasets/gll_psc_v35.fits')
p_data = fermi_fits[1].data
fermi_fulltable = Table(p_data)

#Import LHAASO Catalog
lhaaso_dat = pd.read_csv("datasets/lhaaso_cat.csv", comment="#", header=None)
lhaaso_dat.columns = [
    "Source name", "Components", "RA_2000", "Dec_2000", "Sigma_p95_stat",
    "r_39", "TS", "N0", "Gamma", "TS_100", "Association"
]

def clean_value(val):
    if pd.isna(val):
        return np.nan
    val = str(val).replace('$', '')
    if '<' in val:
        return float(val.replace('<', '').strip())
    elif 'pm' in val or '±' in val or '\\pm' in val:
        parts = re.split(r'±|\\pm|pm', val)
        try:
            return float(parts[0].strip())
        except:
            return np.nan
    try:
        return float(val)
    except:
        return val

for col in ["Source name","r_39", "N0", "Gamma"]:
    lhaaso_dat[col] = lhaaso_dat[col].apply(clean_value)
lhaaso_dat["Source name"] = lhaaso_dat["Source name"].replace(r'^\s*$', np.nan, regex=True).ffill()


def loadmap(filename, coord_sys, coords,*args):
    # print("Coords=",coords)
    with fits.open(filename) as ihdu:
        if 'xyrange' in args:
            e1, e2, e3 , e4 = coords
            cX, cY = (e1+e2)/2, (e3+e4)/2
            xR = int(np.abs(e1-e2)/(1/360))
            yR = int(np.abs(e3-e4)/(1/360))
        if 'origin' in args:
            cX, cY, xR, yR = coords
            xR = int(xR/(1/360))
            yR = int(yR/(1/360))
        # print(cX, cY, xR, yR)
        if coord_sys == 'C':   ###Celestial Coordinate System
            target_header = Header()
            target_header['NAXIS'] = 2
            target_header['NAXIS1'] = xR
            target_header['NAXIS2'] = yR
            target_header['CTYPE1'] = 'RA---MOL'
            target_header['CRPIX1'] = xR/2
            target_header['CRVAL1'] = cX
            target_header['CDELT1'] = -2./360
            target_header['CUNIT1'] = 'deg     '
            target_header['CTYPE2'] = 'DEC--MOL'
            target_header['CRPIX2'] = yR/2
            target_header['CRVAL2'] = cY
            target_header['CDELT2'] = 2./360
            target_header['CUNIT2'] = 'deg     '
            target_header['COORDSYS'] = 'icrs    '
            print("Loading Celestial Map")
        if coord_sys == 'G':  ###Galactic Coordinate System
            target_header = Header()
            target_header['NAXIS'] = 2
            target_header['NAXIS1'] = xR
            target_header['NAXIS2'] = yR
            target_header['CTYPE1'] = 'GLON-AIT'
            target_header['CRPIX1'] = xR/2
            target_header['CRVAL1'] = cX
            target_header['CDELT1'] = -2./360
            target_header['CUNIT1'] = 'deg     '
            target_header['CTYPE2'] = 'GLAT-AIT'
            target_header['CRPIX2'] = yR/2
            target_header['CRVAL2'] = cY
            target_header['CDELT2'] = 2./360
            target_header['CUNIT2'] = 'deg     '
            target_header['COORDSYS'] = 'galactic    '
            print("Loading Galactic Map")
        
        skymap_data = ihdu[1].data["significance"] #.data["significance"]
        ihdu[1].header['COORDSYS'] = 'icrs    '
        wcs = WCS(target_header)
        # print(wcs)
        array, footprint = reproject_from_healpix(ihdu[1],target_header)
        print("Fits File loaded")
        
    return array, footprint, wcs

def plot_4FGL(ax, wcs, ra_center, dec_center, xlength, ylength, npix):
    masks = [fermi_fulltable['GLON'] >= (float(ra_center)-xlength),  fermi_fulltable['GLON'] <= float(ra_center)+xlength,  fermi_fulltable['GLAT'] >= float(dec_center)-ylength,  fermi_fulltable['GLAT'] <= float(dec_center)+ylength]
    full_mask = reduce(np.logical_and, masks)
    fermi_table = fermi_fulltable[full_mask]
    fermi_name = fermi_table['Source_Name']
    fermi_ra = fermi_table['RAJ2000']
    fermi_dec = fermi_table['DEJ2000']
    fermi_semi_major = fermi_table['Conf_68_SemiMajor']
    fermi_semi_minor = fermi_table['Conf_68_SemiMinor']
    fermi_angle = fermi_table['Conf_68_PosAng']
    for i in range(len(fermi_name)):
        coord2 = SkyCoord(ra=fermi_ra[i]*u.deg, dec=fermi_dec[i]*u.deg)
        pixelcoord = astropy_utils.skycoord_to_pixel(coord2, wcs)
        ax.plot(pixelcoord[0], pixelcoord[1], marker='x', markersize=5, color='white')
        ax.annotate(fermi_name[i],xy=(pixelcoord[0], pixelcoord[1]), xycoords='data', xytext=(100, 90),
        textcoords='offset points', arrowprops=dict(arrowstyle="-",color='white', linewidth=2, 
        linestyle='-'), color='cyan', rotation=0, ha='right', va='center' ,
        path_effects=[pe.withStroke(linewidth=2, foreground="gray")])
        center = (float(pixelcoord[0]), float(pixelcoord[1]))
        fermiext = Ellipse(xy=center, width=fermi_semi_major[i]/npix, height=fermi_semi_minor[i]/npix, angle=fermi_angle[i], fc='None', ec='cyan', linewidth=2)
        ax.add_patch(fermiext)

def plot_4hwc1D(ax, wcs, npix):
    try:
        for i in range(len(ax)):
            for i in range(len(hwc4_name)):
                coord2 = SkyCoord(ra=hwc4_ra[i]*u.deg, dec=hwc4_dec[i]*u.deg)
                pixelcoord = astropy_utils.skycoord_to_pixel(coord2, wcs)
                ax[i].plot(pixelcoord[0], pixelcoord[1], marker='x', markersize=5, color='white')
                ax[i].annotate('4HWC '+hwc4_name[i],xy=(pixelcoord[0], pixelcoord[1]), xycoords='data', xytext=(100, 90), 
                textcoords='offset points', arrowprops=dict(arrowstyle="-",color='white', linewidth=2, 
                linestyle='-.'), color='white', rotation=30, ha='right', va='center' ,
                path_effects=[pe.withStroke(linewidth=2, foreground="gray")])
                hawcext = plt.Circle((pixelcoord[0], pixelcoord[1]), hwc4_ext[i]/npix, color='white', linewidth=2, fill=False, linestyle='-')
                ax[i].add_patch(hawcext)
    except:
        for i in range(len(hwc4_name)):
            coord2 = SkyCoord(ra=hwc4_ra[i]*u.deg, dec=hwc4_dec[i]*u.deg)
            pixelcoord = astropy_utils.skycoord_to_pixel(coord2, wcs)
            ax.plot(pixelcoord[0], pixelcoord[1], marker='x', markersize=5, color='white')
            ax.annotate('4HWC '+hwc4_name[i],xy=(pixelcoord[0], pixelcoord[1]), xycoords='data', xytext=(100, -90), 
            textcoords='offset points', arrowprops=dict(arrowstyle="-",color='white', linewidth=2, 
            linestyle='-.'), color='white', rotation=30, ha='right', va='center' ,
            path_effects=[pe.withStroke(linewidth=2, foreground="gray")])
            hawcext = plt.Circle((pixelcoord[0], pixelcoord[1]), hwc4_ext[i]/npix, color='white', linewidth=2, fill=False, linestyle='-')
            ax.add_patch(hawcext)
        hawcext = plt.Circle((pixelcoord[0], pixelcoord[1]), hwc4_ext[i]/npix, color='white', linewidth=2, fill=False, linestyle='-', label='4HWC Extension')
        ax.add_patch(hawcext)

def invrelu(x, floor_min=-3):
	return np.maximum(floor_min, x)

def relu(x, ceil_max=15):
	return np.minimum(ceil_max, x)

def calc_norm_from_act(image, x):
    normalized_data = (x - np.min(image))/(np.max(image) - np.min(image))
    return normalized_data

def calc_act_from_norm(image, x, min, max):
    actual_data = np.min(image) + x * (np.max(image) - np.min(image))
    return actual_data

def check_circle_relation(x1, y1, r1, x2, y2, r2):
    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    
    if distance + r2 <= r1:
        return 0
    elif distance <= r2:
        return 1
    elif distance < 0.6 * (r1 + r2):
        return 2
    else:
        return 3

def plot_ps_blob(ax, ps_blobs, wcs):
    try:
        for i in range(len(ps_blobs)):
                    blob = ps_blobs[i]
                    y, x, r = blob
                    ax.plot(x, y, marker='x', markersize=5, color='green')
                    ax.annotate('Blob '+ str(i),xy=(x, y), xycoords='data', xytext=(100, -90), 
                    textcoords='offset pixels', arrowprops=dict(arrowstyle="-",color='green', linewidth=2, 
                    linestyle='-.'), color='white', rotation=0, ha='right', va='center' ,
                    path_effects=[pe.withStroke(linewidth=2, foreground="green")])
                    c = plt.Circle((x, y), r, color='green', linewidth=3, fill=False)
                    ax.add_patch(c)
    except:
        pass

def plot_ext_blob(ax, ext_blobs, wcs):
    try:
        for i in range(len(ext_blobs)):
                    blob = ext_blobs[i]
                    y, x, r = blob
                    ax.plot(x, y, marker='x', markersize=5, color='white')
                    ax.annotate('Ext Blob '+str(i),xy=(x, y), xycoords='data', xytext=(100, 90), 
                    textcoords='offset pixels', arrowprops=dict(arrowstyle="-",color='gray', linewidth=2, 
                    linestyle='-.'), color='white', rotation=0, ha='right', va='center' ,
                    path_effects=[pe.withStroke(linewidth=2, foreground="purple")])
                    c = plt.Circle((x, y), r, color='gray', linewidth=3, fill=False)
                    ax.add_patch(c)
    except:
        pass

def blob_filter_intensity(blobs, image, min_intensity, wcs, npix):
    filtered_blobs = []
    filtered_coords = []
    filtered_radius = []

    image = np.asarray(image)
    nrows, ncols = image.shape

    for y, x, r in blobs:

        # Extract local patch
        y_min = int(max(y - r, 0))
        y_max = int(min(y + r, nrows))
        x_min = int(max(x - r, 0))
        x_max = int(min(x + r, ncols))

        patch = image[y_min:y_max, x_min:x_max]
        yy, xx = np.ogrid[y_min:y_max, x_min:x_max]
        dist = np.sqrt((yy - y)**2 + (xx - x)**2)
        mask = dist <= (0.9 * r)
        inside_pixels = patch[mask]
        if np.any(inside_pixels > min_intensity):

            coord = astropy_utils.pixel_to_skycoord(x, y, wcs=wcs).icrs

            filtered_blobs.append((y, x, r))
            filtered_coords.append(coord)
            filtered_radius.append(r * npix)

            # print(
            #     f"Blob accepted: max pixel={inside_pixels.max():.3f}, "
            #     f"Coords ({coord.ra}, {coord.dec}), "
            #     f"Radius={r:.2f}, Angular Radius={r*npix:.3f}"
            # )

    return filtered_blobs, filtered_coords, filtered_radius

def blob_filter_overlap(hfiltered_blobs, hfiltered_coords, hfiltered_radius, hfiltered_blobs2, hfiltered_coords2, hfiltered_radius2):
    i=0
    try:
        while i <= len(hfiltered_blobs2):
            c1 = hfiltered_coords2[i]
            # print(c1)
            r1 = hfiltered_radius2[i]
            x1, y1, r1 = c1.ra.deg, c1.dec.deg, r1
            j=0
            while j<=len(hfiltered_blobs):
                c2 = hfiltered_coords[j]
                r2 = hfiltered_radius[j]
                x2, y2, r2 = c2.ra.deg, c2.dec.deg, r2
                x=check_circle_relation(x1, y1, r1, x2, y2, r2)
                # print("j",j)
                if x == 1 or x == 0 or x == 2:
                    hfiltered_blobs.pop(j)
                    hfiltered_coords.pop(j)
                    hfiltered_radius.pop(j)
                    j=j-1
                j = j+1
            i=i+1
    except:
        pass
    return hfiltered_blobs, hfiltered_coords, hfiltered_radius

def SNRCat2():
    url = "http://snrcat.physics.umanitoba.ca/SNRdownload.php?table=SNR"
    tmp = tempfile.NamedTemporaryFile()
    try:
        urllib.request.urlretrieve(url, tmp.name)
    except:
        urllib.urlretrieve(url, tmp.name)
    filename = tmp.name
    f = open(filename)
    outf = open("snrcat_data_%s.txt" % datetime.now().date(), "w")
    assocs = []
    assocs_ra = []
    assocs_dec = []
    i=0
    with open(filename) as f:
        for ln in f.readlines()[2:]:
            col_list = ln.split(';')
            try:
                name_index=col_list.index('G')
                ra_index=col_list.index('J2000_ra (hh:mm:ss)')
                dec_index=col_list.index('J2000_dec (dd:mm:ss)')
            except:
                name_index=name_index
                ra_index=ra_index
                dec_index=dec_index
            if i>1:
                snrcoord_time = SkyCoord(col_list[ra_index], col_list[dec_index], unit=(u.hourangle, u.deg),frame='icrs')
                assocs.append(col_list[name_index])
                assocs_ra.append(snrcoord_time.ra.deg)
                assocs_dec.append(snrcoord_time.dec.deg)
            i=i+1
        return assocs, assocs_ra, assocs_dec

def plot_snrcat(ax, wcs, labels=True):
    assoc, ra, dec = SNRCat2()
    i = 0
    for i in range(len(assoc)):
        coord2 = SkyCoord(ra=ra[i]*u.deg, dec=dec[i]*u.deg)
        pixelcoord = astropy_utils.skycoord_to_pixel(coord2, wcs)
        ax.plot(pixelcoord[0], pixelcoord[1], marker='o', markersize=5, color='cyan')
        if labels:
            ax.annotate(assoc[i],xy=(pixelcoord[0], pixelcoord[1]), xycoords='data', xytext=(100, 90), 
            textcoords='offset points', arrowprops=dict(arrowstyle="-",color='gray', linewidth=2, 
            linestyle='-.'), color='white', rotation=0, ha='right', va='center' ,
            path_effects=[pe.withStroke(linewidth=2, foreground="gray")])
    ax.plot(pixelcoord[0], pixelcoord[1], marker='o', markersize=5, color='white', label='SNR')

def injected_sources_plot(names, ra, dec, ext, ax, wcs):
    for i in range(len(names)):
        coord2 = SkyCoord(ra=ra[i]*u.deg, dec=dec[i]*u.deg)
        pixelcoord = astropy_utils.skycoord_to_pixel(coord2, wcs)
        ax.plot(pixelcoord[0], pixelcoord[1], marker='x', markersize=5, color='blue')
        ax.annotate('Injected '+names[i],xy=(pixelcoord[0], pixelcoord[1]), xycoords='data', fontsize=16, xytext=(100, -90), 
        textcoords='offset pixels', arrowprops=dict(arrowstyle="-",color='gray', linewidth=2, 
        linestyle='-.'), color='white', rotation=0, ha='right', va='center' ,
        path_effects=[pe.withStroke(linewidth=2, foreground="blue")])
        r = ext[i] / 0.0027
        c = plt.Circle((pixelcoord[0], pixelcoord[1]), r, color='blue', linewidth=3, fill=False)
        ax.add_patch(c)

def custom_sources_plot(names, ra, dec, ext, ax, wcs, npix):
    color = cm.Blues(np.linspace(0, 1, len(names)))
    for i in range(len(names)):
        # print(ra[i], dec[i])
        coord2 = SkyCoord(ra=ra[i]*u.deg, dec=dec[i]*u.deg)
        pixelcoord = astropy_utils.skycoord_to_pixel(coord2, wcs)
        ax.plot(pixelcoord[0], pixelcoord[1], marker='x', markersize=5, color='white')
        ax.annotate(names[i],xy=(pixelcoord[0], pixelcoord[1]), xycoords='data', fontsize=12, xytext=(100, 90), 
        textcoords='offset pixels', arrowprops=dict(arrowstyle="-",color='red', linewidth=2, 
        linestyle='-.'), color='white', rotation=30, ha='right', va='center' ,
        path_effects=[pe.withStroke(linewidth=2, foreground="red")])
        r = ext[i] / npix
        c = plt.Circle((pixelcoord[0], pixelcoord[1]), r, color='red', linewidth=3, fill=False)
        ax.add_patch(c)

def custom_sources_plot2(names, ra, dec, ext, ax, wcs, npix):
    color = cm.Blues(np.linspace(0, 1, len(names)))
    for i in range(len(names)):
        coord2 = SkyCoord(ra=ra[i]*u.deg, dec=dec[i]*u.deg)
        pixelcoord = astropy_utils.skycoord_to_pixel(coord2, wcs)
        ax.plot(pixelcoord[0], pixelcoord[1], marker='o', markersize=5, color='white')
        ax.annotate('5HWC '+names[i],xy=(pixelcoord[0], pixelcoord[1]), xycoords='data', fontsize=12, xytext=(100, 90), 
        textcoords='offset pixels', arrowprops=dict(arrowstyle="-",color='red', linewidth=2, 
        linestyle='-.'), color='white', rotation=30, ha='right', va='center' ,
        path_effects=[pe.withStroke(linewidth=2, foreground="red")])
        r = ext[i] / npix
        c = plt.Circle((pixelcoord[0], pixelcoord[1]), r, color='red', linewidth=3, fill=False)
        ax.add_patch(c)

def extract_ra_dec(filename):
    pattern = r'model_\d+_roi_(\d+\.\d+)_(-?\d+\.\d+)\.yaml'
    match = re.search(pattern, filename)
    if match:
        ra, dec = float(match.group(1)), float(match.group(2))
        return ra, dec
    else:
        raise ValueError(f"Filename format is incorrect: {filename}")

def extract_run(filename):
    pattern = r'model_+(\d+)+_roi_(\d+\.\d+)_(-?\d+\.\d+)\.yaml'
    match = re.search(pattern, filename)
    if match:
        run = float(match.group(1))
        return run
    else:
        raise ValueError(f"Filename format is incorrect: {filename}")

def parse_yaml_file(filepath):
    with open(filepath, 'r') as file:
        data = yaml.safe_load(file)
    CU = 6.13e-21
    sources = []
    for source, properties in data.items():
        if 'Gaussian_on_sphere' in properties:
            lon0 = properties['Gaussian_on_sphere']['lon0']['value']
            lat0 = properties['Gaussian_on_sphere']['lat0']['value']
            sigma = properties['Gaussian_on_sphere'].get('sigma', {}).get('value', None)
            try:
                flux_frac = properties['spectrum']['main']['Log_parabola']['K']['value']/CU
                index = np.abs(properties['spectrum']['main']['Log_parabola']['alpha']['value'])
            except:
                flux_frac = properties['spectrum']['main']['Powerlaw']['K']['value']/CU
                index = np.abs(properties['spectrum']['main']['Powerlaw']['index']['value'])
            sources.append({
                'source_name': source,
                'lon0': lon0,
                'lat0': lat0,
                'sigma': sigma,
                'flux_frac': flux_frac,
                'index' : index
            })
        elif 'position' in properties:
            lon0 = properties['position']['ra']['value']
            lat0 = properties['position']['dec']['value']
            sigma = 0
            try:
                flux_frac = properties['spectrum']['main']['Log_parabola']['K']['value']/CU
                index = np.abs(properties['spectrum']['main']['Log_parabola']['alpha']['value'])
            except:
                flux_frac = properties['spectrum']['main']['Powerlaw']['K']['value']/CU
                index = np.abs(properties['spectrum']['main']['Powerlaw']['index']['value'])
            sources.append({
                'source_name': source,
                'lon0': lon0,
                'lat0': lat0,
                'sigma': sigma,
                'flux_frac': flux_frac,
                'index' : index
            })
        else:
            pass
    return sources

def plot_ax_label(ax, coord_sys):
    if coord_sys == 'C':
        ax.set_xlabel(r"$ra^o$")
        ax.set_ylabel(r"$dec^o$")
    elif coord_sys == 'G':
        ax.set_ylabel(r"$b^o$")
        ax.set_xlabel(r"$l^o$")

def create_circular_mask(h, w, center=None, radius=None):
    """
    Create a circular mask for a given height (h), width (w),
    with specified center and radius.
    """
    if center is None:
        center = (int(w / 2), int(h / 2)) 
    if radius is None:
        radius = min(h, w) / 4

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)

    mask = dist_from_center <= radius
    return mask

def parula_cmap():

    cm_data = [[0.2081, 0.1663, 0.5292], [0.2116238095, 0.1897809524, 0.5776761905], 
    [0.212252381, 0.2137714286, 0.6269714286], [0.2081, 0.2386, 0.6770857143], 
    [0.1959047619, 0.2644571429, 0.7279], [0.1707285714, 0.2919380952, 
    0.779247619], [0.1252714286, 0.3242428571, 0.8302714286], 
    [0.0591333333, 0.3598333333, 0.8683333333], [0.0116952381, 0.3875095238, 
    0.8819571429], [0.0059571429, 0.4086142857, 0.8828428571], 
    [0.0165142857, 0.4266, 0.8786333333], [0.032852381, 0.4430428571, 
    0.8719571429], [0.0498142857, 0.4585714286, 0.8640571429], 
    [0.0629333333, 0.4736904762, 0.8554380952], [0.0722666667, 0.4886666667, 
    0.8467], [0.0779428571, 0.5039857143, 0.8383714286], 
    [0.079347619, 0.5200238095, 0.8311809524], [0.0749428571, 0.5375428571, 
    0.8262714286], [0.0640571429, 0.5569857143, 0.8239571429], 
    [0.0487714286, 0.5772238095, 0.8228285714], [0.0343428571, 0.5965809524, 
    0.819852381], [0.0265, 0.6137, 0.8135], [0.0238904762, 0.6286619048, 
    0.8037619048], [0.0230904762, 0.6417857143, 0.7912666667], 
    [0.0227714286, 0.6534857143, 0.7767571429], [0.0266619048, 0.6641952381, 
    0.7607190476], [0.0383714286, 0.6742714286, 0.743552381], 
    [0.0589714286, 0.6837571429, 0.7253857143], 
    [0.0843, 0.6928333333, 0.7061666667], [0.1132952381, 0.7015, 0.6858571429], 
    [0.1452714286, 0.7097571429, 0.6646285714], [0.1801333333, 0.7176571429, 
    0.6424333333], [0.2178285714, 0.7250428571, 0.6192619048], 
    [0.2586428571, 0.7317142857, 0.5954285714], [0.3021714286, 0.7376047619, 
    0.5711857143], [0.3481666667, 0.7424333333, 0.5472666667], 
    [0.3952571429, 0.7459, 0.5244428571], [0.4420095238, 0.7480809524, 
    0.5033142857], [0.4871238095, 0.7490619048, 0.4839761905], 
    [0.5300285714, 0.7491142857, 0.4661142857], [0.5708571429, 0.7485190476, 
    0.4493904762], [0.609852381, 0.7473142857, 0.4336857143], 
    [0.6473, 0.7456, 0.4188], [0.6834190476, 0.7434761905, 0.4044333333], 
    [0.7184095238, 0.7411333333, 0.3904761905], 
    [0.7524857143, 0.7384, 0.3768142857], [0.7858428571, 0.7355666667, 
    0.3632714286], [0.8185047619, 0.7327333333, 0.3497904762], 
    [0.8506571429, 0.7299, 0.3360285714], [0.8824333333, 0.7274333333, 0.3217], 
    [0.9139333333, 0.7257857143, 0.3062761905], [0.9449571429, 0.7261142857, 
    0.2886428571], [0.9738952381, 0.7313952381, 0.266647619], 
    [0.9937714286, 0.7454571429, 0.240347619], [0.9990428571, 0.7653142857, 
    0.2164142857], [0.9955333333, 0.7860571429, 0.196652381], 
    [0.988, 0.8066, 0.1793666667], [0.9788571429, 0.8271428571, 0.1633142857], 
    [0.9697, 0.8481380952, 0.147452381], [0.9625857143, 0.8705142857, 0.1309], 
    [0.9588714286, 0.8949, 0.1132428571], [0.9598238095, 0.9218333333, 
    0.0948380952], [0.9661, 0.9514428571, 0.0755333333], 
    [0.9763, 0.9831, 0.0538]]

    parula_map = LinearSegmentedColormap.from_list('parula', cm_data)
    return parula_map

def setupMilagroColormap(amin, amax, threshold, ncolors):
    thresh = (threshold - amin) / (amax - amin)
    if threshold <= amin or threshold >= amax:
        thresh = 0.
    dthresh = 1 - thresh
    threshDict = { "blue"  : ((0.0, 1.0, 1.0),
                              (thresh, 0.5, 0.5),
                              (thresh+0.077*dthresh, 0, 0),
                              (thresh+0.462*dthresh, 0, 0),
                              (thresh+0.615*dthresh, 1, 1),
                              (thresh+0.692*dthresh, 1, 1),
                              (thresh+0.769*dthresh, 0.6, 0.6),
                              (thresh+0.846*dthresh, 0.5, 0.5),
                              (thresh+0.923*dthresh, 0.1, 0.1),
                              (1, 0, 0)),
                   "green" : ((0.0, 1.0, 1.0),
                              (thresh, 0.5, 0.5),
                              (thresh+0.077*dthresh, 0, 0),
                              (thresh+0.231*dthresh, 0, 0),
                              (thresh+0.308*dthresh, 1, 1),
                              (thresh+0.385*dthresh, 0.8, 0.8),
                              (thresh+0.462*dthresh, 1, 1),
                              (thresh+0.615*dthresh, 0.8, 0.8),
                              (thresh+0.692*dthresh, 0, 0),
                              (thresh+0.846*dthresh, 0, 0),
                              (thresh+0.923*dthresh, 0.1, 0.1),
                              (1, 0, 0)),
                   "red"   : ((0.0, 1.0, 1.0),
                              (thresh, 0.5, 0.5),
                              (thresh+0.077*dthresh, 0.5, 0.5),
                              (thresh+0.231*dthresh, 1, 1),
                              (thresh+0.385*dthresh, 1, 1),
                              (thresh+0.462*dthresh, 0, 0),
                              (thresh+0.692*dthresh, 0, 0),
                              (thresh+0.769*dthresh, 0.6, 0.6),
                              (thresh+0.846*dthresh, 0.5, 0.5),
                              (thresh+0.923*dthresh, 0.1, 0.1),
                              (1, 0, 0)) }

    newcm = LinearSegmentedColormap("thresholdColormap",
                                               threshDict,
                                               ncolors)
    newcm.set_over(newcm(1.0))
    newcm.set_under("w")
    newcm.set_bad("gray")
    textcolor = "#000000"

    return textcolor, newcm

def plot_1lhaaso(ax, wcs, npix):
    for i in range(len(df)):
        if df['RA_2000'][i] != ' ':
            coord2 = SkyCoord(ra=float(df['RA_2000'][i])*u.deg, dec=float(df['Dec_2000'][i])*u.deg)
            coord_gal = coord2.galactic
            pixelcoord = astropy_utils.skycoord_to_pixel(coord_gal, wcs)
            ax.plot(pixelcoord[0], pixelcoord[1], marker='o', markersize=5, color='white')
            ax.annotate(df['Source name'][i],xy=(pixelcoord[0], pixelcoord[1]), xycoords='data', xytext=(100, 90), 
            textcoords='offset points', arrowprops=dict(arrowstyle="-",color='blue', linewidth=2, 
            linestyle='-.'), color='white', rotation=0, ha='right', va='center' ,
            path_effects=[pe.withStroke(linewidth=2, foreground="blue")])
            hawcext = plt.Circle((pixelcoord[0], pixelcoord[1]), float(df['r_39'][i])/npix, color='blue', linewidth=2, fill=False, linestyle='-')
            ax.add_patch(hawcext)

def plot_hgps(ax, wcs, npix):
    for i in range(len(hess_catalog)):
        coord2 = SkyCoord(ra=float(hess_catalog['RAJ2000'][i])*u.deg, dec=float(hess_catalog['DEJ2000'][i])*u.deg)
        coord_gal = coord2.galactic
        pixelcoord = astropy_utils.skycoord_to_pixel(coord_gal, wcs)
        ax.plot(pixelcoord[0], pixelcoord[1], marker='o', markersize=5, color='white')
        ax.annotate(hess_catalog['Source_Name'][i],xy=(pixelcoord[0], pixelcoord[1]), xycoords='data', xytext=(100, 145), 
        textcoords='offset points', arrowprops=dict(arrowstyle="-",color='white', linewidth=2, 
        linestyle='-.'), color='gray', rotation=0, ha='right', va='center' ,
        path_effects=[pe.withStroke(linewidth=2, foreground="white")])
        if hess_catalog['Size'][i] != ' ':
            hawcext = plt.Circle((pixelcoord[0], pixelcoord[1]), float(hess_catalog['Size'][i])/npix, color='white', linewidth=2, fill=False, linestyle='-')
            ax.add_patch(hawcext)
        else:
            hawcext = plt.Circle((pixelcoord[0], pixelcoord[1]), float(0.01)/npix, color='white', linewidth=2, fill=False, linestyle='-')
            ax.add_patch(hawcext)

def parse_model_file(catalog_results):
    with open(catalog_results, "r") as f:
        content = f.read()

    source_blocks = re.findall(r"source_name\s*=\s*\"(.*?)\"(.*?)###################################", content, re.DOTALL)
    sources = []

    for name, block in source_blocks:
        ra_match = re.search(r"source_ra\s*=\s*([\d.]+)", block)
        dec_match = re.search(r"source_dec\s*=\s*([\d.]+)", block)
        sigma_match = re.search(r"shape\s*=\s*astromodels\.Gaussian_on_sphere\(\).*?shape\.sigma\s*=\s*([\d.]+)", block, re.DOTALL)
        source_info = {
            "name": name,
            "ra": float(ra_match.group(1)) if ra_match else None,
            "dec": float(dec_match.group(1)) if dec_match else None,
            "sigma": float(sigma_match.group(1)) if sigma_match else 0.0,
        }

        sources.append(source_info)
    return sources

def plotcatalogs(ax, wcs, *args):
    for k in args:
        print(k)
    if '4hwc' in args:
        print("Plotting 4HWC Catalog")
        plot_4hwc1D(ax, wcs)
    if 'lhaaso' in args:
        print("Plotting LHAASO")
        plot_1lhaaso(ax, wcs)
    if 'fermi' in args:
        plot_4FGL(ax, wcs, ra, dec, xlength, ylength)
    if 'snr' in args:
        plot_snrcat(ax, wcs)
    if 'pulsar' in args:
        plot_pulsar(ax, wcs)
    if 'hgps' in args:
        plot_hgps(ax, wcs)

def plotblobs(ax, wcs, blobs):
    if 'psblobs' in blobs:
        print("Plotting PS blobs")
        plot_ps_blob(ax, blobs['psblobs'], wcs)
    if 'extblobs' in blobs:
        print("Plotting Extended blobs")
        plot_ext_blob(ax, blobs['extblobs'], wcs)
    if 'extblobs2' in blobs:
        print("Plotting Extended blobs")
        plot_ext_blob(ax, blobs['extblobs2'], wcs)

def ultimet(vmin, vmax, threshold, color_map='turbo', n=256, blend_fraction=0.05):

    tnorm = (threshold - vmin) / (vmax - vmin)
    bfrac = blend_fraction
    t_low = max(tnorm - bfrac / 2, 0)
    t_high = min(tnorm + bfrac / 2, 1)

    n1 = int(t_low * n)
    n2 = int((t_high - t_low) * n)
    n3 = n - n1 - n2

    gray_part = plt.cm.binary(np.linspace(0.2, 0.8, max(n1, 1))) 
    blend = np.linspace(0, 1, max(n2, 1))
    cgray = gray_part[-1] if len(gray_part) else plt.cm.binary(0.8)
    cstart = plt.get_cmap(color_map)(0.0)
    blend_part = (cgray * (1 - blend[:, None])) + (cstart * blend[:, None])
    color_part = plt.get_cmap(color_map)(np.linspace(0, 1, max(n3, 1)))

    colors = np.vstack((gray_part, blend_part, color_part))
    cmap = LinearSegmentedColormap.from_list("ultimet_threshold", colors)

    cmap.set_over(cmap(1.0))
    cmap.set_under("white")
    cmap.set_bad("gray")

    return cmap

def parse_pulsar_db(file_contents='./datasets/psrcat_tar/psrcat.db'):
    with open(file_contents, "r") as f:
        db_text = f.read()
    entries = db_text.strip().split('@-----------------------------------------------------------------')
    pulsars = []
    for entry in entries:
        if not entry.strip():
            continue
        lines = entry.strip().splitlines()
        pulsar = {}
        for line in lines:
            parts = line.split()
            if not parts:
                continue
            key = parts[0]
            if key in ["PSRJ", "RAJ", "DECJ"]:
                pulsar[key] = parts[1]
            elif key in ["PSRB", "RAJ", "DECJ"]:
                pulsar[key] = parts[1]
        if all(k in pulsar for k in ['PSRJ', 'RAJ', 'DECJ']):
            pulsars.append(pulsar)
    return pulsars

def make_pulsar_plotter(marker_color='lime', label_color='white', marker_size=6, annotate=True):
    def plot_pulsars(ax, wcs, pulsar_data):
        for pulsar in pulsar_data:
            # print(pulsar)
            coord = SkyCoord(pulsar['RAJ'], pulsar['DECJ'], unit=(u.hourangle, u.deg))
            x, y = coord.to_pixel(wcs)
            ax.plot(x, y, marker='*', color=marker_color, markersize=marker_size)
            if annotate:
                ax.annotate(
                    pulsar['PSRJ'],
                    xy=(x, y),
                    xycoords='data',
                    xytext=(5, 5),
                    textcoords='offset points',
                    arrowprops=dict(arrowstyle="-", color='gray', linewidth=1),
                    color=label_color,
                    ha='left',
                    va='bottom',
                    path_effects=[pe.withStroke(linewidth=2, foreground="black")]
                )
    return plot_pulsars

def radius_to_sigma(R, fraction=0.6827):
    return R / np.sqrt(2 * np.log(1 / (1 - fraction)))
 
def load_hawc_data(filename, x, y, xlength, ylength, coord_sys):
    """
    Load HAWC data from a given directory and observation ID.
    
    Parameters:
    - hawc_dir: Directory containing HAWC data
    - run: Run number
    - obsid: Observation ID
    
    Returns:
    - Dictionary containing the loaded data
    """
    if coord_sys == 'C':
        print(f"ROI center in Celestial Coordintes = {x}, {y}")
    else:
        print(f"ROI center in Galactic Coordintes = {x}, {y}")

    origin = [x, y, xlength, ylength] 

    # Load the data from the specified file
    array, footprint, wcs = loadmap(filename, coord_sys, origin, 'origin')
    xnum = array.shape[1]
    ynum = array.shape[0]
    pixel_size = wcs.wcs.cdelt[1]
    print(f'Degrees per pixel: {pixel_size} ')
    return array, footprint, wcs, xnum, ynum, pixel_size

def find_peak(array, wcs):
    peak_index = np.unravel_index(np.argmax(array), array.shape)
    a = astropy.wcs.utils.pixel_to_skycoord(peak_index[1], peak_index[0], wcs)
    print("Peak intensity pixel location:", peak_index)
    print("Peak intensity sky location:", a)
    print("Peak intensity value:", array[peak_index])
    return array[peak_index]

def find_well(array, wcs):
    peak_index = np.unravel_index(np.argmin(array), array.shape)
    a = astropy.wcs.utils.pixel_to_skycoord(peak_index[1], peak_index[0], wcs)
    print("Well intensity pixel location:", peak_index)
    print("Well intensity sky location:", a)
    print("Well intensity value:", array[peak_index])

def make_plots(array, wcs, npix, coordsys, threshold=4, vmin=-5, vmax=15, blobs=None, contour=False, title=None, hotspots=None, save_dir=None, pdf=False, cmap='inferno', figsize=(10, 6), **kwargs):
    fig = plt.figure(figsize=(figsize[0] , figsize[1]))
    ax = fig.add_subplot(1, 1, 1, projection=wcs)
    if cmap=='ult':
        ult = ultimet(vmin, vmax, threshold)
        im = ax.imshow(array, cmap=ult, vmin=vmin, vmax=vmax)
    else:
        im = ax.imshow(array, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar(im, ax=ax, label=r'Significance($\sigma$)', fraction=0.046, pad=0.04)
    ax.grid(False)


    if hotspots is not None:
        labelA = hotspots['Name']
        raA = hotspots['ra']
        decA = hotspots['dec']
        extA = hotspots['ext']
        custom_sources_plot(labelA, raA, decA, extA, ax, wcs, npix)
    if title:
        ax.set_title(title)
    if contour:
        if np.max(array) > 15:
            levels = [7, 9, 12, 13, 14, 15]
            print(f"Plotting contours {levels}")
            hi_transform = ax.get_transform(wcs)
            ax.contour(array, levels=levels, transform=hi_transform, colors='black')
        elif np.max(array) > 12:
            levels = [5, 7, 9, 11]
            print(f"Plotting contours {levels}")
            hi_transform = ax.get_transform(wcs)
            ax.contour(array, levels=levels, transform=hi_transform, colors='black')
        elif np.max(array) > 5:
            levels = [5, 6, 7]
            print(f"Plotting contours {levels}")
            hi_transform = ax.get_transform(wcs)
            ax.contour(array, levels=levels, transform=hi_transform, colors='black')
        else:
            print("No contours to plot")
    xnum, ynum  = array.shape[1], array.shape[0]
    plot_ax_label(ax, coordsys)
    ax.set_xlim(0, xnum)
    ax.set_ylim(0, ynum)
    ax.coords[0].set_format_unit('deg')
    ax.coords[1].set_format_unit('deg') 
    if 'labels' in kwargs:
        if '4hawc' in kwargs['labels']:
            plot_4hwc1D(ax, wcs, npix)
        if '1lhaaso' in kwargs['labels']:
            plot_1lhaaso(ax, wcs, npix)
        if 'hgps' in kwargs['labels']:
            plot_hgps(ax, wcs, npix)
        if '4fgl' in kwargs['labels']:
            center_x, center_y = array.shape[1] // 2, array.shape[0] // 2
            center_coord = pixel_to_skycoord(center_x, center_y, wcs)
            try:
                plot_4FGL(ax, wcs, center_coord.ra.deg, center_coord.dec.deg, array.shape[1] / 2, array.shape[0] / 2, npix)
            except: 
                plot_4FGL(ax, wcs, center_coord.l.deg, center_coord.b.deg, array.shape[1] / 2, array.shape[0] / 2, npix)
        if 'snr' in kwargs['labels']:
            plot_snrcat(ax, wcs)
        if 'pulsar' in kwargs['labels']:
            pulsar_data = parse_pulsar_db()
            plot_pulsars = make_pulsar_plotter(marker_color='cyan')
            plot_pulsars(ax, wcs, pulsar_data)
    if blobs:
        if 'psblobs' in blobs:
            ps_data = blobs['psblobs']
            plot_ps_blob(ax, ps_data, wcs)
        if 'extblobs' in blobs:
            ext_data = blobs['extblobs']
            plot_ext_blob(ax, ext_data, wcs)
    ax.grid(False)
    plt.tight_layout()
    if save_dir !=None:
        if pdf:
            pdf.savefig(fig, bbox_inches='tight')
        else:
            fig.savefig(save_dir + f'{title}.png')
            print(f"Saved figure to {save_dir + f'{title}.png'}")
    return fig, ax

def make_logplots(array, wcs, npix, coordsys, threshold=4, vmin=-5, vmax=15, blobs=None, contour=False, title=None, hotspots=None, save_dir=None, pdf=False, cmap='inferno', figsize=(10, 6), ax=None, **kwargs):
    max_val = np.max(array)
    min_val = np.min(array)
    threshold = min_val + max_val/8
    fig = plt.figure(figsize=(figsize[0] , figsize[1]))
    ax = fig.add_subplot(1, 1, 1, projection=wcs)
    if cmap=='ult':
        ult = ultimet(min_val, max_val, threshold)
        im = ax.imshow(array, cmap=ult,  norm=LogNorm(vmin=0.01, vmax=max_val))
    else:
        im = ax.imshow(array, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar(im, ax=ax, label=r'log(Significance($\sigma$))', fraction=0.046, pad=0.04)
    ax.grid(False)


    if hotspots is not None:
        labelA = hotspots['Name']
        raA = hotspots['ra']
        decA = hotspots['dec']
        extA = hotspots['ext']
        # print(labelA, raA, decA, extA, ax, wcs, npix)
        custom_sources_plot(labelA, raA, decA, extA, ax, wcs, npix)
    if title:
        ax.set_title(title)
    if contour:
        if np.max(array) > 15:
            levels = [7, 9, 12, 13, 14, 15]
            print(f"Plotting contours {levels}")
            hi_transform = ax.get_transform(wcs)
            ax.contour(array, levels=levels, transform=hi_transform, colors='black')
        elif np.max(array) > 12:
            levels = [5, 7, 9, 11]
            print(f"Plotting contours {levels}")
            hi_transform = ax.get_transform(wcs)
            ax.contour(array, levels=levels, transform=hi_transform, colors='black')
        elif np.max(array) > 5:
            levels = [5, 6, 7]
            print(f"Plotting contours {levels}")
            hi_transform = ax.get_transform(wcs)
            ax.contour(array, levels=levels, transform=hi_transform, colors='black')
        else:
            print("No contours to plot")
    xnum, ynum  = array.shape[1], array.shape[0]
    plot_ax_label(ax, coordsys)
    ax.set_xlim(0, xnum)
    ax.set_ylim(0, ynum)

    if 'labels' in kwargs:
        if '4hawc' in kwargs['labels']:
            plot_4hwc1D(ax, wcs, npix)
        if '1lhaaso' in kwargs['labels']:
            plot_1lhaaso(ax, wcs, npix)
        if 'hgps' in kwargs['labels']:
            plot_hgps(ax, wcs, npix)
        if '4fgl' in kwargs['labels']:
            center_x, center_y = array.shape[1] // 2, array.shape[0] // 2
            center_coord = pixel_to_skycoord(center_x, center_y, wcs)
            try:
                plot_4FGL(ax, wcs, center_coord.ra.deg, center_coord.dec.deg, array.shape[1] / 2, array.shape[0] / 2, npix)
            except: 
                plot_4FGL(ax, wcs, center_coord.l.deg, center_coord.b.deg, array.shape[1] / 2, array.shape[0] / 2, npix)
        if 'snr' in kwargs['labels']:
            plot_snrcat(ax, wcs)
        if 'pulsar' in kwargs['labels']:
            pulsar_data = parse_pulsar_db()
            plot_pulsars = make_pulsar_plotter(marker_color='cyan')
            plot_pulsars(ax, wcs, pulsar_data)
    if blobs:
        if 'psblobs' in blobs:
            ps_data = blobs['psblobs']
            plot_ps_blob(ax, ps_data, wcs)
        if 'extblobs' in blobs:
            ext_data = blobs['extblobs']
            plot_ext_blob(ax, ext_data, wcs)
    ax.grid(False)
    plt.tight_layout()
    plt.show()
    if save_dir !=None:
        if pdf:
            pdf.savefig(fig, bbox_inches='tight')
        else:
            plt.savefig(save_dir + f'{title}.png')
    plt.clf()

def smooth_floor(x, floor_min=-3, sharpness=5):
    """
    Smoothly floors values below floor_min using a logistic function.

    Parameters:
        x         : array input
        floor_min : minimum floor value
        sharpness : how sharp the transition is (higher = sharper)

    Returns:
        Smoothly floored array
    """
    x = np.asarray(x)
    transition = 1 / (1 + np.exp(-sharpness * (x - floor_min)))
    return transition * x + (1 - transition) * floor_min

def soft_floor(x, floor_min=-3, scale=1.0):
    """
    Soft floor using a smooth version of ReLU downward.

    Parameters:
        x         : input array
        floor_min : soft floor value
        scale     : controls smoothness (lower = smoother)

    Returns:
        array with smoothly applied floor
    """
    return floor_min + scale * np.log1p(np.exp((x - floor_min) / scale))

def remove_overlapping_ext_sources(primary_coords, candidate_coords, threshold_deg=0.3):

    primary = SkyCoord([c.ra.deg for c in primary_coords] * u.deg,
                       [c.dec.deg for c in primary_coords] * u.deg)
    filtered_candidates = []

    for cand in candidate_coords:
        sep = cand.separation(primary)
        if np.all(sep.deg >= threshold_deg):
            filtered_candidates.append(cand)

    return filtered_candidates

def filter_overlapping_sources(coords, radii, radius_to_sigma, name_prefix="Drip", max_distance_deg=0.5):

    df = pd.DataFrame({
        'Names': [f'{name_prefix}{i}' for i in range(len(coords))],
        'ra': [c.ra.deg for c in coords],
        'dec': [c.dec.deg for c in coords],
        'Circle Radius': radii,
    })
    df['Sigma Radius'] = [radius_to_sigma(r) for r in radii]

    keep = np.ones(len(df), dtype=bool)
    skycoords = SkyCoord(ra=df['ra'].values * u.deg, dec=df['dec'].values * u.deg)

    for i in range(len(df)):
        if not keep[i]:
            continue
        sep = skycoords[i].separation(skycoords[i+1:]).deg
        for j_offset, dist in enumerate(sep):
            j = i + 1 + j_offset
            if not keep[j]:
                continue
            if dist < max_distance_deg:
                r1 = df.loc[i, 'Sigma Radius']
                r2 = df.loc[j, 'Sigma Radius']
                if r1 > 0.1 and r2 <= 0.1:
                    keep[j] = False
                elif r2 > 0.1 and r1 <= 0.1:
                    keep[i] = False
                else:
                    keep[j if r1 < r2 else i] = False

    return df[keep].reset_index(drop=True)

def remove_overlapping_ext_sources_with_indices(primary_coords, candidate_coords, threshold_deg=1):
    primary = SkyCoord([c.ra.deg for c in primary_coords] * u.deg,
                       [c.dec.deg for c in primary_coords] * u.deg)

    keep_indices = []
    for idx, cand in enumerate(candidate_coords):
        sep = cand.separation(primary)
        if np.all(sep.deg >= threshold_deg):
            keep_indices.append(idx)
    return keep_indices

def remove_ext_sources_with_radius_overlap(t_coords, t_radii, candidate_coords, candidate_radii):
    """
    Removes candidate sources that overlap with any t_coord based on combined radius.

    Parameters:
    ----------
    t_coords : list of SkyCoord
        Existing sources (points or extended).
    t_radii : list of float
        Angular radii (in degrees) for t_coords.
    candidate_coords : list of SkyCoord
        Candidate extended sources to filter.
    candidate_radii : list of float
        Radii for candidate_coords (degrees).

    Returns:
    -------
    keep_indices : list of int
        Indices of candidate_coords that do NOT overlap with any t_coords.
    """
    assert len(candidate_coords) == len(candidate_radii)
    assert len(t_coords) == len(t_radii)

    t_sky = SkyCoord(ra=[c.ra.deg for c in t_coords] * u.deg,
                     dec=[c.dec.deg for c in t_coords] * u.deg)

    keep_indices = []
    for idx, (cand, r_cand) in enumerate(zip(candidate_coords, candidate_radii)):
        sep = cand.separation(t_sky).deg
        combined_radius = r_cand + np.array(t_radii)
        overlaps = sep <= combined_radius
        if not np.any(overlaps):
            keep_indices.append(idx)

    return keep_indices

def gal_to_cel(l, b):
    c = SkyCoord(l=l*u.degree, b=b*u.degree, frame='galactic')
    return c.icrs.ra.degree, c.icrs.dec.degree

def analyze_histogram(dog_image, plot=False, save_pdf=False, pdf=None):
    # Calculate the 1D histogram of the DoG Image

    pixels = dog_image.flatten()
    counts, bin_edges = np.histogram(pixels, bins=200, range=(np.min(pixels), np.max(pixels)))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2 

    def gaussian_fit(x, amplitude, mean, stddev):
        return amplitude * np.exp(-((x - mean)**2) / (2 * stddev**2))

    x_exp = np.linspace(np.min(bin_edges), np.max(bin_edges), 200)
    y_exp = gaussian_fit(x_exp, counts.max(), 0, 0.01)

    initial_guess = [counts.max(), bin_centers[np.argmax(counts)], np.std(pixels)]
    popt, _ = curve_fit(gaussian_fit, bin_centers, counts, p0=initial_guess)
    x_fit = np.linspace(bin_centers[0], bin_centers[-1], 200)
    y_fit = gaussian_fit(x_fit, *popt)


    fit_interp = interp1d(x_fit, y_fit, bounds_error=False, fill_value=0)
    gaussian_at_bins = fit_interp(bin_centers)

    # Compute deviation mask: where histogram exceeds Gaussian expectation on the positive side 
    deviation_mask = (bin_centers > popt[1]) & (counts > 1.1 * gaussian_at_bins)
    # Total excess counts above the fitted Gaussian
    chi_squared = np.sum(((counts[deviation_mask] - gaussian_at_bins[deviation_mask])**2) / 
                     gaussian_at_bins[deviation_mask])

    # Degrees of freedom
    dof = np.sum(deviation_mask)

    # Significance
    SNR = np.sqrt(chi_squared - dof)
    for i in range(len(deviation_mask)):
        if np.all(deviation_mask[i:]):
            j = i
            break
    
    mask_deviation_value = bin_centers[j]
    print(f"Significant deviation from Gaussian on positive side at:{j},  {mask_deviation_value:.5f}")
    excess_pixel_mask = dog_image.flatten() > mask_deviation_value

    # 2. Calculate signal as mean of excess pixels
    signal_pixels = dog_image.flatten()[excess_pixel_mask]
    signal = np.mean(signal_pixels) if len(signal_pixels) > 0 else 0

    # 3. Noise is the fitted Gaussian width (background RMS)
    noise = popt[2]  # This is your sigma_resid

    # 4. SNR
    SNR = signal / noise

    print(f"Signal (mean excess): {signal:.5f}")
    print(f"Noise (background RMS): {noise:.5f}")
    print(f"SNR: {SNR:.2f}")

    # Alternative: Peak SNR (brightest excess pixel)
    if len(signal_pixels) > 0:
        peak_signal = np.max(signal_pixels)
        peak_SNR = peak_signal / noise
        print(f"Peak SNR: {peak_SNR:.2f}")
        
    sigma_resid = popt[2]
    deviation_3sig = 3*sigma_resid
    deviation_2sig = 2*sigma_resid
    print(f"2 sigma deviation from Gaussian fit on positive side {2*sigma_resid:.5f}")
    print(f"3 sigma deviation from Gaussian fit on positive side {deviation_2sig:.5f}")
    print(f"Mask deviation value: {mask_deviation_value:.5f}, Sigma of Gaussian fit: {sigma_resid:.5f}")
    SNR_metric = mask_deviation_value / sigma_resid
    print("Signal-to-noise ratio (SNR) metric based on deviation from Gaussian fit:", SNR_metric)
    print("Signal-to-noise ratio (SNR) metric based blah:", SNR)
    if plot:
        fig=plt.figure(figsize=(10, 8))
        log_counts = np.where(counts > 0, counts, 1) 
        plt.hist(pixels, bins=200, range=(np.min(pixels), np.max(pixels)),  color='fuchsia', edgecolor='green', alpha=0.6, label='Histogram (Counts)',histtype='step', linewidth=3)
        plt.plot(x_fit, y_fit, 'r-', linewidth=2, label=f'Fit\nμ={popt[1]:.5f}, σ={popt[2]:.5f}')
        if np.any(deviation_mask):
            plt.axvline(mask_deviation_value, color='blue', linestyle='--', linewidth=2, 
                        label=f'Deviation from fit ≈ {mask_deviation_value:.5f}')
        plt.axvline(deviation_2sig, label=f'2 $\sigma$  = {3*popt[2]:.5f}', color='black')
        plt.axvline(deviation_3sig, label=f'3 $\sigma$  = {3*popt[2]:.5f}', color='purple')
        plt.axvline(popt[1], label=f'Mean = {popt[1]:.5f}')
        plt.yscale('log')
        plt.ylim(1, 1e8)
        plt.xlim(np.min(bin_edges), np.max(bin_edges))
        plt.xlabel('Pixel Intensity')
        plt.ylabel('Log(Counts)')
        plt.title('Histogram of Gaussian Subtracted Image')
        plt.legend()
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    return mask_deviation_value, deviation_2sig, deviation_3sig

def remove_overlapping_blobs(blobs, coords, radii, overlap_threshold=0.5):
    """
    Remove overlapping blobs, keeping the one with smaller radius.
    Parameters:
    -----------
    blobs : array of shape (n_blobs, 3)
        Each row: [y, x, radius]
    coords : list of SkyCoord objects
        Corresponding sky coordinates
    radii : list of float
        Corresponding radii values
    overlap_threshold : float
        Fraction of overlap to consider blobs as duplicates
    Returns:
    --------
    filtered_blobs, filtered_coords, filtered_radii : kept blobs
    removed_blobs, removed_coords, removed_radii : removed blobs
    """
    if len(blobs) == 0:
        return blobs, coords, radii, np.empty((0, 3)), [], []

    # Sort by radius ascending: smaller blobs first
    sorted_indices = np.argsort(blobs[:, 2])
    sorted_blobs  = blobs[sorted_indices]
    sorted_coords = [coords[i] for i in sorted_indices]
    sorted_radii  = [radii[i] for i in sorted_indices]

    keep_mask = np.ones(len(sorted_blobs), dtype=bool)

    for i in range(len(sorted_blobs)):
        if not keep_mask[i]:
            continue

        for j in range(i + 1, len(sorted_blobs)):
            if not keep_mask[j]:
                continue

            dy = sorted_blobs[i, 0] - sorted_blobs[j, 0]
            dx = sorted_blobs[i, 1] - sorted_blobs[j, 1]
            distance = np.sqrt(dy**2 + dx**2)
            smaller_r = min(sorted_blobs[i, 2], sorted_blobs[j, 2])

            if smaller_r <= 0:
                continue

            overlap = (sorted_blobs[i, 2] + sorted_blobs[j, 2] - distance) / smaller_r

            if overlap > overlap_threshold:
                # i is smaller (sorted), so drop larger/extension blob j
                keep_mask[j] = False

    remove_mask = ~keep_mask

    filtered_blobs  = sorted_blobs[keep_mask]
    filtered_coords = [c for k, c in enumerate(sorted_coords) if keep_mask[k]]
    filtered_radii  = [r for k, r in enumerate(sorted_radii) if keep_mask[k]]

    removed_blobs  = sorted_blobs[remove_mask]
    removed_coords = [c for k, c in enumerate(sorted_coords) if remove_mask[k]]
    removed_radii  = [r for k, r in enumerate(sorted_radii) if remove_mask[k]]

    return filtered_blobs, filtered_coords, filtered_radii, removed_blobs, removed_coords, removed_radii

def combine_blobs(all_blobs, all_coords, all_radii):
    """Flatten per-radius blob lists into single combined arrays."""
    if all_blobs:
        return (
            np.vstack(all_blobs),
            [c for sub in all_coords for c in sub],
            [r for sub in all_radii  for r in sub],
        )
    return np.empty((0, 3)), [], []

def estimate_background_sigma(image, sigma=3, maxiters=5):
    """Sigma-clipped RMS of the DoG residual map."""
    clipped = sigma_clip(image, sigma=sigma, maxiters=maxiters)
    return float(np.std(clipped.data[~clipped.mask]))

def run_ps(dog_final, pixel_size, threshold_val, border_pixels):
    return blob_dog(dog_final,
                    min_sigma=0.15 / pixel_size, max_sigma=0.39 / pixel_size,
                    threshold=threshold_val, exclude_border=border_pixels, overlap=0.7)

def run_ext(extmap, pixel_size, threshold_val, border_pixels):
    return blob_dog(extmap,
                    min_sigma=0.5 / pixel_size, max_sigma=1 / pixel_size,
                    threshold=threshold_val, exclude_border=border_pixels, overlap=0.7)

def compute_bright_frac(image, ly, lx, lr):
    """Fraction of pixels within blob circle brighter than the center pixel."""
    y_min, y_max = int(max(ly - lr, 0)), int(min(ly + lr, image.shape[0]))
    x_min, x_max = int(max(lx - lr, 0)), int(min(lx + lr, image.shape[1]))
    yy, xx       = np.mgrid[y_min:y_max, x_min:x_max]
    mask_circle  = np.sqrt((yy - ly)**2 + (xx - lx)**2) <= lr
    mask_bright  = mask_circle & (image[y_min:y_max, x_min:x_max] > 7)
    return mask_bright.sum() / mask_circle.sum()

def overlap_fraction(ly, lx, lr, sy, sx, sr):
    """Fraction of smaller blob area overlapping with the larger blob."""
    dist = np.sqrt((sy - ly)**2 + (sx - lx)**2)
    if dist >= lr + sr:
        return 0.0
    if dist + sr <= lr:
        return 1.0
    r, R, d  = sr, lr, dist
    alpha    = np.arccos(np.clip((d**2 + r**2 - R**2) / (2*d*r), -1, 1))
    beta     = np.arccos(np.clip((d**2 + R**2 - r**2) / (2*d*R), -1, 1))
    intersection = (r**2 * alpha + R**2 * beta
                    - 0.5 * (r**2 * np.sin(2*alpha) + R**2 * np.sin(2*beta)))
    return intersection / (np.pi * r**2)

def calculate_separation(coord1, coord2):
    """Angular separation in degrees between two SkyCoord objects."""
    return coord1.separation(coord2).deg

def circle_overlap(coord1, r1, coord2, r2, pixel_size):
    """Classify geometric relationship between two sky circles."""
    dist      = calculate_separation(coord1, coord2)
    radii_sum = (r1 + r2) * pixel_size
    radii_dif = abs(r1 - r2) * pixel_size
    if dist > radii_sum and dist > radii_dif:
        return 0.0    # disjoint
    elif dist < radii_dif:
        return 1.0    # one inside the other
    return None       # partial overlap

def blob_to_yaml_record(blob, array, wcs, pixel_size, label):
    """Convert a single blob row to a serialisable dict."""
    y, x, r = blob
    y, x    = int(y), int(x)
    coord   = astropy_utils.pixel_to_skycoord(x, y, wcs=wcs).icrs
    return {
        'label':      label,
        'x_px':       x,
        'y_px':       y,
        'radius_px':  float(r),
        'radius_deg': float(r * pixel_size),
        'l_deg':      float(coord.ra.deg),
        'b_deg':      float(coord.dec.deg),
        'center_ts':  float(array[y, x]),
    }

def serialise_group(group, array, wcs, pixel_size, label):
    """Convert a blob group list to YAML-ready records."""
    records = []
    for item in group:
        try:
            blob = item if (hasattr(item, '__len__') and len(item) == 3
                            and not hasattr(item[0], '__len__')) else item[0][0]
        except (TypeError, ValueError, IndexError):
            blob = item
        records.append(blob_to_yaml_record(blob, array, wcs, pixel_size, label))
    return records

def plot_blob_map(array, wcs, xnum, ynum, kept_ext, kept_ps, ax_title):
    """Overlay blobs on the significance map and return the figure."""
    fig = plt.figure(figsize=(12, 8))
    ax  = fig.add_subplot(111, projection=wcs)
    im  = ax.imshow(array, cmap='magma', vmin=-5, vmax=15)
    plt.colorbar(im, ax=ax, label=r'Significance ($\sigma$)',
                 fraction=0.046, pad=0.04)

    for lb in kept_ext:
        ly, lx, lr = lb
        ly, lx = int(ly), int(lx)
        ax.add_patch(plt.Circle((lx, ly), lr, fill=False,
                                edgecolor='white', linewidth=2, linestyle='--'))
        ax.scatter(lx, ly, s=80, c='black', marker='*', zorder=5)
        ax.text(lx, ly + lr + 10, f'TS={array[ly,lx]:.1f}',
                color='white', fontsize=9, ha='center')

    for sb in kept_ps:
        try:
            sy, sx, sr = sb
        except (TypeError, ValueError):
            sy, sx, sr = sb[0][0]
        sy, sx = int(sy), int(sx)
        ax.add_patch(plt.Circle((sx, sy), sr, fill=False,
                                edgecolor='cyan', linewidth=2, linestyle='-'))
        ax.text(sx + sr + 20, sy + sr + 10, f'TS={array[sy,sx]:.1f}',
                color='black', fontsize=9, ha='center')

    ax.legend(handles=[
        Line2D([0],[0], color='white', linewidth=2,   linestyle='--', label='Extended blob'),
        Line2D([0],[0], color='cyan',  linewidth=1.5, linestyle='-',  label='Point source blob'),
    ], fontsize=8, loc='upper right')
    ax.set_xlim(0, xnum);  ax.set_ylim(0, ynum)
    ax.set_title(ax_title, fontsize=9)
    ax.set_xlabel('X (px)');  ax.set_ylabel('Y (px)')
    plt.tight_layout()
    return fig

def deduplicate_ps_group(ps_filtered_group, wcs, sep_threshold_deg=0.4):
    """
    Remove smaller PS blobs from ps_filtered_group when two blobs are
    within sep_threshold_deg of each other, keeping the larger one.

    Parameters
    ----------
    ps_filtered_group : list of (y, x, r) tuples
    array             : significance map (for TS lookup)
    wcs               : WCS object
    pixel_size        : degrees per pixel
    sep_threshold_deg : angular separation threshold in degrees

    Returns
    -------
    kept    : list of blobs that survived
    removed : list of blobs that were culled
    """
    if len(ps_filtered_group) <= 1:
        return list(ps_filtered_group), []

    # Normalise entries — ps_filtered_group can contain raw tuples or
    # nested [(blob, frac)] lists from the grouping loop
    blobs = []
    for item in ps_filtered_group:
        try:
            y, x, r = item
            blobs.append((y, x, r))
        except (TypeError, ValueError):
            y, x, r = item[0][0]
            blobs.append((y, x, r))

    # Sort largest-radius first so we always keep the bigger detection
    blobs = sorted(blobs, key=lambda b: b[2], reverse=True)

    keep_mask = [True] * len(blobs)

    for i in range(len(blobs)):
        if not keep_mask[i]:
            continue
        yi, xi, ri = blobs[i]
        coord_i    = astropy_utils.pixel_to_skycoord(int(xi), int(yi), wcs=wcs).galactic

        for j in range(i + 1, len(blobs)):
            if not keep_mask[j]:
                continue
            yj, xj, rj = blobs[j]
            coord_j     = astropy_utils.pixel_to_skycoord(int(xj), int(yj), wcs=wcs).galactic
            sep         = calculate_separation(coord_i, coord_j)
            if sep < sep_threshold_deg:
                print(f"  PS dedup: removing smaller blob @ ({xj},{yj}) r={rj:.1f}px "
                      f"sep={sep:.3f}° < {sep_threshold_deg}° threshold")
                keep_mask[j] = False

    kept    = [b for b, k in zip(blobs, keep_mask) if     k]
    removed = [b for b, k in zip(blobs, keep_mask) if not k]
    print(f"  PS dedup: {len(blobs)} → {len(kept)} kept, {len(removed)} removed")
    return kept, removed

def threeML_model_from_sources(filtered_df):
    ra_m = []
    dec_m = []
    for xra, xdec in zip(filtered_df['ra'], filtered_df['dec']):
        ra_m.append(xra)
        dec_m.append(xdec)
    source_name = {}
    sources = {}
    spectra = {}
    morphologies = {}

    for i in range(len(filtered_df)):
        if filtered_df['Sigma Radius'][i]<0.12:
            sources[f"spectrum{i}"] = Powerlaw()
            sources[f"source{i}"] = PointSource(f"Source{i}", ra=filtered_df['ra'][i], dec=filtered_df['dec'][i], spectral_shape=  sources[f"spectrum{i}"])
            fluxUnit = 1./(u.keV * u.cm ** 2 * u.s)

            # sources[f"spectrum{i}"].K = 1e-14
            sources[f"spectrum{i}"].K = 1e-22 * fluxUnit
            sources[f"spectrum{i}"].K.fix = False
            sources[f"spectrum{i}"].K.bounds = (1e-26, 1e-20) * fluxUnit
            sources[f"spectrum{i}"].index = -2.5 
            sources[f"spectrum{i}"].index.fix = False
            sources[f"spectrum{i}"].index.bounds = (-3, -1)
            sources[f"spectrum{i}"].piv = 2 * u.TeV
            sources[f"spectrum{i}"].piv.fix = True

            sources[f"source{i}"].position.ra.free = True
            sources[f"source{i}"].position.ra.bounds = ((ra_m[i] - 3), (ra_m[i]+3)) * u.degree
            sources[f"source{i}"].position.dec.free = True
            sources[f"source{i}"].position.dec.bounds = ((dec_m[i] - 3), (dec_m[i]+3)) * u.degree
        else:
            sources[f"spectrum{i}"] = Powerlaw()
            shape = Gaussian_on_sphere()

            shape.lon0 = filtered_df['ra'][i]
            shape.lon0.fix = False
            shape.lon0.bounds = ( (filtered_df['ra'][i] - 3), (filtered_df['ra'][i] + 3))

            shape.lat0 = filtered_df['dec'][i]
            shape.lat0.fix = False
            shape.lat0.bounds = ( (filtered_df['dec'][i] - 3), (filtered_df['dec'][i] + 3))


            shape.sigma = filtered_df['Sigma Radius'][i]
            shape.sigma.fix = False
            shape.sigma.bounds = (0.01, 3)

            sources[f"source{i}"] = ExtendedSource(
                f"Source{i}", spatial_shape=shape, spectral_shape=sources[f"spectrum{i}"]
            )

            fluxUnit = 1. / (u.keV * u.cm**2 * u.s)
            sources[f"spectrum{i}"].K = 1e-22 * fluxUnit
            sources[f"spectrum{i}"].K.fix = False
            sources[f"spectrum{i}"].K.bounds = (1e-26, 1e-20) * fluxUnit
            sources[f"spectrum{i}"].index = -2.5
            sources[f"spectrum{i}"].index.fix = False
            sources[f"spectrum{i}"].index.bounds = (-3, -1)
            sources[f"spectrum{i}"].piv = 2 * u.TeV
            sources[f"spectrum{i}"].piv.fix = True

    keys_to_remove = [key for key in sources if key.startswith("spectrum")]

    # Remove those keys from the dictionary
    for key in keys_to_remove:
        del sources[key]
    allmodel = Model(*sources.values())

    return allmodel, sources

def hdf_to_fits(hdf_path, fits_path):
    """Convert HDF5 blob data to FITS format for 3ML."""
    with h5py.File(hdf_path, 'r') as hdf:
        blobs = hdf['blobs'][:]
        ra    = hdf['ra'][:]
        dec   = hdf['dec'][:]
        sigma = hdf['sigma'][:]

    col1 = fits.Column(name='RA',  format='E', array=ra)
    col2 = fits.Column(name='DEC', format='E', array=dec)
    col3 = fits.Column(name='Sigma', format='E', array=sigma)
    col4 = fits.Column(name='Blob_Radius_PX', format='E', array=blobs[:, 2])

    cols = fits.ColDefs([col1, col2, col3, col4])
    hdu  = fits.BinTableHDU.from_columns(cols)
    hdu.writeto(fits_path, overwrite=True)

def convert_coords(region, system='galactic'):
    new_region=[]

    for i,coord in enumerate(region):
        x=coord[0]
        y=coord[1]
        c = SkyCoord(x,y, frame=system, unit='deg')
        point=( c.icrs.ra.value, c.icrs.dec.value )
        new_region.append(point)

    return new_region

def coord_vectors(region):
    region_vectors=[]
    convert=np.pi/180.0

    for i in region:
        ra=i[0]
        dec=i[1]
        vec=hp.ang2vec(np.pi/2.-dec*convert,ra*convert,lonlat=False)
        region_vectors.append(vec)

    return region_vectors

def save_ROI(pix, NSIDE, name):
    m=np.zeros(hp.nside2npix(NSIDE))
    m[pix]=1
    hp.write_map(name, m, nest=False, coord="C", partial=False, overwrite=True )  

