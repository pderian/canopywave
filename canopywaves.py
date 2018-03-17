"""This module handles the processing of "canopy wave" data in lidar scans by the REAL.

[WIP]

Note: this is not a standalone module, it requires the Python "Toolbox" developed for the REAL,
and acess to the scan database.

Written by P. DERIAN 2018-01-02 |contact@pierrederian.net | www.pierrederian.net
"""
### STL
import collections
import datetime
import os
import sys
### External
import numpy
import pycuda.driver as cuda
import scipy.signal as signal
import skimage.filters as filters
import skimage.feature as feature
import skimage.measure as measure
import skimage.morphology as morphology
### Custom
# Dirty hack to make sure we're not loading SAMPLE's modules
sys.path.insert(0,'/home/pderian/Code/Python/Toolbox')
import lidarRunTime.sqltools as sqltools
import PyCudaTools.CuMedianFilter.CuMedianFilter as cuMedianFilter
###

class CanopyWaveCase:
    """Represents a single case (event) of canopy wave.

    Note: the various products (hard target masks, autocorrelations, etc) are memoized
    to avoid computing multiple times the same thing.

    Written by P. DERIAN 2018-02-02.
    """
    ### Constants

    # path to the CSV file describing the different wave events
    WAVECASES_CSV_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                      'resources/Fifty_three_wave_cases.csv')
    # region of interest (RoI) and grid parameters
    # - (x0, y0) are the coordinates of the center of the RoI.
    # - (xdim , ydim) are the dimensions of the RoI.
    # - resolution is the cartesian grid resolution.
    # note: the true xdim, ydim can be in practice slightly different from values specified
    # here (+/- resolution), as we want and odd number of points in each dimension.
    WAVECASES_GRID = {
        # default values: 1 km x 1km RoI centered on the met tower.
        'default': {'x0': 0., #[m];
                    'y0': -1610., #[m], i.e. 1.61 km south of the REAL;
                    'xdim': 1000., #[m] centered on x0;
                    'ydim': 1000., #[m] centered on y0;
                    'resolution': 4., #[m/pixel];
                    },
        # case-specific values
        11: {'x0': 0., #[m];
             'y0': -1610., #[m], i.e. 1.61 km south of the REAL;
             'xdim':400., #[m] centered on x0;
             'ydim':400., #[m] centered on y0;
             'resolution': 2., #[m/pixel];
             },
        12: {'x0': 0., #[m];
             'y0': -1610., #[m], i.e. 1.61 km south of the REAL;
             'xdim':400., #[m] centered on x0;
             'ydim':400., #[m] centered on y0;
             'resolution': 2., #[m/pixel];
             },
        121: {'x0': 0., #[m];
              'y0': -1660., #[m], i.e. 1.66 km south of the REAL;
              'xdim':300., #[m] centered on x0;
              'ydim':300., #[m] centered on y0;
              'resolution': 1., #[m/pixel];
              },
        36: {'x0': 0., #[m];
             'y0': -1710., #[m];
             'xdim':400., #[m] centered on x0;
             'ydim':600., #[m] centered on y0;
             'resolution': 2., #[m/pixel];
             },
        37: {'x0': 0., #[m];
             'y0': -1710., #[m];
             'xdim':400., #[m] centered on x0;
             'ydim':600., #[m] centered on y0;
             'resolution': 2., #[m/pixel];
             },
        38: {'x0': 0., #[m];
             'y0': -1710., #[m];
             'xdim':400., #[m] centered on x0;
             'ydim':600., #[m] centered on y0;
             'resolution': 2., #[m/pixel];
             },
        40: {'x0': 0., #[m];
             'y0': -1710., #[m];
             'xdim':400., #[m] centered on x0;
             'ydim':600., #[m] centered on y0;
             'resolution': 2., #[m/pixel];
             },
        51: {'x0': 0., #[m];
             'y0': -1710., #[m];
             'xdim':400., #[m] centered on x0;
             'ydim':600., #[m] centered on y0;
             'resolution': 2., #[m/pixel];
             },
    }
    # altitude of the anemometer used in each case.
    # source: big spreadsheet from Shane (Spreadsheet_01_22_18.xls)
    WAVECASES_ANEMOMETER_ALTITUDE = {
        'default': 18., #[m]
        11: 14.,
        12: 14.,
        121: 14.,
        13: 14.,
        14: 14.,
        36: 14.,
        37: 14.,
        38: 12.5,
        40: 12.5,
        41: 12.5,
        }
    # names, values of wave types
    # 'asym' stands for asymmetric, and the suffix is the propagation direction;
    # 'sym' means symmetric (sinusoidal).
    WAVETYPE_NAMES = {'asym_east':1, 'asym_west':-1, 'sym':0}
    WAVETYPE_VALUES = {v:k for k, v in WAVETYPE_NAMES.items()}

    ### High-level functions

    def __init__(self, case_id, datetime_start, datetime_stop):
        """Case constructor.

        :param case_id: the case id,
        :param datetime_start:
        :param datetime_stop:

        Written by P. DERIAN 2018-01-02.
        Updated by P. DERIAN 2018-02-21: grid have odd number of points in each dimension.
        """
        ### case parameters (e.g. from the CSV file)
        self.case_id = case_id
        self.datetime_start = datetime_start
        self.datetime_stop = datetime_stop

        ### the RoI and cartesian grid
        # parameters
        self.param_grid = self.WAVECASES_GRID[self.case_id if (self.case_id in self.WAVECASES_GRID)
                                              else 'default']
        # grid lower bounds
        self.param_grid['xmin'] = self.param_grid['x0'] - self.param_grid['xdim']/2
        self.param_grid['ymin'] = self.param_grid['y0'] - self.param_grid['ydim']/2
        # number of points (=> odd)
        nx = int(numpy.ceil(self.param_grid['xdim']/self.param_grid['resolution']))
        if not nx%2: nx += 1
        ny = int(numpy.ceil(self.param_grid['ydim']/self.param_grid['resolution']))
        if not ny%2: ny += 1
        # grid upper bounds (note: mostly used for plotting?)
        self.param_grid['xmax'] = self.param_grid['xmin'] + float(nx)*self.param_grid['resolution']
        self.param_grid['ymax'] = self.param_grid['ymin'] + float(ny)*self.param_grid['resolution']
        # 1d coordinate values
        self.x = self.param_grid['resolution']*numpy.arange(nx, dtype=float) + self.param_grid['xmin']
        self.y = self.param_grid['resolution']*numpy.arange(ny, dtype=float) + self.param_grid['ymin']
        self.grid_shape = (ny, nx)
        # 2d values
        self.grid_x, self.grid_y = numpy.meshgrid(self.x, self.y)
        self.grid_ranges = numpy.sqrt(self.grid_x**2 + self.grid_y**2) #range from the REAL
        self.grid_angles = numpy.rad2deg(numpy.arctan2(self.grid_x, self.grid_y)) #note: switched y and x to respect meteo convention (0=North)
        self.grid_angles[self.grid_angles<0.] += 360. #so that we only have positive angles
        self.grid_yx = numpy.concatenate((self.grid_y.reshape((-1,1)),
                                          self.grid_x.reshape((-1,1))), axis=1) #for interpolation

        ### scan information from the database
        self.scan_info = sqltools.get_scan_sequence_info(self.datetime_start,
                                                         self.datetime_stop,
                                                         scanType='PPI', fullPath=True,
                                                         allInfo=False)
        ### scan data
        self.scan_data = []
        self.scan_valid = []
        self.grid_scans = []
        self.grid_masks = []
        self.grid_global_mask = None
        ### products
        self.products = {}

    def __str__(self):
        """Used with print()

        :return: a str describing the instance.

        Written by P. DERIAN 2018-02-02.
        """
        name_str = '{} #{}'.format(self.__class__.__name__, self.case_id)
        data_str = '{:%Y-%m-%d %H:%M:%S} - {:%Y-%m-%d %H:%M:%S} UTC ({} scans)'.format(
            self.datetime_start, self.datetime_stop, len(self.scan_info))
        grid_str = '[{xmin}, {xmax}]x[{ymin}, {ymax}] m^2 ({nx}x{ny} pixel^2 @ {resolution} m/pixel)'.format(
            nx=self.grid_shape[1], ny=self.grid_shape[0], **self.param_grid)
        return '{}\n\tdata: {}\n\tgrid: {}'.format(name_str, data_str, grid_str)

    def load_scan_data(self, device_id=0, cuda_context=None, min_domain=.25):
        """Load, preprocess and grid the scan data for the given case.

        :param device_id: the CUDA device id for preprocessing filters;
        :param cuda_context: optional existing CUDA context [TODO] test.
        :param min_domain: min fraction of the grid domain covered by the scan. Below said
                           fraction, the scan is rejected.

        Written by P. DERIAN 2018-02-03.
        Updated by P. DERIAN 2018-02-07: dealt with non-valid scans.
        """
        print('* (Case #{:02d}) Loading scan data'.format(self.case_id))
        ### Initialize
        self.scan_data = []
        self.scan_valid = []
        self.grid_scans = []
        self.grid_masks = []
        self.grid_global_mask = None
        ### Setup CUDA
        if cuda_context is None:
            print('Initializing CUDA')
            cuda.init()
            cuda_device  = cuda.Device(device_id)
            cuda_context = cuda_device.make_context()
            own_context = True
        else:
            own_context = False
        try: #we use try block to catch exception and allow cuda to close properly
            ### setup filters
            lp_median = cuMedianFilter.CuMedianFilter(7)
            hp_median = cuMedianFilter.CuMedianFilter(333, high_pass=True)
            ### now for each scan
            bad_scan_indices = []
            for n, tmp_info in enumerate(self.scan_info):
                # read scan data
                tmp_data = bscan.readPreprocScan(tmp_info['path'],
                                                 hpMedianFilter=hp_median, lpMedianFilter=lp_median,
                                                 removeHardTarget=False, verbose=True, decimate=0,
                                                 withSNR=False, withImgSNR=False)
                # interpolate
                tmp_scan = self.interpolate(tmp_data['x'], tmp_data['y'], tmp_data['scan'])
                tmp_mask = self.domain_mask(tmp_data['azimuth'], tmp_data['range'])
                # check the domain, move on to next scan if too little covered
                tmp_valid = tmp_mask.mean() < (1.-min_domain)
                if not tmp_valid:
                    print('[!] domain covered by scan is below threshold')
                # append
                self.scan_data.append(tmp_data)
                self.scan_valid.append(tmp_valid)
                self.grid_scans.append(tmp_scan)
                self.grid_masks.append(tmp_mask)
            ### compute the global domain mask
            # take the stack as an array (only valid scans), sum across scans, cast to bool
            self.grid_global_mask = numpy.asarray(self.grid_masks)[self.scan_valid].sum(axis=0).astype(bool)
        ### If failure, clear and pass
        except Exception as e:
            print('[!] load_scan_data() failed with error: {}'.format(e))
            self.scan_data = []
            self.scan_valid = []
            self.grid_scans = []
            self.grid_masks = []
            pass
        ### detach CUDA before leaving
        # but only if it was created here
        if own_context:
            cuda_context.detach()

    ### Products

    def hard_target_mask(self, threshold=3.):
        """Compute a hard target mask from the gridded scan data.

        :param threshold: threshold above which signal is considered hard-target.
        :return: a mask, same shape as the grid (self.grid_shape).

        Note: return a mask in numpy.ma sense, i.e. True where masked.

        Written by P. DERIAN 2018-02-03.
        Updated by P. DERIAN 2018-02-12: added custom mask for tower shadow.
        """
        if 'hard_target_mask' not in self.products:
            print('* (Case #{:02d}) Generating hard-target mask'.format(self.case_id))
            # compute the max over the stack of (valid) gridded scans
            max_scan = numpy.asarray(self.grid_scans)[self.scan_valid].max(axis=0)
            # set out-of-domain values of max_scan to zero, just in case
            max_scan *= numpy.logical_not(self.grid_global_mask).astype(float)
            # threshold
            auto_mask = max_scan>threshold
            # custom mask: manually discrad the tower shadow
            custom_mask = numpy.logical_and(self.grid_y<-1610.,
                                            numpy.abs(self.grid_x)<2.*self.param_grid['resolution'])
            # the combination of both masks
            self.products['hard_target_mask'] = numpy.logical_or(auto_mask, custom_mask)
        return self.products['hard_target_mask']

    def autocorrelations(self):
        """Compute autocorrelation functions for each scan.

        :return: autocorrel, (xlag, ylag)

        Written by P. DERIAN 2018-02-07.
        """
        if 'autocorrelations' not in self.products:
            print('* (Case #{:02d}) Computing autocorrelations'.format(self.case_id))
            ### the displacements (lag)
            ny, nx = self.grid_shape
            xl = self.param_grid['resolution']*(nx//2)
            yl = self.param_grid['resolution']*(ny//2)
            xlag = numpy.linspace(-xl, xl, nx)
            ylag = numpy.linspace(-yl, yl, ny)
            xlag, ylag = numpy.meshgrid(xlag, ylag)
            ### the autocorrelations
            # hard targets
            hard_target_mask = self.hard_target_mask()
            # tapering window
            window = numpy.outer(signal.hann(ny), signal.hann(nx))
            # now for each scan
            autocorrelations = []
            for idx_scan, (scan, mask) in enumerate(zip(self.grid_scans, self.grid_masks)):
                # full mask (domain + hard targets)
                full_mask = numpy.logical_or(mask, hard_target_mask)
                # full window (tapering window + mask)
                tmp_window = window*numpy.logical_not(full_mask).astype(float)
                tmp_scan = scan*tmp_window
                # compute auto-coorrelation in Fourrier domain
                tmp_fft = numpy.fft.fft2(tmp_scan - tmp_scan.mean())
                tmp_autocorrel = numpy.real(numpy.fft.ifft2(tmp_fft*numpy.conj(tmp_fft)))
                autocorrelations.append(numpy.fft.fftshift(tmp_autocorrel))
            self.products['autocorrelations'] = autocorrelations, (xlag, ylag)
        return self.products['autocorrelations']

    def wave_parameters(self,  max_wavelength=120.):
        """Estimate wave parameters (wavelength, propagation direction) from the
        autocorrelation function (AF).

        :param max_wavelength: in [m].
        :return: waves, mean_wave
            waves: a list of dict of wave parameters, one for each scan;
            mean_wave: a dict wave parameters, computed from the mean AF.

        Written by P. DERIAN 2017-02-07.
        Updated by P. DERIAN 2018-02-21: fixed bug when some bins had zero count.
        Updated by P. DERIAN 2018-03-17: two passes to improve robustness.
        """
        if 'wave_parameters' not in self.products:
            print('* (Case #{:02d}) Estimating wave parameters'.format(self.case_id))
            ### the autocorrelation
            # compute autocorrelations
            autocorrelations, (xlag, ylag) = self.autocorrelations()
            # average
            mean_autocorrelation = numpy.asarray(autocorrelations)[self.scan_valid].mean(axis=0)
            ### the bins for spectra computation
            # radius bins, indices and spectrum initialization
            radii = numpy.sqrt(xlag**2 + ylag**2) #values on the grid
            nbins = max_wavelength//2.
            rbins, dr = numpy.linspace(0., max_wavelength,
                                       num=nbins+1, retstep=True) #bin edges
            rvalues = 0.5*(rbins[1:] + rbins[:-1]) #bin values (center of each bin)
            rindices = numpy.digitize(radii, rbins)
            # direction bins, indices and spectrum initialization
            angles = numpy.rad2deg(numpy.arctan2(xlag, ylag)) #meteo convention (0=North)
            directions = angles
            directions[directions<0] += 180. #we use directions in [0, 180[ degree
            directions = directions%360.
            dbins, dd = numpy.linspace(2.5, 177.5, num=36, retstep=True) #bin edges, every 5 deg (note: 0 avoided due to artifacts)
            dvalues = 0.5*(dbins[1:] + dbins[:-1]) #bin values (center of each bin)
            dindices = numpy.digitize(directions, dbins)
            ### estimate wave parameters
            def find_parameters(autocorrel, firstguess_wavenumber=None,
                                firstguess_orientation=None):
                ### initialize spectra
                iso_spectrum = numpy.zeros(rbins.size-1)
                iso_count = numpy.zeros_like(iso_spectrum)
                dir_spectrum = numpy.zeros(dbins.size-1)
                dir_count = numpy.zeros_like(dir_spectrum)
                ### isotropic spectrum
                if firstguess_orientation is not None:
                    # consider only points within +/- 45 deg of the given firstguess
                    test_orientation = lambda x: numpy.abs((x - firstguess_orientation + 90.)%180. - 90.)<45.
                else:
                    # else take any point
                    test_orientation = lambda x: True
                for j, k in enumerate(rindices.flat):
                    oj = directions.flat[j] #orientation at current point j
                    # if the point belongs to a radius bin
                    if k>0 and k<rbins.size and test_orientation(oj):
                        iso_spectrum[k-1] += autocorrel.flat[j]
                        iso_count[k-1] += 1.
                # normalize, smooth, compute derivatives
                nz_count = iso_count > 0
                iso_rvalues = rvalues[nz_count]
                iso_spectrum = iso_spectrum[nz_count]
                iso_spectrum /= iso_count[nz_count]
                iso_spectrum /= iso_spectrum.max()
                iso_spectrum = ndimage.uniform_filter1d(iso_spectrum, 3, mode='nearest')
                deriv1_iso_spectrum = numpy.gradient(iso_spectrum, dr)
                deriv2_iso_spectrum = numpy.gradient(deriv1_iso_spectrum, dr)
                # first strategy: find 1st derivative crossing zero with negative 2nd-order derivative
                peak_found = False
                for i_r, isp in enumerate(iso_spectrum[1:-1],1):
                    if (deriv1_iso_spectrum[i_r]>=.0 and deriv1_iso_spectrum[i_r+1]<=0.):
                        peak_found = True
                        break
                # second strategy: take the smallest one with a negative second derivative
                if not peak_found:
                    isort = numpy.argsort(numpy.abs(deriv1_iso_spectrum))
                    for i_r in isort:
                        if deriv2_iso_spectrum[i_r]<0:
                            peak_found = True
                            break
                # we say that the wavelength is the middle of the bin...
                wavelength = iso_rvalues[i_r] if peak_found else numpy.nan
                ### directional spectrum
                if peak_found:
                    rmin = 3.*self.param_grid['resolution'] #to allow for sufficient decay from 0-lag peak
                    rmax = wavelength
                elif firstguess_wavelength is not None:
                    rmin = max(0.5*firstguess_wavelength, 3.*self.param_grid['resolution'])
                    rmax = min(1.5*firstguess_wavelength, max_wavelength)
                else:
                    rmin = 3.*self.param_grid['resolution']
                    rmax = max_wavelength
                test_radius = lambda x: (x>=rmin) and (x<=rmax)
                for j, k in enumerate(dindices.flat):
                    rj = radii.flat[j] #radius at current point
                    # if the point belongs to a direction bin and is within our radius range of interest
                    if k>0 and k<dbins.size and test_radius(rj):
                        dir_spectrum[k-1] += autocorrel.flat[j]
                        dir_count[k-1] += 1.
                # normalize, smooth
                nz_count = dir_count > 0
                dir_dvalues = dvalues[nz_count]
                dir_spectrum = dir_spectrum[nz_count]
                dir_spectrum /= dir_count[nz_count]
                dir_spectrum /= dir_spectrum.max()
                dir_spectrum = ndimage.uniform_filter1d(dir_spectrum, 3, mode='wrap') #'wrap' because circular data
                # find max, take 9-point window centered on it
                i_d = numpy.argmax(dir_spectrum)
                i_win = range(i_d-4, i_d+5)
                i_win = [i if i<dir_spectrum.size else i%dir_spectrum.size for i in i_win]
                d_win = dir_dvalues[i_win]
                # fix directions values if crossing 180-0 degree, so they are increasing
                if d_win[0]>d_win[-1]:
                    i0 = 0
                    while d_win[i0]<d_win[i0+1]:
                        i0 += 1
                    d_win[:i0+1] -= 180.
                # fit 2nd-order polynomial over 9 points (40 deg)
                p = numpy.polyfit(d_win, dir_spectrum[i_win], 2)
                # take the coordinate of its maximum: this is the direction along which lies the crest
                crest_dir = (-p[1]/(2.*p[0]) + 180.)%180.
                # so that the propagation direction is perpendicular.
                propagation_dir = (crest_dir + 90.)%180.
                ###
                return {'wavelength': wavelength,
                        'crest_direction': crest_dir,
                        'propagation_direction': propagation_dir,
                        'iso_spectrum': iso_spectrum,
                        'dir_spectrum': dir_spectrum,
                        'radii': iso_rvalues,
                        'directions': dir_dvalues,
                        }
            waves = []
            for autocorrelation, valid in zip(autocorrelations, self.scan_valid):
                wave = None
                if valid:
                    wave = find_parameters(autocorrelation) #first pass
                    wave = find_parameters(autocorrelation, wave['wavelength'],
                                           wave['propagation_direction']) #second pass
                    waves.append(wave)
            mean_wave = find_parameters(mean_autocorrelation)
            self.products['wave_parameters'] = waves, mean_wave
        return self.products['wave_parameters']

    def crest_locations(self, correl_threshold=0.1):
        """Attempt to locate wave crests.

        DEPRECATED!

        :param correl_threshold:
        :return: crests, patterns, correl_fields
            crests: a list of binary images (1 where potential crest point), one for
                each scan of the case.
            patterns: a list of wave pattern, one for each scan.
            correl_fields: a list of correlation fields, one for eachs can. These fields
                are the correlation between the scan and its synthetic wave pattern.

        Written by P. DERIAN 2018-02-12.
        """
        if 'crest_locations' not in self.products:
            print('* (Case #{:02d}) Locating crests'.format(self.case_id))
            ### Detect wave parameters
            waves, mean_wave = self.wave_parameters()
            wavelengths_fixed, directions_fixed, _ = self.fix_wave_outliers(waves, mean_wave)
            ### hard target mask
            hard_target_mask = self.hard_target_mask()
            # dilate the hard-target mask to minimize artifacts
            dilation_kernel = morphology.disk(3, dtype=bool)
            dilated_hard_target_mask = morphology.binary_dilation(hard_target_mask,
                                                                  dilation_kernel)
            ### scans
            crests = []
            patterns = []
            correl_fields = []
            resolution = self.param_grid['resolution']
            for (scan_info, scan_data, scan_valid, grid_scan, grid_mask,
                 wavelength, propagation_direction) in zip(self.scan_info, self.scan_data,
                                                           self.scan_valid, self.grid_scans,
                                                           self.grid_masks, wavelengths_fixed,
                                                           directions_fixed):
                ### synthetic wave pattern
                n = numpy.int(numpy.ceil(wavelength/resolution))
                if not n%2: #make sure its dimension is odd
                    n += 1
                x = resolution*(numpy.arange(n) - float(n//2))
                x, y = numpy.meshgrid(x, x)
                theta = numpy.deg2rad(propagation_direction)
                wave_pattern = numpy.cos( (x*numpy.sin(theta) + y*numpy.cos(theta))*(2.*numpy.pi/wavelength) )
                ### scan
                # compute full mask, apply
                full_scan_mask = numpy.logical_or(grid_mask, hard_target_mask)
                full_scan_window = numpy.logical_not(full_scan_mask).astype(float)
                ### correlate (in spatial domain)
                correl = ndimage.correlate(grid_scan*full_scan_window, wave_pattern,
                                           mode='constant', cval=0.)
                correl *= 2./float(wave_pattern.size) #normalize
                ### find crests
                # by skeletonization
                full_crest_mask = numpy.logical_or(grid_mask, dilated_hard_target_mask)
                is_crest = numpy.logical_and(morphology.skeletonize_3d(correl>correl_threshold),
                                             numpy.logical_not(full_crest_mask))
                # append
                crests.append(is_crest)
                patterns.append(wave_pattern)
                correl_fields.append(correl)
            self.products['crest_locations'] = crests, patterns, correl_fields
        return self.products['crest_locations']

    def crest_locations2(self, correl_threshold=0.2, area_threshold=6, max_angle_diff=20.,
                         asym_threshold=1.05):
        """Attempt to locate wave crests.

        :param correl_threshold: min correlation value for crest detection;
        :param area_threshold: min number of points in a crest line;
        :param asym_threshold: min asymmetry ratio;
        :param max_angle_diff: max angle difference in [deg] when excluding crests;
        :return: crests, a list of dict with fields:

        Written by P. DERIAN 2018-02-21.
        """
        if 'crest_locations' not in self.products:
            print('* (Case #{:02d}) Locating crests v2'.format(self.case_id))
            ### Detect wave parameters
            waves, mean_wave = self.wave_parameters()
            wavelengths_fixed, directions_fixed, _ = self.fix_wave_outliers(waves, mean_wave)
            ### hard target mask
            hard_target_mask = self.hard_target_mask()
            # dilate the hard-target mask to minimize artifacts
            dilation_kernel = morphology.disk(3, dtype=bool)
            dilated_hard_target_mask = morphology.binary_dilation(hard_target_mask,
                                                                  dilation_kernel)
            ### scans
            crest_location_results = []
            resolution = self.param_grid['resolution']
            for (scan_info, scan_data, scan_valid, grid_scan, grid_mask,
                 wavelength, propagation_direction) in zip(self.scan_info, self.scan_data,
                                                           self.scan_valid, self.grid_scans,
                                                           self.grid_masks, wavelengths_fixed,
                                                           directions_fixed):
                ### scan
                # compute full mask, apply
                full_crest_mask = numpy.logical_or(grid_mask, dilated_hard_target_mask)
                full_scan_mask = numpy.logical_or(grid_mask, hard_target_mask)
                full_scan_window = numpy.logical_not(full_scan_mask).astype(float)
                # normalize for correlation computation
                tmp_scan = grid_scan*full_scan_window
                tmp_scan = (tmp_scan - tmp_scan.mean())/tmp_scan.std()
                ### synthetic wave patterns
                n = numpy.int(numpy.ceil(wavelength/resolution))
                if not n%2: #make sure its dimension is odd
                    n += 1
                x = resolution*(numpy.arange(n) - float(n//2))
                x, y = numpy.meshgrid(x, x)
                theta = numpy.deg2rad(propagation_direction)
                z = x*numpy.sin(theta) + y*numpy.cos(theta)
                # one symmetric and two asymmetric (for each direction)
                wave_patterns = {'sym': self.symmetric_wave_cos(z, wavelength),
                                 'asym_east': self.breaking_wave_exp(z, wavelength),
                                 'asym_west': self.breaking_wave_exp(-z, wavelength)}
                crest_direction = (propagation_direction - 90.)%180. #[deg]
                ### correlate and detect crests
                wave_correlations = {}
                wave_crests = {}
                wave_scores = {}
                for pattern_name, pattern in wave_patterns.items():
                    ### correlate (in spatial domain)
                    # normalize for correlation computation
                    tmp_pattern =  tmp_pattern = (pattern - pattern.mean())/pattern.std()
                    # compute correlation
                    correl = ndimage.correlate(tmp_scan, tmp_pattern, mode='constant', cval=0.)
                    correl *= 1./float(tmp_pattern.size) #normalize
                    ### find crests, remove spurious
                    # apply dilated mask to remove crest points near hard targets
                    crest_map = numpy.logical_and(morphology.skeletonize_3d(correl>correl_threshold),
                                                  numpy.logical_not(full_crest_mask))
                    # remove spurious crests
                    crest_labels = measure.label(crest_map)
                    crest_properties = measure.regionprops(crest_labels)
                    for item in crest_properties:
                        # remove if too small
                        if item['area']<area_threshold:
                            crest_map[item['coords'][:,0], item['coords'][:,1]] = False
                            continue
                        # or if the structure orientation differs too much from the
                        # wave crest direction. Note: item['orientation'] is not in same reference.
                        ellipse_orientation = (90. + numpy.rad2deg(item['orientation']))%180.
                        angle_diff = (crest_direction - ellipse_orientation + 90.)%180. - 90.
                        if numpy.abs(angle_diff)>max_angle_diff:
                            #print('[!]', pattern_name, angle_diff) #[DEBUG]
                            crest_map[item['coords'][:,0], item['coords'][:,1]] = False
                            continue
                    ### finally compute score
                    crest_score = grid_scan[crest_map].mean()
                    ### store results
                    wave_correlations[pattern_name] = correl
                    wave_crests[pattern_name] = crest_map
                    wave_scores[pattern_name] = crest_score
                ### select best pattern
                sorted_scores = sorted(wave_scores.items(), key=lambda x:x[1])
                best_match, _ = sorted_scores[-1] #pattern with highest score
                # check the asymmetry: if scores too close, set symmetry as the best
                asym_scores = (wave_scores['asym_east'], wave_scores['asym_west'])
                asym_ratio = max(asym_scores)/min(asym_scores)
                if (best_match is not 'sym') and (asym_ratio<asym_threshold):
                    best_match = 'sym'
                ### re-label the best crest map
                crest_map = wave_crests[best_match]
                crest_labels = measure.label(crest_map)
                crest_properties = measure.regionprops(crest_labels)
                ### append
                crest_location_results.append({
                    'wavelength': wavelength,
                    'propagation_direction': propagation_direction,
                    'wave_type': best_match,
                    'wave_pattern': wave_patterns[best_match],
                    'correlation_field': wave_correlations[best_match],
                    'crest_map': crest_map,
                    'crest_labels': crest_labels,
                    'crest_properties': crest_properties,
                    'score': wave_scores[best_match],
                    'asymmetry_ratio': asym_ratio,
                    })
            self.products['crest_locations'] = crest_location_results
        return self.products['crest_locations']

    ### Helpers

    def interpolate(self, x, y, values):
        """Interpolate the given field on the cartesian grid (nearest neighbor interpolation).

        :param x:
        :param y:
        :param values:
        :return: values interpolated on the grid (self.grid_shape).

        Written by P. DERIAN 2018-02-02.
        """
        # interpolate
        input_yx = numpy.concatenate((y.reshape((-1,1)), x.reshape((-1,1))), axis=1)
        nn_interp = interpolate.NearestNDInterpolator(input_yx, values.ravel())
        # return
        return nn_interp(self.grid_yx).reshape(self.grid_shape)

    def domain_mask(self, angles, ranges):
        """Compute the grid domain mask from given scan polar grid.

        :param angles: array of scan angles;
        :param ranges: array of range values;
        :return: a mask, same shape as the grid (self.shape)

        Note: return a mask in numpy.ma sense, i.e. True where masked.

        Written by P. DERIAN 2018-02-02.
        Updated by P. DERIAN 2018-03-17: switched to measure.points_in_poly() for robustness.
        """
        ### coordinates of the polar grid
        angles_r = numpy.deg2rad(angles)
        xp = numpy.outer(ranges, numpy.sin(angles_r))
        yp = numpy.outer(ranges, numpy.cos(angles_r))
        ### boundary polygon (counterclockwise)
        # Note: simplified polygon, approximated using only the point at half-arcs
        polygon_idx = [(0,0), (-1,0), (-1,angles.size//2), (-1,-1), (0,-1), (0,angles.size//2)]
        polygon_yx = numpy.array([[yp[idx], xp[idx]] for idx in polygon_idx])
        ### test
        inside_domain = measure.points_in_poly(self.grid_yx, polygon_yx).reshape(self.grid_shape)
        return numpy.logical_not(inside_domain)

    @classmethod
    def generate_standard_cases(self, csv_path=WAVECASES_CSV_PATH):
        """Generate instances for the various standard wave cases.

        :param csv_path: path to the csv file listing the variosu cases;
        :return: a dict of cases. Keys are the case indices as in [1].

        References:
        [1] S. D. Mayor, "Observations of microscale internal gravity waves in very stable
            atmospheric boundary layers over an orchard canopy". Agricultural and Forest Meteorology,
            2017.

        Written by P. DERIAN 2018-01-02.
        """
        print('* Generating standard cases')
        ### load data
        # skip the first row which is the column names.
        data = numpy.loadtxt(csv_path, dtype=int, delimiter=',', skiprows=1)
        ### Header states:
        # Year,Month,Day,Hour,Min,Sec,Hour,Min,Sec,Hour,Min,Sec,Hour,Min,Sec
        # so there are 4 times for each episode. From Shane:
        #   "[...] the row number corresponds to the episode number in the infamous canopy waves paper.
        #    The first time is the beginning of the 5-minute period used to analyze in situ data.
        #    The second time is the ending of the 5-minute period used to analyze in situ data.
        #    The third time is the beginning of when waves became apparent in the lidar scans.
        #    The fourth time is the ending of when waves were apparent in the lidar scans.
        #    You'll want to use these last two times."
        # Note: this is UTC time.
        cases = {}
        for k, (year, month, day, _, _, _, _, _, _, hour_start, min_start, sec_start,
                hour_stop, min_stop, sec_stop) in enumerate(data, 1):
            # the UTC datetimes
            datetime_start_UTC = datetime.datetime(year, month, day,
                                                   hour_start, min_start, sec_start)
            datetime_stop_UTC = datetime.datetime(year, month, day,
                                                  hour_stop, min_stop, sec_stop)
            # now instantiate
            cases[k] = CanopyWaveCase(k, datetime_start_UTC, datetime_stop_UTC)
        ### return
        return cases

    @classmethod
    def fix_wave_outliers(self, waves, default_wave, interp_kind='linear'):
        """Detect and fix outliers in the sequence of waves parameters.

        :param waves:
        :param default_wave:
        :param interp_kind:
        :return: wavelengths, directions, is_fixed

        [TODO] robustify.

        Written by P. DERIAN 2018-02-08.
        """
        ### extract data, fill missing
        directions = numpy.array([wave['propagation_direction'] if (wave is not None)
                                 else numpy.nan for wave in waves])
        wavelengths = numpy.array([wave['wavelength'] if (wave is not None)
                                   else numpy.nan for wave in waves])
        directions[numpy.isnan(directions)] = default_wave['propagation_direction']
        wavelengths[numpy.isnan(wavelengths)] = default_wave['wavelength']
        ### transform wavelength, direction => coordinates
        directions_rad = numpy.deg2rad(directions)
        x = wavelengths*numpy.sin(directions_rad)
        y = wavelengths*numpy.cos(directions_rad)
        ### test each coordinate, combine
        test_x = self.find_ouliers_MAD(x)
        test_y = self.find_ouliers_MAD(y)
        is_bad = numpy.logical_or(test_x, test_y)
        is_valid = numpy.logical_not(is_bad)
        ### interpolate where outliers
        idx = numpy.arange(len(is_bad))
        interp = interpolate.interp1d(idx[is_valid], x[is_valid], kind=interp_kind,
                                      fill_value='extrapolate')
        xi = interp(idx)
        interp = interpolate.interp1d(idx[is_valid], y[is_valid], kind=interp_kind,
                                      fill_value='extrapolate')
        yi = interp(idx)
        wavelengths = numpy.sqrt(xi**2 + yi**2)
        directions = numpy.rad2deg(numpy.arctan2(xi, yi))
        directions[directions<0] += 180.
        return wavelengths, directions, is_bad

    ### Static

    @staticmethod
    def find_ouliers_MAD(z, coeff=3.):
        """Detect outliers based on MAD (median of absolute deviations).

        :param z: the sequence;
        :param coeff: the tolerance coefficient;
        :return: is_outlier (True where flagged as outlier)

        Written by P. DERIAN 2018-02-08.
        """
        median_z = numpy.median(z)
        abs_res = numpy.abs(z - median_z)
        median_abs_res = numpy.median(abs_res)
        return abs_res>coeff*median_abs_res

    @staticmethod
    def wind_direction(u, v):

        """
        Computes the direction of a wind field/time-series.
        Uses Meteo convention: 0 is North, 90 is East, etc.
        The given angle is the direction the wind COMES from.

        :param u: horizontal wind component
        :param v: vertical wind component
        :return: the direction in [degree]

        Written by P. DERIAN 2018-02-13.
        """
        # Note: reverse u, v order to get 0 = North
        # and adding -pi for the direction convention (wind COMES from)
        theta = numpy.arctan2(numpy.atleast_1d(u), numpy.atleast_1d(v)) - numpy.pi
        theta[ theta<0] += 2.*numpy.pi #so we get only positive values in [0, 2*pi]
        theta = numpy.rad2deg(theta) #convert to degrees
        return theta

    @staticmethod
    def breaking_wave_linear(x, w):
        """Piecewise-linear profile of a breaking wave (asymmetric profile).

        :param x: input points in [m];
        :param w: wavelength [m];
        :return: the wave profile.

        Written by P. DERIAN 2018.02-21.
        """
        xcut = w/5. #where the change occurs
        xp = (x + xcut)%w #periodize
        part1 = xp<=xcut
        part2 = numpy.logical_not(part1)
        result = numpy.zeros_like(x)
        result[part1] = (2./xcut)*xp[part1] - 1.
        result[part2] = (-2./(w-xcut))*(xp[part2] - w) - 1.
        return result

    @staticmethod
    def breaking_wave_exp(x, w):
        """Piecewise-exponential profile of a breaking wave (asymmetric profile).

        :param x: input points in [m];
        :param w: wavelength [m];
        :return: the wave profile.

        Written by P. DERIAN 2018.02-21.
        """
        xcut = w/5. #where the change occurs
        tau1 = xcut/numpy.pi #decay of the increasing part
        tau2 = (w - xcut)/numpy.pi #decay of the decreasing part
        xp = (x + xcut)%w #periodize
        part1 = xp<=xcut
        part2 = numpy.logical_not(part1)
        result = numpy.zeros_like(x)
        result[part1] = 1. - 2.*numpy.exp((-1./tau1)*xp[part1])
        result[part2] = 2.*numpy.exp((-1./tau2)*(xp[part2] - xcut)) - 1.
        return result

    @staticmethod
    def symmetric_wave_cos(x, w):
        """Cosine profile of a wave (symmetric profile).

        :param x: input points in [m];
        :param w: wavelength [m];
        :return: the wave profile.

        Written by P. DERIAN 2018.02-21.
        """
        return numpy.cos((2.*numpy.pi/w)*x)


### MAIN FUNCTIONS ###

if __name__=="__main__":
    ###
    import glob
    import pprint
    import shutil
    import subprocess
    ###
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.patches as patches
    import matplotlib.pyplot as pyplot
    import scipy.ndimage as ndimage
    import scipy.interpolate as interpolate
    ###
    import lidarIO.bscan as bscan
    ###
    OUTPUT_ROOT = '/bulk_storage/pderian_workspace/CanopyWaves/output'
    ###

    ### MAIN FUNCTIONS ###

    def export_products(case, root_dir=OUTPUT_ROOT):
        """Save products.

        :param case: a CanopyWaveCase instance;
        :param root_dir: root directory for plots;

        Written by P. DERIAN 2018-02-19.
        """
        ### Output directory
        output_dir = os.path.join(root_dir, 'products', 'case_{:02d}'.format(case.case_id))
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir, mode=0o755)
            print('Created output directory: {}'.format(output_dir))
        ### Hard-target mask
        # compute
        case.hard_target_mask()
        # save
        output_file = os.path.join(output_dir, 'hard_target_mask.png')
        pyplot.imsave(output_file, (255*case.products['hard_target_mask']).astype('uint8'),
                      format='png', origin='upper', cmap='gray', vmin=0, vmax=255)
        print('saved {}'.format(output_file))
        ### Products
        # compute
        case.crest_locations2()
        for scan_info, grid_scan, grid_mask, crest_data in zip(case.scan_info,
                                                               case.grid_scans,
                                                               case.grid_masks,
                                                               case.products['crest_locations'],
                                                               ):
            basename, _ = os.path.splitext(os.path.basename(scan_info['path']))
            # the grid mask
            output_file = os.path.join(output_dir, 'domain_{}.png'.format(basename))
            pyplot.imsave(output_file, (255*grid_mask).astype('uint8'),
                          format='png', origin='upper', cmap='gray', vmin=0, vmax=255)
            print('saved {}'.format(output_file))
            # the grid scan
            output_file = os.path.join(output_dir, 'scan_{}.png'.format(basename))
            pyplot.imsave(output_file, numpy.logical_not(grid_mask).astype(float)*grid_scan,
                          format='png', origin='upper', cmap='gray', vmin=-1., vmax=3.)
            print('saved {}'.format(output_file))
            # the crests
            output_file = os.path.join(output_dir, 'crests_{}.png'.format(basename))
            pyplot.imsave(output_file, (255*crest_data['crest_map']).astype('uint8'),
                          format='png', origin='upper', cmap='gray', vmin=0, vmax=255)
            print('saved {}'.format(output_file))

    def plot_case_scans(case, root_dir=OUTPUT_ROOT):
        """Plot the scans for the given case.

        :param case: a CanopyWaveCase instance;
        :param root_dir: root directory for plots;

        Written by P. DERIAN 2018-02-02.
        Updated by P. DERIAN 2018-02-07: added warning when scan not valid.
        """
        ### output directory for scans
        output_dir = os.path.join(root_dir, 'scans', 'case_{:02d}'.format(case.case_id))
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir, mode=0o755)
            print('Created output directory: {}'.format(output_dir))
        ### plot
        # get the hard target mask for this case
        hard_target_mask = case.hard_target_mask()
        # now for each scan
        for scan_info, scan_data, scan_valid, grid_scan, grid_mask in zip(case.scan_info,
                                                                          case.scan_data,
                                                                          case.scan_valid,
                                                                          case.grid_scans,
                                                                          case.grid_masks):
            # full mask
            full_mask = numpy.logical_or(hard_target_mask, grid_mask)
            ### figure
            dpi = 72
            fig, axes = pyplot.subplots(1,2,
                                        gridspec_kw={'left':0.09, 'right':0.97, 'wspace':0.25},
                                        figsize=(768./dpi, 480./dpi))
            # the polar data
            ax = axes[0]
            ax.set_aspect('equal')
            ax.set_title('Polar data')
            p0 = ax.pcolormesh(scan_data['x'], scan_data['y'], scan_data['scan'],
                               cmap='nipy_spectral')
            p0.set_clim(-1., 3.)
            # and the gridded data
            ax = axes[1]
            ax.set_title('Gridded data with hard-target mask ({} m/px)'.format(
                         case.param_grid['resolution']))
            p1 = ax.imshow(numpy.ma.array(grid_scan, mask=full_mask), interpolation='nearest',
                           vmin=-1., vmax=3., cmap='nipy_spectral',
                           origin='bottom', extent=[case.x[0], case.x[-1], case.y[0], case.y[-1]])
            if not scan_valid:
                ax.text(0.5, 0.5, '[!] not valid for grid', ha='center', va='center', size='medium',
                        transform=ax.transAxes)
            # style
            pyplot.figtext(0.5, 0.98,
                           'Canopy wave case #{:02d} - {:%Y %b %d - %H:%M:%S} UTC'.format(
                               case.case_id, scan_data['time'][0]),
                           ha='center', va='top', size='large')
            for ax in axes:
                ax.set_xlim(xmin=case.x[0], xmax=case.x[-1])
                ax.set_ylim(ymin=case.y[0], ymax=case.y[-1])
                ax.set_xlabel('x (m)')
                ax.set_ylabel('y (m)')
            # save
            basename, _ = os.path.splitext(os.path.basename(scan_info['path']))
            output_file = os.path.join(output_dir, basename + '.png')
            fig.savefig(output_file, format='png', dpi=dpi)
            print('saved {}'.format(output_file))
            pyplot.close(fig)

    def plot_case_waves(case, root_dir=OUTPUT_ROOT):
        """Plot the wave data for the given case.

        :param case: a CanopyWaveCase instance;
        :param root_dir: root directory for plots;

        Written by P. DERIAN 2018-02-02.
        Updated by P. DERIAN 2018-02-07: added warning when scan not valid.
        """
        ### output directory for waves
        output_dir = os.path.join(root_dir, 'waves', 'case_{:02d}'.format(case.case_id))
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir, mode=0o755)
            print('Created output directory: {}'.format(output_dir))
        ### wave parameters
        autocorrelations, (xlag, ylag) = case.autocorrelations()
        waves, mean_wave = case.wave_parameters() #Note: last one is the mean
        mean_autocorrelation = numpy.asarray(autocorrelations).mean(axis=0)
        for k, (wave, autocorrelation) in enumerate(zip(waves,#+[mean_wave,],
                                                        autocorrelations)):#+[mean_autocorrelation,])):
            if wave is None:
                continue
            wavelength = wave['wavelength'] if (wave is not None) else None
            crest_direction = wave['crest_direction'] if (wave is not None) else None
            propagation_direction = wave['propagation_direction'] if (wave is not None) else None
            max_radius = wave['radii'][-1]
            # is it the last one?
            if k==len(waves):
                autocorrelation_title = 'mean auto-correlation'
                description_str = 'results of mean autocorrel.'
                output_basename = 'wave_mean_case_{:02d}.png'.format(case.case_id)
            else:
                autocorrelation_title = 'auto-correlation'
                description_str = '{:%Y %b %d - %H:%M:%S} UTC'.format(
                    case.scan_data[k]['time'][0])
                scan_basename, _ = os.path.splitext(os.path.basename(case.scan_info[k]['path']))
                output_basename = 'wave_{}.png'.format(scan_basename)
            ### plot
            fig, axes_spec = pyplot.subplots(2,1,
                                             gridspec_kw={'left':.55, 'right':.97,
                                                          'bottom':.1,'top':.95,
                                                          'hspace':.5})
            # auto-correlation
            ax_corr = fig.add_axes([.1, .1, .35, .9])
            ax_corr.set_title(autocorrelation_title)
            p = ax_corr.imshow(autocorrelation, interpolation='nearest', origin='bottom',
                               extent=[xlag[0,0], xlag[0,-1], ylag[0,0], ylag[-1,0]])
            ax_corr.set_xlim(-max_radius, max_radius)
            ax_corr.set_ylim(-max_radius, max_radius)
            # isotropic spectrum
            ax = axes_spec[0]
            ax.set_xlim(0., max_radius)
            ax.set_ylim(-0.1, 1.)
            ax.set_xlabel('length (m)')
            ax.plot(wave['radii'], wave['iso_spectrum'])
            if wavelength is not numpy.nan:
                ax.axvline(wavelength, ls='--', color='k' ,lw=0.5)
                ax.text(wavelength, 1.01, 'wavelength: {:.0f} m'.format(wavelength),
                        ha='center', va='bottom', transform=ax.get_xaxis_transform())
                # plot on autocorr
                ax_corr.add_artist(patches.Circle((0., 0.), radius=wavelength, ls='--',
                                                  fc='none', ec='w', lw=0.75))
            # directional spectrum
            ax = axes_spec[1]
            ax.set_xlim(0., 180.)
            ax.set_ylim(-0.1, 1.)
            ax.set_xlabel('orientation (deg)')
            ax.plot(wave['directions'], wave['dir_spectrum'])
            if crest_direction is not None:
                ax.axvline(crest_direction, ls='--', color='k', lw=0.5)
                ax.text(crest_direction, 1.01, 'crest: {:.0f} deg'.format(crest_direction),
                        ha='center', va='bottom', transform=ax.get_xaxis_transform())
            # plot on autocorr
            if propagation_direction is not numpy.nan:
                if propagation_direction>0.:
                    slope = 1./numpy.tan(numpy.deg2rad(propagation_direction))
                    x0 = -max_radius
                    y0 = slope*x0
                    x1 = max_radius
                    y1 = slope*x1
                else:
                    x0 = 0.
                    y0 = -max_radius
                    x1 = 0.
                    y1 = max_radius
                ax_corr.plot([x0, x1], [y0, y1], color='w', ls='--', lw=0.5) #propagation direction
                ax_corr.plot([-y0, -y1], [x0, x1], color='k', ls='--', lw=0.5) #crest direction
            # labels
            wavelength_str = ('{:.0f} m'.format(wavelength) if
                              (wavelength is not None) else 'not found' )
            propagation_str = ('{:.0f} ({:.0f}) deg'.format(propagation_direction,
                                                            propagation_direction+180.) if
                               (propagation_direction is not None) else 'not found')

            pyplot.figtext(0.025, 0.2,
                           'Case #{:02d} - {}\nwavelength = {}\npropagation direction = {}'.format(
                               case.case_id, description_str, wavelength_str, propagation_str),
                           ha='left', va='top', size='medium', color='k')
            # save
            output_file = os.path.join(output_dir, output_basename)
            fig.savefig(output_file, format='png')
            print('saved {}'.format(output_file))
            pyplot.close(fig)

    def plot_case_fixedwaves(case, root_dir=OUTPUT_ROOT):
        """Plot fixed wave parameters.

        :param case: a CanopyWaveCase instance;
        :param root_dir: root directory for plots;

        Written by P. DERIAN 2018-02-08.
        """
        ### output directory for waves
        output_dir = os.path.join(root_dir, 'waves', 'case_{:02d}'.format(case.case_id))
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir, mode=0o755)
            print('Created output directory: {}'.format(output_dir))
        ### wave parameters
        # Note: last one is the mean
        waves, mean_wave = case.wave_parameters()
        ### fix
        wavelengths_fixed, directions_fixed, is_fixed = case.fix_wave_outliers(waves,
                                                                               mean_wave)
        ### display
        # extract estimated wavelength, directions (excluding mean wave)
        wavelengths_estimated = numpy.array([wave['wavelength'] if (wave is not None)
                                            else numpy.nan for wave in waves])
        directions_estimated = numpy.array([wave['propagation_direction'] if (wave is not None)
                                           else numpy.nan for wave in waves])
        idx = numpy.arange(len(waves))
        fig, axes = pyplot.subplots(1,2, gridspec_kw={'right':.97, 'wspace':.25})
        axes[0].plot(idx, directions_estimated, '+-')
        axes[0].plot(idx[is_fixed], directions_estimated[is_fixed], '*r')
        axes[0].plot(idx, directions_fixed, 'o--k', markerfacecolor='none')
        axes[0].set_ylim(0., 180.)
        axes[0].set_xlabel('scan index')
        axes[0].set_ylabel('direction (deg)')
        axes[0].set_title('Propagation orientation')
        axes[1].plot(idx, wavelengths_estimated, '+-')
        axes[1].plot(idx[is_fixed], wavelengths_estimated[is_fixed], '*r')
        axes[1].plot(idx, wavelengths_fixed, 'o--k', markerfacecolor='none')
        axes[1].set_ylim(10., 120.)
        axes[1].set_xlabel('scan index')
        axes[1].set_ylabel('wavelength (m)')
        axes[1].set_title('Wavelength')
        pyplot.figtext(0.5, 0.97, 'Case #{:02d}'.format(case.case_id),
                       ha='center', va='top', size='large')
        output_file = os.path.join(output_dir, 'fixwave_case_{:02d}.png'.format(case.case_id))
        fig.savefig(output_file, format='png')
        print('saved {}'.format(output_file))
        pyplot.close(fig)

    def plot_case_crests(case, root_dir=OUTPUT_ROOT):
        """Plot potential crests.

        :param case: a CanopyWaveCase instance;
        :param root_dir: root directory for plots;

        Written by P. DERIAN 2018-02-08.
        """
        ### output directory for crests
        output_dir = os.path.join(root_dir, 'crests', 'case_{:02d}'.format(case.case_id))
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir, mode=0o755)
            print('Created output directory: {}'.format(output_dir))
        ### Locate crests
        case_crests, case_patterns, case_correl_fields = case.crest_locations()
        ### scans
        hard_target_mask = case.hard_target_mask()
        for (scan_info, scan_data, scan_valid, grid_scan, grid_mask,
             crest, pattern, correl_field) in zip(case.scan_info, case.scan_data,
                                                  case.scan_valid, case.grid_scans,
                                                  case.grid_masks, case_crests,
                                                  case_patterns, case_correl_fields):
            full_mask = numpy.logical_or(grid_mask, hard_target_mask)
            not_crest = numpy.logical_not(crest)
            ### plot
            dpi = 72
            fig, axes = pyplot.subplots(1,2,
                                        gridspec_kw={'left':0.09, 'right':0.97, 'wspace':0.25},
                                        figsize=(768./dpi, 480./dpi))
            # scan
            ax = axes[0]
            ax.set_title('Gridded scan, wave pattern, crests')
            p1 = ax.imshow(numpy.ma.array(grid_scan, mask=full_mask), interpolation='nearest',
                           vmin=-1., vmax=3., cmap='nipy_spectral',
                           origin='bottom', extent=[case.x[0], case.x[-1], case.y[0], case.y[-1]],
                           zorder=1)
            # crests?
            p3 = ax.imshow(numpy.ma.array(crest, mask=not_crest), interpolation='nearest',
                           vmin=0., vmax=1., cmap='gray_r',
                           origin='bottom', extent=[case.x[0], case.x[-1], case.y[0], case.y[-1]],
                           zorder=3)
            # synthetic wave in the bottom-left corner
            pxdim = case.param_grid['resolution']*pattern.shape[1]
            pydim = case.param_grid['resolution']*pattern.shape[0]
            xmin = case.x[0] + 5.
            xmax = case.x[0] + 5. + pxdim
            ymin = case.y[0] + 5.
            ymax = case.y[0] + 5. + pydim
            ax.add_artist(patches.Rectangle((xmin-5., ymin-5.), pxdim+10., pydim+10.,
                                            ec='none', fc='w', zorder=2)) #white background
            p2 = ax.imshow(pattern, interpolation='nearest',
                           vmin=-1., vmax=3., cmap='nipy_spectral',
                           origin='bottom', extent=[xmin, xmax, ymin, ymax],
                           zorder=3) #patch
            ax.add_artist(patches.Rectangle((xmin, ymin), pxdim, pydim,
                                            ec='k', fc='none', lw=1., zorder=4)) #black line
            # correlation
            ax = axes[1]
            ax.set_title('Correlation (scan, wave pattern)')
            p = ax.imshow(numpy.ma.array(correl_field, mask=grid_mask), interpolation='nearest',
                          vmin = -1., vmax=1., cmap='RdYlBu_r',
                          origin='bottom', extent=[case.x[0], case.x[-1], case.y[0], case.y[-1]])
            # style
            scan_str = 'Canopy wave case #{:02d} - {:%Y %b %d - %H:%M:%S} UTC'.format(
                case.case_id, scan_data['time'][0])
            pyplot.figtext(0.5, 0.98, '{}'.format(scan_str),
                           ha='center', va='top', size='large')
            for ax in axes:
                ax.set_xlim(xmin=case.x[0], xmax=case.x[-1])
                ax.set_ylim(ymin=case.y[0], ymax=case.y[-1])
                ax.set_xlabel('x (m)')
                ax.set_ylabel('y (m)')
            basename, _ = os.path.splitext(os.path.basename(scan_info['path']))
            output_file = os.path.join(output_dir, 'crests_{}.png'.format(basename))
            fig.savefig(output_file, format='png', dpi=dpi)
            print('saved {}'.format(output_file))
            pyplot.close(fig)

    def plot_case_crests2(case, root_dir=OUTPUT_ROOT):
        """Plot potential crests.

        :param case: a CanopyWaveCase instance;
        :param root_dir: root directory for plots;

        Written by P. DERIAN 2018-02-08.
        """
        ### output directory for crests
        output_dir = os.path.join(root_dir, 'crests2', 'case_{:02d}'.format(case.case_id))
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir, mode=0o755)
            print('Created output directory: {}'.format(output_dir))
        ### Locate crests
        case_crests = case.crest_locations2()
        ### scans
        hard_target_mask = case.hard_target_mask()
        for (scan_info, scan_data, scan_valid, grid_scan, grid_mask,
             crest_data) in zip(case.scan_info, case.scan_data, case.scan_valid, case.grid_scans,
                                case.grid_masks, case_crests):
            ### data
            full_mask = numpy.logical_or(grid_mask, hard_target_mask)
            crest = crest_data['crest_map']
            not_crest = numpy.logical_not(crest)
            ### plot
            dpi = 72
            fig, axes = pyplot.subplots(1,2,
                                        gridspec_kw={'left':0.09, 'right':0.97,
                                                     'bottom':0.12, 'wspace':0.25},
                                        figsize=(768./dpi, 480./dpi))
            # scan
            ax = axes[0]
            ax.set_title('Gridded scan, wave pattern, crests')
            p1 = ax.imshow(numpy.ma.array(grid_scan, mask=full_mask), interpolation='nearest',
                           vmin=-1., vmax=1., cmap='copper',
                           origin='bottom', extent=[case.x[0], case.x[-1], case.y[0], case.y[-1]],
                           zorder=1)
            # crests?
            p3 = ax.imshow(numpy.ma.array(crest, mask=not_crest), interpolation='nearest',
                           vmin=0., vmax=1., cmap='gray_r',
                           origin='bottom', extent=[case.x[0], case.x[-1], case.y[0], case.y[-1]],
                           zorder=3)
            # synthetic wave in the bottom-left corner
            pxdim = case.param_grid['resolution']*crest_data['wave_pattern'].shape[1]
            pydim = case.param_grid['resolution']*crest_data['wave_pattern'].shape[0]
            xmin = case.x[0] + 5.
            xmax = case.x[0] + 5. + pxdim
            ymin = case.y[0] + 5.
            ymax = case.y[0] + 5. + pydim
            ax.add_artist(patches.Rectangle((xmin-5., ymin-5.), pxdim+10., pydim+10.,
                                            ec='none', fc='w', zorder=2)) #white background
            p2 = ax.imshow(crest_data['wave_pattern'], interpolation='nearest',
                           vmin=-1., vmax=1., cmap='copper',
                           origin='bottom', extent=[xmin, xmax, ymin, ymax],
                           zorder=3) #patch
            ax.add_artist(patches.Rectangle((xmin, ymin), pxdim, pydim,
                                            ec='k', fc='none', lw=1., zorder=4)) #black line
            # correlation
            ax = axes[1]
            ax.set_title('Correlation (scan, wave pattern)')
            p = ax.imshow(numpy.ma.array(crest_data['correlation_field'], mask=grid_mask),
                          interpolation='nearest', vmin = -1., vmax=1., cmap='RdYlBu_r',
                          origin='bottom', extent=[case.x[0], case.x[-1], case.y[0], case.y[-1]])
            # style
            scan_str = 'Canopy wave case #{:02d} - {:%Y %b %d - %H:%M:%S} UTC'.format(
                case.case_id, scan_data['time'][0])
            pyplot.figtext(0.5, 0.98, scan_str, ha='center', va='top', size='large')
            pattern_str = 'wave pattern type: "{}", wavelength={:.0f} m, propagation={:.0f} ({:.0f}) deg'.format(
                crest_data['wave_type'], crest_data['wavelength'],
                crest_data['propagation_direction'],
                crest_data['propagation_direction']+180.)
            pyplot.figtext(0.02, 0.02, pattern_str, ha='left', va='bottom', size='medium')
            for ax in axes:
                ax.set_xlim(xmin=case.x[0], xmax=case.x[-1])
                ax.set_ylim(ymin=case.y[0], ymax=case.y[-1])
                ax.set_xlabel('x (m)')
                ax.set_ylabel('y (m)')
            basename, _ = os.path.splitext(os.path.basename(scan_info['path']))
            output_file = os.path.join(output_dir, 'crests_{}.png'.format(basename))
            fig.savefig(output_file, format='png', dpi=dpi)
            print('saved {}'.format(output_file))
            pyplot.close(fig)
        ### wave type distribution
        # count the types
        wave_types = [item['wave_type'] for item in case_crests] #timeseries of types
        wave_values = [case.WAVETYPE_NAMES[t] for t in wave_types] #timeseries of associated values
        wave_count = collections.Counter(wave_types) #counts of each type in the series
        type_names = sorted(case.WAVETYPE_NAMES.keys()) #sorted list of types
        type_values = sorted(case.WAVETYPE_VALUES.keys()) #sorted list of values
        # plot
        fig, axes = pyplot.subplots(gridspec_kw={'left':.09, 'right':.97})
        ax = axes
        for v in type_values:
            ax.bar(v, wave_count[case.WAVETYPE_VALUES[v]])
        ax.set_xticks(type_values)
        ax.set_xticklabels([' '.join(case.WAVETYPE_VALUES[v].split('_')) for v in ax.get_xticks()])
        ax.set_xlabel('wave type')
        ax.set_ylabel('count')
        case_str = 'Canopy wave case #{:02d} - distribution of wave types'.format(case.case_id)
        ax.set_title(case_str)
        # and save
        output_file = os.path.join(output_dir, 'wave_types_case_{:02d}.png'.format(case.case_id))
        fig.savefig(output_file, format='png', dpi=72)
        print('saved {}'.format(output_file))
        pyplot.close(fig)

    def make_movies(case, root_dir=OUTPUT_ROOT, tmp_dir='/tmp/canopywave_pderian'):
        """Make movies from plots.

        :param case: a CanopyWaveCase instance;
        :param root_dir: root directory for plots;

        Written by P. DERIAN 2018-02-24.
        """
        def clear_tmp():
            for f in os.listdir(tmp_dir):
                file_path = os.path.join(tmp_dir, f)
                try:
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                except Exception as e:
                    print(e)
        ### output directory for movies
        output_dir = os.path.join(root_dir, 'movies', 'case_{:02d}'.format(case.case_id))
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir, mode=0o755)
            print('Created output directory: {}'.format(output_dir))
        ### temp directory
        if not os.path.isdir(tmp_dir):
            os.makedirs(tmp_dir, mode=0o755)
            print('Created tmp directory: {}'.format(tmp_dir))
        ### scan movies
        ### crests movies
        crest_dir = os.path.join(root_dir, 'crests2', 'case_{:02d}'.format(case.case_id))
        crest_images = sorted(glob.glob(os.path.join(crest_dir, 'crests*.png')))
        print('Found {} images'.format(len(crest_images)))
        # copy and rename to tmp
        clear_tmp()
        for k, f in enumerate(crest_images):
            shutil.copy(f, os.path.join(tmp_dir, '{:03d}.png'.format(k)))
        print('Found {} frames'.format(len(glob.glob(os.path.join(tmp_dir, '*.png')))))
        # output file
        output_file = os.path.join(output_dir, 'crests_case_{:02d}.mp4'.format(case.case_id))
        subprocess.call(['ffmpeg',
                         '-y', #force Yes to overwrite
                         '-r', '6', #input framerate
                         '-i', os.path.join(tmp_dir, '%03d.png'),
                         '-vcodec', 'libx264', #output codec
                         '-pix_fmt', 'yuv420p', #pixel format
                         output_file
                         ])
        print('Rendered {}'.format(output_file))

    ### OTHER PLOTS ###

    def plot_synthetic_wave_profiles(output_dir=OUTPUT_ROOT):
        """Plot 1D profiles of synthetic waves used to highlight the wave field.

        Written by P. DERIAN 2018-02-21.
        """
        # parameters
        wavelength = 25. #wavelength in [m]
        resolution = 2. #[m/px]
        # grid
        n = numpy.int(numpy.ceil(wavelength/resolution))
        if not n%2:
            n += 1
        x = resolution*(numpy.arange(n) - float(n//2)) #center around 0.
        # profiles
        y_lin = CanopyWaveCase.breaking_wave_linear(x, wavelength)
        y_exp = CanopyWaveCase.breaking_wave_exp(x, wavelength)
        y_wav = CanopyWaveCase.symmetric_wave_cos(x, wavelength)
        # plot
        fig, axes = pyplot.subplots(gridspec_kw={'right':0.97, 'top':.92})
        axes.plot(x, y_wav, '+-', label='cosine')
        axes.plot(x, y_lin, '+-', label='piecewise-linear')
        axes.plot(x, y_exp, '+-', label='piecewise-exponential')
        axes.axhline(0., color='k', ls='--', label='_nolabel_')
        axes.axvline(0., color='k', ls='--', label='_nolabel_')
        axes.set_xlim(-wavelength/2., wavelength/2.)
        axes.set_xlabel('x (m)')
        axes.set_title('Synthetic wave profiles for wavelength={} m'.format(wavelength))
        axes.legend(frameon=False)
        # save
        output_file = os.path.join(output_dir, 'synthetic_wave_profiles.png')
        fig.savefig(output_file, format='png', dpi=72)
        print('saved {}'.format(output_file))
        pyplot.close(fig)

    ### MISC TESTS - DEPRECATED STUFF ###

    def test_hard_target_mask(case):
        """
        Written by P. DERIAN 2018-02-02.
        """

        grid_scans = numpy.asarray(case.grid_scans)
        not_valid = numpy.asarray(case.grid_masks)[case.valid_scans].sum(axis=0).astype(bool)
        max_scan = grid_scans.max(axis=0)
        mean_scan = grid_scans.mean(axis=0)

        ### [WIP] display
        if 1:
            fig, axes = pyplot.subplots(2,2)
            for n, (var, label) in enumerate(zip([mean_scan, max_scan], ['mean', 'max'])):
                axes[0,n].imshow(numpy.ma.array(var, mask=not_valid), interpolation='nearest',
                                 vmin=-1., vmax=3., cmap='nipy_spectral',
                                 origin='bottom', extent=[case.x[0], case.x[-1], case.y[0], case.y[-1]],
                )
                axes[0,n].set_title(label)
                axes[1,n].imshow(numpy.ma.array(var>5., mask=not_valid), interpolation='nearest',
                                 vmin=0, vmax=1, cmap='gray',
                                 origin='bottom', extent=[case.x[0], case.x[-1], case.y[0], case.y[-1]],
                )
            for ax in axes.ravel():
                ax.set_facecolor('.8')
                ax.set_xticklabels([])
                ax.set_yticklabels([])
            fig.savefig('../tmp/mean_{:02d}.png'.format(case.case_id))
            pyplot.close(fig)

    def test_wave_parameters_spatial(case, max_wavelength=120.):
        """
        Written by P. DERIAN 2018-02-05.
        """
        ### the mask
        hard_target_mask = case.hard_target_mask()
        ### the displacements
        resolution = case.param_grid['resolution']
        ny, nx = case.grid_shape
        sx = resolution*nx//2
        sy = resolution*ny//2
        shiftx = numpy.linspace(-sx, sx, nx)
        shifty = numpy.linspace(-sy, sy, ny)
        shiftx, shifty = numpy.meshgrid(shiftx, shifty)
        # range indices and isotropic spectrum
        ranges = numpy.sqrt(shiftx**2 + shifty**2)
        rbins, dr = numpy.linspace(0, 120, num=61, retstep=True)
        rvalues = 0.5*(rbins[1:] + rbins[:-1])
        rindices = numpy.digitize(ranges, rbins)
        iso_spectrum = numpy.zeros(rbins.size-1)
        iso_count = numpy.zeros_like(iso_spectrum)
        # angle indices and directional spectrum
        angles = numpy.rad2deg(numpy.arctan2(shiftx, shifty)) #meteo convention (0=North)
        directions = angles
        directions[directions<0] += 180.
        directions = directions%360.
        dbins, dd = numpy.linspace(2.5, 177.5, num=36, retstep=True)
        dvalues = 0.5*(dbins[1:] + dbins[:-1])
        dindices = numpy.digitize(directions, dbins)
        dir_spectrum = numpy.zeros(dbins.size-1)
        dir_count = numpy.zeros_like(dir_spectrum)
        ### the autocorrelation
        # taper window
        window = numpy.outer(signal.hann(ny), signal.hann(nx))
        # now for each scan
        res = 0.
        for idx_scan, (scan, mask) in enumerate(zip(case.grid_scans, case.grid_masks)):
            not_valid = numpy.logical_or(mask, hard_target_mask)
            is_valid = window*numpy.logical_not(not_valid).astype(float)
            tmp_scan = scan*is_valid
            tmp_fft = numpy.fft.fft2(tmp_scan - tmp_scan.mean())
            tmp_res = numpy.real(numpy.fft.ifft2(tmp_fft*numpy.conj(tmp_fft)))
            res += numpy.fft.fftshift(tmp_res)
        ### as isotropic spectrum
        for j, k in enumerate(rindices.flat):
            if k>0 and k<rbins.size:
                iso_spectrum[k-1] += res.flat[j]
                iso_count[k-1] += 1.
        # normalize, smooth, compute derivatives
        iso_spectrum /= iso_count
        iso_spectrum /= iso_spectrum.max()
        iso_spectrum = ndimage.uniform_filter1d(iso_spectrum, 3, mode='nearest')
        deriv1_iso_spectrum = numpy.gradient(iso_spectrum, dr)
        deriv2_iso_spectrum = numpy.gradient(deriv1_iso_spectrum, dr)
        # first strategy: find 1st derivative crossing zero with negative 2nd-order derivative
        peak_found = False
        for i_r, isp in enumerate(iso_spectrum[1:-1],1):
            if (deriv1_iso_spectrum[i_r]>=.0 and deriv1_iso_spectrum[i_r+1]<=0.):
                peak_found = True
                break
        # second strategy: take the smallest one with a negative second derivative
        if not peak_found:
            isort = numpy.argsort(numpy.abs(deriv1_iso_spectrum))
            for i_r in isort:
                if deriv2_iso_spectrum[i_r]<0:
                    peak_found = True
                    break
        # we say that the radius is the middle of the bin...
        peak_radius = rvalues[i_r]
        ### as directional spectrum
        if peak_found:
            rmin = 3.*resolution
            rmax = peak_radius
        else:
            rmin = 3.*resolution
            rmax = max_wavelength
        for j, k in enumerate(dindices.flat):
            r = ranges.flat[j]
            if k>0 and k<dbins.size and r>=rmin and r<=rmax:
                dir_spectrum[k-1] += res.flat[j]
                dir_count[k-1] += 1.
        # normalize, smooth
        dir_spectrum /= dir_count
        dir_spectrum /= dir_spectrum.max()
        dir_spectrum = ndimage.uniform_filter1d(dir_spectrum, 3, mode='wrap')
        # find max, take 9-point window centered on it
        i_d = numpy.argmax(dir_spectrum)
        i_win = range(i_d-4, i_d+5)
        d_win = dvalues[i_win]
        # fix directions if crossing 180-0 degree
        if d_win[0]>d_win[-1]:
            i0 = 0
            while d_win[i0]<d_win[i0+1]:
                i0 += 1
            d_win[:i0+1] -= 180.
        # fit 2nd-order polynomial over 9 points (40 deg)
        p = numpy.polyfit(d_win, dir_spectrum[i_win], 2)
        crest_dir = (-p[1]/(2.*p[0]) + 180.)%180.
        propag_dir = (crest_dir + 90.)%180.
        ### plot
        fig, axes_spec = pyplot.subplots(2,1,
                                         gridspec_kw={'left':.55, 'right':.9, 'bottom':.1,'top':.95,
                                                      'hspace':.5})
        # auto-correlation
        ax_corr = fig.add_axes([.1, .1, .35, .9])
        ax_corr.set_title('mean auto-correlation')
        p = ax_corr.imshow(res, interpolation='nearest', origin='bottom', extent=[-sx, sx, -sy, sy])
        ax_corr.set_xlim(-max_wavelength, max_wavelength)
        ax_corr.set_ylim(-max_wavelength, max_wavelength)
        # isotropic spectrum
        ax = axes_spec[0]
        ax.set_xlim(0., max_wavelength)
        #ax.set_ylim(-0.1, 1.)
        ax.set_xlabel('length (m)')
        ax.plot(rvalues, iso_spectrum)
        ax = ax.twinx()
        ax.plot(rvalues, deriv1_iso_spectrum, color='r')
        ax.axhline(0, ls='--', color='r', lw=0.5)
        if peak_found:
            ax.axvline(peak_radius, ls='--', color='k' ,lw=0.5)
            print('peak at {} m, {:.0f} deg'.format(peak_radius, propag_dir))
            ax.text(peak_radius, 1.01, 'peak: {:.0f} m'.format(peak_radius), ha='center', va='bottom',
                    transform=ax.get_xaxis_transform())
            # plot on autocorr
            ax_corr.add_artist(patches.Circle((0., 0.), radius=peak_radius, ls='--',
                                              fc='none', ec='w', lw=0.75))
        # directional spectrum
        ax = axes_spec[1]
        ax.set_xlim(0., 180.)
        ax.set_ylim(-0.1, 1.)
        ax.set_xlabel('direction (deg)')
        ax.plot(dvalues, dir_spectrum)
        ax.axvline(crest_dir, ls='--', color='k', lw=0.5)
        ax.text(crest_dir, 1.01, 'crest: {:.0f} deg'.format(crest_dir), ha='center', va='bottom',
                transform=ax.get_xaxis_transform())
        # plot on autocorr
        if propag_dir>0.:
            slope = 1./numpy.tan(numpy.deg2rad(propag_dir))
            x0 = -max_wavelength
            y0 = slope*x0
            x1 = max_wavelength
            y1 = slope*x1
        else:
            x0 = 0.
            y0 = -max_wavelength
            x1 = 0.
            y1 = max_wavelength
        ax_corr.plot([x0, x1], [y0, y1], color='w', ls='--', lw=0.5) #propagation direction
        ax_corr.plot([-y0, -y1], [x0, x1], color='k', ls='--', lw=0.5) #crest direction
        # labels
        pyplot.figtext(0.025, 0.2,
                       'Case {:02d}\nwavelength = {:.0f} m\npropagation direction = {:.0f} ({:.0f}) deg'.format(
                           case.case_id, peak_radius, propag_dir, propag_dir+180.),
                       ha='left', va='top', size='medium', color='k')
        pyplot.savefig('../tmp/autocorr_{:02d}.png'.format(case.case_id))

    def test_wave_outliers(case):
        """Test detection and fixing of outliers in wave parameters.

        Written by P. DERIAN 2018-02-08.
        """
        def fix_outliers(waves, default_wave, interp_kind='linear'):
            """Detect and fix outliers in the sequence of waves parameters.

            :return: wavelengths, directions, is_fixed

            Written by P. DERIAN 2018-02-08.
            """
            def find_ouliers_MAD(z, coeff=3., size=5):
                """Detect outliers based on MAD (median of absolute deviations).

                :param z: the sequence;
                :param coeff: the tolerance coefficient;
                :param size: the size of the median filter;
                :return: is_outlier (True where flagged as outlier)

                Written by P. DERIAN 2018-02-08.
                """
                #median_z = ndimage.median_filter(z, size=size)
                median_z = numpy.median(z)
                abs_res = numpy.abs(z - median_z)
                #median_abs_res = ndimage.median_filter(abs_res, size=size)
                median_abs_res = numpy.median(abs_res)
                return abs_res>coeff*median_abs_res
            ### extract data, fill missing
            directions = numpy.array([wave['propagation_direction'] if (wave is not None)
                                     else numpy.nan for wave in waves])
            wavelengths = numpy.array([wave['wavelength'] if (wave is not None)
                                       else numpy.nan for wave in waves])
            directions[numpy.isnan(directions)] = default_wave['propagation_direction']
            wavelengths[numpy.isnan(wavelengths)] = default_wave['wavelength']
            ### transform wavelength, direction => coordinates
            directions_rad = numpy.deg2rad(directions)
            x = wavelengths*numpy.sin(directions_rad)
            y = wavelengths*numpy.cos(directions_rad)
            ### test each coordinate, combine
            test_x = find_ouliers_MAD(x)
            test_y = find_ouliers_MAD(y)
            is_bad = numpy.logical_or(test_x, test_y)
            is_valid = numpy.logical_not(is_bad)
            ### interpolate where outliers
            idx = numpy.arange(len(is_bad))
            interp = interpolate.interp1d(idx[is_valid], x[is_valid], kind=interp_kind,
                                          fill_value='extrapolate')
            xi = interp(idx)
            interp = interpolate.interp1d(idx[is_valid], y[is_valid], kind=interp_kind,
                                          fill_value='extrapolate')
            yi = interp(idx)
            wavelengths = numpy.sqrt(xi**2 + yi**2)
            directions = numpy.rad2deg(numpy.arctan2(xi, yi))
            directions[directions<0] += 180.
            return wavelengths, directions, is_bad

        print(case)
        ### find wave parameters
        # note: last wave is the mean
        waves = case.wave_parameters()
        ### fix
        wavelengths_fixed, directions_fixed, is_fixed = fix_outliers(waves[:-1], waves[-1])
        ### display
        # extract estimated wavelength, directions (excluding mean wave)
        wavelengths_estimated = numpy.array([wave['wavelength'] if (wave is not None)
                                            else numpy.nan for wave in waves[:-1]])
        directions_estimated = numpy.array([wave['propagation_direction'] if (wave is not None)
                                           else numpy.nan for wave in waves[:-1]])
        idx = numpy.arange(len(waves)-1)
        fig, axes = pyplot.subplots(1,2)
        axes[0].plot(idx, directions_estimated, '+-')
        axes[0].plot(idx[is_fixed], directions_estimated[is_fixed], '*r')
        axes[0].plot(idx, directions_fixed, 'o--k', markerfacecolor='none')
        axes[0].set_ylim(0., 180.)
        axes[1].plot(idx, wavelengths_estimated, '+-')
        axes[1].plot(idx[is_fixed], wavelengths_estimated[is_fixed], '*r')
        axes[1].plot(idx, wavelengths_fixed, 'o--k', markerfacecolor='none')
        axes[1].set_ylim(10., 120.)
        fig.savefig('../tmp/fixwave_case_{:02d}.png'.format(case.case_id))
        pyplot.close(fig)

    def test_detect_crest(case):
        """
        Written by P. DERIAN 2018-02-07.
        """
        print(case)
        ### output directory for crests
        output_dir = os.path.join('..', 'tmp', 'crests', 'case_{:02d}'.format(case.case_id))
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir, mode=0o755)
            print('Created output directory: {}'.format(output_dir))
        ### Detect wave parameters
        waves = case.wave_parameters()
        wavelengths_fixed, directions_fixed, _ = case.fix_wave_outliers(waves[:-1], waves[-1])
        ### scans
        resolution = case.param_grid['resolution']
        hard_target_mask = case.hard_target_mask()
        for (scan_info, scan_data, scan_valid, grid_scan, grid_mask,
             wavelength, propagation_direction) in zip(case.scan_info, case.scan_data,
                                                       case.scan_valid, case.grid_scans,
                                                       case.grid_masks, wavelengths_fixed,
                                                       directions_fixed):
            ### synthetic wave pattern
            n = numpy.int(numpy.ceil(wavelength/resolution))
            if not n%2:
                n += 1
            x = resolution*(numpy.arange(n) - float(n//2))
            x, y = numpy.meshgrid(x, x)
            theta = numpy.deg2rad(propagation_direction)
            wave_pattern = numpy.cos( (x*numpy.sin(theta) + y*numpy.cos(theta))*(2.*numpy.pi/wavelength) )
            # normalize for correlation computation
            tmp_pattern =  (wave_pattern - wave_pattern.mean())/wave_pattern.std()
            ### scan
            # compute full mask, apply
            full_mask = numpy.logical_or(grid_mask, hard_target_mask)
            full_window = numpy.logical_not(full_mask).astype(float)
            # normalize for correlation computation
            tmp_scan = numpy.ma.array(grid_scan*full_window, mask=full_mask)
            tmp_mean = tmp_scan.mean()
            tmp_std = tmp_scan.std()
            tmp_scan = tmp_scan - tmp_scan.mean()/tmp_scan.std()
            ### correlate (in spatial domain)
            corr = ndimage.correlate(tmp_scan, tmp_pattern, mode='constant', cval=0.)/float(tmp_pattern.size)
            corr *= full_window
            ### find crests
            # dilate the hard-target mask to minimize artifacts
            dilation_kernel = morphology.disk(3, dtype=bool)
            dilated_hard_target_mask = morphology.binary_dilation(hard_target_mask,
                                                                  dilation_kernel)
            # by skeletonization
            is_crest = numpy.logical_and(morphology.skeletonize_3d(corr>0.05),
                                         numpy.logical_not(dilated_hard_target_mask))
            not_crest = numpy.logical_not(is_crest)
            ### plot
            dpi = 72
            fig, axes = pyplot.subplots(1,2,
                                        gridspec_kw={'left':0.09, 'right':0.97, 'wspace':0.25},
                                        figsize=(768./dpi, 480./dpi))
            # scan
            ax = axes[0]
            ax.set_title('Gridded scan, wave pattern, crests')
            p1 = ax.imshow(numpy.ma.array(grid_scan, mask=full_mask), interpolation='nearest',
                           vmin=-1., vmax=3., cmap='nipy_spectral',
                           origin='bottom', extent=[case.x[0], case.x[-1], case.y[0], case.y[-1]],
                           zorder=1)
            # crests?
            p3 = ax.imshow(numpy.ma.array(is_crest, mask=not_crest), interpolation='nearest',
                           vmin=0., vmax=1., cmap='gray_r',
                           origin='bottom', extent=[case.x[0], case.x[-1], case.y[0], case.y[-1]],
                           zorder=3)
            # synthetic wave in the bottom-left corner
            pdim = n*resolution
            xmin = case.x[0] + 5.
            xmax = case.x[0] + 5. + pdim
            ymin = case.y[0] + 5.
            ymax = case.y[0] + 5. + pdim
            ax.add_artist(patches.Rectangle((xmin-5., ymin-5.), pdim+10., pdim+10.,
                                            ec='none', fc='w', zorder=2)) #white background
            p2 = ax.imshow(wave_pattern, interpolation='nearest',
                           vmin=-1., vmax=3., cmap='nipy_spectral',
                           origin='bottom', extent=[xmin, xmax, ymin, ymax],
                           zorder=3) #patch
            ax.add_artist(patches.Rectangle((xmin, ymin), pdim, pdim,
                                            ec='k', fc='none', lw=1., zorder=4)) #black line
            # correlation
            ax = axes[1]
            ax.set_title('Correlation')
            p = ax.imshow(numpy.ma.array(corr, mask=full_mask), interpolation='nearest',
                          vmin = -1., vmax=1., cmap='RdYlBu_r',
                          origin='bottom', extent=[case.x[0], case.x[-1], case.y[0], case.y[-1]])
            # style
            scan_str = 'Canopy wave case #{:02d} - {:%Y %b %d - %H:%M:%S} UTC'.format(
                case.case_id, scan_data['time'][0])
            wave_str = 'wavelength={:.0f} m, propagation direction={:.0f} ({:.0f}) deg'.format(
                wavelength, propagation_direction, propagation_direction+180.)
            pyplot.figtext(0.5, 0.98, '{}\n{}'.format(scan_str, wave_str),
                           ha='center', va='top', size='large')
            for ax in axes:
                ax.set_xlim(xmin=case.x[0], xmax=case.x[-1])
                ax.set_ylim(ymin=case.y[0], ymax=case.y[-1])
                ax.set_xlabel('x (m)')
                ax.set_ylabel('y (m)')
            basename, _ = os.path.splitext(os.path.basename(scan_info['path']))
            output_file = os.path.join(output_dir, 'crests_{}.png'.format(basename))
            fig.savefig(output_file, format='png', dpi=dpi)
            print('saved {}'.format(output_file))
            pyplot.close(fig)

    def test_crest_motion(case):
        """
        Written by P. DERIAN 2018-02-13.
        """
        ### parameters
        max_timedelta = 60. #[s] max time difference between two scans
        min_points = 6 #[px] min number of points in a candidate crests
        crest_subsample = 2 #subsampling factor of crest points for motion estimation
        coeff_mad = 4. #tolerance for MAD test
        ### motion estimation
        hann_windows = {w: numpy.outer(signal.hann(w), signal.hann(w)) for w in [128, 64, 32, 16, 8]}
        def displacement(xy, im0, im1, window_sizes=[64, 32, 16, 8]):
            ydim, xdim = im0.shape
            x0, y0 = xy
            x1, y1 = xy
            u = 0.
            v = 0.
            for k, win in enumerate(window_sizes, 1):
                # patch coordinates
                half_win = win//2
                x0min = x0 - half_win
                x0max = x0min + win
                y0min = y0 - half_win
                y0max = y0min + win
                x1min = x1 - half_win
                x1max = x1min + win
                y1min = y1 - half_win
                y1max = y1min + win
                # stop if out-of-domain
                # Note: we only check (x1, y1). at first it's the same as (x0, y0),
                # and then it is the only one to be updated.
                if (x1min<0) or (y1min<0) or (x1max>=xdim) or (y1max>=ydim):
                    continue
                # extract patch
                p0 = im0[y0min:y0max, x0min:x0max]
                p1 = im1[y1min:y1max, x1min:x1max]
                # normalize
                p0 = (p0 - p0.mean())/p0.std()
                p1 = (p1 - p1.mean())/p1.std()
                # fill masked entries with zeros, apply hann window
                p0 = p0.filled(0.)*hann_windows[win]
                p1 = p1.filled(0.)*hann_windows[win]
                # estimate translation, upsample at last step
                shifts, _, _ = feature.register_translation(
                    p1, p0, upsample_factor=(10 if (k==len(window_sizes)) else 1))
                dv, du = shifts
                # update vectors
                u = du if (u==numpy.nan) else u+du
                v = dv if (v==numpy.nan) else v+dv
                # and center of 2nd patch
                x1 = int(numpy.floor(x0 + u))
                y1 = int(numpy.floor(y0 + v))
            return (u, v)
        ### output directory for crests
        output_dir = os.path.join('..', 'tmp', 'motions', 'case_{:02d}'.format(case.case_id))
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir, mode=0o755)
            print('Created output directory: {}'.format(output_dir))
        ### detect crest locations
        crest_results = case.crest_locations2()
        ### now for each scan
        hard_target_mask = case.hard_target_mask()
        for idx_scan in range(len(case.grid_scans) - 1):
            ### estimate motions
            dt_scan = (case.scan_data[idx_scan+1]['time'][0] -
                       case.scan_data[idx_scan]['time'][0]).total_seconds()
            # skip if either scan is not valid
            if (not(case.scan_valid[idx_scan] and case.scan_valid[idx_scan+1]) or
                dt_scan>max_timedelta):
                continue
            # masks
            domain_mask = numpy.logical_or(case.grid_masks[idx_scan],
                                           case.grid_masks[idx_scan+1])
            full_mask = numpy.logical_or(domain_mask, hard_target_mask)
            # images
            if 1:
                im0 = numpy.ma.array(case.grid_scans[idx_scan], mask=full_mask)
                im1 = numpy.ma.array(case.grid_scans[idx_scan+1], mask=full_mask)
                cmap = 'nipy_spectral'
                vmin, vmax = -1., 3.
                prefix = 'motion'
            else:
                im0 = numpy.ma.array(wave_fields[idx_scan], mask=full_mask)
                im1 = numpy.ma.array(wave_fields[idx_scan+1], mask=full_mask)
                cmap = 'RdYlBu_r'
                vmin, vmax = -1., 1.
                prefix = 'altmotion'
            # labels crest
            crest_properties = crest_results[idx_scan]['crest_properties']
            # crest image coordinates
            idx_crests = [crest['coords'][::crest_subsample] for crest in crest_properties]
            if len(idx_crests)>0:
                idx_crests = numpy.vstack(idx_crests)
                idx_crests_y = idx_crests[:,0]
                idx_crests_x = idx_crests[:,1]
                # crest world coordinates
                crests_x = case.grid_x[idx_crests_y, idx_crests_x]
                crests_y = case.grid_y[idx_crests_y, idx_crests_x]
                # motion
                uv = []
                for idx_xy in idx_crests[:,::-1]:
                    uv.append(displacement(idx_xy, im0, im1))
                uv = numpy.atleast_2d(uv)*(case.param_grid['resolution']/dt_scan)
                speed = numpy.sqrt(uv[:,0]**2 + uv[:,1]**2)
                direction = CanopyWaveCase.wind_direction(uv[:,0], uv[:,1])
                # outliers
                test_u = CanopyWaveCase.find_ouliers_MAD(uv[:,0], coeff=coeff_mad)
                test_v = CanopyWaveCase.find_ouliers_MAD(uv[:,1], coeff=coeff_mad)
                is_bad = numpy.logical_or(test_u, test_v)
                is_valid = numpy.logical_not(is_bad)
                # done
                has_estimates = True
            else:
                has_estimates = False
            ### Retrieve met tower measurements
            tower_height = CanopyWaveCase.WAVECASES_ANEMOMETER_ALTITUDE[case.case_id if
                (case.case_id in CanopyWaveCase.WAVECASES_ANEMOMETER_ALTITUDE) else 'default']
            tower_data = sqltools.get_chats_tower(case.scan_data[idx_scan]['time'][0],
                                                  case.scan_data[idx_scan+1]['time'][0],
                                                  tower_height, toDict=True)
            tower_data['mean_u'] = numpy.mean(tower_data['u'])
            tower_data['mean_v'] = numpy.mean(tower_data['v'])
            tower_data['mean_speed'] = numpy.sqrt(tower_data['mean_u']**2 + tower_data['mean_v']**2)
            tower_data['mean_direction'] = CanopyWaveCase.wind_direction(tower_data['mean_u'],
                                                                         tower_data['mean_v'])
            ### display
            dpi = 72
            fig, axes = pyplot.subplots(2,1,
                                        gridspec_kw={'left':.65, 'right':.97,
                                                     'bottom':.1,'top':.85,
                                                     'hspace':.4},
                                        figsize=(768./dpi, 480./dpi))
            # scan
            ax_scan = fig.add_axes([.1, .1, .5, .8])
            ax_scan.set_title('Gridded scan, crest displacement')
            p1a = ax_scan.imshow(im0,
                                 interpolation='nearest', vmin=vmin, vmax=vmax, cmap=cmap,
                                 origin='bottom', extent=[case.x[0], case.x[-1], case.y[0], case.y[-1]],
                                 zorder=1)
            # vectors
            if has_estimates:
                p1c = ax_scan.quiver(crests_x, crests_y, uv[:,0], uv[:,1],
                                     color=['k' if valid else 'r' for valid in is_valid],
                                     pivot='middle', units='xy', angles='xy', zorder=2)
            # style
            ax_scan.set_xlim(xmin=case.x[0], xmax=case.x[-1])
            ax_scan.set_ylim(ymin=case.y[0], ymax=case.y[-1])
            ax_scan.set_xlabel('x (m)')
            ax_scan.set_ylabel('y (m)')
            # histograms
            ax = axes[0]
            if has_estimates:
                ax.hist([speed[is_valid], speed[is_bad]], bins=numpy.linspace(0., 3., 31))
            ax.axvline(tower_data['mean_speed'], ls='--', color='k', lw=1.)
            ax.set_xlabel('wind speed (m/s)')
            ax.set_xlim(-0.1, 3.1)
            ax = axes[1]
            if has_estimates:
                ax.hist([direction[is_valid], direction[is_bad]], bins=numpy.linspace(0., 360., 37))
            ax.axvline(tower_data['mean_direction'], ls='--', color='k', lw=1.)
            ax.set_xlabel('wind direction (deg)')
            ax.set_xlim(-10., 370.)
            # labels
            scan_str = 'Canopy wave case #{:02d} - {:%Y %b %d - %H:%M:%S} UTC, dt={:.2f} s'.format(
                case.case_id, case.scan_data[idx_scan]['time'][0], dt_scan)
            tower_str = 'Estimations, mean met tower data @ {} m'.format(tower_height)
            pyplot.figtext(0.5, 0.98, '{}'.format(scan_str),
                           ha='center', va='top', size='large')
            pyplot.figtext(.81, 0.9, '{}'.format(tower_str),
                           ha='center', va='top', size='large')
            # and save
            basename, _ = os.path.splitext(os.path.basename(case.scan_info[idx_scan]['path']))
            output_file = os.path.join(output_dir, '{}_{}.png'.format(prefix, basename))
            fig.savefig(output_file, format='png', dpi=dpi)
            print('saved {}'.format(output_file))
            pyplot.close(fig)

    def test_classify_wave(case):
        """
        Written by P. DERIAN 2018-02-20.
        """
        print(case)
        ### output directory for crests
        output_dir = os.path.join('..', 'tmp', 'classify', 'case_{:02d}'.format(case.case_id))
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir, mode=0o755)
            print('Created output directory: {}'.format(output_dir))
        ### Detect wave parameters
        waves, mean_wave = case.wave_parameters()
        wavelengths_fixed, directions_fixed, _ = case.fix_wave_outliers(waves, mean_wave)
        ### scans
        resolution = case.param_grid['resolution']
        hard_target_mask = case.hard_target_mask()
        for (scan_info, scan_data, scan_valid, grid_scan, grid_mask,
             wavelength, propagation_direction) in zip(case.scan_info, case.scan_data,
                                                       case.scan_valid, case.grid_scans,
                                                       case.grid_masks, wavelengths_fixed,
                                                       directions_fixed):
            ### synthetic wave pattern
            n = numpy.int(numpy.ceil(wavelength/resolution))
            if not n%2:
                n += 1
            x = resolution*(numpy.arange(n) - float(n//2))
            x, y = numpy.meshgrid(x, x)
            theta = numpy.deg2rad(propagation_direction)
            wave_pattern = numpy.cos( (x*numpy.sin(theta) + y*numpy.cos(theta))*(2.*numpy.pi/wavelength) )
            sin_pattern = numpy.sin( (x*numpy.sin(theta) + y*numpy.cos(theta))*(2.*numpy.pi/wavelength) )
            wbreaking_pattern = wave_pattern.copy()
            wbreaking_pattern[sin_pattern>0.] =  -1.
            ebreaking_pattern = wave_pattern.copy()
            ebreaking_pattern[sin_pattern<0.] =  -1.
            ### scan
            # compute full mask, apply
            full_scan_mask = numpy.logical_or(grid_mask, hard_target_mask)
            full_scan_window = numpy.logical_not(full_scan_mask).astype(float)
            ### correlate (in spatial domain)
            correl_wave = ndimage.correlate(grid_scan*full_scan_window, wave_pattern,
                                            mode='constant', cval=0.)
            correl_wave *= 2./float(wave_pattern.size) #normalize
            correl_ebreaking = ndimage.correlate(grid_scan*full_scan_window, ebreaking_pattern,
                                                 mode='constant', cval=0.)
            correl_ebreaking *= 2./float(ebreaking_pattern.size) #normalize
            correl_wbreaking = ndimage.correlate(grid_scan*full_scan_window, wbreaking_pattern,
                                                 mode='constant', cval=0.)
            correl_wbreaking *= 2./float(wbreaking_pattern.size) #normalize
            ### plot
            dpi = 72
            fig, axes = pyplot.subplots(2,3,
                                        gridspec_kw={'left':0.09, 'right':0.97, 'wspace':0.25},
                                        figsize=(768./dpi, 480./dpi))
            for i, (pattern, correl, label) in enumerate(zip([wave_pattern, ebreaking_pattern,
                                                              wbreaking_pattern],
                                                             [correl_wave, correl_ebreaking,
                                                              correl_wbreaking],
                                                             ['Sine wave', 'Breaking E',
                                                              'Breaking W'])):

                tmp_correl = correl*full_scan_window
                # pattern
                ax = axes[0,i]
                ax.set_title('{}: {:.2f}'.format(label, numpy.sum(tmp_correl[tmp_correl>=0.1])))
                p2 = ax.imshow(pattern, interpolation='nearest',
                               vmin=-1., vmax=3., cmap='nipy_spectral',
                               origin='bottom', extent=[x[0,0], x[0,-1], y[0,0], y[-1,0]])
                # correlation
                ax = axes[1,i]
                p = ax.imshow(correl, interpolation='nearest',
                              vmin = -1., vmax=1., cmap='RdYlBu_r',
                              origin='bottom', extent=[case.x[0], case.x[-1], case.y[0], case.y[-1]])
                # style
                scan_str = 'Canopy wave case #{:02d} - {:%Y %b %d - %H:%M:%S} UTC'.format(
                    case.case_id, scan_data['time'][0])
                wave_str = 'wavelength={:.0f} m, propagation direction={:.0f} ({:.0f}) deg'.format(
                    wavelength, propagation_direction, propagation_direction+180.)
                pyplot.figtext(0.5, 0.98, '{}\n{}'.format(scan_str, wave_str),
                               ha='center', va='top', size='large')
            for ax in axes.ravel():
                #ax.set_xlim(xmin=case.x[0], xmax=case.x[-1])
                #ax.set_ylim(ymin=case.y[0], ymax=case.y[-1])
                ax.set_xlabel('x (m)')
                ax.set_ylabel('y (m)')
            basename, _ = os.path.splitext(os.path.basename(scan_info['path']))
            output_file = os.path.join(output_dir, 'classify_{}.png'.format(basename))
            fig.savefig(output_file, format='png', dpi=dpi)
            print('saved {}'.format(output_file))
            pyplot.close(fig)

    ###
    def main():
        ### Misc plots
        #plot_synthetic_wave_profiles()
        ### Main processing
        standard_cases = CanopyWaveCase.generate_standard_cases()
        special_cases = {121: CanopyWaveCase(121,
                                             datetime.datetime(2007, 04, 24, 13, 17, 25),
                                             datetime.datetime(2007, 04, 24, 13, 18, 40))
                         }
        cases = standard_cases
        cases.update(special_cases)
        set1 = [7, 11, 12, 121, 36, 37, 40]
        set2 = [2, 5, 6, 38, 51,121]
        for k in [36,]:#set1+set2:
            case = cases[k]
            case.load_scan_data()
            #export_products(case)
            #plot_case_scans(case)
            #plot_case_waves(case)
            #plot_case_fixedwaves(case)
            #plot_case_crests2(case)
            #make_movies(case)
            #test_hard_target_mask(case)
            #test_wave_parameters_spatial(case)
            #test_wave_outliers(case)
            #test_detect_crest(case)
            test_crest_motion(case)
            #test_classify_wave(case)
    main()
