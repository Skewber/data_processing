"""Class for performin data reduction. Optimized to use it with vlt or LFOA data"""

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from functools import lru_cache
from itertools import combinations
import logging
import os
import shutil
from pathlib import Path
from warnings import filterwarnings

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
from scipy.ndimage import median_filter, shift

import astroalign as aa
from astropy import units
from astropy.io import fits
from astropy.nddata import CCDData
from astropy.stats import mad_std
from astropy.wcs import WCS
from ccdproc import (ImageFileCollection, combine, subtract_bias,
                     subtract_dark, flat_correct, wcs_project,
                     cosmicray_lacosmic)

# FIXME: do not ignore all warnings
filterwarnings(action="ignore", module="ccdproc")
filterwarnings(action="ignore", module="astropy")

logger = logging.getLogger("data_reduction")
logger.propagate = False

# TODO: Check if the necessary entries in the header exist
class DataReduction():
    """Basic class for Data Reduction.
    For this class to work properly the used .fits files need the following keywords in the header:
    'EXPTIME', 'FILTER', 'OBJECT'.
    Optionally it can also contain wcs informations if an alignment of those is desired.
    If not provided the alignemt can also be done by matching stars in the image.
    If the status informations should be tracked a logger with the name 'data_reduction'
    should be defined.
    If this logger is not defined in advance, a basic console logger will be used.

    :param foldername_data: Name of the folder with the raw data that should be processed.
    :type foldername_data: str
    :param foldername_reduced: Name of the folder where the reduced data should be stored.
                               If it does not exist, this folder will be created. Defaults to "reduced_data"
    :type foldername_reduced: str, optional
    :param bias: Keyword in the headers for bias frames, defaults to ''
    :type bias: str, optional
    :param dark: Keyword in the headers for dark frames, defaults to ''
    :type dark: str, optional
    :param flat: Keyword in the headers for flat frames, defaults to ''
    :type flat: str, optional
    :param light: Keyword in the headers for light frames, defaults to ''
    :type light: str, optional
    """
    def __init__(self, foldername_data: str, foldername_reduced: str="reduced_data",
                 bias='', dark='', flat='', light='') -> None:
        """constructor method
        """
        # setup for a console handler if there are none
        if not logger.hasHandlers():
            logger.setLevel(logging.INFO)
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            formatter = logging.Formatter('%(levelname)s - %(message)s', style='%')
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        # set up for raw data path
        ## Path to the raw data
        self.raw_path: Path = Path('.', foldername_data)
        ## Collection of all images that are stored in the raw data path
        self.ifc_raw: ImageFileCollection = ImageFileCollection(self.raw_path)
        logger.info('Detected %(len(self.ifc_raw.files))s files in the raw data directory.')

        # set up for reduced data path
        ## Path to the reduced data, which will be created if it does not exist
        self.reduced_path: Path = Path('.', foldername_reduced)
        self.reduced_path.mkdir(exist_ok=True)
        ## Collection of all images that are allready in the reduced folder
        self.ifc_reduced: ImageFileCollection = ImageFileCollection(self.reduced_path)
        logger.info('Detected %(len(self.ifc_reduced.files))s files in the reduced data directory.')

        # check if all framenames are provided
        if bias=='' or dark=='' or flat=='' or light=='':
            ## Defines the keywords in the header of the used fits files
            self.imagetypes: dict[str, str] = self.__get_framenames(foldername_data)
        else:
            self.imagetypes: dict[str, str] = {'bias':bias, 'dark':dark, 'flat':flat, 'light':light}
            for hdu, fname in self.ifc_raw.hdus(return_fname=True):
                # FIXME: There might be a different unit. Which will be overwritten by this.
                # add 'bunit' to every header. Needed later on
                hdu.header['bunit'] = 'adu'
                hdu.writeto(Path('.', foldername_data, fname), overwrite=True)
            logger.debug("Added keyword 'BUNIT' with the value 'adu' to each header.")

        # set up dict for masters
        ## Dictionary to store the master frames for a more convenient access
        self.master_frames: dict = {'bias':None, 'dark':None, 'flat':None, 'light':None}
        logger.info("Finished setup.\n")

    def __get_framenames(self, foldername_data: str) -> dict[str, str]:
        """Determines the headerkeywords for bias, dark, flat and lightframes from the header
        of the given files.

        :param foldername_data: Name of the folder with the data
        :type foldername_data: str
        :raises ValueError: raises ValueError if for at least one frametyp no headerkeyword
        could be found
        :return: dictionary of headerkeywords for bias, dark, flat and lightframes
        :rtype: dict[str,str]
        """
        # TODO: determine imagetypes from image statistics rather than header keywords
        imagetypes: dict[str, str] = {'bias':'', 'dark':'', 'flat':'', 'light':''}

        keys = imagetypes.keys()
        frame_names = imagetypes.values()

        ## determine the strings that specify the imagetype
        for hdu, fname in self.ifc_raw.hdus(return_fname=True):
            # FIXME: There might be a different unit. Which will be overwritten by this.
            # add 'bunit' to every header. Needed later on
            hdu.header['bunit'] = 'adu'
            hdu.writeto(Path('.', foldername_data, fname), overwrite=True)

            # set names for imagetype
            img_type: str = hdu.header['imagetyp']
            if not img_type in frame_names:
                for key in keys:
                    imagetypes[key] = (img_type if key.lower() in img_type.lower()
                                        else imagetypes[key])
        logger.debug("Added keyword 'BUNIT' with the value 'adu' to each header.")
        logger.debug("Image types were detected from the headers.\n%(imagetypes)s")
        # raise an error if for at least one type ther is no name found
        if '' in frame_names:
            message = ("No name for the imagetype was detected for at least one imagetype." +
                    "The following types are determined:\n%(imagetypes)s.\nCheck your data again.")
            logger.error(message)
            raise ValueError(message)

        return imagetypes

    def __combine(self, to_combine: list[CCDData | str], frame: str='frame',
                  obj: str='', filt: str='', exposure: str='',
                  mem_lim: float=8e9, filenames=None) -> CCDData:
        """Combines the given images.

        :param to_combine: Data that will be stacked
        :type to_combine: list[CCDData  |  str]
        :param frame: Name of the frame that is stacked (bias, dark, flat, light),
        defaults to 'frame'
        :type frame: str, optional
        :param obj: Name of the object in the image, defaults to ''
        :type obj: str, optional
        :param filt: Name of the used filter, defaults to ''
        :type filt: str, optional
        :param exposure: Value of the exposure, defaults to ''
        :type exposure: str, optional
        :param mem_lim: Maximum of the used memory during the stacking process, defaults to 8e9
        :type mem_lim: float, optional
        :param filenames: The files that are stacked.
        Only relevant if 'to_combine' is a list of CCDData objects and not the actual filenames,
        defaults to None
        :type filenames: list[str], optional
        :return: Stacked image
        :rtype: CCDData
        """
        master: CCDData
        if frame != 'flat':
            master = combine(to_combine,
                            method='average',
                            sigma_clip=True, sigma_clip_low_thresh=5, sigma_clip_high_thresh=5,
                            sigma_clip_func=np.ma.median, sigma_clip_dev_func=mad_std,
                            mem_limit=mem_lim)
        else:
            # images are scaled by the inverse of the median
            master = combine(to_combine,
                            method='average', scale=lambda x: 1/np.meadian(x),
                            sigma_clip=True, sigma_clip_low_thresh=5, sigma_clip_high_thresh=5,
                            sigma_clip_func=np.ma.median, sigma_clip_dev_func=mad_std,
                            mem_limit=mem_lim)

        if isinstance(to_combine[0], CCDData):
            master.header = to_combine[0].header
        else:
            with fits.open(to_combine[0]) as hdul:
                master.header = hdul[0].header
        # adding an entry to the header
        master.meta['nframes'] = len(to_combine)
        master.meta['combined'] = True


        # generate the right filename depending on what arguments are given
        file_name: str = f"master_{frame}"
        if obj != '':
            file_name = f"{file_name}_{obj}"
        if filt != '':
            file_name = f"{file_name}_filter_{filt}"
        if exposure != '':
            file_name = f"{file_name}_{exposure}s"
        file_name = f"{file_name}.fits"

        # save the image
        master.write(self.reduced_path / file_name, overwrite=True)

        if isinstance(to_combine[0], str):
            # FIXME: test if this message is actually correctly printed
            logger.debug("Combined %s frames to one master %(frame)s frame.\n"+
                         "Master filename:\t%(file_name)s\nObject:\t%(obj)s\n"+
                         "Filter:\t%(filt)s\nExposure:\t%(exposure)s\n"+
                         "The used frames are:\n\t%s", len(to_combine, '\n\t'.join(to_combine)))
        elif filenames is not None:
            logger.debug("Combined %(nframes)s frames to one master %(frame)s frame.\n"+
                         "Master filename:\t%(file_name)s\nObject:\t%(obj)s\n"+
                         "Filter:\t%(filt)s\nExposure:\t%(exposure)s\n"+
                         "The used frames are:\n\t%(files)s")
        else:
            logger.debug("Combined %(nframes)s frames to one master %(frame)s frame.\n"+
                         "Master filename:\t%(file_name)s\nObject:\t%(obj)s\n"+
                         "Filter:\t%(filt)s\nExposure:\t%(exposure)s")
        return master

    def __rm_master(self, filelist: list[str]) -> list[str]:
        """Removes the master frames from the filelist

        :param filelist: list with the filenames
        :type filelist: list[str]
        :return: return list of filename without the master frames
        :rtype: list[str]
        """
        combined: list[str] = self.ifc_reduced.files_filtered(combined=True, include_path=True)
        filelist = [file for file in filelist if not file in combined]
        return filelist

    @staticmethod
    def align_astroalign(to_combine: list[str|CCDData]) -> list[CCDData]:
        """Aligns the given images using the astroalign library

        :param to_combine: list of filenames that should be aligned
        :type to_combine: list[str|CCDData]
        :return: list of all aligned images.
        :rtype: list[CCDData]
        """
        logger.debug("Started aligning images with 'astroalign'.")
        # the first image is the target
        # all others will be aligned to this one
        with fits.open(to_combine[0]) as file:
            target = file[0].data.astype(float)

        def align_img(img, target):
            with fits.open(img) as file:
                source = file[0].data.astype(float)
            registered_img = CCDData(aa.register(source=source, target=target,
                                                max_control_points=50, detection_sigma=5)[0],
                                                unit='adu')
            return registered_img

        with ProcessPoolExecutor() as executor:
            aligned: list[CCDData] = list(executor.map(lambda img: align_img(img, target),
                                                       to_combine))
        logger.debug("Finished aligning")
        return aligned

    @staticmethod
    def align_simple(to_combine: list[str]) -> list[CCDData]:
        """Aligns the given images using an input from the user to detect a star for alignment.
        Since it only uses one star no rotational alignment is possible.

        :param to_combine: list of filenames that should be aligned
        :type to_combine: list[str]
        :return: list of images that are aligne to the first
        :rtype: list[CCDData]
        """
        # define a helping function for the estimation of the first star center
        def estimate_px(data) -> tuple[int, int]:
            """Returns an estimate for the center posintion after a user input

            :param data: the array with the stars, has to be in a form to call plt.imshow(data)
            :type data: array-like
            :return: estimated x and y values
            :rtype: tuple[int, int]
            """

            marker = None   # marker to show the position of the click
            # a callback function that handles the click in the image
            def onclick(event):
                """
                Writes the current x and y positions to the coresponding variables
                """
                nonlocal x, y, marker
                if fig.canvas.toolbar.mode == '':
                    if event.inaxes is not None:
                        x = int(event.xdata)
                        y = int(event.ydata)
                        logger.info("Clicked on: %(x)s\t%(y)s")

                        # remove marker if it exists
                        if marker is not None:
                            marker.remove()

                        marker = ax.plot(x, y, 'ro', markersize=10, markerfacecolor='none',
                                         label="Selected Position")[0]
                        fig.canvas.draw()

            # show the data
            fig, ax = plt.subplots()
            ax.set_title("Click in the image to set an estimate for the first star center.\n"
                         "The last click before closing the window will be used.")
            ax.imshow(data, cmap='Greys', norm=LogNorm(vmin=0.0001))

            x, y = 0, 0
            # connect the event handling to allow the interaction with the mouse
            cid = fig.canvas.mpl_connect('button_press_event', onclick)
            plt.show()

            fig.canvas.mpl_disconnect(cid)
            return x, y

        # helping function to determine the limits for a cutout
        def limits(x, y, halve_size):
            x_min = x - halve_size
            x_max = x + halve_size
            y_min = y - halve_size
            y_max = y + halve_size
            return x_min, x_max, y_min, y_max

        # open the taget image
        with fits.open(to_combine[0]) as file:
            target = file[0].data

        # getting the first estimate of the center of the star
        x, y = estimate_px(target)

        # getting the cutout limits for the fit
        # # FIXME: determine the size of the cutout dynamically
        x_min, x_max, y_min, y_max = limits(x, y, 50)

        # only use the cutout to decrease the data for the fit
        fit_data = target[y_min:y_max, x_min:x_max]

        # find the position of the max
        max_idx = np.where(fit_data == np.max(fit_data))
        x = int(max_idx[1])
        y = int(max_idx[0])
        # determine the position in the whole image
        x0 = x + x_min
        y0 = y + y_min

        # generate a list of aligned images
        aligned = []
        x_prev = x0
        y_prev = y0

        for img in to_combine:
            # open the current file
            with fits.open(img) as file:
                source = file[0].data

            # make a cutout of the new image
            x_min, x_max, y_min, y_max = limits(x_prev, y_prev, 50)
            fit_data = source[y_min:y_max, x_min:x_max]
            # apply a median filter to eliminate outliers
            med_data = median_filter(fit_data, size=10)

            # find the position of the new max
            max_idx = np.where(med_data == np.max(med_data))

            x1 = int(max_idx[1][0]) + x_min
            y1 = int(max_idx[0][0]) + y_min

            # calculate the difference to the target image
            dx = x0 - x1
            dy = y0 - y1
            logger.debug("Shift: %(dx)s %(dy)s")

            # shift the data to match the target
            shifted_data = shift(source, [dy, dx])
            shifted_data = CCDData(data=shifted_data, unit='adu')
            aligned.append(shifted_data)
            logger.debug("%(img)s aligned successfully")

            # update the previouse value to center the next cutout
            x_prev = x1
            y_prev = y1

        return aligned

    @staticmethod
    def align_wcs(to_combine: list[CCDData]) -> list[CCDData]:
        """Aligns the given images using the wcs information from the header

        :param to_combine: list of filenames that should be aligned
        :type to_combine: list[CCDData]
        :return: list of aligned CCDData objects
        :rtype: list[CCDData]
        """
        # generating a target wcs to which all images are projected to
        with fits.open(to_combine[0]) as file:
            target_wcs: WCS = WCS(header=file[0].header)

        def align_img(img, target_wcs):
            with fits.open(img) as file:
                ccd = CCDData(data=file[0].data, wcs=WCS(header=file[0].header), unit='adu')
                new_image = wcs_project(ccd, target_wcs)
                logger.debug("%(img)s aligned successfully.")
                return new_image

        aligned: list[CCDData] = [align_img(img, target_wcs) for img in to_combine]

        return aligned

    def best_dark(self, target: float) -> CCDData:
        """Determines the best dark for a given exposure time

        :param target: exposure time for which the best darkframe should be found
        :type target: float
        :raises AttributeError: raises an AttributeError if no darkframes exist
        :return: the best darkframe
        :rtype: CCDData
        """
        if not self.master_frames['dark'] is None:
            dark_exposures: list[float] = sorted(self.master_frames['dark'].keys())
            best_idx: int = int(np.digitize(target, dark_exposures))
            best_exposure: float
            if best_idx == len(dark_exposures):
                best_exposure = dark_exposures[-1]
            else:
                best_exposure = dark_exposures[best_idx]
            best_dark: CCDData = self.master_frames['dark'][best_exposure]
            return best_dark
        else:
            raise AttributeError("Expected dark frames to exist."+
                                 "Ensure that there exist at least one dark frame "+
                                 "before running this function.")

    def check_master(self, frametype:str, master:str='') -> bool:
        """Checks if a specific master frame exists

        :param frametype: Specifies which frametype should be checked.
                          Valid are 'bias', 'dark', 'flat' and 'light'
        :type frametype: str
        :param master: name of a specific master frame, defaults to ''
        :type master: str, optional
        :raises ValueError: if 'frametype' does not have a valid value
        :return: True if the file exist, False otherwise
        :rtype: bool
        """
        # no master filename is given
        if master == '':
            if self.master_frames[frametype] is not None:
                return True
            else:
                masters: list[str] = self.ifc_reduced.files_filtered(imagetyp=self.imagetypes[frametype],
                                                                     combined=True)
                if len(masters) == 0:    # no masters found
                    return False
                else:
                    return True
        # a filename for a master is given
        else:
            if os.path.exists(self.reduced_path / master):
                return True
            else:
                message = ("The given filename for a master bias is not valid." +
                           "Make sure not to include the foldername and only the filename." +
                           "'{master}' was given")
                logger.error(message)
                raise ValueError(message)

    def reduce_bias(self, force_new_master:bool=False, keep_files:bool=False) -> CCDData:
        """Calibrates the bias frames and stacks them to a master bias.
        If there is already a masterbias in the folder for the reduced data this one will be used
        instead of creating a new one.

        :param force_new_master: True if a new master shall be created even if there is already one,
                                 defaults to False
        :type force_new_master: bool, optional
        :param keep_files: True if the individual file should be kept in the reduced folder,
                           False if the files should be removed after stacking.
                           The raw data will not be altered,
                           only the files in the reduced path, defaults to False
        :type keep_files: bool, optional
        :raises RuntimeError: if more than one master bias is found.
        :return: stacked master bias
        :rtype: CCDData
        """
        logger.info("Started reduction of bias frames")
        self.ifc_reduced.refresh()
        if not self.check_master("bias") or force_new_master:
            # There is no need for somehow calibrating them. Just copy to the calibrated path
            # select only bias frames
            biases: list[str] = self.ifc_raw.files_filtered(imagetyp=self.imagetypes['bias'],
                                                            include_path=True)

            # copy the biasframes to the folder with calibrated data
            with ThreadPoolExecutor() as executor:
                executor.map(lambda file: shutil.copy(file, self.reduced_path), biases)

            # stack them to a master bias
            self.ifc_reduced.refresh()
            reduced_biases: list[str] = self.ifc_reduced.files_filtered(imagetyp=self.imagetypes['bias'],
                                                                        include_path=True)

            if force_new_master:
                # remove existing master from the imagelist, since those should not be stacked
                reduced_biases = self.__rm_master(reduced_biases)

            self.master_frames['bias'] = self.__combine(reduced_biases, frame='bias')

            # remove all single calibrate bias frames
            if not keep_files:
                with ThreadPoolExecutor() as executor:
                    executor.map(lambda file: os.remove(Path('.', file)), reduced_biases)

        # in all other cases there is already a master bias that can be used
        else:
            logger.info("One or more master bias were detected.")
            master_bias: list[str] = self.ifc_reduced.files_filtered(imagetyp=self.imagetypes['bias'],
                                                                     combined=True)
            if len(master_bias) > 1:
                n = '\n'
                logger.error("More than one master bias are detected." +
                             "Make sure that at most one frame is available." +
                              "The found files are %s", (n.join(master_bias)))
                raise RuntimeError("More than one master bias are detected." +
                                   "Make sure that at most one frame is available." +
                                   "The found files are %(master_bias)s")
            else:
                with fits.open(self.reduced_path / master_bias[0]) as file:
                    self.master_frames['bias'] = CCDData(file[0].data, unit='adu')
                logger.debug("One master bias was detected and will be used.\n" +
                             "Filename: %s", Path('.', f'{self.reduced_path}/{master_bias[0]}'))

        logger.info("Finished reduction of bias frames.\n")
        return self.master_frames['bias']

    # TODO: implement given master_bias filename
    def reduce_darks(self, force_new_master:bool=False, keep_files:bool=False,
                     master_bias:str|CCDData=None) -> dict[float, CCDData]:
        """Calibrates the dark frames and stacks them to a master dark.

        :param force_new_master: True if a new master should be created even if there is already one,
                                defaults to False
        :type force_new_master: bool, optional
        :param keep_files: True if the individual file should be kept in the reduced folder,
                            False if the files should be removed after stacking.
                            The raw data will not be altered, only the files in the reduced path,
                            defaults to False
        :type keep_files: bool, optional
        :param master_bias: filename or CCDData object of the frame that should be used as bias,
                            defaults to None
        :type master_bias: str | CCDData, optional
        :raises RuntimeError: if more than one master dark for the same exposure time is found
        :return: dict with the eexposuretime as key and the stacked CCDData object as the value
        :rtype: dict[float, CCDData]
        """
        logger.info("Started reduction of dark frames.")
        self.ifc_reduced.refresh()
        # 1) check for existing
        if not self.check_master("dark") or force_new_master:
            # ensure that a master bias exists
            if self.master_frames['bias'] is None:
                self.reduce_bias()

            dark_times: set[float] = set(h['exptime'] for h in
                                         self.ifc_raw.headers(imagetyp=self.imagetypes['dark']))

            # 2) reduce the dark frames
            def correct(ccd, fname):
                ccd = subtract_bias(ccd, self.master_frames['bias'])
                ccd.write(self.reduced_path / fname, overwrite=True)
            with ThreadPoolExecutor() as executor:
                executor.map(lambda args: correct(*args),
                             self.ifc_raw.ccds(imagetyp=self.imagetypes['dark'], return_fname=True))

            # 3) stack the frames and use dict for different exp times
            self.ifc_reduced.refresh()
            for exp_time in dark_times:
                reduced_darks: list[str] = self.ifc_reduced.files_filtered(imagetyp=self.imagetypes['dark'],
                                                                           exptime=exp_time, include_path=True)

                if force_new_master:
                # remove existing master from the imagelist, since those should not be stacked
                    reduced_darks = self.__rm_master(reduced_darks)

                self.__combine(reduced_darks, frame='dark', exposure=str(int(exp_time)))

            # 4) clean up
            # remove all single calibrate bias frames
            reduced_darks: list[str] = self.ifc_reduced.files_filtered(imagetyp=self.imagetypes['dark'],
                                                                       include_path=True)
            reduced_darks = self.__rm_master(reduced_darks)

            if not keep_files:
                with ThreadPoolExecutor() as executor:
                    executor.map(lambda file: os.remove(Path('.', file)), reduced_darks)
                logger.info("Removed all reduced dark frame files.")

        # in all other cases there is already a master dark that can be used
        else:
            logger.info("One or more master darks were detected.")
            master_darks = self.ifc_reduced.files_filtered(imagetyp=self.imagetypes['dark'],
                                                           combined=True, include_path=True)

            exptimes: list[float] = []
            for dark in master_darks:
                # iterate over the files and check for duplicates for the same exposure time
                with fits.open(dark) as file:
                    exp_time = int(file[0].header['exptime'])
                    if exp_time in exptimes:
                        logger.error("More than one master dark for the same exposure time are detected. "+
                                     "Make sure that at most one file for each exposure time is available. "+
                                     "The found files are %s", '\n'.join(master_darks))
                        raise RuntimeError("More than one master dark for the same exposure time are detected. "+
                                           "Make sure that at most one file for each exposure time is available. "+
                                           f"The found files are {master_darks}")
                    else:
                        self.master_frames['dark'] = {exp_time:CCDData(file[0].data, unit='adu')}
                        logger.debug("One master dark with %(exp_time)s s exposure was detected and will be used."+
                                     "\nFilename: %(dark)s")
                        exptimes = list(self.master_frames['dark'].keys())

        # create a dictionary for better access
        self.ifc_reduced.refresh()
        self.master_frames['dark'] = {ccd.header['exptime']: ccd for ccd in
                                      self.ifc_reduced.ccds(imagetyp=self.imagetypes['dark'], combined=True)}

        logger.info("Finished reduction of dark frames.\n")
        return self.master_frames['dark']

    # TODO: implement given master_bias filename
    # TODO: implement given master_dark filename
    def reduce_flats(self, force_new_master:bool=False, keep_files:bool=False,
                     master_bias:str|CCDData=None, master_dark:str|CCDData=None) -> dict[str, CCDData]:
        """ Calibrates the flat frames and stacks them to a master flat.

        :param force_new_master: True if a new master should be created even if there is already one,
                                 defaults to False
        :type force_new_master: bool, optional
        :param keep_files: True if the individual file should be kept in the reduced folder,
                           False if the files should be removed after stacking.
                           The raw data will not be altered, only the files in the reduced path,
                           defaults to False
        :type keep_files: bool, optional
        :param master_bias: filename or CCDData object of the frame that should be used for bias subtraction,
                            defaults to None
        :type master_bias: str | CCDData, optional
        :param master_dark: filename or CCDData object of the frame that should be used for dark subtraction,
                            defaults to None
        :type master_dark: str | CCDData, optional
        :raises RuntimeError: if more than one master flat for the same filter is found
        :return: dict of the form {filtername:flat data}.
        :rtype: dict[str, CCDData]
        """
        logger.info("Started reduction of flat frames.")
        # 1) check for existing
        if not self.check_master('flat') or force_new_master:
            # check necessary files
            if self.master_frames['bias'] is None:
                self.reduce_bias(force_new_master, keep_files)
            if self.master_frames['dark'] is None:
                self.reduce_darks(force_new_master, keep_files)

        # 2) calibrate the flat frames
                # ccd.write(self.reduced_path / fname, overwrite=True)
            def correct(hdu, fname):
                ccd = CCDData(hdu.data, meta=hdu.header, unit='adu')
                ccd = subtract_bias(ccd, self.master_frames['bias'])
                ccd = subtract_dark(ccd, self.best_dark(hdu.header['exptime']),
                                    exposure_time='exptime', exposure_unit=units.second, scale=True)
                ccd.write(self.reduced_path / fname, overwrite=True)
            with ThreadPoolExecutor() as executor:
                executor.map(lambda args: correct(*args), self.ifc_raw.hdus(imagetyp=self.imagetypes['flat'],
                                                                            return_fname=True))

        # 3) stack them
            self.ifc_reduced.refresh()

            # creating a set of all filters to distinguish the flats
            flat_filters: set[str] = set(h['filter'] for h in
                                         self.ifc_reduced.headers(imagetyp=self.imagetypes['flat']))

            for filt in flat_filters:
                reduced_flats: list[str] = self.ifc_reduced.files_filtered(imagetyp=self.imagetypes['flat'],
                                                                           filter=filt, include_path=True)

                # remove existing master from the imagelist, since those should not be stacked
                if force_new_master:
                    reduced_flats = self.__rm_master(reduced_flats)
                self.__combine(reduced_flats, frame='flat', filt=filt)

        # 4) clean up
            # remove not stacked files if not needed
            reduced_flats: list[str] = self.ifc_reduced.files_filtered(imagetyp=self.imagetypes['flat'],
                                                                       include_path=True)
            reduced_flats = self.__rm_master(reduced_flats)
            if not keep_files:
                with ThreadPoolExecutor() as executor:
                    executor.map(os.remove, reduced_flats)
                logger.info("Removed all reduced flat frame files.")

        # in all other cases there are already master flats
        else:
            logger.info("One or more master flats were detected.")
            master_flats = self.ifc_reduced.files_filtered(imagetyp=self.imagetypes['flat'],
                                                           combined=True, include_path=True)
            filters: list[str] = []
            for flat in master_flats:
                # iterate over filters and check for duplicates
                with fits.open(flat) as file:
                    used_filter = file[0].header['filter']
                    if used_filter in filters:
                        logger.error("More than one master flat for the same filter are detected. "+
                                     "Make shure that at most one file for each filter is available. "+
                                     "The found files are %s", '\n'.join(master_flats))
                        raise RuntimeError("More than one master flat for the same filter are detected. "+
                                           "Make shure that at most one file for each filter is available. "+
                                           "The found files are %(master_flats)s")
                    else:
                        self.master_frames['flat'] = {used_filter:CCDData(file[0].data, unit='adu')}
                        logger.debug("One master flat for filter %(used_filter)s was detected and will be used."+
                                     "\nFilename: %(flat)s")

        # create dictionary for better access
        self.ifc_reduced.refresh()
        self.master_frames['flat'] = {ccd.header['filter']: ccd for ccd in
                                      self.ifc_reduced.ccds(imagetyp=self.imagetypes['flat'], combined=True)}

        logger.info("Finished reduction of flat frames.\n")
        return self.master_frames['flat']

    # TODO: implement given master_bias filename
    # TODO: implement given master_dark filename
    # TODO: implement given master_flat filename
    def reduce_lights(self, correct_cosmics=False, master_bias:str|CCDData=None,
                      master_dark:str|CCDData=None, master_flat:str|CCDData=None) -> None:
        """Corrects the light frames by subtracting bias and dark frames and dividing by the flat.
        This does NOT create a master frame.

        :param correct_cosmics: whether to correct cosmic rays, defaults to True
        :type correct_cosmics: bool, optional
        :param master_bias: filename or CCDData object of the frame that should be used for bias subtraction,
                            defaults to None
        :type master_bias: str | CCDData, optional
        :param master_dark: filename or CCDData object of the frame that should be used for dark subtraction,
                            defaults to None
        :type master_dark: str | CCDData, optional
        :param master_flat: filename or CCDData object of the frame that should be used for flat division,
                            defaults to None
        :type master_flat: str | CCDData, optional
        """
        logger.info("Started reduction of light frames.")
        # ensure all necessary files are available
        if self.master_frames['bias'] is None:
            self.reduce_bias()
        if self.master_frames['dark'] is None:
            self.reduce_darks()
        if self.master_frames['flat'] is None:
            self.reduce_flats()

        # correction of light frames
        def correct(light, fname, corr_cos):
            if corr_cos:
                self.cosmic_correct(data=light, fname=fname)
            light = subtract_bias(light, self.master_frames['bias'])
            light = subtract_dark(light, self.best_dark(light.header['exptime']),
                                  exposure_time='exptime', exposure_unit=units.second, scale=True)
            light = flat_correct(light, self.master_frames['flat'][light.header['filter']])
            light.write(self.reduced_path / f"reduced_{fname}", overwrite=True)
        with ThreadPoolExecutor() as executor:
            executor.map(lambda args: correct(*args, correct_cosmics),
                         self.ifc_raw.ccds(imagetyp=self.imagetypes['light'], return_fname=True))

        logger.info("Finished reduction of light frames.\n")

    @classmethod
    def align_images(cls, filenames: list[str], alignment: str) -> list[CCDData]:
        """Aligns the given images in preparation for later stacking.

        :param filenames: names of the files that should be aligned
        :type filenames: list[str]
        :param alignment: type of alignment to use. Valid are 'none', 'astroalign', 'singlestar' and 'wcs'.
                          'none' does not align the images, 'astroalign' uses a dedicated library to align the images,
                          'singlestar' shows the first images and the user has to select the position of one star
                          that should be used for alignment,
                          'wcs' uses the WCS information from the header to align the images.
        :type alignment: str
        :raises ValueError: if the provided alignment method is invalid
        :return: aligned images as CCDData objects
        :rtype: list[CCDData]
        """
        match alignment:
            case 'none':
                aligned = filenames
                logger.info("No alignment will be used for combination.")

            case 'star1':
                logger.info("The frames will be aligned by the 'astroalign' library.")
                aligned = cls.align_astroalign(filenames)

            case 'star2':
                logger.info("The frames will be aligned with one reference star selected by the user.")
                aligned = cls.align_simple(filenames)

            case 'wcs':
                aligned = cls.align_wcs(filenames)
                logger.info("The frames will be aligned according to their WCS information from the headers.")

            case _:
                logger.error("Invalid alignemntmethod was given. "+
                             "Valid are 'none', 'star1', 'star2', and 'wcs' but '%(alignment)s' was given.")
                raise ValueError("Invalid alignement method! "+
                                 "Valid values are 'none', 'star1', 'star2' and 'wcs' but '%(alignment)s' was given.")

        return aligned

    def stack_lights(self, alignment: str='none', filelists: list[list[str]]=None, out_names: list[str]=None) -> None:
        """Stacks the light frames and aligns them if desired.

        :param alignment: alignment method to use, see DataReduction.align_images() for more detail, defaults to 'none'
        :type alignment: str, optional
        :param filelists: list that contains one list with filenames for every object that should be stacked seperatly,
                          defaults to None
        :type filelists: list[list[str]], optional
        :param out_names: list of the names for the individual images.
                          The resulting filename will be 'master_<outname>.fits,
                          defaults to None
        :type out_names: list[str], optional
        :raises ValueError: if the number of items in 'filelist' and the number of items in 'out_names' does not match.
        """
        logger.info("Started stacking of light frames.")
        self.ifc_reduced.refresh()

        if filelists is None:
            # create a set of the filters used for the light frames
            used_filters = set(h['filter'] for h in self.ifc_reduced.headers(imagetyp=self.imagetypes['light']))

            # create a set of the observed objects
            observed_objects = set(h['object'] for h in self.ifc_reduced.headers(imagetyp=self.imagetypes['light']))

            for obj in observed_objects:
                for filt in used_filters:
                    filenames: list[str] = self.ifc_reduced.files_filtered(imagetyp=self.imagetypes['light'],
                                                                           filter=filt, object=obj, include_path=True)

                    # for some combinations there might not be any files
                    if len(filenames) == 0:
                        continue

                    # aligning the images
                    to_combine = self.align_images(filenames, alignment)

                    self.__combine(to_combine, 'light', obj, filt, filenames=filenames)

        else:
            # check length of input
            if len(filelists) == 1 and isinstance(out_names, str):
                to_combine = self.align_images(filelists[0], alignment)
                self.__combine(to_combine, 'light', out_names, filenames=filelists[0])
            elif len(filelists) == len(out_names):
                for files, name in zip(filelists, out_names):
                    to_combine = self.align_images(files, alignment)
                    self.__combine(to_combine, 'light', name, filenames=files)
            else:
                logger.error("The number of input filelists and output filenames does not match. "+
                             "You provided %s filelists and %s output filenames.", len(filelists), len(out_names))
                raise ValueError("The provided filnames does not match the provided number of output filenames. "+
                                 f"You provided {len(filelists)} filelists and {len(out_names)}.")
        logger.info("Finished stacking light frames.\n")

    @property
    @lru_cache
    def readnoise(self):
        """calculates an estimate of the readoise based on the standard deviation of the bias frames.

        :return: estimate for the readnoise
        :rtype: float
        """
        return np.mean([np.std(bias) for bias in self.ifc_raw.data(imagetyp=self.imagetypes['bias'])])

    @property
    @lru_cache
    def gain_estimate(self):
        """calculates an estimate for the gain using the flat frames.
        This is very inaccurate and should not be used if there is another way to get this value.

        :return: calculated estimate for the gain
        :rtype: float
        """
        logger.info("Started gain estimation.")
        if self.master_frames['bias'] is None:
            self.reduce_bias()

        gains = []
        for flat1, flat2 in combinations(self.ifc_raw.data(imagetyp=self.imagetypes['flat']), 2):
            diff = flat1 - flat2
            xcenter, ycenter = [x//4 for x in flat1.shape]
            # std from central 100x100 pixels
            variance = np.std(diff[xcenter:3*xcenter, ycenter:3*ycenter])**2 / 2
            # subtract bias from both frames
            corrected1 = flat1 - self.master_frames['bias']
            corrected2 = flat2 - self.master_frames['bias']
            # mean from central 100x100 pixels
            mean1 = np.mean(corrected1[xcenter:3*xcenter, ycenter:3*ycenter])
            mean2 = np.mean(corrected2[xcenter:3*xcenter, ycenter:3*ycenter])
            # i actually don't know exactly why this is the gain now \_(°_°)_/
            # https://www.photometrics.com/learn/imaging-topics/gain
            gains.append(mean1/variance)
            gains.append(mean2/variance)
        gain = np.mean(gains)
        logger.debug("Estimated the gain of the data to: %(gain)s e/ADU")
        return gain

    def cosmic_correct(self, data=None, gain=None, niter=4, readnoise=None, save_cosmics=False, fname='.fits'):
        """Removes cosmics from the image data.

        :param data: data to remove cosmics on.
                     If 'None' every registerd light will be corrected and saved as a new file,
                     defaults to None
        :type data: CCDData, optional
        :param gain: value of the gain of the data. If None than the gain will be read from the header.
                     If there is no entry for 'EGAIN' the gain will be estimated using the registered flat frames.
        :type gain: float, optional
        :param niter: number of iterations to perform the L.A. cosmic algorithm, defaults to 4
        :type niter: int, optional
        :param readnoise: Value for the readnoise, if not given it will be calculated from the registered bias frames,
                          defaults to None
        :type readnoise: float, optional
        :param save_cosmics: Whether or not the detected cosmics sould be saved as an additional image,
                             defaults to False
        :type save_cosmics: bool, optional
        :param fname: name of the image file to save the cosmic corrected image, defaults to '.fits'
        :type fname: str, optional
        :raises ValueError: if 'data' is neither None nor a CCDData object.
        :return: None if 'data' is None, otherwise the data without the cosmics
        :rtype: CCDData
        """
        if readnoise is None:
            readnoise = self.readnoise

        if data is None:
            logger.info("Started cosmic correction")
            def correct(hdu, fname, gain, save_cosmics):
                if gain is None:
                    try:
                        gain = hdu.header['egain']
                        # not sure if this statement is necessarily true
                        gain = gain if gain >= 1 else 1/gain
                    except KeyError:
                        gain = self.gain_estimate
                else:
                    gain = 1.
                # determine the saturation level
                satlevel = 2**(np.abs(hdu.header['bitpix']/2)) - 1

                hdu.data, cosmics = cosmicray_lacosmic(hdu.data, sigclip=4.5, sigfrac=0.3, objlim=5.0,
                                                        gain=gain, readnoise=readnoise, satlevel=satlevel,
                                                        niter=niter, cleantype='meanmask', fsmode='median',
                                                        gain_apply=False)
                hdu.header['cosmics'] = 'corrected'
                if save_cosmics:
                    fits.HDUList([fits.PrimaryHDU(data=cosmics*1)]).writeto(Path('.', self.reduced_path,
                                                                                 f"cosmics_{fname}"), overwrite=True)
                hdu.writeto(Path('.', self.reduced_path, f"coscorr_{fname}"), overwrite=True)
            with ThreadPoolExecutor() as executer:
                executer.map(lambda args: correct(*args, gain, save_cosmics),
                             self.ifc_raw.hdus(imagetyp=self.imagetypes['light'],
                                               return_fname=True))
            logger.info("Finished cosmic correction")

        elif isinstance(data, CCDData):
            try:
                gain = data.header['egain']
                gain = gain if gain >= 1 else 1/gain
            except KeyError:
                gain = self.gain_estimate

            satlevel = 2**(np.abs(data.header['bitpix']/2)) - 1

            data.data, cosmics = cosmicray_lacosmic(data.data, sigclip=4.5, sigfrac=0.3, objlim=5.0,
                                                    gain=gain, readnoise=readnoise, satlevel=satlevel,
                                                    niter=niter, cleantype='meanmask', fsmode='median',
                                                    gain_apply=False)

            data.header['cosmics'] = 'corrected'
            if save_cosmics:
                fits.HDUList([fits.PrimaryHDU(data=cosmics*1)]).writeto(Path('.', self.reduced_path,
                                                                             f"cosmics_{fname}"), overwrite=True)

        else:
            message = ("You provided an invalid type for 'data' to perform cosmic correction on. "+
                      f"You provided '{type(data)}'.")
            logger.error(message)
            raise ValueError(message)

        return data
