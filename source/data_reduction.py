from pathlib import Path
import shutil
import numpy as np
from ccdproc import ImageFileCollection, combine, subtract_bias, subtract_dark
from astropy.io import fits
from astropy.nddata import CCDData
from astropy.stats import mad_std
from astropy import units
import os

class DataReduction():
    """! Basic class for Data Reduction. For this class to work properly the used .fits files need a the following keywords in the header: 'EXPTIME', 'FILTER', 'OBJECT'. Optionally it can also contain wcs informations if an alignment of those is desired. If not provided the alignemt can also be done by matching stars in the image.
    """
    def __init__(self, foldername_data: str, foldername_reduced: str="reduced_data") -> None:
        """! Set up of general information

        @param foldername_data (str): name of the folder with the raw data to
        @param foldername_reduced (str): (optional) name of the folder the reduced data should be stored. If it does not exist, this folder will be created. Defaults to "reduced_data".
        """
        # set up for raw data path
        ## Path to the raw data
        self.raw_path: Path = Path('.', foldername_data)
        ## Collection of all images that are stored in the raw data path
        self.ifc_raw: ImageFileCollection = ImageFileCollection(self.raw_path)

        # set up for reduced data path
        ## Path to the reduced data, which will be created if it does not exist
        self.reduced_path: Path = Path('.', foldername_reduced)
        self.reduced_path.mkdir(exist_ok=True)
        ## Collection of all images that are allready in the reduced folder
        self.ifc_reduced: ImageFileCollection = ImageFileCollection(self.reduced_path)

        ## Defines the keywords in the header of the used fits files
        self.imagetypes: dict[str, str] = {'bias':'', 'dark':'', 'flat':'', 'light':''}
        
        keys = self.imagetypes.keys()
        frame_names = self.imagetypes.values()

        ## determine the strings that specify the imagetype
        for hdu, fname in self.ifc_raw.hdus(return_fname=True):
            # add 'bunit' to every header. Needed later on
            hdu.header['bunit'] = 'adu'
            hdu.writeto(Path('.', foldername_data, fname), overwrite=True)

            # set names for imagetype
            img_type: str = hdu.header['imagetyp']
            # FIXME: try to remove nesting
            if not img_type in frame_names:
                for key in keys:
                    if key.lower() in img_type.lower():
                        self.imagetypes[key] = img_type
        
        # raise an error if for at least one type ther is no name found
        if '' in frame_names:
            raise ValueError(f"No name for the imagetype was found for at least one imagetype. The following types are determined:\n{self.imagetypes}.\nCheck your data again.")

        # set up dict for masters
        ## Dictionary to store the master frames for a more convenient access
        self.master_frames: dict = {'bias':None, 'dark':None, 'flat':None, 'light':None}
                
    def __combine(self, to_combine: list[CCDData | str], frame: str='frame', obj: str='', filt: str='', exposure: str='', mem_lim: float=8e9) -> CCDData:
        """!
        Combines the given images

        @param to_combine (list) : Names or CCDData objects of the files that should be stacked
        @param frame (string) : Name of the frame (bias, dark, flat, light)
        @param obj (string) : Name of the object
        @param filt (string) : Name of the filter used
        """

        master: CCDData
        if frame != 'flat':
            master = combine(to_combine,
                            method='average',
                            sigma_clip=True, sigma_clip_low_thresh=5, sigma_clip_high_thresh=5,
                            sigma_clip_func=np.ma.median, sigma_clip_dev_func=mad_std,
                            mem_limit=mem_lim)
        else:
            # defining function that returns the invers of the median
            # needed to scale the flat frames to ~1 so the division does not change the values to much
            inv_median = lambda x: 1/np.median(x)

            master = combine(to_combine,
                            method='average', scale=inv_median,
                            sigma_clip=True, sigma_clip_low_thresh=5, sigma_clip_high_thresh=5,
                            sigma_clip_func=np.ma.median, sigma_clip_dev_func=mad_std,
                            mem_limit=mem_lim)
        
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

        return master

    def __rm_master(self, filelist: list[str]) -> list[str]:
        """!Removes the master frame from the filelist

        @param filelist (list) : list of filenames
        
        @return filelist (list) : list of filenames without the master frames
        """
        combined: list[str] = self.ifc_reduced.files_filtered(combined=True, include_path=True)
        filelist = [file for file in filelist if not file in combined]
        return filelist
    
    def best_dark(self, target: float) -> CCDData:
        """!Finds the best dark

        @param target (float) : exposure time for which the best darkframe should be found
        
        @return bast_dark (CCDData) : A CCDData object of the best dark
        """
        if not self.master_frames['dark'] == None:
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
            raise AttributeError("Expected dark frames to exist. Ensure that there exist at least one dark frame before running this function.")
            
    def check_master(self, frametype: str, master: str='') -> bool:
        """!Checks if  a specific master frame exist
        
        @param frametype (str) : specify which frametype should be checked. Valid are 'bias', 'dark', 'flat', 'light'
        @param master (str) : (optional) the filename of a masterframe. Check if the file exist
            
        @return (bool) : True if the file is available, False if not"""
        # no master filename is given
        if master == '':
            if self.master_frames[frametype] != None:
                return True
            else:
                masters: list[str] = self.ifc_reduced.files_filtered(imagetyp=self.imagetypes[frametype], combined=True)
                if len(masters) == 0:    # no masters found
                    return False
                else:
                    return True
        # a filename for a master is given
        else:
            if os.path.exists(self.reduced_path / master):
                return True
            else:
                raise ValueError("The given filename for a master bias is not valid. Make sure not to include the foldername and only the filename. '{}' was given".format(master))

    def reduce_bias(self, force_new_master: bool=False, keep_files: bool=False) -> CCDData:
        """!
        Calibrates the bias frames and stacks them to a master bias. If there is already a masterbias in the folder for reduced data. This one will be used instead of computing it new

        @param force_new_master (bool) : (optional) If True, an existing master will be ignored and overwritten. Otherwise the existing will be used.
        @param keep_files (bool) : (optional) If True, all calibrated files will be kept. Default is 'False'
        
        @return master_bias (CCDData) : master bias as a CCDData object
        """
        self.ifc_reduced.refresh()
        if not self.check_master("bias") or force_new_master:
            # There is no need for somehow calibrating them. Just copy to the calibrated path
            # select only bias frames
            biases: list[str] = self.ifc_raw.files_filtered(imagetyp=self.imagetypes['bias'], include_path=True)

            # copy the biasframes to the folder with calibrated data
            for bias in biases:
                shutil.copy(bias, self.reduced_path)

            # stack them to a master bias
            self.ifc_reduced.refresh()
            reduced_biases: list[str] = self.ifc_reduced.files_filtered(imagetyp=self.imagetypes['bias'], include_path=True)

            if force_new_master:
                # remove existing master from the imagelist, since those should not be stacked
                reduced_biases = self.__rm_master(reduced_biases)

            self.master_frames['bias'] = self.__combine(reduced_biases, frame='bias')

            # remove all single calibrate bias frames
            if not keep_files:
                for file in reduced_biases:
                    os.remove(Path('.', file))

        # in all other cases there is already a master bias that can be used
        else:
            print("There is already a master bias.")
            master_bias: list[str] = self.ifc_reduced.files_filtered(imagetyp=self.imagetypes['bias'], combined=True)
            if len(master_bias) > 1:
                raise RuntimeError(f"More than one master bias are detected. Make shure that at most one frame is available. The found files are {master_bias}")
            else:
                with fits.open(self.reduced_path / master_bias[0]) as file:
                    self.master_frames['bias'] = CCDData(file[0].data, unit='adu')

        return self.master_frames['bias']
    
    def reduce_darks(self, force_new_master: bool=False, keep_files: bool=False) -> dict[float, CCDData]:
        """!Calibrates the dark frames and stacks them to a master dark
        
        @param force_new_master (bool) : (optional) If True, an existing master will be ignored and overwritten. Otherwise the existing will be used.
        @param keep_files (bool) : (optional) If True, all calibrated files will be kept. Default is 'False'
        @param master_bias (str) : (optional) filename of a masterbias that should be used. Defaulte is 'None'

        @return (dict) : dictionary with master darks. key=exposure time, value CCDData object
        """
        self.ifc_reduced.refresh()
        # 1) check for existing
        if not self.check_master("dark") or force_new_master:
            # ensure that a master bias exists
            if self.master_frames['bias'] == None:
                self.reduce_bias()

            dark_times: set[float] = set(h['exptime'] for h in self.ifc_raw.headers(imagetyp=self.imagetypes['dark']))

            # 2) reduce the dark frames
            for ccd, fname in self.ifc_raw.ccds(imagetyp=self.imagetypes['dark'], return_fname=True):
                ccd = subtract_bias(ccd, self.master_frames['bias'])
                ccd.write(self.reduced_path / fname, overwrite=True)

            # 3) stack the frames and use dict for different exp times
            self.ifc_reduced.refresh()
            for exp_time in dark_times:
                reduced_darks: list[str] = self.ifc_reduced.files_filtered(imagetyp=self.imagetypes['dark'], exptime=exp_time, include_path=True)

                if force_new_master:
                # remove existing master from the imagelist, since those should not be stacked
                    reduced_darks = self.__rm_master(reduced_darks)

                self.__combine(reduced_darks, frame='dark', exposure=str(int(exp_time)))
            
            # 4) clean up
            # remove all single calibrate bias frames
            reduced_darks: list[str] = self.ifc_reduced.files_filtered(imagetyp=self.imagetypes['dark'], include_path=True)
            reduced_darks = self.__rm_master(reduced_darks)

            if not keep_files:
                [os.remove(Path('.', file)) for file in reduced_darks]

        # in all other cases there is already a master dark that can be used
        else:
            print("There is already a master dark.")
            master_darks = self.ifc_reduced.files_filtered(imagetyp=self.imagetypes['dark'], combined=True, include_path=True)

            exptimes: list[float] = []
            for dark in master_darks:
                # iterate over the files and check for duplicates for the same exposure time
                with fits.open(dark) as file:
                    exp_time = int(file[0].header['exptime'])
                    if exp_time in exptimes:
                        raise RuntimeError(f"More than one master dark for the same exposure time are detected. Make sure that at most one file for each exposure time is available. The found files are {master_darks}")
                    else:
                        self.master_frames['dark'] = {exp_time:CCDData(file[0].data, unit='adu')}
                        exptimes = list(self.master_frames['dark'].keys())

        # create a dictionary for better access
        self.ifc_reduced.refresh()
        self.master_frames['dark'] = {ccd.header['exptime']: ccd for ccd in self.ifc_reduced.ccds(imagetyp=self.imagetypes['dark'], combined=True)}

        return self.master_frames['dark']

    def reduce_flats(self, force_new_master: bool=False, keep_files: bool=False) -> dict[str, CCDData]:
        """!Calibrates the flat frames and stacks them to a master flat
        
        @param force_new_master (bool) : (optional) If True, an existing master will be ignored and overwritten. Otherwise the existing will be used.
        @param keep_files (bool) : (optional) If True, all calibrated files will be kept. Default is 'False'
        @param master_bias (str) : (optional) filename of a masterbias that should be used. Defaulte is 'None'

        @return (dict) : dictionary with master flats. key=filter, value CCDData object
        """
        # 1) check for existing
        if not self.check_master('flat') or force_new_master:
            # check necessary files
            if self.master_frames['bias'] == None:
                self.reduce_bias(force_new_master, keep_files)
            if self.master_frames['dark'] == None:
                self.reduce_darks(force_new_master, keep_files)
        
        # 2) calibrate the flat frames
            for hdu, fname in self.ifc_raw.hdus(imagetyp=self.imagetypes['flat'], return_fname=True):
                ccd = CCDData(hdu.data, meta=hdu.header, unit='adu')
                ccd = subtract_bias(ccd, self.master_frames['bias'])
                ccd = subtract_dark(ccd, self.best_dark(hdu.header['exptime']), exposure_time='exptime', exposure_unit=units.second, scale=True)
                ccd.write(self.reduced_path / fname, overwrite=True)
                
        # 3) stack them
            self.ifc_reduced.refresh()

            # creating a set of all filters to distinguish the flats
            flat_filters: set[str] = set(h['filter'] for h in self.ifc_reduced.headers(imagetyp=self.imagetypes['flat']))

            for filt in flat_filters:
                reduced_flats: list[str] = self.ifc_reduced.files_filtered(imagetyp=self.imagetypes['flat'], filter=filt, include_path=True)

                # remove existing master from the imagelist, since those should not be stacked
                if force_new_master:
                    reduced_flats = self.__rm_master(reduced_flats)
                self.__combine(reduced_flats, frame='flat', filt=filt)

        # 4) clean up
            # remove not stacked files if not needed
            reduced_flats: list[str] = self.ifc_reduced.files_filtered(imagetyp=self.imagetypes['flat'], include_path=True)
            reduced_flats = self.__rm_master(reduced_flats)
            if not keep_files:
                [os.remove(file) for file in reduced_flats]
        
        # in all other cases there are already master flats
        else:
            print("There is already a master flat.")
            master_flats = self.ifc_reduced.files_filtered(iamgetyp=self.imagetypes['flat'], combined=True, include_path=True)
            filters: list[str] = []
            for flat in master_flats:
                # iterate over filters and check for duplicates
                with fits.open(flat) as file:
                    used_filter = file[0].header['filter']
                    if used_filter in filters:
                        raise RuntimeError(f"More than one master flat for the same filter are detected. Make shure that at most one file for each filter is available. The found files are {master_flats}")
                    else:
                        self.master_frames['flat'] = {used_filter:CCDData(file[0].data, unit='adu')}
        
        # create dictionary for better access
        self.ifc_reduced.refresh()
        self.master_frames['flat'] = {ccd.header['filter']: ccd for ccd in self.ifc_reduced.ccds(imagetyp=self.imagetypes['flat'], combined=True)}

        return self.master_frames['flat']