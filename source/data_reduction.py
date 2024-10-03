from pathlib import Path
import shutil
import numpy as np
from ccdproc import ImageFileCollection, combine
from astropy.io import fits
from astropy.nddata import CCDData
from astropy.stats import mad_std
import os

class DataReduction():
    """! Basic class for Data Reduction. For this class to work properly the used .fits files need a the following keywords in the header: 'EXPTIME', 'FILTER', 'OBJECT'. Optionally it can also contain wcs informations if an alignment of those is desired. If not provided the alignemt can also be done by matching stars in the image.
    """
    def __init__(self, foldername_data: str, foldername_reduced: str="reduced_data") -> None:
        """! Set up of general information

        @param foldername_data (str): name of the folder with the raw data to
        @param foldername_reduced (str, optional): (optional) name of the folder the reduced data should be stored. If it does not exist, this folder will be created. Defaults to "reduced_data".
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
        self.ifc_reduced = ImageFileCollection(self.reduced_path)

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
    
    def check_master(self, frametype: str, master: str='') -> bool:
        """!Checks if  a specific master frame exist
        
        @param frametype (string) : specify which frametype should be checked. Valid are 'bias', 'dark', 'flat', 'light'
        @param master (string) : optional, the filename of a masterframe. Check if the file exist
            
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