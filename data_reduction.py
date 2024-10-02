from pathlib import Path
from ccdproc import ImageFileCollection

class DataReduction():
    """
    Basic class for Data Reduction. For this class to work properly the used .fits files need a the following keywords in the header: 'EXPTIME', 'FILTER', 'OBJECT'. Optionally it can also contain wcs informations if an alignment of those is desired. If not provided the alignemt can also be done by matching stars in the image.
    """
    def __init__(self, foldername_data: str, foldername_reduced: str="reduced_data") -> None:
        """Set up of general information

        Args:
            foldername_data (str): name of the folder with the raw data to
            foldername_reduced (str, optional): (optional) name of the folder the reduced data should be stored. If it does not exist, this folder will be created. Defaults to "reduced_data".
        """
        # set up for raw data path
        self.raw_path = Path('.', foldername_data)
        self.ifc_raw = ImageFileCollection(self.raw_path)

        # set up for reduced data path
        self.reduced_path = Path('.', foldername_reduced)
        self.reduced_path.mkdir(exist_ok=True)
        self.ifc_reduced = ImageFileCollection(self.reduced_path)

        self.imagetypes: dict[str, str] = {'bias':'',
                                        'dark':'',
                                        'flat':'',
                                        'light':''}
        
        keys = self.imagetypes.keys()
        frame_names = self.imagetypes.values()

        # determine the strings that specify the imagetype
        for hdu, fname in self.ifc_raw.hdus(return_fname=True):
            # add 'bunit' to every header. Needed later on
            hdu.header['bunit'] = 'adu'
            hdu.writeto(Path('.', foldername_data, fname), overwrite=True)

            # set names for imagetype
            img_type: str = hdu.header['imagetyp']
            # FIXME: try to remove nesting
            if not img_type in frame_names:
                for key in key:
                    if key.lower() in img_type.lower():
                        self.imagetypes[key] = img_type
        
        # raise an error if for at least one type ther is no name found
        if '' in frame_names:
            raise ValueError(f"No name for the imagetype was found for at least one imagetype. The following types are determined:\n{self.imagetypes}.\nCheck your data again.")

        # set up dict for masters
        self.master_frames: dict = {'bias':None,
                                'dark':None,
                                'flat':None,
                                'light':None}
                