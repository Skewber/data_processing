import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

class FWHMestimator():
    def __init__(self, data):
        self.data = data
    
    @classmethod
    def fromfits(cls, fname: str):
        """!creat an instance from a given *.fits file

        @param fname (str): filename of the file that should be used for estimation
        """
        with fits.open(fname) as file:
            data = file[0].data
        return cls(data)
    
    def __circular_median(self, cx, cy, r):
        counts = []
        npixel = 0
        x = 0
        y = -r
        p = -r
        while x < -y:
            if p > 0:
                y += 1
                p += 2*(x+y) + 1
            else:
                p += 2*x + 1

            counts.append(self.data[cx + x, cy + y])
            counts.append(self.data[cx - x, cy + y])
            counts.append(self.data[cx + x, cy - y])
            counts.append(self.data[cx - x, cy - y])
            counts.append(self.data[cx + y, cy + x])
            counts.append(self.data[cx + y, cy - x])
            counts.append(self.data[cx - y, cy + x])
            counts.append(self.data[cx - y, cy - x])

            x += 1
            npixel += 8
        
        return np.median(counts)
    
    @staticmethod
    def psf_from_points(data, xy, fwhm):
        """!Generates a model PSF from given star locations by taking their mean

            @param data (array): image data
            @param xy (array): array or list of pairs of the positions that should be used
            @param fwhm (int): estimated value for the FWHM, them model will have the size of 2xFWHM x 2xFWHM

            @return (array): 2d array of the modeled PSF
    """
        fwhm = int(fwhm)
        models = np.zeros((len(xy), 2*fwhm, 2*fwhm))
        for i, (x, y) in enumerate(xy):
            # scale max to 1 --> brigther stars do not have mor impact on the mean
            star = data[x-fwhm:x+fwhm, y-fwhm:y+fwhm]
            max_val = np.max(star)
            models[i] = star / max_val
        return np.mean(models, axis=0)


    # TODO: add safety if too many stars are skipped --> possible infinity loop
    def estimate(self, nstars:int, fwhm_max:float=25, skip_max:int=10, full_output:bool=False) -> int:
        """!Estimates the FWHM for a given number of stars. Minimum FWHM is 2. Otherwise it is considered a cosmic and is skipped. This method can only estimate in even steps.

 
        @param nstars (int): number of stars that should be used for the estimation
        @param fwhm_max (float): highest possible value of the fwhm
        @param skip_max (int): maximum number of stars that should be skipped.
        @param full_output (bool): 

        @return (int|list[int], int): estimated FWHM value, or if full_output is set to True a list of estimated FWHM values and the number of skipped stars.
        """
        star = 0    # counter for detected stars
        positions = []
        estimates = []
        skipped = 0 # counter for skipped stars
        mask = np.zeros_like(self.data) # empty mask to cover detected stars
        while star < nstars and skipped < skip_max:
            # find the max of the uncovered area
            masked_array = np.ma.masked_array(self.data, mask)
            max_idx = np.unravel_index(np.argmax(masked_array), masked_array.shape)
            fwhm_value = self.data[*max_idx] * 0.5  # half the max value
            fwhm = 0
            median = self.data[*max_idx]    # initial value, median has to drop below fwhm_value
            while median > fwhm_value and fwhm <= fwhm_max/2:
                # increase the radius until median is too low
                fwhm += 1
                median = self.__circular_median(*max_idx, fwhm)
            # mask data for next iteration
            mask[max_idx[0]-fwhm_max:max_idx[0]+fwhm_max, max_idx[1]-fwhm_max:max_idx[1]+fwhm_max] = True
            if fwhm > 1:
                star += 1
                positions.append(max_idx)
                estimates.append(fwhm*2)
            else:
                skipped += 1

        # psf_model = self.psf_from_points(self.data, positions, np.mean(estimates))
        # plt.imshow(psf_model)
        # plt.show()
        if not full_output:
            return np.mean(estimates)
        else:
            return estimates, skipped, positions


estimator = FWHMestimator.fromfits('./testdata/data_comb/reduced_data/master_light_NGC2281_filter_V.fits')
print(estimator.estimate(55, full_output=True))