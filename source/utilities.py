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


    # TODO: add safety if too many stars are skipped --> possible infinity loop
    def estimate(self, nstars:int, fwhm_max:float=25, skip_max:int=10, full_output:bool=False) -> int:
        """!Estimates the FWHM for a given number of stars. Minimum FWHM is 2. Otherwise it is considered a cosmic and is skipped. This method can only estimate in even steps.

 
        @param nstars (int): number of stars that should be used for the estimation
        @param fwhm_max (float): highest possible value of the fwhm
        @param skip_max (int): maximum number of stars that should be skipped.
        @param full_output (bool): 

        @return (int|list[int], int): estimated FWHM value, or if full_output is set to True a list of estimated FWHM values and the number of skipped stars.
        """
        star = 0
        positions = []
        estimates = []
        skipped = 0
        mask = np.zeros_like(self.data)
        while star < nstars and skipped < skip_max:
            masked_array = np.ma.masked_array(self.data, mask)
            x = np.unravel_index(np.argmax(masked_array), masked_array.shape)
            fwhm_value = self.data[*x] * 0.5
            fwhm = 0
            median = self.data[*x]
            while median > fwhm_value and fwhm <= fwhm_max/2:
                fwhm += 1
                median = self.__circular_median(*x, fwhm)
            # prepare data for next iteration
            mask[x[0]-25:x[0]+25, x[1]-25:x[1]+25] = True
            if fwhm > 1:
                star += 1
                positions.append(x)
                estimates.append(fwhm*2)
            else:
                skipped += 1
        if not full_output:
            return np.mean(estimates)
        else:
            return estimates, skipped


estimator = FWHMestimator.fromfits('./testdata/data_comb/reduced_data/master_light_NGC2281_filter_V.fits')
print(estimator.estimate(10, full_output=True))