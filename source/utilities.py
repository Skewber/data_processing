import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.ndimage import maximum_filter
from scipy.optimize import curve_fit

class FWHMestimator():
    def __init__(self, data):
        self.data = data
    
    @classmethod
    def fromfits(cls, fname: str):
        """!create an instance from a given *.fits file

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
        models = np.zeros((len(xy), 4*fwhm, 4*fwhm))
        for i, (x, y) in enumerate(xy):
            # scale max to 1 --> brigther stars do not have mor impact on the mean
            star = data[x-2*fwhm:x+2*fwhm, y-2*fwhm:y+2*fwhm]
            max_val = np.max(star)
            models[i] = star / max_val
        return np.mean(models, axis=0)

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
        # fimd maxima
        max_data = maximum_filter(self.data, fwhm_max)
        max_mask = (self.data == max_data)
        max_coords = np.column_stack(np.where(max_mask))
        max_coords = sorted(max_coords, key=lambda coord: self.data[tuple(coord)], reverse=True)
        edge_buffer = fwhm_max // 2
        max_coords = [coord for coord in max_coords
                      if coord[0] > edge_buffer and coord[0] < self.data.shape[0] - edge_buffer
                      and coord[1] > edge_buffer and coord[1] < self.data.shape[1] - edge_buffer]
        max_coords = max_coords[:nstars+skip_max]

        for i, star in enumerate(max_coords):
            # estimate fwhm
            fwhm_value = self.data[*star] * 0.5 # half the max value
            fwhm = 0
            median = self.data[*star]   # initial value, median has to drop below fwhm_value
            while median > fwhm_value and fwhm <= fwhm_max/2:
                # increas the radius until the median is too low
                fwhm += 1
                median = self.__circular_median(*star, fwhm)
            
            if fwhm > 1:
                positions.append(star)
                estimates.append(fwhm*2)
            else:
                skipped += 1

            if i == nstars or skipped == skip_max:
                break

        estimate = np.mean(estimates)
        psf_model = self.psf_from_points(self.data, positions, np.mean(estimates))

        # perform a gauss fit to psf model to get the final estimate
        def gauss(x, mu, sigma, amp, offset):
            return amp * np.exp(-(x-mu)**2 / (2 * sigma**2)) + offset
        
        bounds = [[0, 0, 0, 0],
                  [psf_model.shape[0], psf_model.shape[0]/2, 1, 1]]

        pixel = np.arange(0, psf_model.shape[0])
        poptx, _ = curve_fit(gauss, pixel, psf_model[int(estimate//2),:], bounds=bounds)
        popty, _ = curve_fit(gauss, pixel, psf_model[:, int(estimate//2)], bounds=bounds)

        sig_to_fwhm = 2 * np.sqrt(2*np.log(2))  # conversion factor from sigma to fwhm
        fwhm = (sig_to_fwhm * poptx[1] + sig_to_fwhm * popty[1]) / 2    # average over x and y direction

        if not full_output:
            return fwhm
        else:
            return fwhm, skipped, positions, psf_model
    