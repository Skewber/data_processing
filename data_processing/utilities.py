import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.stats import SigmaClip
from scipy.ndimage import median_filter
from photutils.segmentation import SourceCatalog, detect_sources, detect_threshold, deblend_sources
from photutils.background import Background2D, MedianBackground
from scipy.optimize import curve_fit

def gauss(x, amp, mu, sigma, base):
    return amp * np.exp(-(x - mu)**2 / (2*sigma**2)) + base

def fit_gauss(x, y, p0=None, bounds=None):
    if p0 == None:
        p0 = [np.max(y), x[0]+(x[-1]-x[0])/2, (x[-1]-x[0])/2, np.min(y)]
    if bounds == None:
        bounds = [[np.min(y), x[0], 0, np.min(y)],
                  [np.max(y), x[-1], x[-1]-x[0], np.max(y)]]
        
    popt, _ = curve_fit(gauss, x, y, p0=p0, bounds=bounds)
    return popt

def fwhm_from_std(std):
    return 2 * np.sqrt(2 * np.log(2)) * std

def estimate_fwhm(data, threshold_sigma=3, median_size=3, npixels=9, cutout=15, plot_sources=False):
    #subtrackt background
    sigma_clip = SigmaClip(sigma=3)
    bkg_estimator = MedianBackground()
    bkg = Background2D(data, data.shape, filter_size=3, sigma_clip=sigma_clip, bkg_estimator=bkg_estimator)
    data = data - bkg.background

    # define threshold for detection
    threshold = detect_threshold(data, threshold_sigma)

    # apply filter to reduce artifacts
    data = median_filter(data, median_size)

    # detect sources and deblend them
    segments = detect_sources(data, threshold, npixels=npixels)
    segments = deblend_sources(data, segments, npixels=npixels, nlevels=32, contrast=0.001)

    # create source catalog to work with
    catalog = SourceCatalog(data, segments)

    xlist, ylist = [], []
    sources = []

    edge_buffer = cutout

    for source in catalog:
        x, y = int(source.xcentroid), int(source.ycentroid)

        if x > edge_buffer and x < data.shape[1] - edge_buffer and y > edge_buffer and y < data.shape[0] - edge_buffer:
            xlist.append(source.xcentroid)
            ylist.append(source.ycentroid)

            half_size = cutout
            # min/max as a safty if the star is too close to the edge
            y_min = max(0, y - half_size)
            y_max = min(data.shape[0], y + half_size)
            x_min = max(0, x - half_size)
            x_max = min(data.shape[1], x + half_size)
            sources.append(data[y_min:y_max, x_min:x_max]/data[y,x])
        
    print(np.array(sources).shape)
    epsf = np.median(sources, axis=0)
    print(epsf.shape)

    # extract a slice in x and y direction
    center = epsf.shape[0]//2
    slice_x = epsf[center, :]
    slice_y = epsf[:, center]

    # perform fit
    pixels = np.arange(0, len(slice_x))
    poptx = fit_gauss(pixels, slice_x)
    popty = fit_gauss(pixels, slice_y)

    x = np.linspace(0, pixels[-1], 1000)

    plt.scatter(pixels, slice_x, color='blue', label='x')
    plt.scatter(pixels, slice_y, color='red', label='y')
    plt.plot(x, gauss(x, *poptx), color='blue')
    plt.plot(x, gauss(x, *popty), color='red')
    plt.legend()
    plt.show()
    plt.clf()

    fwhm1 = fwhm_from_std(poptx[2])
    fwhm2 = fwhm_from_std(popty[2])
    fwhm = (fwhm1 + fwhm2) / 2

    if plot_sources:
        # calculateate grid dimensions (rows x cols) as close to a square as possible
        n_sources = len(sources)
        cols = int(np.ceil(np.sqrt(n_sources)))
        rows = int(np.ceil(n_sources / cols))

        # Plot each source in the grid
        fig, axes = plt.subplots(rows, cols, figsize=(10, 10))
        axes = axes.flatten()  # Flatten for easy indexing

        for i, ax in enumerate(axes):
            if i < n_sources:
                ax.imshow(sources[i], origin='lower', cmap='viridis')
                # ax.axis('off')  # Hide axes for a cleaner look
            else:
                ax.axis('off')  # Hide unused subplots if any

        plt.tight_layout()
        plt.savefig("sources.png")

        plt.clf()
        plt.imshow(epsf)
        plt.savefig("epsf.png")

    return fwhm

with fits.open("./testdata/lfoa/20120328T195140.st10.object.fits") as hdul:
    data = hdul[0].data

print(estimate_fwhm(data, 3, 3, 9, 15, True))