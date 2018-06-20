from builtins import zip
import numpy as np
from lsst.sims.utils import m5_flat_sed
from lsst.sims.photUtils import LSSTdefaults


def generate_percentiles(nbins=20):
    """
    Make histograms of the 5-sigma limiting depths for each point and each filter.
    """

    filters = ['u', 'g', 'r', 'i', 'z', 'y']

    restore_file = 'healpix/59560_59772.npz'
    disk_data = np.load(restore_file)
    required_mjds = disk_data['header'][()]['required_mjds'].copy()
    dict_of_lists = disk_data['dict_of_lists'][()].copy()
    disk_data.close()

    sb_file = 'healpix/59560_59772.npy'
    sky_brightness = np.load(sb_file)

    npix = sky_brightness['r'].shape[-1]

    histograms = np.zeros((nbins, npix), dtype=list(zip(filters, [float]*6)))
    histogram_npts = np.zeros(npix, dtype=list(zip(filters, [int]*6)))

    # find the indices of all the evenly spaced mjd values
    even_mjd_indx = np.in1d(dict_of_lists['mjds'], required_mjds)
 
    # make an array to hold only the sky brightness values for evenly spaced mjds
    sky_brightness_reduced = np.ones((even_mjd_indx.sum(), npix),
                                      dtype=sky_brightness.dtype)

    for key in dict_of_lists:
        dict_of_lists[key] = dict_of_lists[key][even_mjd_indx]
    for key in filters:
        sky_brightness_reduced[key] = sky_brightness[key][even_mjd_indx, :]
    del sky_brightness

    for filtername in filters:
        # convert surface brightness to m5
        FWHMeff = LSSTdefaults().FWHMeff(filtername)
        # increase as a function of airmass
        airmass_correction = np.power(dict_of_lists['airmass'], 0.6)
        FWHMeff *= airmass_correction
        m5_arr = m5_flat_sed(filtername, sky_brightness_reduced[filtername], FWHMeff, 30.,
                             dict_of_lists['airmass'])

        for indx in np.arange(npix):
            m5s = m5_arr[:, indx]
            m5s = m5s[np.isfinite(m5s)]
            m5s = np.sort(m5s)
            percentile_points = np.round(np.linspace(0, m5s.size-1, nbins))
            if m5s.size > percentile_points.size:
                histograms[filtername][:, indx] = m5s[percentile_points.astype(int)]
                histogram_npts[filtername][indx] = m5s.size
            # make the histogram for this point in the sky
            # histograms[filtername][:, indx] += np.histogram(m5s[np.isfinite(m5s)],
            #                                                bins=bins[filtername])[0]

    np.savez('percentile_m5_maps.npz', histograms=histograms, histogram_npts=histogram_npts)


if __name__ == '__main__':

    generate_percentiles()
