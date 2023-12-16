# @author paul-bd and Arthur Astier
# https://github.com/paul-bd/MIP-PET.git
import os
import shutil
import matplotlib.pylab as plt
import numpy as np
import scipy
from scipy import ndimage


def create_mip_from_3d(img, patient_reference: str, nb_image=40, duration=0.1, is_mask=False, borne_max=None):
    ls_mip = []

    img_data = img.get_fdata()
    img_data += 1e-5

    for angle in np.linspace(0, 360, nb_image):
        vol_angle = scipy.ndimage.interpolation.rotate(img_data, angle)

        MIP = np.amax(vol_angle, axis=1)
        MIP -= 1e-5
        MIP[MIP < 1e-5] = 0
        MIP = np.flipud(MIP.T)
        ls_mip.append(MIP)
    try:
        shutil.rmtree(patient_reference + '/')
    except:
        pass
    os.mkdir(patient_reference + '/')

    for mip, i in zip(ls_mip, range(len(ls_mip))):
        fig, ax = plt.subplots()
        ax.set_axis_off()
        if borne_max is None:
            if is_mask == True:
                borne_max = 1
            else:
                borne_max = 15000
        plt.imshow(mip, cmap='Greys', vmax=borne_max)
        fig.savefig(patient_reference + '/MIP' + '%04d' % (i) + '.png')
        plt.close(fig)
