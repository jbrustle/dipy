#!//anaconda/bin/python
from __future__ import division
import nibabel as nib
import numpy as np
import dipy
import os

from dipy.core.ndindex import ndindex
from mygradients import gradient as mygrad
from numpy import linalg
from dipy.segment.mask import median_otsu
from dipy.core.gradients import gradient_table_from_bvals_bvecs
from dipy.io.gradients import read_bvals_bvecs
from dipy.reconst.dti import (TensorModel, color_fa, fractional_anisotropy,
                              mean_diffusivity, axial_diffusivity, radial_diffusivity,
                              lower_triangular, mode)



data_file = '/Users/brut1901/.dipy/stanford_hardi/HARDI150.nii.gz'
b_vals_file = '/Users/brut1901/.dipy/stanford_hardi/HARDI150.bval'
b_vecs_file = '/Users/brut1901/.dipy/stanford_hardi/HARDI150.bvec'

# Load data
img = nib.load(data_file)
data = img.get_data()
affine = img.get_affine()

# Setting suffix savename
filename = ""

if os.path.exists(filename + 'fa.nii.gz'):
    print (filename + "fa.nii.gz", " already exists and will be overwritten.")

print("No mask specified. Computing mask with median_otsu.")
from dipy.segment.mask import median_otsu
data, mask = median_otsu(data)
mask_img = nib.Nifti1Image(mask.astype(np.float32), affine)
nib.save(mask_img, filename + 'mask.nii.gz')


# Get tensors
print('Tensor estimation...')
b_vals, b_vecs = read_bvals_bvecs(b_vals_file, b_vecs_file)
print b_vals.shape
print b_vecs.shape
print data.shape

# Choose some epsilon for mask
eps = 0.1
b0_mask = b_vals < eps
b0_dwi = data[:,:,:,b0_mask]
dwi = data[:,:,:,np.logical_not(b0_mask)]
vecs = b_vecs[np.logical_not(b0_mask)]
vals = b_vals[np.logical_not(b0_mask)]

# Minimize risk of noise by taking the average of b0 values
b0_moy = np.average(b0_dwi, axis=3)
print b0_dwi.shape
print dwi.shape

# Make Gradients
grads=mygrad(vals,vecs)

# Design matrix
B = np.array([   vecs[:,0]*vecs[:,0],
        2.*vecs[:,0]*vecs[:,1],
        2.*vecs[:,0]*vecs[:,2],
        vecs[:,1]*vecs[:,1],
        2.*vecs[:,1]*vecs[:,2],
        vecs[:,2]*vecs[:,2] ]).T


# Stable inv. function pinv to find pseudo-inv
B_inv=np.linalg.pinv(B)
data_shape=[data.shape[0],data.shape[1],data.shape[2],6]

# Create tensor matrix
D=np.zeros(data_shape)

# Create signal matrix
X=np.zeros_like(dwi)
X=X.astype(np.float32)

# Iterate over the 3 dimensions
"""
for x in range(data_shape[0]):
    for y in range(data_shape[1]):
        for z in range(data_shape[2]):
            if ((b0_moy[x,y,z]==0.) or (not(all(dwi[x,y,z])))):
                D[x,y,z,:]=[0.,0.,0.,0.,0.,0.]
            else:
                X[x,y,z,:] = -1/vals*np.log(dwi[x,y,z,:]/b0_moy[x,y,z,np.newaxis])
                D[x,y,z,:]=np.dot(B_inv,X[x,y,z,:])
"""

for w in ndindex((data.shape[0], data.shape[1], data.shape[2])):
    if ((b0_moy[w]==0.) or (not(all(dwi[w])))):
        D[w]=[0.,0.,0.,0.,0.,0.]
    else:
        X[w,:] = -1/vals*np.log(dwi[w,:]/b0_moy[w+(np.newaxis)])
        D[w,:]=np.dot(B_inv,X[w,:])



# Save tensor
D_img = nib.Nifti1Image(D.astype(np.float32), affine)
nib.save(D_img, filename + 'D.nii.gz')















"""
E=np.array([data.shape[0],data.shape[1],data.shape[2],3])
E=E.astype(np.float32)
for x in range(data_shape[0]):
    for y in range(data_shape[1]):
        for z in range(data_shape[2]):
            D[x,y,z].reshape(2,3)

"""





"""
gtab = gradient_table_from_bvals_bvecs(b_vals, b_vecs)
tenmodel = TensorModel(gtab)
tenfit = tenmodel.fit(data, mask)
"""

"""
# FA
print('Computing FA...')
FA = fractional_anisotropy(tenfit.evals)
FA[np.isnan(FA)] = 0

# RGB
print('Computing RGB...')
FA = np.clip(FA, 0, 1)
RGB = color_fa(FA, tenfit.evecs)
"""



"""
print('Computing Diffusivities...')
# diffusivities
MD = mean_diffusivity(tenfit.evals)
AD = axial_diffusivity(tenfit.evals)
RD = radial_diffusivity(tenfit.evals)

print('Computing Mode...')
MODE = mode(tenfit.quadratic_form)

print('Saving tensor coefficients and metrics...')
# Get the Tensor values and format them for visualisation in the Fibernavigator.
tensor_vals = lower_triangular(tenfit.quadratic_form)
correct_order = [0, 1, 3, 2, 4, 5]
tensor_vals_reordered = tensor_vals[..., correct_order]
fiber_tensors = nib.Nifti1Image(tensor_vals_reordered.astype(np.float32), affine)
nib.save(fiber_tensors, filename + 'tensors.nii.gz')

# Save - for some reason this is not read properly by the FiberNav
md_img = nib.Nifti1Image(MD.astype(np.float32), affine)
nib.save(md_img, filename + 'md.nii.gz')
ad_img = nib.Nifti1Image(AD.astype(np.float32), affine)
nib.save(ad_img, filename + 'ad.nii.gz')
rd_img = nib.Nifti1Image(RD.astype(np.float32), affine)
nib.save(rd_img, filename + 'rd.nii.gz')
mode_img = nib.Nifti1Image(MODE.astype(np.float32), affine)
nib.save(mode_img, filename + 'mode.nii.gz')
"""

"""
fa_img = nib.Nifti1Image(FA.astype(np.float32), affine)
nib.save(fa_img, filename + 'fa.nii.gz')
rgb_img = nib.Nifti1Image(np.array(255 * RGB, 'uint8'), affine)
nib.save(rgb_img, filename + 'rgb.nii.gz')
"""    