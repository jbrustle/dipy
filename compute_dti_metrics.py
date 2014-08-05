#! /usr/bin/env python
from __future__ import division, print_function

import nibabel as nib
import numpy as np
import argparse
import os

from dipy.segment.mask import median_otsu
from dipy.core.gradients import gradient_table_from_bvals_bvecs
from dipy.io.gradients import read_bvals_bvecs
from dipy.reconst.dti import (TensorModel, color_fa, fractional_anisotropy,
                              mean_diffusivity, axial_diffusivity, radial_diffusivity,
                              lower_triangular, mode)

DESCRIPTION = """
    Convenient script to compute all of the Diffusion Tensor Imaging (DTI) metrics. If -all option is given fractional anisotropy (FA), axial diffusivisty (AD), radial diffusivity (radial diffusivity), mean diffusivity (MD), mode, colored red-green-blue FA (rgb) and tensor coefficients (dxx, dxy, dxz, dyy, dyz, dzz) are saved.
    """


def buildArgsParser():

    p = argparse.ArgumentParser(description=DESCRIPTION)

    p.add_argument('input', action='store', metavar='input', type=str,
                   help='Path of the input diffusion volume.')

    p.add_argument('bvals', action='store', metavar='bvals',
                   help='Path of the bvals file, in FSL format.')

    p.add_argument('bvecs', action='store', metavar='bvecs',
                   help='Path of the bvecs file, in FSL format.')

    p.add_argument('-mask', action='store', dest='mask',
                   metavar='mask', required=False, default=None, type=str,
                   help='Path to a binary mask. Only data inside the mask will be used \
                   for computations and reconstruction.')

    p.add_argument('-o', action='store', dest='savename',
                   metavar='savename', required=False, default=None, type=str,
                   help='Path and prefix for the saved metrics files. The name is always appended \
                   with _(metric_name).nii.gz, where (metric_name) if the name of the computed metric.')

    p.add_argument('-f', action='store_true', dest='overwrite', required=False,
                   help='If True, the saved files volume will be overwritten \
                   if they already exist.')

    p.add_argument('-all', action='store_true', dest='all', required=False,
                   help='If True, saves the fa, ad, rd, md, mode, rgb and tensor coefficients.')

    return p


def main():
    parser = buildArgsParser()
    args = parser.parse_args()

    # Load data
    img = nib.load(args.input)
    data = img.get_data()
    affine = img.get_affine()

    # Setting suffix savename
    if args.savename is None:
        filename = ""
    else:
        filename = args.savename + "_"

    if os.path.exists(filename + 'fa.nii.gz'):
        if not args.overwrite:
            raise ValueError("File " + filename + "fa.nii.gz" 
                             + " already exists. Use -f option to overwrite.")

        print (filename + "fa.nii.gz", " already exists and will be overwritten.")

    if args.mask is not None:
        mask = nib.load(args.mask).get_data()
    else:
        print("No mask specified. Computing mask with median_otsu.")
        data, mask = median_otsu(data)
        mask_img = nib.Nifti1Image(mask.astype(np.float32), affine)
        nib.save(mask_img, filename + 'mask.nii.gz')

    # Get tensors
    print('Tensor estimation...')
    b_vals, b_vecs = read_bvals_bvecs(args.bvals, args.bvecs)
    gtab = gradient_table_from_bvals_bvecs(b_vals, b_vecs)
    tenmodel = TensorModel(gtab)
    tenfit = tenmodel.fit(data, mask)

    # FA
    print('Computing FA...')
    FA = fractional_anisotropy(tenfit.evals)
    FA[np.isnan(FA)] = 0

    # RGB
    print('Computing RGB...')
    FA = np.clip(FA, 0, 1)
    RGB = color_fa(FA, tenfit.evecs)

    if args.all :
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

    fa_img = nib.Nifti1Image(FA.astype(np.float32), affine)
    nib.save(fa_img, filename + 'fa.nii.gz')
    rgb_img = nib.Nifti1Image(np.array(255 * RGB, 'uint8'), affine)
    nib.save(rgb_img, filename + 'rgb.nii.gz')
        

if __name__ == "__main__":
    main()



