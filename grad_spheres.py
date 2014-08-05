#!//anaconda/bin/python

import numpy as np
from dipy.core.sphere import disperse_charges, Sphere, HemiSphere
n_pts = 64
theta = np.pi * np.random.rand(n_pts)
phi = 2 * np.pi * np.random.rand(n_pts)
hsph_initial = HemiSphere(theta=theta, phi=phi)
hsph_updated, potential = disperse_charges(hsph_initial, 5000)
from dipy.viz import fvtk
ren = fvtk.ren()
ren.SetBackground(1, 1, 1)
fvtk.add(ren, fvtk.point(hsph_initial.vertices, fvtk.colors.red, point_radius=0.05))
fvtk.add(ren, fvtk.point(hsph_updated.vertices, fvtk.colors.green, point_radius=0.05))

print('Saving illustration as initial_vs_updated.png')
fvtk.record(ren, out_path='initial_vs_updated.png', size=(300, 300))
sph = Sphere(xyz = np.vstack((hsph_updated.vertices, -hsph_updated.vertices)))

fvtk.rm_all(ren)
fvtk.add(ren, fvtk.point(sph.vertices, fvtk.colors.green, point_radius=0.05))

print('Saving illustration as full_sphere.png')
fvtk.record(ren, out_path='full_sphere.png', size=(300, 300))
from dipy.core.gradients import gradient_table

vertices = hsph_updated.vertices
values = np.ones(vertices.shape[0])
bvecs = np.vstack((vertices, vertices))
bvals = np.hstack((1000 * values, 2500 * values))
bvecs = np.insert(bvecs, (0, bvecs.shape[0]), np.array([0, 0, 0]), axis=0)
bvals = np.insert(bvals, (0, bvals.shape[0]), 0)
gtab = gradient_table(bvals, bvecs)

fvtk.rm_all(ren)
colors_b1000 = fvtk.colors.blue * np.ones(vertices.shape)
colors_b2500 = fvtk.colors.cyan * np.ones(vertices.shape)
colors = np.vstack((colors_b1000, colors_b2500))
colors = np.insert(colors, (0, colors.shape[0]), np.array([0, 0, 0]), axis=0)
colors = np.ascontiguousarray(colors)
fvtk.add(ren, fvtk.point(gtab.gradients, colors, point_radius=100))
fvtk.show(ren, title='Dipy', size=(300, 300), png_magnify=1)

