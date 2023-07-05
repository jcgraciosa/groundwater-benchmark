# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Coupled groundwater and temperature model
# - converted from underworld2 code

# %%
from petsc4py import PETSc
import underworld3 as uw
import numpy as np
import sympy

options = PETSc.Options()

# %%
minX, maxX = 0.0, 2.0
minY, maxY = -1.0, 0.0
res = 64

max_pressure = 1.

mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(minX, minY), maxCoords=(maxX, maxY), cellSize = 1/res, qdegree=3)

# mesh = uw.meshing.StructuredQuadBox(elementRes=(20,20),
#                                       minCoords=(minX,minY),
#                                       maxCoords=(maxX,maxY),)


p_soln = uw.discretisation.MeshVariable("P", mesh, 1, degree=2)
t_soln = uw.discretisation.MeshVariable("T", mesh, 1, degree=3)
v_soln = uw.discretisation.MeshVariable("U", mesh, mesh.dim, degree=1)

# x and y coordinates
x = mesh.N.x
y = mesh.N.y

# %%
1/64

# %%

if uw.mpi.size == 1:

    # plot the mesh
    import numpy as np
    import pyvista as pv
    import vtk

    pv.global_theme.background = "white"
    pv.global_theme.window_size = [750, 750]
    pv.global_theme.antialiasing = 'ssaa'
    pv.global_theme.jupyter_backend = "panel"
    pv.global_theme.smooth_shading = True

    mesh.vtk("tmp_mesh.vtk")
    pvmesh = pv.read("tmp_mesh.vtk")

    pl = pv.Plotter()

    pl.add_mesh(pvmesh, cmap="coolwarm", edge_color="Black", show_edges=True, use_transparency=False)

    pl.show(cpos="xy")

# %%
# Create Darcy Solver
darcy = uw.systems.SteadyStateDarcy(mesh, u_Field=p_soln, v_Field=v_soln)
darcy.petsc_options.delValue("ksp_monitor")
darcy.petsc_options["snes_rtol"] = 1.0e-6  # Needs to be smaller than the contrast in properties
darcy.constitutive_model = uw.systems.constitutive_models.DiffusionModel(mesh.dim)
# set the diffusivity below


# %%
# set up two materials

interfaceX = 1

from sympy import Piecewise, ceiling, Abs

k1 = 1.0        # permeable
k2 = 1.0e-4     # impermeable

# The piecewise version
hydraulicDiffFunc = Piecewise((k1, x < interfaceX), (k2, x >= interfaceX), (1.0, True))

# A smooth version
# kFunc = k2 + (k1-k2) * (0.5 + 0.5 * sympy.tanh(100.0*(y-interfaceY)))

darcy.constitutive_model.Parameters.diffusivity=hydraulicDiffFunc
darcy.f = 0.0
darcy.s = sympy.Matrix([0., 0.]).T # no gravitational body force

# set up boundary conditions
darcy.add_dirichlet_bc(0.0, "Top")
darcy.add_dirichlet_bc(-1.0 * minY * max_pressure, "Bottom") # still 1

# Zero pressure gradient at sides / base (implied bc)

darcy._v_projector.petsc_options["snes_rtol"] = 1.0e-6
darcy._v_projector.petsc_options["snes_max_it"] = 500
darcy._v_projector.smoothing = 1.0e-6
darcy._v_projector.add_dirichlet_bc(0.0, "Left", 0)
darcy._v_projector.add_dirichlet_bc(0.0, "Right", 0)
# %%
# Create Poisson solver - steady state heat equation

thermalDiff = 1.0
coeff = 1.0

poisson = uw.systems.Poisson(
    mesh,
    u_Field=t_soln,
    solver_name="poisson",
)
poisson.constitutive_model = uw.systems.constitutive_models.DiffusionModel(mesh.dim)
poisson.constitutive_model.Parameters.diffusivity = thermalDiff
poisson.f = -coeff*(sympy.diff(t_soln.sym[0], x)*v_soln.sym[0] + sympy.diff(t_soln.sym[0], y)*v_soln.sym[1])

# Dirichlet boundary conditions for temperature
poisson.add_dirichlet_bc(0.0, "Top")
poisson.add_dirichlet_bc(1.0, "Bottom")


# %%
# Groundwater pressure initial field


with mesh.access(p_soln):
    for i, coord in enumerate(p_soln.coords):
        p_soln.data[i] = -1.*coord[1]*max_pressure
        
# temperature initial field

with mesh.access(t_soln):
    for i, coord in enumerate(t_soln.coords):
        t_soln.data[i] = -1.*coord[1]

# %%
import matplotlib.pyplot as plt

with mesh.access(p_soln):
    fig, ax = plt.subplots(dpi = 100)
    out = ax.scatter(x = p_soln.coords[:, 0], y = p_soln.coords[:, 1], c = p_soln.data, cmap = "viridis",s = 15)
    plt.colorbar(out)


with mesh.access(t_soln):
    fig, ax = plt.subplots(dpi = 100)
    out = ax.scatter(x = t_soln.coords[:, 0], y = t_soln.coords[:, 1], c = t_soln.data, cmap = "viridis",s = 15)
    plt.colorbar(out)

# %%
# Solve time
darcy.solve()

# %%
poisson.solve()

# %%


if uw.mpi.size == 1:

    import numpy as np
    import pyvista as pv
    import vtk

    pv.global_theme.background = "white"
    pv.global_theme.window_size = [1250, 750]
    pv.global_theme.antialiasing = True
    pv.global_theme.jupyter_backend = "panel"
    pv.global_theme.smooth_shading = True

    mesh.vtk("tmp_mesh.vtk")
    pvmesh = pv.read("tmp_mesh.vtk")

    with mesh.access():
        usol = v_soln.data.copy()

    pvmesh.point_data["P"] = uw.function.evaluate(p_soln.fn, mesh.data, mesh.N)
    pvmesh.point_data["T"] = uw.function.evaluate(t_soln.fn, mesh.data, mesh.N)
    #pvmesh.point_data["K"] = uw.function.evaluate(kFunc, mesh.data, mesh.N)
    # pvmesh.point_data["S"]  = uw.function.evaluate(sympy.log(v_soln.fn.dot(v_soln.fn)), mesh.data)

    arrow_loc = np.zeros((v_soln.coords.shape[0], 3))
    arrow_loc[:, 0:2] = v_soln.coords[...]

    arrow_length = np.zeros((v_soln.coords.shape[0], 3))
    arrow_length[:, 0:2] = usol[...]

    # point sources at cell centres

    points = np.zeros((mesh._centroids.shape[0], 3))
    points[:, 0] = mesh._centroids[:, 0]
    points[:, 1] = mesh._centroids[:, 1]
    point_cloud = pv.PolyData(points[::3])

    v_vectors = np.zeros((mesh.data.shape[0], 3))
    v_vectors[:, 0:2] = uw.function.evaluate(v_soln.fn, mesh.data)
    pvmesh.point_data["V"] = v_vectors

    pvstream = pvmesh.streamlines_from_source(
        point_cloud,
        vectors="V",
        integrator_type=45,
        integration_direction="both",
        max_steps=1000,
        max_time=0.1,
        initial_step_length=0.001,
        max_step_length=0.01,
    )

    pl = pv.Plotter()

    pl.add_mesh(
        pvmesh, cmap="coolwarm", edge_color="Black", show_edges=True, scalars="T", use_transparency=False, opacity=1.0
    )

    #pl.add_mesh(pvstream, line_width=10.0)

    #pl.add_arrows(arrow_loc, arrow_length, mag=0.005, opacity=0.75)

    pl.show(cpos="xy")


# %%
# set up interpolation coordinates
ycoords = np.linspace(minY + 0.001 * (maxY - minY), maxY - 0.001 * (maxY - minY), 100)
xcoords = np.full_like(ycoords, -1)
xy_coords = np.column_stack([xcoords, ycoords])

pressure_interp = uw.function.evaluate(p_soln.sym[0], xy_coords)


# %%
La = -1.0 * interfaceY
Lb = 1.0 + interfaceY
dP = max_pressure

S = 1
Pa = (dP / Lb - S + k1 / k2 * S) / (1.0 / Lb + k1 / k2 / La)
pressure_analytic = np.piecewise(
    ycoords,
    [ycoords >= -La, ycoords < -La],
    [lambda ycoords: -Pa * ycoords / La, lambda ycoords: Pa + (dP - Pa) * (-ycoords - La) / Lb],
)

S = 0
Pa = (dP / Lb - S + k1 / k2 * S) / (1.0 / Lb + k1 / k2 / La)
pressure_analytic_noG = np.piecewise(
    ycoords,
    [ycoords >= -La, ycoords < -La],
    [lambda ycoords: -Pa * ycoords / La, lambda ycoords: Pa + (dP - Pa) * (-ycoords - La) / Lb],
)

# %%
import matplotlib.pyplot as plt

# %matplotlib inline

fig = plt.figure()
ax1 = fig.add_subplot(111, xlabel="Pressure", ylabel="Depth")
ax1.plot(pressure_interp, ycoords, linewidth=3, label="Numerical solution")
ax1.plot(pressure_analytic, ycoords, linewidth=3, linestyle="--", label="Analytic solution")
ax1.plot(pressure_analytic_noG, ycoords, linewidth=3, linestyle="--", label="Analytic (no gravity)")
ax1.grid("on")
ax1.legend()
# %%

