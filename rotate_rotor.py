import numpy as np
from dolfinx import fem
from dolfinx.io import VTXWriter, XDMFFile
from dolfinx.mesh import create_submesh
from mpi4py import MPI

from utils import par_print

comm = MPI.COMM_WORLD
degree = 1

with XDMFFile(comm, "team24_horseshoe.xdmf", "r") as xdmf:
    mesh = xdmf.read_mesh()
    ct   = xdmf.read_meshtags(mesh, name="Cell_markers")
    tdim = mesh.topology.dim
    fdim = tdim - 1
    mesh.topology.create_entities(fdim)
    ft   = xdmf.read_meshtags(mesh, name="Facet_markers")

par_print(comm, f"Number of cells in the mesh: {mesh.topology.index_map(tdim).size_global}")


def rotation_matrix_z(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def set_mesh_rotation(msh, ref_coords, theta, centre):
    """Place ``msh`` at absolute angle ``theta`` about the z-axis."""
    R = rotation_matrix_z(theta)
    msh.geometry.x[:] = (ref_coords - centre) @ R.T + centre


domains = {
    "air": 1,
    "coil1": 2,
    "coil2": 3,
    "rotor": 4,
    "stator": 5,
}

rotor_domain = domains["rotor"]
submesh_rotor, rotor_to_parent = create_submesh(mesh, tdim, ct.find(rotor_domain))[:2]

rotor_ref_coords = submesh_rotor.geometry.x.copy()
centre = np.array([0.0, 0.0, 0.0])

V_rotor = fem.functionspace(submesh_rotor, ("DG", 0))
q_rotor = fem.Function(V_rotor)
q_rotor.x.array[:] = 1.0

V_bg = fem.functionspace(mesh, ("DG", 0))
q_bg = fem.Function(V_bg)

file = VTXWriter(mesh.comm, "qbg.bp", [q_bg], "BP4")
file.write(0.0)

combined_cells = np.unique(
    np.concatenate([ct.find(domains["rotor"]), ct.find(domains["air"])])
).astype(np.int32) # This is the set of cells that are either in the rotor or in the air domain

angles = np.deg2rad(np.linspace(0.0, 180.0, 20, endpoint=False))


for theta in angles:
    set_mesh_rotation(submesh_rotor, rotor_ref_coords, theta, centre)

    idata = fem.create_interpolation_data(V_bg, V_rotor, combined_cells, padding=1e-14) # This interpolates between the 2 meshes

    q_bg.x.array[:] = 0.0
    q_bg.interpolate_nonmatching(q_rotor, combined_cells, interpolation_data=idata)
    q_bg.x.scatter_forward()

    par_print(comm, f"theta = {np.rad2deg(theta):6.1f} deg, "
                    f"covered cells = {int(np.sum(q_bg.x.array > 0.5))}")
    
    q_bg.x.scatter_forward()
    file.write(theta)
