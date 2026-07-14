import numpy as np
import ufl
from dolfinx import default_scalar_type, fem
from dolfinx.fem.petsc import LinearProblem
from dolfinx.io import VTXWriter, XDMFFile
from dolfinx.mesh import create_submesh, exterior_facet_indices
from mpi4py import MPI

from utils import L2_norm, par_print

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

V_background = fem.functionspace(mesh, ("DG", 0))
q_bg = fem.Function(V_background)


file = VTXWriter(mesh.comm, "rotor_geometry.bp", [q_bg], "BP4")
file.write(0.0)

combined_cells = np.unique(
    np.concatenate([ct.find(domains["rotor"]), ct.find(domains["air"])])
).astype(np.int32)

angles = np.deg2rad(np.linspace(0.0, 180.0, 20, endpoint=False))


V_poisson = fem.functionspace(mesh, ("Lagrange", degree))
u = ufl.TrialFunction(V_poisson)
v = ufl.TestFunction(V_poisson)

a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
L = ufl.inner(q_bg, v) * ufl.dx

mesh.topology.create_connectivity(fdim, tdim)
boundary_facets = exterior_facet_indices(mesh.topology)
boundary_dofs = fem.locate_dofs_topological(V_poisson, fdim, boundary_facets)
bc = fem.dirichletbc(default_scalar_type(0.0), boundary_dofs, V_poisson)

uh = fem.Function(V_poisson)
uh.name = "u"

problem = LinearProblem(
    a,
    L,
    bcs=[bc],
    u=uh,
    petsc_options={"ksp_type": "cg", "pc_type": "gamg", "ksp_rtol": 1e-8},
    petsc_options_prefix="poisson_"
)

theta = 0.0

u_file = VTXWriter(mesh.comm, "results.bp", [uh], "BP4")
u_file.write(theta)

Lagrange_rot = fem.functionspace(submesh_rotor, ("Lagrange", degree))

smsh_cell_imap = submesh_rotor.topology.index_map(tdim)
smsh_cells = np.arange(smsh_cell_imap.size_local + smsh_cell_imap.num_ghosts)
parent_cells = rotor_to_parent.sub_topology_to_topology(smsh_cells, inverse=False)
u_rotor = fem.Function(Lagrange_rot)
u_rotor.interpolate(
    uh,
    cells0=parent_cells,
    cells1=smsh_cells
)

u_file_rotor = VTXWriter(mesh.comm, "u_field_rotor.bp", u_rotor, "BP4")
u_file_rotor.write(theta)


for theta in angles:
    set_mesh_rotation(submesh_rotor, rotor_ref_coords, theta, centre)

    idata = fem.create_interpolation_data(V_background, V_rotor, combined_cells, padding=1e-14)

    q_bg.x.array[:] = 0.0
    q_bg.interpolate_nonmatching(q_rotor, combined_cells, interpolation_data=idata)
    q_bg.x.scatter_forward()
    file.write(theta)


    problem.solve()
    uh.x.scatter_forward()
    u_file.write(theta)

    u_rotor.interpolate(
    uh,
    cells0=parent_cells,
    cells1=smsh_cells
    )
    u_file_rotor.write(theta)

    par_print(comm, f"|u|_L2 = {L2_norm(uh):.6e}")


file.close()
u_file.close()
u_file_rotor.close()
