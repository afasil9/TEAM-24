import numpy as np
import ufl
from basix.ufl import element
from dolfinx import default_scalar_type, fem
from dolfinx.fem.petsc import LinearProblem
from dolfinx.io import VTXWriter, XDMFFile
from dolfinx.mesh import create_submesh, meshtags
from mpi4py import MPI

from utils import convert_facet_tags, interpolate_by_tags, par_print

comm = MPI.COMM_WORLD

with XDMFFile(comm, "team24_horseshoe.xdmf", "r") as xdmf:
    mesh = xdmf.read_mesh()
    ct = xdmf.read_meshtags(mesh, name="Cell_markers")
    tdim = mesh.topology.dim
    fdim = tdim - 1
    mesh.topology.create_entities(fdim)
    ft = xdmf.read_meshtags(mesh, name="Facet_markers")

mesh.topology.create_connectivity(fdim, tdim)
mesh.topology.create_connectivity(tdim, tdim)
mesh.topology.create_connectivity(tdim - 1, 0)

par_print(comm, f"Number of cells: {mesh.topology.index_map(tdim).size_global}")

domains = {
    "air": 1,
    "coil1": 2,
    "coil2": 3,
    "rotor": 4,
    "stator": 5,
}

boundary = {
    "outer": 1,
    "symmetry": 2,
    "coil1_in": 3,
    "coil1_out": 4,
    "coil2_in": 5,
    "coil2_out": 6,
    "rotor_outer": 7,
    "stator_outer": 8,
}

degree = 1

target_tags = [domains["coil1"], domains["coil2"], domains["rotor"], domains["stator"]]
conductive_cells = np.unique(
    np.concatenate([ct.find(tag) for tag in target_tags])
).astype(np.int32)

submesh, submesh_to_domain = create_submesh(mesh, tdim, conductive_cells)[:2]
submesh.topology.create_connectivity(fdim, tdim)

conductive_ft = convert_facet_tags(submesh, submesh_to_domain, ft)

smsh_cell_imap = submesh.topology.index_map(tdim)
all_smsh_cells = np.arange(smsh_cell_imap.size_local, dtype=np.int32)
parent_cells = submesh_to_domain.sub_topology_to_topology(all_smsh_cells, inverse=False)
conductive_ct = meshtags(submesh, tdim, all_smsh_cells, ct.values[parent_cells])

k_copper = 400.0  # W/(m·K)
k_iron = 50.0  # W/(m·K)

k_values = {
    domains["coil1"]: fem.Constant(submesh, default_scalar_type(k_copper)),
    domains["coil2"]: fem.Constant(submesh, default_scalar_type(k_copper)),
    domains["rotor"]: fem.Constant(submesh, default_scalar_type(k_iron)),
    domains["stator"]: fem.Constant(submesh, default_scalar_type(k_iron)),
}

DG0 = fem.functionspace(submesh, ("DG", 0))
k = fem.Function(DG0)
interpolate_by_tags(k, k_values, conductive_ct)

VT = fem.functionspace(submesh, element("Lagrange", submesh.basix_cell(), degree))
T = ufl.TrialFunction(VT)
phi = ufl.TestFunction(VT)

dx = ufl.Measure("dx", domain=submesh, subdomain_data=conductive_ct)
ds = ufl.Measure("ds", domain=submesh, subdomain_data=conductive_ft)

x = ufl.SpatialCoordinate(submesh)
r2 = ufl.dot(x, x)
f = 1.0e6 * ufl.exp(-r2 / 0.01) + 1.0e4

a = ufl.inner(k * ufl.grad(T), ufl.grad(phi)) * dx
L = ufl.inner(f, phi) * dx

h_conv = fem.Constant(submesh, default_scalar_type(10.0))  # W/(m²·K)
T_amb = fem.Constant(submesh, default_scalar_type(293.15))  # K

for tag in (boundary["rotor_outer"], boundary["stator_outer"]):
    a += h_conv * T * phi * ds(tag)
    L += h_conv * T_amb * phi * ds(tag)

T_ref = 293.15  # K

outer_tags_heat = [
    boundary["coil1_in"],
    boundary["coil1_out"],
    boundary["coil2_in"],
    boundary["coil2_out"],
]
heat_boundary_facets = np.unique(
    np.concatenate([conductive_ft.find(tag) for tag in outer_tags_heat])
)
bdofs_T = fem.locate_dofs_topological(VT, fdim, heat_boundary_facets)
bc_T = fem.dirichletbc(default_scalar_type(T_ref), bdofs_T, VT)

problem = LinearProblem(
    a,
    L,
    bcs=[bc_T],
    petsc_options={
        "ksp_type": "cg",
        "pc_type": "hypre",
        "pc_hypre_type": "boomeramg",
    },
    petsc_options_prefix="heat_",
)
T_h = problem.solve()
T_h.name = "Temperature"

with VTXWriter(submesh.comm, "T_field.bp", T_h, "BP4") as T_file:
    T_file.write(0.0)

par_print(comm, "Heat solve done")
