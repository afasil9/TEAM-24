#%%
from mpi4py import MPI
import numpy as np
from petsc4py import PETSc
import ufl
from dolfinx import fem, default_scalar_type
from dolfinx.io import XDMFFile, VTXWriter
from dolfinx.mesh import GhostMode, create_submesh
from basix.ufl import element
from dolfinx.fem import form, Function
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, apply_lifting, set_bc
from utils import L2_norm, my_monitor, interpolate_by_tags, par_print

comm = MPI.COMM_WORLD

with XDMFFile(comm, "team24_horseshoe.xdmf", "r") as xdmf:
    mesh = xdmf.read_mesh()
    ct   = xdmf.read_meshtags(mesh, name="Cell_markers")
    tdim = mesh.topology.dim
    fdim = tdim - 1
    mesh.topology.create_entities(fdim)
    ft   = xdmf.read_meshtags(mesh, name="Facet_markers")

mesh.topology.create_connectivity(fdim, tdim)
mesh.topology.create_connectivity(tdim, tdim)
mesh.topology.create_connectivity(tdim - 1, 0)

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

DG0 = fem.functionspace(mesh, ("DG", 0))  # Piecewise constant function space

sigma = fem.Function(DG0)

sigma_air = fem.Constant(mesh, default_scalar_type(0.0))
sigma_copper = fem.Constant(mesh, default_scalar_type(5.96e7))
sigma_stator = fem.Constant(mesh, default_scalar_type(1e5))
sigma_rotor = fem.Constant(mesh, default_scalar_type(1e5))

sigma_values = {
    domains["air"]: sigma_air,
    domains["coil1"]: sigma_copper,
    domains["coil2"]: sigma_copper,
    domains["rotor"]: sigma_rotor,
    domains["stator"]: sigma_stator,
}

interpolate_by_tags(sigma, sigma_values, ct)

degree = 1

# Scalar CG space for u_n1
lagrange_elem = element("Lagrange", mesh.basix_cell(), degree)
V1 = fem.functionspace(mesh, lagrange_elem)

# Dirichlet boundary conditions on V1
gdim = mesh.geometry.dim
facet_dim = gdim - 1


outer_boundary_facets = ft.find(boundary["outer"])

bdofs1 = fem.locate_dofs_topological(
    V1, entity_dim=facet_dim, entities=outer_boundary_facets
)
u_bc_V1 = fem.Function(V1)
u_bc_V1.x.array[:] = 0.0
bc1 = fem.dirichletbc(u_bc_V1, bdofs1)

# Upper (facet tag 3) -> 10.0
coil1_in = ft.find(boundary["coil1_out"])
bdofs_coil1_in = fem.locate_dofs_topological(V1, entity_dim=facet_dim, entities=coil1_in)
coil1_in_func = fem.Function(V1)
coil1_in_func.x.array[:] = 10.0
bc2 = fem.dirichletbc(coil1_in_func, bdofs_coil1_in)

coil1_out = ft.find(boundary["coil1_in"])
bdofs_coil1_out = fem.locate_dofs_topological(V1, entity_dim=facet_dim, entities=coil1_out)
coil1_out_func = fem.Function(V1)
coil1_out_func.x.array[:] = 0.0
bc3 = fem.dirichletbc(coil1_out_func, bdofs_coil1_out)

coil2_in = ft.find(boundary["coil2_out"])
bdofs_coil2_in = fem.locate_dofs_topological(V1, entity_dim=facet_dim, entities=coil2_in)
coil2_in_func = fem.Function(V1)
coil2_in_func.x.array[:] = -10.0
bc4 = fem.dirichletbc(coil2_in_func, bdofs_coil2_in)

coil2_out = ft.find(boundary["coil2_in"])
bdofs_coil2_out = fem.locate_dofs_topological(V1, entity_dim=facet_dim, entities=coil2_out)
coil2_out_func = fem.Function(V1)
coil2_out_func.x.array[:] = 0.0
bc5 = fem.dirichletbc(coil2_out_func, bdofs_coil2_out)

bcs = [bc1, bc2, bc3, bc4, bc5]

u1 = ufl.TrialFunction(V1)
v1 = ufl.TestFunction(V1)
dx = ufl.Measure("dx", domain=mesh, subdomain_data=ct)


lhs = ufl.inner(sigma * ufl.grad(u1), ufl.grad(v1)) * dx
rhs = fem.Constant(mesh, PETSc.ScalarType(0.0)) * v1 * dx

a = form(lhs)
L = form(rhs)


# Assemble system
A = assemble_matrix(a, bcs=bcs)
A.assemble()
b = assemble_vector(L)
apply_lifting(b, [a], bcs=[bcs])
b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
set_bc(b, bcs)


# Solve with PETSc
ksp = PETSc.KSP().create(mesh.comm)
ksp.setOperators(A)

ksp.setType("gmres")
ksp.setTolerances(rtol=1e-10, atol=1e-50, max_it=1000)

pc = ksp.getPC()
pc.setType("hypre")
pc.setHYPREType("boomeramg")

ksp.setMonitor(my_monitor)

pc.setUp()
ksp.setUp()

u_n1 = fem.Function(V1)
ksp.solve(b, u_n1.x.petsc_vec)
u_n1.x.scatter_forward()

reason = ksp.getConvergedReason()
print(f"KSP converged with reason {reason}")

#
# Output
t = 0.0
u_n1_file = VTXWriter(mesh.comm, "V_field.bp", u_n1, "BP4")
u_n1_file.write(t)
u_n1_file.close()

comm = MPI.COMM_WORLD
par_print(comm, f"L2 norm of u_n1 field {L2_norm(u_n1)}")

res = b - A * u_n1.x.petsc_vec



vector_vis = fem.functionspace(
    mesh, ("Discontinuous Lagrange", degree, (mesh.geometry.dim,))
)

E = -ufl.grad(u_n1)
E_vis = fem.Function(vector_vis)
Eexpr = fem.Expression(E, vector_vis.element.interpolation_points)
E_vis.interpolate(Eexpr)

J = sigma * E
J_vis = fem.Function(vector_vis)
Jexpr = fem.Expression(J, vector_vis.element.interpolation_points)
J_vis.interpolate(Jexpr)

target_tags = [tag for name, tag in domains.items() if name != "air"]
cell_lists = [ct.find(tag) for tag in target_tags]
target_cells = np.unique(np.concatenate(cell_lists)).astype(np.int32)

submesh, subdomain_motor_to_domain = create_submesh(mesh, tdim, target_cells)[:2]
smsh_cell_imap = submesh.topology.index_map(tdim)
smsh_cells = np.arange(smsh_cell_imap.size_local + smsh_cell_imap.num_ghosts)
parent_cells = subdomain_motor_to_domain.sub_topology_to_topology(smsh_cells, inverse=False)

# Output

u_n1_submesh = Function(fem.functionspace(submesh, ("DG", degree)))
u_n1_submesh.interpolate(u_n1, cells0=parent_cells, cells1=smsh_cells)
u_n1_file = VTXWriter(mesh.comm, "V_field.bp", u_n1_submesh, "BP4")
u_n1_file.write(t)
u_n1_file.close()


DG_submesh_vis = fem.functionspace(
    submesh, ("Discontinuous Lagrange", degree, (submesh.geometry.dim,))
)


E_vis_submesh = fem.Function(DG_submesh_vis)
E_vis_submesh.interpolate(
    E_vis, cells0=parent_cells, cells1=smsh_cells
)
E_file_submesh = VTXWriter(mesh.comm, "E_field.bp", E_vis_submesh, "BP4")
E_file_submesh.write(t)
E_file_submesh.close()


J_vis_submesh = fem.Function(DG_submesh_vis)
J_vis_submesh.interpolate(
    J_vis, cells0=parent_cells, cells1=smsh_cells
)
J_file_submesh = VTXWriter(mesh.comm, "J_field.bp", J_vis_submesh, "BP4")
J_file_submesh.write(t)
J_file_submesh.close()





# %%
