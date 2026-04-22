# %%
import numpy as np
import ufl
from basix.ufl import element
from dolfinx import default_scalar_type, fem
from dolfinx.cpp.fem.petsc import discrete_gradient, interpolation_matrix
from dolfinx.fem import (
    Function,
    bcs_by_block,
    extract_function_spaces,
    form,
)
from dolfinx.fem.petsc import (
    apply_lifting,
    assemble_matrix,
    assemble_vector,
    set_bc,
)
from dolfinx.io import VTXWriter, XDMFFile
from dolfinx.mesh import create_submesh
from mpi4py import MPI
from petsc4py import PETSc
from ufl import (
    curl,
    grad,
)

from utils import (
    interpolate_by_tags,
    my_monitor,
    par_print,
)

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

sigma_air = fem.Constant(mesh, default_scalar_type(0.0))
sigma_copper = fem.Constant(mesh, default_scalar_type(5.96e7))

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

mu = 4e-7 * np.pi
nu_value = fem.Constant(mesh, default_scalar_type(1 / mu))

nu_values = {
    domains["air"]: nu_value,
    domains["coil1"]: nu_value,
    domains["coil2"]: nu_value,
    domains["rotor"]: nu_value,
    domains["stator"]: nu_value,
}

DG0 = fem.functionspace(mesh, ("DG", 0))  # Piecewise constant function space

sigma = fem.Function(DG0)
nu = fem.Function(DG0)

interpolate_by_tags(sigma, sigma_values, ct)
interpolate_by_tags(nu, nu_values, ct)


degree = 1

nedelec_elem = element("N1curl", mesh.basix_cell(), degree)
V = fem.functionspace(mesh, nedelec_elem)

lagrange_elem = element("Lagrange", mesh.basix_cell(), degree)
V1 = fem.functionspace(mesh, lagrange_elem)

# Dirichlet boundary conditions on V1
gdim = mesh.geometry.dim
facet_dim = gdim - 1

outer_boundaries = [
    boundary["outer"],
    boundary["symmetry"],
    boundary["coil1_out"],
    boundary["coil2_out"],
    boundary["coil1_in"],
    boundary["coil2_in"],
]
outer_boundary = np.unique(
    np.concatenate([ft.find(tag) for tag in outer_boundaries])
)


V_scale = 10.0


bdofs_outer = fem.locate_dofs_topological(
    V, entity_dim=facet_dim, entities=outer_boundary
)
outer_func = fem.Function(V)
outer_func.x.array[:] = 0.0
bc1 = fem.dirichletbc(outer_func, bdofs_outer)


coil1_in = ft.find(boundary["coil1_in"])
bdofs_coil1_in = fem.locate_dofs_topological(
    V1, entity_dim=facet_dim, entities=coil1_in
)
coil1_in_func = fem.Function(V1)
coil1_in_func.x.array[:] = 1.0
bc2 = fem.dirichletbc(coil1_in_func, bdofs_coil1_in)

coil1_out = ft.find(boundary["coil1_out"])
bdofs_coil1_out = fem.locate_dofs_topological(
    V1, entity_dim=facet_dim, entities=coil1_out
)
coil1_out_func = fem.Function(V1)
coil1_out_func.x.array[:] = 0.0
bc3 = fem.dirichletbc(coil1_out_func, bdofs_coil1_out)

coil2_in = ft.find(boundary["coil2_in"])
bdofs_coil2_in = fem.locate_dofs_topological(
    V1, entity_dim=facet_dim, entities=coil2_in
)
coil2_in_func = fem.Function(V1)
coil2_in_func.x.array[:] = -1.0
bc4 = fem.dirichletbc(coil2_in_func, bdofs_coil2_in)

coil2_out = ft.find(boundary["coil2_out"])
bdofs_coil2_out = fem.locate_dofs_topological(
    V1, entity_dim=facet_dim, entities=coil2_out
)
coil2_out_func = fem.Function(V1)
coil2_out_func.x.array[:] = 0.0
bc5 = fem.dirichletbc(coil2_out_func, bdofs_coil2_out)

bc = [bc1, bc2, bc3, bc4, bc5]

u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

u1 = ufl.TrialFunction(V1)
v1 = ufl.TestFunction(V1)

u_n = fem.Function(V)
u_n1 = fem.Function(V1)

dx = ufl.Measure("dx", domain=mesh, subdomain_data=ct)

whole = (1, 2, 3, 4, 5)
omega_c = (2, 3, 4, 5)

# whole = (domains["air"], domains["coil1"], domains["coil2"], domains["rotor"], domains["stator"])
# omega_c = (domains["coil1"], domains["coil2"], domains["rotor"], domains["stator"])


# %%

a00 = ufl.inner(nu * curl(u), curl(v)) * dx(whole)
a01 = ufl.inner(sigma * grad(u1), v) * dx(omega_c)
a11 = ufl.inner(sigma * ufl.grad(u1), ufl.grad(v1)) * dx(omega_c)

zero_vec = fem.Constant(mesh, PETSc.ScalarType((0.0, 0.0, 0.0)))
L0 = ufl.inner(zero_vec, v) * dx(whole)
L1 = fem.Constant(mesh, PETSc.ScalarType(0.0)) * v1 * dx(omega_c)

a = form([[a00, a01], [None, a11]])

A_mat = assemble_matrix(a, bcs=bc)
A_mat.assemble()

L = form([L0, L1])

b = assemble_vector(L)
bcs1 = bcs_by_block(extract_function_spaces(a, 1), bc)
apply_lifting(b, a, bcs=bcs1)
b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
bcs0 = bcs_by_block(extract_function_spaces(L), bc)
set_bc(b, bcs0)

par_print(comm, "Assembling system matrix...")


a_p = form([[a00, None], [None, a11]])

P = assemble_matrix(a_p, bcs=bc)
P.assemble()

# Create functions to split A, S

offset = V.dofmap.index_map.size_local * V.dofmap.index_map_bs

u_map = V.dofmap.index_map
u1_map = V1.dofmap.index_map

offset_u = u_map.local_range[0] * V.dofmap.index_map_bs + u1_map.local_range[0]
offset_u1 = offset_u + u_map.size_local * V.dofmap.index_map_bs

is_u = PETSc.IS().createStride(
    u_map.size_local * V.dofmap.index_map_bs, offset_u, 1, comm=mesh.comm
)
is_u1 = PETSc.IS().createStride(u1_map.size_local, offset_u1, 1, comm=mesh.comm)


ksp = PETSc.KSP().create(mesh.comm)
ksp.setOperators(A_mat, P)
ksp.setType("gmres")
ksp.setTolerances(rtol=1e-10, atol=1e-50, max_it=100)
ksp.setNormType(PETSc.KSP.NormType.PRECONDITIONED)

pc = ksp.getPC()
pc.setType("fieldsplit")
pc.setFieldSplitType(PETSc.PC.CompositeType.MULTIPLICATIVE)
pc.setFieldSplitIS(("u", is_u), ("u1", is_u1))

ksp.setUp()

ksp_u, ksp_u1 = pc.getFieldSplitSubKSP()

ksp_u.setType("preonly")
ksp_u.getPC().setType("hypre")
ksp_u.getPC().setHYPREType("ams")

W = fem.functionspace(mesh, ("Lagrange", degree))
G = discrete_gradient(W._cpp_object, V._cpp_object)
G.assemble()
ksp_u.getPC().setHYPREDiscreteGradient(G)

if degree == 1:
    cvec_0 = Function(V)
    cvec_0.interpolate(
        lambda x: np.vstack(
            (np.ones_like(x[0]), np.zeros_like(x[0]), np.zeros_like(x[0]))
        )
    )
    cvec_1 = Function(V)
    cvec_1.interpolate(
        lambda x: np.vstack(
            (np.zeros_like(x[0]), np.ones_like(x[0]), np.zeros_like(x[0]))
        )
    )
    cvec_2 = Function(V)
    cvec_2.interpolate(
        lambda x: np.vstack(
            (np.zeros_like(x[0]), np.zeros_like(x[0]), np.ones_like(x[0]))
        )
    )
    ksp_u.getPC().setHYPRESetEdgeConstantVectors(
        cvec_0.x.petsc_vec, cvec_1.x.petsc_vec, cvec_2.x.petsc_vec
    )

else:
    shape = (mesh.geometry.dim,)
    Q = fem.functionspace(mesh, ("Lagrange", degree, shape))
    Pi = interpolation_matrix(Q._cpp_object, V._cpp_object)
    Pi.assemble()
    ksp_u.getPC().setHYPRESetInterpolations(dim=mesh.geometry.dim, ND_Pi_Full=Pi)

ksp_u.getPC().setHYPRESetBetaPoissonMatrix(None)

opts = PETSc.Options()
opts[f"{ksp_u.prefix}pc_hypre_ams_cycle_type"] = 13
opts[f"{ksp_u.prefix}pc_hypre_ams_tol"] = 0
opts[f"{ksp_u.prefix}pc_hypre_ams_max_iter"] = 1
opts[f"{ksp_u.prefix}pc_hypre_ams_amg_beta_theta"] = 0.25
opts[f"{ksp_u.prefix}pc_hypre_ams_print_level"] = 1
opts[f"{ksp_u.prefix}pc_hypre_ams_amg_alpha_options"] = "10,1,6,6,4"
opts[f"{ksp_u.prefix}pc_hypre_ams_amg_beta_options"] = "10,1,6,6,4"
opts[f"{ksp_u.prefix}pc_hypre_ams_relax_type"] = 2
opts[f"{ksp_u.prefix}pc_hypre_ams_relax_weight"] = 1.0
opts[f"{ksp_u.prefix}pc_hypre_ams_relax_times"] = 1
opts[f"{ksp_u.prefix}pc_hypre_ams_omega"] = 1.0

ksp_u.setFromOptions()

ksp_u1.setType("preonly")
ksp_u1.getPC().setType("hypre")
ksp_u1.getPC().setHYPREType("boomeramg")

ksp_u1.setFromOptions()

ksp.setUp()
ksp_u.getPC().setUp()
ksp_u1.getPC().setUp()

ksp.setMonitor(my_monitor)

pc.setUp()
ksp.setUp()

#%%
sol = A_mat.createVecRight()

par_print(comm, "about to solve")
ksp.solve(b, sol)

reason = ksp.getConvergedReason()
print(f"KSP converged with reason {reason}")


uh, uh1 = Function(V), Function(V1)
offset = V.dofmap.index_map.size_local * V.dofmap.index_map_bs

uh.x.array[:offset] = sol.array_r[:offset]
uh1.x.array[: (len(sol.array_r) - offset)] = sol.array_r[offset:]

uh.x.scatter_forward()
uh1.x.scatter_forward()

u_n.x.array[:] = uh.x.array
u_n1.x.array[:] = uh1.x.array

u_n.x.scatter_forward()
u_n1.x.scatter_forward()


t = 0.0

vector_vis = fem.functionspace(
    mesh, ("Discontinuous Lagrange", degree, (mesh.geometry.dim,))
)

B = ufl.curl(u_n)
B_vis = Function(vector_vis)
B_file = VTXWriter(mesh.comm, "B_field.bp", B_vis, "BP4")
Bexpr = fem.Expression(B, vector_vis.element.interpolation_points)
B_vis.interpolate(Bexpr)
B_file.write(t)
B_file.close()

E = -ufl.grad(u_n1)
E_vis = fem.Function(vector_vis)
Eexpr = fem.Expression(E, vector_vis.element.interpolation_points)
E_vis.interpolate(Eexpr)


J = sigma * E
J_vis = fem.Function(vector_vis)
Jexpr = fem.Expression(J, vector_vis.element.interpolation_points)
J_vis.interpolate(Jexpr)

target_tags = [domains["coil1"], domains["coil2"]]
cell_lists = [ct.find(tag) for tag in target_tags]
target_cells = np.unique(np.concatenate(cell_lists)).astype(np.int32)

submesh, subdomain_motor_to_domain = create_submesh(mesh, tdim, target_cells)[:2]
smsh_cell_imap = submesh.topology.index_map(tdim)
smsh_cells = np.arange(smsh_cell_imap.size_local + smsh_cell_imap.num_ghosts)
parent_cells = subdomain_motor_to_domain.sub_topology_to_topology(
    smsh_cells, inverse=False
)

# Output

u_n1_submesh = Function(fem.functionspace(submesh, ("DG", degree)))
u_n1_submesh.interpolate(u_n1, cells0=parent_cells, cells1=smsh_cells)
u_n1_file = VTXWriter(mesh.comm, "V_field.bp", u_n1_submesh, "BP4")
u_n1_file.write(t)
u_n1_file.close()


DG_submesh_vis = fem.functionspace(
    submesh, ("Discontinuous Lagrange", degree, (submesh.geometry.dim,))
)


B_vis_submesh = fem.Function(DG_submesh_vis)
B_vis_submesh.interpolate(B_vis, cells0=parent_cells, cells1=smsh_cells)
B_file_submesh = VTXWriter(mesh.comm, "B_submesh.bp", B_vis_submesh, "BP4")
B_file_submesh.write(t)
B_file_submesh.close()

E_vis_submesh = fem.Function(DG_submesh_vis)
E_vis_submesh.interpolate(E_vis, cells0=parent_cells, cells1=smsh_cells)
E_file_submesh = VTXWriter(mesh.comm, "E_submesh.bp", E_vis_submesh, "BP4")
E_file_submesh.write(t)
E_file_submesh.close()


J_vis_submesh = fem.Function(DG_submesh_vis)
J_vis_submesh.interpolate(J_vis, cells0=parent_cells, cells1=smsh_cells)
J_file_submesh = VTXWriter(mesh.comm, "J_submesh.bp", J_vis_submesh, "BP4")
J_file_submesh.write(t)
J_file_submesh.close()
