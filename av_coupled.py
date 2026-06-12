#%%
from mpi4py import MPI
from dolfinx import fem
from dolfinx.fem import (
    Function,
    form,
)
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, apply_lifting, set_bc
import numpy as np
from ufl import curl, inner, grad
from basix.ufl import element
from petsc4py import PETSc
from dolfinx.cpp.fem.petsc import discrete_gradient, interpolation_matrix
from utils import L2_norm, par_print
import ufl
from dolfinx.fem import (
    functionspace,
    bcs_by_block,
    extract_function_spaces,
)
from utils import convert_facet_tags
from dolfinx.mesh import create_submesh
from dolfinx.io import XDMFFile
from utils import interpolate_by_tags
from dolfinx import default_scalar_type
from dolfinx.io import VTXWriter
from dolfinx.fem import assemble_scalar
import json
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, apply_lifting, set_bc, LinearProblem

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

ti = 0.0  # Start time
T = 0.1  # End time

frequency = 50.0
steps_per_period = 100
d_t = 1.0 / (frequency * steps_per_period)

n_cycles_warmup = 10
n_cycles_measure = 1
num_steps = int((n_cycles_warmup + n_cycles_measure) / (frequency * d_t))

warmup_steps = n_cycles_warmup * steps_per_period


dt = fem.Constant(mesh, d_t)
t = fem.Constant(mesh, ti)


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
mu_r_iron = 1000.0

nu_value_air = fem.Constant(mesh, default_scalar_type(1.0 / mu))
nu_value_iron = fem.Constant(mesh, default_scalar_type(1.0 / (mu_r_iron * mu)))

nu_values = {
    domains["air"]:    nu_value_air,
    domains["coil1"]:  nu_value_air,
    domains["coil2"]:  nu_value_air,
    domains["rotor"]:  nu_value_iron,
    domains["stator"]: nu_value_iron,
}

DG0 = fem.functionspace(mesh, ("DG", 0))  # Piecewise constant function space

sigma = fem.Function(DG0)
nu = fem.Function(DG0)

interpolate_by_tags(sigma, sigma_values, ct)
interpolate_by_tags(nu, nu_values, ct)


target_tags = [domains["coil1"], domains["coil2"], domains["rotor"], domains["stator"]]
cell_lists = [ct.find(tag) for tag in target_tags]
conductive_cells = np.unique(np.concatenate(cell_lists)).astype(np.int32)

submesh_conductive, subdomain_conductive_to_domain = create_submesh(mesh, tdim, conductive_cells)[:2]
entity_maps = [subdomain_conductive_to_domain]

nedelec_elem = element("N1curl", mesh.basix_cell(), degree)
V = fem.functionspace(mesh, nedelec_elem)
V_submesh = functionspace(submesh_conductive, nedelec_elem)

lagrange_elem = element("Lagrange", submesh_conductive.basix_cell(), degree)
V1 = fem.functionspace(submesh_conductive, lagrange_elem)


# Boundary conditions

omega = 2.0 * np.pi * frequency
V_in = 10.0


outer_boundaries = [boundary["outer"], boundary["symmetry"], boundary["coil1_out"], boundary["coil2_out"], boundary["coil1_in"], boundary["coil2_in"]]
outer_boundary_tags = np.unique(np.concatenate([ft.find(tag) for tag in outer_boundaries]))

bdofs_outer = fem.locate_dofs_topological(V, entity_dim=fdim, entities=outer_boundary_tags)
outer_func = fem.Function(V)
outer_func.x.array[:] = 0.0
bc1 = fem.dirichletbc(outer_func, bdofs_outer)

conductive_ft = convert_facet_tags(submesh_conductive, subdomain_conductive_to_domain, ft)
submesh_conductive.topology.create_connectivity(fdim, tdim)

coil1_in = conductive_ft.find(boundary["coil1_in"])
bdofs_coil1_in = fem.locate_dofs_topological(V1, entity_dim=fdim, entities=coil1_in)
coil1_in_func = fem.Function(V1)
# coil1_in_func.x.array[:] = 1.0
high_expr = fem.Expression(
    V_in * ufl.sin(omega * t), V1.element.interpolation_points
)
coil1_in_func.interpolate(high_expr)
bc2 = fem.dirichletbc(coil1_in_func, bdofs_coil1_in)

coil1_out = conductive_ft.find(boundary["coil1_out"])
bdofs_coil1_out = fem.locate_dofs_topological(V1, entity_dim=fdim, entities=coil1_out)
coil1_out_func = fem.Function(V1)
coil1_out_func.x.array[:] = 0.0
bc3 = fem.dirichletbc(coil1_out_func, bdofs_coil1_out)

coil2_in = conductive_ft.find(boundary["coil2_in"])
bdofs_coil2_in = fem.locate_dofs_topological(V1, entity_dim=fdim, entities=coil2_in)
coil2_in_func = fem.Function(V1)
# coil2_in_func.x.array[:] = -1.0
low_expr = fem.Expression(
    -V_in * ufl.sin(omega * t), V1.element.interpolation_points
)
coil2_in_func.interpolate(low_expr)
bc4 = fem.dirichletbc(coil2_in_func, bdofs_coil2_in)

coil2_out = conductive_ft.find(boundary["coil2_out"])
bdofs_coil2_out = fem.locate_dofs_topological(V1, entity_dim=fdim, entities=coil2_out)
coil2_out_func = fem.Function(V1)
coil2_out_func.x.array[:] = 0.0
bc5 = fem.dirichletbc(coil2_out_func, bdofs_coil2_out)

bcs = [bc1, bc2, bc3, bc4, bc5]


u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

u1 = ufl.TrialFunction(V1)
v1 = ufl.TestFunction(V1)

u_n = fem.Function(V)
u_n1 = fem.Function(V1)
u_n_prev  = fem.Function(V)
u_n_submesh = fem.Function(V_submesh)


u_n_prev.x.array[:] = u_n.x.array[:]
u_n_prev.x.scatter_forward()

u_n_submesh_prev = fem.Function(V_submesh)
u_n_submesh_prev.x.array[:] = u_n_submesh.x.array[:]
u_n_submesh_prev.x.scatter_forward()


dx = ufl.Measure("dx", domain=mesh, subdomain_data=ct)

whole = (1,2,3,4,5)
omega_c = (2,3,4,5)

a00 = dt * inner(nu * curl(u), curl(v)) * dx(whole) + inner(sigma * u, v) * dx(whole)

a01 = dt * inner(sigma * grad(u1), v) * dx(omega_c)
a10 = inner(sigma * grad(v1), u) * dx(omega_c)

a11 = dt * inner(sigma * grad(u1), grad(v1)) * dx(omega_c)

L0 = inner(sigma * u_n, v) * dx(whole)
L1 = inner(grad(v1), sigma * u_n) * dx(omega_c)

a = form([[a00, a01], [a10, a11]], entity_maps=entity_maps)
L = form([L0, L1], entity_maps=entity_maps)

# Solver steps

A_mat = assemble_matrix(a, bcs=bcs)
A_mat.assemble()

b = assemble_vector(L)
bcs1 = bcs_by_block(extract_function_spaces(a, 1), bcs)
apply_lifting(b, a, bcs=bcs1)
b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
bcs0 = bcs_by_block(extract_function_spaces(L), bcs)
set_bc(b, bcs0)

a_p = form([[a00, None], [None, a11]], entity_maps=entity_maps)
P = assemble_matrix(a_p, bcs=bcs)
P.assemble()

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
ksp.setGMRESRestart(200)
ksp.setTolerances(rtol=1e-7, atol=1e-7, max_it=200)
ksp.setNormType(PETSc.KSP.NormType.UNPRECONDITIONED)
ksp.getPC().setType("fieldsplit")
ksp.getPC().setFieldSplitType(PETSc.PC.CompositeType.ADDITIVE)
ksp.getPC().setFieldSplitIS(("u", is_u), ("u1", is_u1))
ksp_u, ksp_u1 = ksp.getPC().getFieldSplitSubKSP()

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


ksp.setOptionsPrefix("main_") # Add this line

opts = PETSc.Options()
# opts[f"{ksp.getOptionsPrefix()}ksp_monitor_true_residual"] = None
opts[f"{ksp_u.prefix}pc_hypre_ams_cycle_type"] = 1
opts[f"{ksp_u.prefix}pc_hypre_ams_tol"] = 0
opts[f"{ksp_u.prefix}pc_hypre_ams_max_iter"] = 4
opts[f"{ksp_u.prefix}pc_hypre_ams_amg_beta_theta"] = 0.25
opts[f"{ksp_u.prefix}pc_hypre_ams_print_level"] = 0
opts[f"{ksp_u.prefix}pc_hypre_ams_amg_alpha_options"] = "10,2,6,6,6"
opts[f"{ksp_u.prefix}pc_hypre_ams_amg_beta_options"] = "10,1,6,6,4"
opts[f"{ksp_u.prefix}pc_hypre_ams_relax_type"] = 8
opts[f"{ksp_u.prefix}pc_hypre_ams_relax_weight"] = 1.0
opts[f"{ksp_u.prefix}pc_hypre_ams_relax_times"] = 2
opts[f"{ksp_u.prefix}pc_hypre_ams_omega"] = 1.0
opts[f"{ksp_u.prefix}pc_hypre_ams_projection_frequency"] = 100000000

V_interior = fem.functionspace(mesh, ("CG", degree))
interior_nodes_array = fem.Function(V_interior)

interior_nodes_array.x.array[:] = 1.0
interior_nodes_array.x.scatter_forward()

dofmap = W.dofmap
num_dofs_per_cell = dofmap.dof_layout.num_dofs
cell_dofs = dofmap.list.reshape(-1, num_dofs_per_cell)

tags = omega_c
tagged_cells = np.unique(np.concatenate([ct.find(tag) for tag in tags]))

tagged_cell_dofs = cell_dofs[tagged_cells].flatten()
unique_dofs = np.unique(tagged_cell_dofs)

interior_nodes_array.x.array[unique_dofs] = 0.0
interior_nodes_array.x.scatter_forward()

ksp_u.getPC().setHYPREAMSSetInteriorNodes(interior_nodes_array.x.petsc_vec)

ksp_u.setFromOptions()

ksp_u1.setType("preonly")
ksp_u1.getPC().setType("hypre")
ksp_u1.getPC().setHYPREType("boomeramg")

ksp_u1.setFromOptions()

ksp.setFromOptions()
ksp.setUp()
ksp_u.getPC().setUp()
ksp_u1.getPC().setUp()

sol = A_mat.createVecRight()


par_print(mesh.comm, "about to solve")
ksp.solve(b, sol)


reason = ksp.getConvergedReason()
par_print(mesh.comm, f"KSP converged with reason {reason}")

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

par_print(mesh.comm, f"Residual norm: {ksp.getResidualNorm()}")

# Output results for visualization

vector_vis = fem.functionspace(
    mesh, ("Discontinuous Lagrange", degree, (mesh.geometry.dim,))
)

B = ufl.curl(u_n)
B_vis = Function(vector_vis)
B_file = VTXWriter(mesh.comm, "B_field_mixed.bp", B_vis, "BP4")
Bexpr = fem.Expression(B, vector_vis.element.interpolation_points)
B_vis.interpolate(Bexpr)
B_file.write(t)

u_n1_file = VTXWriter(mesh.comm, "V_field_mixed.bp", u_n1, "BP4")
u_n1_file.write(t)

DG_submesh_vis = fem.functionspace(
    submesh_conductive, ("Discontinuous Lagrange", degree, (submesh_conductive.geometry.dim,))
)

smsh_cell_imap = submesh_conductive.topology.index_map(tdim)
smsh_cells = np.arange(smsh_cell_imap.size_local + smsh_cell_imap.num_ghosts)
parent_cells = subdomain_conductive_to_domain.sub_topology_to_topology(smsh_cells, inverse=False)

u_n_submesh.interpolate(u_n, cells0=parent_cells, cells1=smsh_cells)
dt_submesh = fem.Constant(submesh_conductive, d_t)
da_dt_submesh = (u_n_submesh - u_n_submesh_prev) / dt_submesh


# B on submesh
B_vis_submesh = fem.Function(DG_submesh_vis)
B_vis_submesh.interpolate(
    B_vis, cells0=parent_cells, cells1=smsh_cells
)
B_file_submesh = VTXWriter(mesh.comm, "B_submesh.bp", B_vis_submesh, "BP4")
B_file_submesh.write(t)

E = -grad(u_n1) - da_dt_submesh


DG_0_submesh_vis = fem.functionspace(submesh_conductive, ("DG", 0))
sigma_submesh = fem.Function(DG_0_submesh_vis)
sigma_submesh.interpolate(
    sigma, cells0=parent_cells, cells1=smsh_cells
)

J = sigma_submesh * E
J_vis = fem.Function(DG_submesh_vis)
Jexpr = fem.Expression(J, DG_submesh_vis.element.interpolation_points)
J_vis.interpolate(Jexpr)
J_file = VTXWriter(mesh.comm, "J_field_mixed.bp", J_vis, "BP4")
J_file.write(t)


diagnostics = []
diag_path = "diagnostics_av_mixed.json"


# Solving for heat

rho_air = fem.Constant(mesh, default_scalar_type(1.225))
rho_copper = fem.Constant(mesh, default_scalar_type(8960.0))
rho_stator = fem.Constant(mesh, default_scalar_type(7870.0))
rho_rotor = fem.Constant(mesh, default_scalar_type(7870.0))

rho_values = {
    domains["air"]: rho_air,
    domains["coil1"]: rho_copper,
    domains["coil2"]: rho_copper,
    domains["rotor"]: rho_rotor,
    domains["stator"]: rho_stator,
}

k_copper= 385.0 # W/(m·K) – thermal conductivity
k_iron = 30.0  # W/(m·K)
k_air = 0.0257     # W/(m·K)

k_values = {
    domains["air"]: k_air,
    domains["coil1"]: k_copper,
    domains["coil2"]: k_copper,
    domains["rotor"]: k_iron,
    domains["stator"]: k_iron,
}

cp_copper = fem.Constant(mesh, default_scalar_type(385.0))
cp_iron   = fem.Constant(mesh, default_scalar_type(460.0))
cp_air    = fem.Constant(mesh, default_scalar_type(1005.0))
cp_values = {
    domains["air"]: cp_air,
    domains["coil1"]: cp_copper,
    domains["coil2"]: cp_copper,
    domains["rotor"]: cp_iron,
    domains["stator"]: cp_iron,
}

theta_amb    = 293.15 

rho = fem.Function(DG0)
kappa = fem.Function(DG0)
cp = fem.Function(DG0)

interpolate_by_tags(rho, rho_values, ct)
interpolate_by_tags(kappa, k_values, ct)
interpolate_by_tags(cp, cp_values, ct)

Q_expr = sigma_submesh * ufl.dot(E, E)

V_thermal = fem.functionspace(submesh_conductive, lagrange_elem)

# Constants
dt_th = fem.Constant(submesh_conductive, default_scalar_type(d_t))
h_stator = fem.Constant(submesh_conductive, default_scalar_type(500.0))
h_rotor = fem.Constant(submesh_conductive, default_scalar_type(25.0))
h_air = fem.Constant(submesh_conductive, default_scalar_type(10.0))

theta_amb_c = fem.Constant(submesh_conductive, default_scalar_type(theta_amb))

# Measures
dx_sm = ufl.Measure("dx", domain=submesh_conductive)
ds_sm = ufl.Measure("ds", domain=submesh_conductive,subdomain_data=conductive_ft)


# Trial and test functions for thermal field
theta = ufl.TrialFunction(V_thermal) 
phi = ufl.TestFunction(V_thermal)
theta_n = fem.Function(V_thermal)

theta_n_file = VTXWriter(mesh.comm, "theta_mixed.bp", theta_n, "BP4")
theta_n_file.write(t)


def solve_thermal(Q_expr, theta_n):
    F = dt_th * kappa * inner(ufl.grad(theta), ufl.grad(phi)) * dx_sm
    F += inner(rho * cp * theta, phi) * dx_sm
    F -= inner(rho * cp * theta_n, phi) * dx_sm
    F -= dt_th * inner(Q_expr, phi) * dx_sm
    F -= dt_th * h_rotor * inner(theta - theta_amb_c, phi) * ds_sm(boundary["rotor_outer"])
    F -= dt_th * h_stator * inner(theta - theta_amb_c, phi) * ds_sm(boundary["stator_outer"])
    F -= dt_th * h_air * inner(theta - theta_amb_c, phi) * ds_sm


    a_th = ufl.lhs(F)
    L_th = ufl.rhs(F)

    problem_T = LinearProblem(
        a_th, L_th, bcs=[],
        petsc_options={
            "ksp_type": "cg",
            "pc_type": "hypre",
            "pc_hypre_type": "boomeramg",
            # "ksp_monitor_true_residual": None,
            "ksp_converged_reason": None,
            "ksp_rtol": 1e-8,
        },
        petsc_options_prefix="heat_",
        entity_maps=entity_maps
    )

    # problem_T.solver.setMonitor(
    #     lambda ksp, it, rnorm: par_print(mesh.comm, f"[Heat] it {it}: rnorm = {rnorm:.3e}")
    # )
    T_h = problem_T.solve()

    theta_n.x.array[:] = T_h.x.array
    theta_n.x.scatter_forward()

    return theta_n


#%%
last_steps = 100

def gscalar(expr, entity_maps=None):
    return comm.allreduce(assemble_scalar(form(expr, entity_maps=entity_maps)), op=MPI.SUM)
dx_c = dx(tuple(target_tags))

for n in range(num_steps):

    t.value += d_t

    par_print(comm, "\n")
    par_print(comm, f"Time step {n+1}: t = {t.value}")

    ksp_u.getPC().HYPREAMSResetSolveCounter()

    u_n_prev.x.array[:] = u_n.x.array[:]
    u_n_submesh_prev.x.array[:] = u_n_submesh.x.array[:]

    uh.x.array[:] = 0
    uh1.x.array[:] = 0

    high_expr = fem.Expression(
    V_in * ufl.sin(omega * t), V1.element.interpolation_points)

    low_expr = fem.Expression(
    -V_in * ufl.sin(omega * t), V1.element.interpolation_points
    )

    coil1_in_func.interpolate(high_expr)
    coil2_in_func.interpolate(low_expr)


    bcs = [bc1, bc2, bc3, bc4, bc5]

    b = assemble_vector(L)
    bcs1 = bcs_by_block(extract_function_spaces(a, 1), bcs)
    apply_lifting(b, a, bcs=bcs1)
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    bcs0 = bcs_by_block(extract_function_spaces(L), bcs)
    set_bc(b, bcs0)

    sol = A_mat.createVecRight()

    ksp.solve(b, sol)

    uh.x.array[:offset] = sol.array_r[:offset]
    uh1.x.array[:(len(sol.array_r) - offset)] = sol.array_r[offset:]

    uh.x.scatter_forward()
    uh1.x.scatter_forward()

    u_n.x.array[:] = uh.x.array
    u_n1.x.array[:] = uh1.x.array

    u_n.x.scatter_forward()
    u_n1.x.scatter_forward()

    reason = ksp.getConvergedReason()
    rel_norm = ksp.getResidualNorm() / b.norm()

    u_n_submesh.interpolate(u_n, cells0=parent_cells, cells1=smsh_cells)


    B = curl(u_n)
    u_n_submesh.interpolate(u_n, cells0=parent_cells, cells1=smsh_cells)
    da_dt_submesh = -(u_n_submesh - u_n_submesh_prev) / dt_submesh
    E_gal = -grad(u_n1)
    
    E = E_gal + da_dt_submesh
    J_ind = sigma_submesh * E

    Q = sigma_submesh * ufl.dot(ufl.grad(u_n1), ufl.grad(u_n1))
    theta_n = solve_thermal(Q, theta_n)


    # par_print(comm, f"Converged reason: {reason}")

    # par_print(comm, f"L2 norm of u_n is {L2_norm(u_n)}")
    # par_print(comm, f"L2 norm of B is {L2_norm(B)}")
    # par_print(comm, f"L2 norm of u_n1 is {L2_norm(u_n1)}")
    # par_print(comm, f"L2 norm of E_gal is {L2_norm(E_gal)}")
    # par_print(comm, f"L2 norm of da_dt is {L2_norm(da_dt_submesh)}")
    # par_print(comm, f"L2 norm of E is {L2_norm(E)}")
    # par_print(comm, f"L2 norm of J is {L2_norm(J)}")

    # par_print(comm, f"L2 norm of sigma*grad(phi) = {L2_norm(sigma_submesh*E_gal)}")
    # par_print(comm, f"L2 norm of sigma*dA_dt = {L2_norm(sigma_submesh*da_dt_submesh)}")

    W_mag   = 0.5 * gscalar(nu * inner(B, B) * dx)
    P_joule = gscalar(sigma_submesh * inner(E, E) * dx_c, entity_maps=entity_maps)
    P_eddy  = gscalar(sigma_submesh * inner(da_dt_submesh, da_dt_submesh) * dx_c, entity_maps=entity_maps)   # ∫σ|∂A/∂t|²
    P_cond  = gscalar(sigma_submesh * inner(E_gal, E_gal) * dx_c, entity_maps=entity_maps)   # ∫σ|∇V|²

    record = {
        "step": n + 1,
        "t": float(t.value),
        "ksp_reason": reason,
        "ksp_rel_residual": rel_norm,
        "Iterations": ksp.getIterationNumber(),
        "W_mag": W_mag,
        "P_joule": P_joule,
        "P_eddy": P_eddy,
        "P_cond": P_cond,
        "L2_u_n": L2_norm(u_n),
        "L2_u_n1": L2_norm(u_n1),
        "L2_B": L2_norm(B),
        "L2_E": L2_norm(E),
        "L2_J": L2_norm(J),
        "L2_E_gal": L2_norm(E_gal),
        "L2_da_dt": L2_norm(da_dt_submesh),
        "L2_delta_A": L2_norm(u_n - u_n_prev),
        "L2_delta_B": L2_norm(curl(u_n) - curl(u_n_prev)),
        "L2_Q": L2_norm(Q),
        "L2_theta": L2_norm(theta_n),
    }

    diagnostics.append(record)

    with open(diag_path, "w") as diag_file:
        json.dump(diagnostics, diag_file, indent=4)

    if n >= num_steps - last_steps:
    # if n >0:
        par_print(comm, "Writing output files...")
        
        Bexpr = fem.Expression(B, vector_vis.element.interpolation_points)
        B_vis.interpolate(Bexpr)
        B_file.write(t.value)

        B_vis_submesh.interpolate(B_vis, cells0=parent_cells, cells1=smsh_cells)
        B_file_submesh.write(t)

        Jexpr = fem.Expression(J_ind, vector_vis.element.interpolation_points)
        J_vis.interpolate(Jexpr)
        J_file.write(t.value)
        
        u_n1_file.write(t.value)

        theta_n_file.write(t.value)


B_file.close()
B_file_submesh.close()
J_file.close()
u_n1_file.close()
theta_n_file.close()
# E_file.close()