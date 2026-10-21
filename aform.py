#%%
import ufl
import numpy as np
from basix.ufl import element
from dolfinx import default_scalar_type, fem
from dolfinx.fem import Function, form
from dolfinx.fem.petsc import (
    apply_lifting,
    assemble_matrix,
    assemble_vector,
    discrete_gradient,
    interpolation_matrix,
    set_bc,
)
from dolfinx.io import VTXWriter, XDMFFile
from dolfinx.mesh import create_submesh
from mpi4py import MPI
from petsc4py import PETSc
from ufl import curl, inner

from current_density import coil_thickness, coil_width, coil_current
from utils import interpolate_by_tags, par_print

comm = MPI.COMM_WORLD
degree = 1

with XDMFFile(comm, "team24.xdmf", "r") as xdmf:
    mesh = xdmf.read_mesh()
    ct = xdmf.read_meshtags(mesh, name="Cell tags")
    tdim = mesh.topology.dim
    fdim = tdim - 1
    mesh.topology.create_entities(fdim)
    ft = xdmf.read_meshtags(mesh, name="Facet tags")

mesh.topology.create_connectivity(fdim, tdim)
mesh.topology.create_connectivity(tdim, tdim)
mesh.topology.create_connectivity(tdim - 1, 0)

domains = {
    "air": 14,
    "coil1": 8,
    "coil2": 12,
    "rotor": 2,
    "stator": 4,
}
boundary = {
    "outer": 13,
    "rotor_outer": 1,
    "stator_outer": 3,
}

J_unit = coil_current(mesh, ct, current=coil_width * coil_thickness)
mesh.geometry.x[:] *= 1e-3

with VTXWriter(mesh.comm, "J_coils.bp", J_unit, "BP4") as f:
    f.write(0.0)

sigma_non_conducting = fem.Constant(mesh, default_scalar_type(0.0))
sigma_stator = fem.Constant(mesh, default_scalar_type(4.54e6))
sigma_rotor = fem.Constant(mesh, default_scalar_type(4.54e6))

sigma_values = {
    domains["air"]: sigma_non_conducting,
    domains["coil1"]: sigma_non_conducting,
    domains["coil2"]: sigma_non_conducting,
    domains["rotor"]: sigma_rotor,
    domains["stator"]: sigma_stator,
}

mu_0 = 4 * np.pi * 1e-7

nu_air = fem.Constant(mesh, default_scalar_type(1 / (1.0 * mu_0)))
nu_stator = fem.Constant(mesh, default_scalar_type(1 / (280.0 * mu_0)))
nu_coil_1 = fem.Constant(mesh, default_scalar_type(1 / (1.0 * mu_0)))
nu_coil_2 = fem.Constant(mesh, default_scalar_type(1 / (1.0 * mu_0)))
nu_rotor = fem.Constant(mesh, default_scalar_type(1 / (280.0 * mu_0)))

nu_values = {
    domains["air"]: nu_air,
    domains["stator"]: nu_stator,
    domains["coil1"]: nu_coil_1,
    domains["coil2"]: nu_coil_2,
    domains["rotor"]: nu_rotor,
}

DG0 = fem.functionspace(mesh, ("DG", 0))
sigma = fem.Function(DG0)
nu = fem.Function(DG0)

interpolate_by_tags(sigma, sigma_values, ct)
interpolate_by_tags(nu, nu_values, ct)

nedelec_elem = element("N1curl", mesh.basix_cell(), degree)
V_A = fem.functionspace(mesh, nedelec_elem)

ti = 0.0    # start time
T_end = 0.3  # end time
d_t = 0.005

dt = fem.Constant(mesh, d_t)
t = fem.Constant(mesh, ti)

par_print(comm, f"Number of cells: {mesh.topology.index_map(tdim).size_global}")

N = 350
S = 24 * 17 * 1e-6 # coil cross-section [m^2]

I_table_t = np.array([0.0, 0.002, 0.005, 0.010, 0.015, 0.020, 0.030, 0.040, 0.050, 0.060, 0.070,
                      0.080, 0.090, 0.100, 0.120, 0.140, 0.160, 0.180, 0.200, 0.240, 0.300])
I_table_A = np.array([0.0, 0.91, 1.80, 2.95, 3.82, 4.49, 5.45, 6.05, 6.45, 6.71, 6.91,
                      7.04, 7.15, 7.22, 7.31, 7.36, 7.38, 7.40, 7.40, 7.41, 7.41])


def coil_current_at(time):
    return np.interp(time, I_table_t, I_table_A)


N_o_S = fem.Constant(mesh, default_scalar_type(N / S))
I_coil = fem.Constant(mesh, default_scalar_type(coil_current_at(ti)))

dx = ufl.Measure("dx", domain=mesh, subdomain_data=ct)
coil_dx = dx((domains["coil1"], domains["coil2"]))

A_n = fem.Function(V_A)
A_n.x.array[:] = 0.0

A = ufl.TrialFunction(V_A)
v = ufl.TestFunction(V_A)

a = (nu * inner(curl(A), curl(v)) * dx
     + inner(sigma * (A / dt), v) * dx)

L = (inner(sigma * (A_n / dt), v) * dx
     + N_o_S * I_coil * inner(J_unit, v) * coil_dx)

a_form = form(a)
L_form = form(L)

outer_facets = ft.find(boundary["outer"])
outer_dofs = fem.locate_dofs_topological(V_A, fdim, outer_facets)

A_bc = fem.Function(V_A)
A_bc.x.array[:] = 0.0
bc_outer = fem.dirichletbc(A_bc, outer_dofs)
bcs = [bc_outer]

A_mat = assemble_matrix(a_form, bcs=bcs)
A_mat.assemble()

b = assemble_vector(L_form)
apply_lifting(b, [a_form], bcs=[bcs])
b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
set_bc(b, bcs)

ksp = PETSc.KSP().create(mesh.comm)
ksp.setOptionsPrefix("main_")
ksp.setOperators(A_mat)
ksp.setType("fgmres")
ksp.setGMRESRestart(100)
ksp.setTolerances(rtol=1e-8, atol=1e-8, max_it=100)
ksp.setNormType(PETSc.KSP.NormType.UNPRECONDITIONED)

pc = ksp.getPC()
pc.setType("hypre")
pc.setHYPREType("ams")

W = fem.functionspace(mesh, ("Lagrange", degree))
G = discrete_gradient(W, V_A)
G.assemble()

# Cleaning Discrete values
ai, aj, av = G.getValuesCSR()
keep = np.abs(av) > 1e-12
csum = np.concatenate(([0], np.cumsum(keep)))
new_ai = csum[ai].astype(ai.dtype)
new_aj = aj[keep]
new_av = av[keep]

G_clean = PETSc.Mat().createAIJ(size=G.getSizes(), csr=(new_ai, new_aj, new_av), comm=G.comm)
G_clean.assemble()
pc.setHYPREDiscreteGradient(G_clean)

if degree == 1:
    cvec_0 = Function(V_A)
    cvec_0.interpolate(
        lambda x: np.vstack((np.ones_like(x[0]), np.zeros_like(x[0]), np.zeros_like(x[0])))
    )
    cvec_1 = Function(V_A)
    cvec_1.interpolate(
        lambda x: np.vstack((np.zeros_like(x[0]), np.ones_like(x[0]), np.zeros_like(x[0])))
    )
    cvec_2 = Function(V_A)
    cvec_2.interpolate(
        lambda x: np.vstack((np.zeros_like(x[0]), np.zeros_like(x[0]), np.ones_like(x[0])))
    )
    pc.setHYPRESetEdgeConstantVectors(cvec_0.x.petsc_vec, cvec_1.x.petsc_vec, cvec_2.x.petsc_vec)
else:
    shape = (mesh.geometry.dim,)
    Q = fem.functionspace(mesh, ("Lagrange", degree, shape))
    Pi = interpolation_matrix(Q, V_A)
    Pi.assemble()
    pc.setHYPRESetInterpolations(dim=mesh.geometry.dim, ND_Pi_Full=Pi)

omega_c = (domains["rotor"], domains["stator"])
interior_nodes = Function(W)
interior_nodes.x.array[:] = 1.0
conducting_cells = np.unique(np.concatenate([ct.find(tag) for tag in omega_c]))
conducting_dofs = np.unique(W.dofmap.list[conducting_cells].ravel())
interior_nodes.x.array[conducting_dofs] = 0.0
interior_nodes.x.array[fem.locate_dofs_topological(W, fdim, outer_facets)] = 0.0
interior_nodes.x.scatter_forward()
pc.setHYPREAMSSetInteriorNodes(interior_nodes.x.petsc_vec)

opts = PETSc.Options()
prefix = ksp.getOptionsPrefix()
opts[f"{prefix}ksp_monitor_true_residual"] = None
opts[f"{prefix}pc_hypre_ams_cycle_type"] = 1
opts[f"{prefix}pc_hypre_ams_tol"] = 0
opts[f"{prefix}pc_hypre_ams_max_iter"] = 4
opts[f"{prefix}pc_hypre_ams_amg_beta_theta"] = 0.25
opts[f"{prefix}pc_hypre_ams_print_level"] = 0
opts[f"{prefix}pc_hypre_ams_amg_alpha_options"] = "10,2,6,6,6"
opts[f"{prefix}pc_hypre_ams_amg_beta_options"] = "10,1,6,6,4"
opts[f"{prefix}pc_hypre_ams_relax_type"] = 8
opts[f"{prefix}pc_hypre_ams_relax_weight"] = 1.0
opts[f"{prefix}pc_hypre_ams_relax_times"] = 2
opts[f"{prefix}pc_hypre_ams_omega"] = 1.0
opts[f"{prefix}pc_hypre_ams_projection_frequency"] = 10
ksp.setFromOptions()
ksp.setUp()

sol = A_mat.createVecRight()

vector_vis = fem.functionspace(
    mesh, ("Discontinuous Lagrange", degree, (mesh.geometry.dim,))
)

B = ufl.curl(A_n)
B_vis = Function(vector_vis)
B_file = VTXWriter(mesh.comm, "B_field.bp", B_vis, "BP4")
Bexpr = fem.Expression(B, vector_vis.element.interpolation_points)
B_vis.interpolate(Bexpr)
B_file.write(t)

target_tags = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
cell_lists = [ct.find(tag) for tag in target_tags]
target_cells = np.unique(np.concatenate(cell_lists)).astype(np.int32)

submesh, subdomain_motor_to_domain = create_submesh(mesh, tdim, target_cells)[:2]
smsh_cell_imap = submesh.topology.index_map(tdim)
smsh_cells = np.arange(smsh_cell_imap.size_local + smsh_cell_imap.num_ghosts)
parent_cells = subdomain_motor_to_domain.sub_topology_to_topology(smsh_cells, inverse=False)

DG_submesh_vis = fem.functionspace(
    submesh, ("Discontinuous Lagrange", degree, (submesh.geometry.dim,))
)
B_vis_submesh = fem.Function(DG_submesh_vis)
B_vis_submesh.interpolate(B_vis, cells0=parent_cells, cells1=smsh_cells)
B_file_submesh = VTXWriter(mesh.comm, "B_submesh.bp", B_vis_submesh, "BP4")
B_file_submesh.write(t)

num_steps = int(np.ceil(T_end / d_t))

for n in range(num_steps):
    t.value += d_t
    I_coil.value = coil_current_at(float(t.value))
    par_print(comm, f"step {n + 1:4d}/{num_steps}  t={t.value:.4f}s  I_coil={float(I_coil.value):+.4f} A")

    with b.localForm() as b_loc:
        b_loc.set(0.0)
    assemble_vector(b, L_form)
    apply_lifting(b, [a_form], bcs=[bcs])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    set_bc(b, bcs)

    ksp.solve(b, sol)
    par_print(comm, f"KSP converged reason {ksp.getConvergedReason()} in {ksp.getIterationNumber()} iterations")

    A_n.x.array[: len(sol.array_r)] = sol.array_r
    A_n.x.scatter_forward()

    B_vis.interpolate(Bexpr)
    B_file.write(t)

    B_vis_submesh.interpolate(B_vis, cells0=parent_cells, cells1=smsh_cells)
    B_file_submesh.write(t)


B_file.close()
B_file_submesh.close()

