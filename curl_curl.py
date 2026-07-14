import numpy as np
import ufl
from dolfinx import default_scalar_type, fem
from dolfinx.fem import Function
from dolfinx.fem.petsc import (assemble_matrix, assemble_vector, set_bc, apply_lifting,
                               discrete_gradient, interpolation_matrix)
from dolfinx.io import VTXWriter, XDMFFile
from dolfinx.mesh import create_submesh
from mpi4py import MPI
from petsc4py import PETSc

from utils import par_print, interpolate_by_tags


comm = MPI.COMM_WORLD
degree = 1

with XDMFFile(comm, "team24_horseshoe.xdmf", "r") as xdmf:
    mesh = xdmf.read_mesh()
    ct = xdmf.read_meshtags(mesh, name="Cell_markers")
    tdim = mesh.topology.dim
    fdim = tdim - 1
    mesh.topology.create_entities(fdim)
    ft = xdmf.read_meshtags(mesh, name="Facet_markers")

par_print(comm, f"Number of cells in the mesh: {mesh.topology.index_map(tdim).size_global}")

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

sigma_non = 0.0
sigma_copper_val = 5.96e7
sigma_iron_val = 1e5

mu = 4e-7 * np.pi
mu_r_iron = 1000.0
nu_air_val = 1.0 / mu
nu_iron_val = 1.0 / (mu_r_iron * mu)

sigma_values = {
    domains["air"]:    fem.Constant(mesh, default_scalar_type(sigma_non)),
    domains["coil1"]:  fem.Constant(mesh, default_scalar_type(sigma_copper_val)),
    domains["coil2"]:  fem.Constant(mesh, default_scalar_type(sigma_copper_val)),
    domains["rotor"]:  fem.Constant(mesh, default_scalar_type(sigma_iron_val)),
    domains["stator"]: fem.Constant(mesh, default_scalar_type(sigma_iron_val)),
}

nu_values = {
    domains["air"]:    fem.Constant(mesh, default_scalar_type(nu_air_val)),
    domains["coil1"]:  fem.Constant(mesh, default_scalar_type(nu_air_val)),
    domains["coil2"]:  fem.Constant(mesh, default_scalar_type(nu_air_val)),
    domains["rotor"]:  fem.Constant(mesh, default_scalar_type(nu_iron_val)),
    domains["stator"]: fem.Constant(mesh, default_scalar_type(nu_iron_val)),
}

DG0 = fem.functionspace(mesh, ("DG", 0))

sigma = fem.Function(DG0)
interpolate_by_tags(sigma, sigma_values, ct)

nu = fem.Function(DG0)
interpolate_by_tags(nu, nu_values, ct)

V = fem.functionspace(mesh, ("N1curl", degree))

A = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

A_h = fem.Function(V)
A_n = fem.Function(V)
A_h.name = "A"

NI = 3000.0
coil_thickness_y = 17.0e-3
T0_mag = NI / coil_thickness_y

coil_cells = np.concatenate(
    [ct.find(domains["coil1"]), ct.find(domains["coil2"])]
).astype(np.int32)

T0 = fem.Function(V)
T0.x.array[:] = 0.0
T0.interpolate(
    lambda x: np.vstack(
        (
            np.zeros_like(x[0]),
            np.full_like(x[0], T0_mag),
            np.zeros_like(x[0]),
        )
    ),
    coil_cells,
)
T0.x.scatter_forward()

dx = ufl.Measure("dx", domain=mesh, subdomain_data=ct)
dt = fem.Constant(mesh, default_scalar_type(1e-3))
a_form = fem.form(dt * ufl.inner(nu * ufl.curl(A), ufl.curl(v)) * dx + ufl.inner(sigma * A, v) * dx)
L_form = fem.form(dt * ufl.inner(T0, ufl.curl(v)) * dx)


outer_boundaries = [boundary["outer"], boundary["symmetry"],
                    boundary["coil1_out"], boundary["coil2_out"],
                    boundary["coil1_in"], boundary["coil2_in"]]
outer_boundary_tags = np.unique(
    np.concatenate([ft.find(tag) for tag in outer_boundaries])
)
boundary_dofs = fem.locate_dofs_topological(V, entity_dim=fdim,
                                            entities=outer_boundary_tags)

A_bc = fem.Function(V)
A_bc.x.array[:] = 0.0
bc = fem.dirichletbc(A_bc, boundary_dofs)

W = fem.functionspace(mesh, ("Lagrange", degree))
G = discrete_gradient(W, V)
G.assemble()

ai, aj, av = G.getValuesCSR()
keep = np.abs(av) > 1e-12
csum = np.concatenate(([0], np.cumsum(keep)))
G_clean = PETSc.Mat().createAIJ(size=G.getSizes(),
                                csr=(csum[ai].astype(ai.dtype), aj[keep], av[keep]),
                                comm=G.comm)
G_clean.assemble()

if degree == 1:
    cvecs = []
    for d in range(3):
        c = Function(V)
        c.interpolate(lambda x, d=d: np.vstack(
            [np.ones_like(x[0]) if i == d else np.zeros_like(x[0]) for i in range(3)]
        ))
        cvecs.append(c)
else:
    shape = (mesh.geometry.dim,)
    Q = fem.functionspace(mesh, ("Lagrange", degree, shape))
    Pi = interpolation_matrix(Q, V)
    Pi.assemble()


interior_nodes_array = fem.Function(W)

dofmap = W.dofmap
num_dofs_per_cell = dofmap.dof_layout.num_dofs
cell_dofs = dofmap.list.reshape(-1, num_dofs_per_cell)

bdofs_outer = fem.locate_dofs_topological(W, entity_dim=fdim,
                                          entities=outer_boundary_tags)

fixed_conductor_cells = np.unique(
    np.concatenate([ct.find(domains[name]) for name in ("coil1", "coil2", "stator")])
)


def update_interior_nodes(rotor_cells):
    interior_nodes_array.x.array[:] = 1.0
    conductor_cells = np.unique(np.concatenate([fixed_conductor_cells, rotor_cells]))
    conductor_dofs = np.unique(cell_dofs[conductor_cells].flatten())
    interior_nodes_array.x.array[conductor_dofs] = 0.0
    interior_nodes_array.x.array[bdofs_outer] = 0.0
    interior_nodes_array.x.scatter_forward()


opts = PETSc.Options()
opts["main_ksp_monitor_true_residual"] = None
opts["main_pc_hypre_ams_cycle_type"] = 1
opts["main_pc_hypre_ams_tol"] = 0
opts["main_pc_hypre_ams_max_iter"] = 1
opts["main_pc_hypre_ams_amg_beta_theta"] = 0.25
opts["main_pc_hypre_ams_print_level"] = 0
opts["main_pc_hypre_ams_amg_alpha_options"] = "10,2,6,6,6"
opts["main_pc_hypre_ams_amg_beta_options"] = "10,1,6,6,4"
opts["main_pc_hypre_ams_relax_type"] = 8
opts["main_pc_hypre_ams_relax_weight"] = 1.0
opts["main_pc_hypre_ams_relax_times"] = 2
opts["main_pc_hypre_ams_omega"] = 1.0
opts["main_pc_hypre_ams_projection_frequency"] = 100000


rotor_cells0 = ct.find(domains["rotor"]).astype(np.int32)
air_cells = ct.find(domains["air"]).astype(np.int32)

combined_cells = np.unique(np.concatenate([rotor_cells0, air_cells])).astype(np.int32)

theta = 0.0

A_mat = assemble_matrix(a_form, bcs=[bc])
A_mat.assemble()
b = assemble_vector(L_form)


def reassemble_system():
    A_mat.zeroEntries()
    assemble_matrix(A_mat, a_form, bcs=[bc])
    A_mat.assemble()

    with b.localForm() as b_loc:
        b_loc.set(0.0)
    assemble_vector(b, L_form)
    apply_lifting(b, [a_form], bcs=[[bc]])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    set_bc(b, [bc])
    b.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)


def solve_at_current_state(rotor_cells, use_initial_guess=False):
    update_interior_nodes(rotor_cells)
    reassemble_system()

    ksp = PETSc.KSP().create(mesh.comm)
    ksp.setOptionsPrefix("main_")
    ksp.setOperators(A_mat)
    ksp.setType("gmres")
    ksp.setGMRESRestart(200)
    ksp.setTolerances(rtol=1e-6, atol=1e-6, max_it=200)
    ksp.setNormType(PETSc.KSP.NormType.UNPRECONDITIONED)

    pc = ksp.getPC()
    pc.setType("hypre")
    pc.setHYPREType("ams")
    pc.setHYPREDiscreteGradient(G_clean)
    if degree == 1:
        pc.setHYPRESetEdgeConstantVectors(cvecs[0].x.petsc_vec,
                                          cvecs[1].x.petsc_vec,
                                          cvecs[2].x.petsc_vec)
    else:
        pc.setHYPRESetInterpolations(dim=mesh.geometry.dim, ND_Pi_Full=Pi)
    if sigma_non == 0.0:
        pc.setHYPREAMSSetInteriorNodes(interior_nodes_array.x.petsc_vec)

    ksp.setFromOptions()
    if use_initial_guess:
        ksp.setInitialGuessNonzero(True)

    return ksp


ksp = solve_at_current_state(rotor_cells0)
ksp.solve(b, A_h.x.petsc_vec)
A_h.x.scatter_forward()
par_print(comm, f"[theta =  0.0 deg] KSP converged reason {ksp.getConvergedReason()} "
                f"in {ksp.getIterationNumber()} iterations")
ksp.destroy()


with XDMFFile(mesh.comm, "interior_nodes.xdmf", "w") as interior_nodes_file:
    interior_nodes_file.write_mesh(mesh)
    interior_nodes_file.write_function(interior_nodes_array)


A_n.x.array[:] = A_h.x.array

vector_vis = fem.functionspace(
    mesh, ("Discontinuous Lagrange", degree, (mesh.geometry.dim,))
)
B_vis = Function(vector_vis)
B_file = VTXWriter(mesh.comm, "B_field.bp", B_vis, "BP4")
Bexpr = fem.Expression(ufl.curl(A_h), vector_vis.element.interpolation_points)
B_vis.interpolate(Bexpr)
B_file.write(theta)


submesh_rotor, rotor_to_parent = create_submesh(mesh, tdim, rotor_cells0)[:2]

DG_rot = fem.functionspace(
    submesh_rotor,
    ("Discontinuous Lagrange", degree, (mesh.geometry.dim,))
)

smsh_cell_imap = submesh_rotor.topology.index_map(tdim)
smsh_cells = np.arange(smsh_cell_imap.size_local + smsh_cell_imap.num_ghosts)
parent_cells = rotor_to_parent.sub_topology_to_topology(smsh_cells, inverse=False)
B_rotor = fem.Function(DG_rot)
B_rotor.interpolate(
    B_vis,
    cells0=parent_cells,
    cells1=smsh_cells
)

B_file_rotor = VTXWriter(mesh.comm, "B_field_rotor.bp", B_rotor, "BP4")
B_file_rotor.write(theta)


def rotation_matrix_z(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def set_mesh_rotation(msh, ref_coords, theta, centre):
    R = rotation_matrix_z(theta)
    msh.geometry.x[:] = (ref_coords - centre) @ R.T + centre



V_submesh = fem.functionspace(submesh_rotor, ("N1curl", degree))
rotor_ref_coords = submesh_rotor.geometry.x.copy()
centre = np.array([0.0, 0.0, 0.0])

smsh_cell_imap = submesh_rotor.topology.index_map(tdim)
smsh_cells = np.arange(smsh_cell_imap.size_local + smsh_cell_imap.num_ghosts,
                       dtype=np.int32)
parent_cells = rotor_to_parent.sub_topology_to_topology(smsh_cells, inverse=False)

DG0_sub = fem.functionspace(submesh_rotor, ("DG", 0))
chi_sub = fem.Function(DG0_sub)
chi_sub.x.array[:] = 1.0


def rotor_cells_at_current_angle(padding=1e-14):
    chi = fem.Function(DG0)
    chi.x.array[:] = 0.0
    idata_chi = fem.create_interpolation_data(DG0, DG0_sub, combined_cells,
                                              padding=padding)
    chi.interpolate_nonmatching(chi_sub, combined_cells,
                                interpolation_data=idata_chi)
    chi.x.scatter_forward()
    return combined_cells[chi.x.array[combined_cells] > 0.5].astype(np.int32)


def set_materials_for_rotor(rotor_cells):
    sigma.x.array[combined_cells] = sigma_non
    nu.x.array[combined_cells] = nu_air_val
    sigma.x.array[rotor_cells] = sigma_iron_val
    nu.x.array[rotor_cells] = nu_iron_val
    sigma.x.scatter_forward()
    nu.x.scatter_forward()


# Rotating the rotor to 90 degrees
theta = np.deg2rad(90.0)

A_submesh = fem.Function(V_submesh)
A_submesh.interpolate(A_n, cells0=parent_cells, cells1=smsh_cells)
A_submesh.x.scatter_forward()

set_mesh_rotation(submesh_rotor, rotor_ref_coords, theta, centre)

rotor_cells_new = rotor_cells_at_current_angle()
n_local = len(rotor_cells_new)
par_print(comm, f"Rotor covers {comm.allreduce(n_local, op=MPI.SUM)} parent cells "
                f"at theta = {np.rad2deg(theta):.1f} deg")

set_materials_for_rotor(rotor_cells_new)

A_h.x.array[:] = A_n.x.array
idata_A = fem.create_interpolation_data(V, V_submesh, combined_cells,
                                        padding=1e-8)
A_h.interpolate_nonmatching(A_submesh, combined_cells,
                            interpolation_data=idata_A)
A_h.x.scatter_forward()


ksp = solve_at_current_state(rotor_cells_new, use_initial_guess=True)
ksp.solve(b, A_h.x.petsc_vec)
A_h.x.scatter_forward()
par_print(comm, f"[theta = 90.0 deg] KSP converged reason {ksp.getConvergedReason()} "
                f"in {ksp.getIterationNumber()} iterations")
ksp.destroy()

with XDMFFile(mesh.comm, "interior_nodes_90.xdmf", "w") as f:
    f.write_mesh(mesh)
    f.write_function(interior_nodes_array)

B_vis.interpolate(Bexpr)
B_file.write(theta)
B_file.close()

B_rotor.interpolate(
    B_vis,
    cells0=parent_cells,
    cells1=smsh_cells
)

B_file_rotor.write(theta)
