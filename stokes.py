# %%
import numpy as np
from basix.ufl import element
from dolfinx import default_scalar_type
from dolfinx.fem import (
    Constant,
    Function,
    bcs_by_block,
    dirichletbc,
    extract_function_spaces,
    form,
    functionspace,
    locate_dofs_topological,
)
from dolfinx.fem.petsc import apply_lifting, assemble_matrix, assemble_vector, set_bc
from dolfinx.io import XDMFFile, VTXWriter
from dolfinx.mesh import create_submesh
from mpi4py import MPI
from petsc4py import PETSc
from ufl import (
    Measure,
    TestFunction,
    TrialFunction,
    ZeroBaseForm,
    div,
    grad,
    inner,
)
from utils import convert_facet_tags, par_print
from matplotlib import pyplot as plt

comm = MPI.COMM_WORLD
degree = 2

output_file = "stokes"

U = 1.0  # Mean inlet speed
rho = 1.2  # Air density
mu_value = 1.8e-5  # Dynamic viscosity

rtol = 1e-8
max_it = 2000

vol_ids = {"air": 1, "coil1": 2, "coil2": 3, "rotor": 4, "stator": 5}
boundary_ids = {
    "outer": 1,
    "symmetry": 2,
    "coil1_in": 3,
    "coil1_out": 4,
    "coil2_in": 5,
    "coil2_out": 6,
    "rotor_outer": 7,
    "stator_outer": 8,
    "inlet": 9,
    "outlet": 10,
    "coil1": 11,
    "coil2": 12,
    "bottom": 13,
}

inlet = boundary_ids["inlet"]

no_slip = (
    boundary_ids["outer"],
    boundary_ids["symmetry"],
    boundary_ids["bottom"],
    boundary_ids["rotor_outer"],
    boundary_ids["stator_outer"],
    boundary_ids["coil1"],
    boundary_ids["coil2"],
)

with XDMFFile(comm, "team24_horseshoe.xdmf", "r") as xdmf:
    domain = xdmf.read_mesh()
    ct = xdmf.read_meshtags(domain, name="Cell_markers")
    tdim = domain.topology.dim
    fdim = tdim - 1
    domain.topology.create_entities(fdim)
    ft = xdmf.read_meshtags(domain, name="Facet_markers")

domain.topology.create_connectivity(fdim, tdim)
domain.topology.create_connectivity(tdim, fdim)

air_cells = ct.find(vol_ids["air"])
fluid, fluid_to_domain = create_submesh(domain, tdim, air_cells)[:2]
fluid.topology.create_connectivity(fdim, tdim)
fluid.topology.create_connectivity(tdim, fdim)

par_print(comm, f"Mesh: {domain.topology.index_map(tdim).size_global} cells")
par_print(comm, f"Air submesh: {fluid.topology.index_map(tdim).size_global} cells")

par_print(comm, f"Number of dofs on mesh: {domain.topology.index_map(fdim).size_global}")
par_print(comm, f"Number of dofs on fluid submesh: {fluid.topology.index_map(tdim).size_global}")

ft_fluid = convert_facet_tags(fluid, fluid_to_domain, ft)

u_elem = element("Lagrange", fluid.basix_cell(), degree, shape=(fluid.geometry.dim,))
p_elem = element("Lagrange", fluid.basix_cell(), degree - 1)
V = functionspace(fluid, u_elem)
Q = functionspace(fluid, p_elem)

dofs_u = V.dofmap.index_map.size_global * V.dofmap.index_map_bs
dofs_p = Q.dofmap.index_map.size_global
par_print(comm, f"dofs: u {dofs_u:,}  p {dofs_p:,}  total {dofs_u + dofs_p:,}")

dx = Measure("dx", fluid)

mu = Constant(fluid, default_scalar_type(mu_value))
f = Constant(fluid, np.zeros(3, dtype=default_scalar_type))

u = TrialFunction(V)
v = TestFunction(V)
p = TrialFunction(Q)
q = TestFunction(Q)

a00 = mu * inner(grad(u), grad(v)) * dx
a01 = -inner(p, div(v)) * dx
a10 = -inner(div(u), q) * dx
a11 = 1 / mu * inner(p, q) * dx


a = form([[a00, a01], [a10, None]])
L = form([inner(f, v) * dx, ZeroBaseForm((q,))])
a_p = form([[a00, None], [None, a11]])

u_wall = Function(V)
dofs_wall = locate_dofs_topological(
    V, fdim, np.unique(np.concatenate([ft_fluid.find(t) for t in no_slip]))
)
bc_wall = dirichletbc(u_wall, dofs_wall)


x = fluid.geometry.x
lo = np.array([comm.allreduce(x[:, i].min(), op=MPI.MIN) for i in range(3)])
hi = np.array([comm.allreduce(x[:, i].max(), op=MPI.MAX) for i in range(3)])

def inlet_profile(x):
    fy = 6 * (x[1] - lo[1]) * (hi[1] - x[1]) / (hi[1] - lo[1]) ** 2
    fz = 6 * (x[2] - lo[2]) * (hi[2] - x[2]) / (hi[2] - lo[2]) ** 2
    ux = U * fy * fz
    return np.vstack((ux, np.zeros_like(ux), np.zeros_like(ux)))

def plot_inlet_profile():
    z = np.linspace(lo[2], hi[2], 200)

    x = np.zeros((3, len(z)))
    x[2] = z

    velocity = inlet_profile(x)
    ux = velocity[0]

    plt.plot(z, ux)
    plt.xlabel("z")
    plt.ylabel("$u_x$")
    plt.title("Inlet velocity profile")
    plt.grid(True)
    plt.show()


u_in = Function(V)
u_in.interpolate(inlet_profile)
dofs_inlet = locate_dofs_topological(V, fdim, ft_fluid.find(inlet))
bc_inlet = dirichletbc(u_in, dofs_inlet)

bc = [bc_wall, bc_inlet]

A = assemble_matrix(a, bcs=bc, kind="mpi")
A.assemble()

P = assemble_matrix(a_p, bcs=bc, kind="mpi")
P.assemble()

b = assemble_vector(L, kind="mpi")
apply_lifting(b, a, bcs=bcs_by_block(extract_function_spaces(a, 1), bc))
b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
set_bc(b, bcs_by_block(extract_function_spaces(L), bc))

u_map = V.dofmap.index_map
p_map = Q.dofmap.index_map
n_u_local = u_map.size_local * V.dofmap.index_map_bs

offset_u = u_map.local_range[0] * V.dofmap.index_map_bs + p_map.local_range[0]
offset_p = offset_u + n_u_local

is_u = PETSc.IS().createStride(n_u_local, offset_u, 1, comm=fluid.comm)
is_p = PETSc.IS().createStride(p_map.size_local, offset_p, 1, comm=fluid.comm)
is_u.setBlockSize(V.dofmap.index_map_bs)

ksp = PETSc.KSP().create(fluid.comm)
ksp.setOptionsPrefix("stokes_")
ksp.setOperators(A, P)
ksp.setType("minres")
ksp.setTolerances(rtol=rtol, max_it=max_it)
ksp.getPC().setType("fieldsplit")
ksp.getPC().setFieldSplitType(PETSc.PC.CompositeType.ADDITIVE)
ksp.getPC().setFieldSplitIS(("u", is_u), ("p", is_p))
ksp_u, ksp_p = ksp.getPC().getFieldSplitSubKSP()

opts = PETSc.Options()
opts[f"{ksp.getOptionsPrefix()}ksp_monitor_true_residual"] = None
ksp.setFromOptions()

ksp_u.setType("preonly")
pc0 = ksp_u.getPC()
pc0.setType("gamg")

ksp_p.setType("preonly")
pc1 = ksp_p.getPC()
pc1.setType("jacobi")

ksp.setUp()
pc0.setUp()
pc1.setUp()


uh, ph = Function(V, name="u"), Function(Q, name="p")

sol = A.createVecRight()
ksp.solve(b, sol)

uh.x.array[:n_u_local] = sol.array_r[:n_u_local]
ph.x.array[: (len(sol.array_r) - n_u_local)] = sol.array_r[n_u_local:]

uh.x.scatter_forward()
ph.x.scatter_forward()

par_print(comm, f"Number of iterations: {ksp.getIterationNumber()}")
par_print(comm, f"Converged reason: {ksp.getConvergedReason()}")

par_print(comm, f"Bounding box: {lo} -> {hi}")
L_ref = float(hi[0] - lo[0])
par_print(comm, f"Re = {rho * U * L_ref / mu_value:.3g} -- Stokes assumes Re << 1")


with VTXWriter(comm, f"{output_file}_u.bp", uh, "bp4") as file:
    file.write(0.0)

with VTXWriter(comm, f"{output_file}_p.bp", ph, "bp4") as file:
    file.write(0.0)

