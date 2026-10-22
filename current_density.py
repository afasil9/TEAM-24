import numpy as np
import ufl
from dolfinx import fem, la
from dolfinx.fem.petsc import LinearProblem
from dolfinx.io import VTXWriter, XDMFFile
from dolfinx.mesh import create_submesh
from mpi4py import MPI

coil_width = 24.0
coil_thickness = 17.0
window = (-17.0, 17.0, -3.0, 28.0)

coils = {"coil1": (8, 1.0), "coil2": (12, 1.0)}


def coil_direction(x, sign):
    """Unit tangent of the coil current at points x (shape (3, N))."""
    x0, x1, z0, z1 = window
    dx = x[0] - np.clip(x[0], x0, x1)
    dz = x[2] - np.clip(x[2], z0, z1)
    r = np.sqrt(dx**2 + dz**2)
    r[r == 0.0] = 1.0
    return sign * np.vstack((dz / r, np.zeros_like(r), -dx / r))


def coil_current(mesh, ct, current=1.0):
    """Return a cellwise constant, discretely divergence-free coil current."""
    J0_mag = current / (coil_width * coil_thickness)

    tdim = mesh.topology.dim
    coil_cells = np.sort(np.concatenate([ct.find(tag) for tag, _ in coils.values()]))
    coil_mesh, entity_map, _, _ = create_submesh(mesh, tdim, coil_cells)
    cmap = coil_mesh.topology.index_map(tdim)
    sub_cells = np.arange(cmap.size_local + cmap.num_ghosts, dtype=np.int32)
    parent_cells = entity_map.sub_topology_to_topology(sub_cells, inverse=False)
    cell_tag = np.full(mesh.topology.index_map(tdim).size_local + mesh.topology.index_map(tdim).num_ghosts, -1)
    cell_tag[ct.indices] = ct.values
    sub_tag = cell_tag[parent_cells]

    W_c = fem.functionspace(coil_mesh, ("DG", 0, (3,)))
    J0 = fem.Function(W_c, name="J0")
    for tag, sign in coils.values():
        J0.interpolate(lambda x, s=sign: J0_mag * coil_direction(x, s), cells0=sub_cells[sub_tag == tag])

    V = fem.functionspace(coil_mesh, ("Lagrange", 1))
    phi, q = ufl.TrialFunction(V), ufl.TestFunction(V)
    a = ufl.inner(ufl.grad(phi), ufl.grad(q)) * ufl.dx
    L = ufl.inner(J0, ufl.grad(q)) * ufl.dx

    imap = V.dofmap.index_map
    pinned = []
    for tag, _ in coils.values():
        dofs = np.unique(V.dofmap.list[sub_cells[sub_tag == tag]].ravel()).astype(np.int32)
        gdofs = imap.local_to_global(dofs)
        gmin = mesh.comm.allreduce(gdofs.min() if len(gdofs) > 0 else np.iinfo(np.int64).max, op=MPI.MIN)
        pinned.append(dofs[gdofs == gmin])
    bc = fem.dirichletbc(0.0, np.sort(np.concatenate(pinned)), V)

    problem = LinearProblem(
        a,
        L,
        bcs=[bc],
        petsc_options_prefix="coil_current_",
        petsc_options={"ksp_type": "cg", "pc_type": "hypre", "ksp_rtol": 1e-12},
    )
    phi_h = problem.solve()

    J_c = fem.Function(W_c)
    J_c.interpolate(fem.Expression(J0 - ufl.grad(phi_h), W_c.element.interpolation_points))

    W = fem.functionspace(mesh, ("DG", 0, (3,)))
    J = fem.Function(W, name="J")
    J.interpolate(J_c, cells0=sub_cells, cells1=parent_cells)
    J.x.scatter_forward()
    return J


if __name__ == "__main__":
    with XDMFFile(MPI.COMM_WORLD, "team24.xdmf", "r") as xdmf:
        mesh = xdmf.read_mesh()
        ct = xdmf.read_meshtags(mesh, "Cell tags")

    I = 1.0
    J = coil_current(mesh, ct, current=I)

    V = fem.functionspace(mesh, ("Lagrange", 1))
    q = ufl.TestFunction(V)
    div_res = fem.assemble_vector(fem.form(ufl.inner(J, ufl.grad(q)) * ufl.dx))
    div_res.scatter_reverse(la.InsertMode.add)
    res = np.sqrt(mesh.comm.allreduce(np.sum(div_res.array[: V.dofmap.index_map.size_local] ** 2)))

    dx = ufl.Measure("dx", domain=mesh, subdomain_data=ct)
    x = ufl.SpatialCoordinate(mesh)
    top_leg = ufl.conditional(ufl.And(ufl.lt(abs(x[0]), 17.0), ufl.gt(x[2], 28.0)), 1.0, 0.0)
    if mesh.comm.rank == 0:
        print(f"Discrete divergence residual: {res:.3e}")
    for name, (tag, _) in coils.items():
        Jx_int = mesh.comm.allreduce(fem.assemble_scalar(fem.form(top_leg * J[0] * dx(tag))), op=MPI.SUM)
        if mesh.comm.rank == 0:
            print(f"{name}: current through top leg = {Jx_int / 34.0:.6f} A (target {I} A)")

    with VTXWriter(mesh.comm, "coil_current.bp", [J], engine="BP4") as vtx:
        vtx.write(0.0)
