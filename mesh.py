# Generate TEAM24 benchmark mesh
#
# Usage:
#   python3 mesh.py                      # characteristic size 5 mm
#   python3 mesh.py --size 3             # finer mesh
#   python3 mesh.py --refine 1           # split every tetrahedron into 8
#   python3 mesh.py --size 3 --refine 1 --output team24_fine
import argparse

import numpy as np
import gmsh
from dolfinx.io import XDMFFile
from dolfinx.io.gmsh import model_to_mesh
from mpi4py import MPI


def coil(r, dx, dy, dz):
    b1 = gmsh.model.occ.addBox(0.0, r, 0.0, 2 * r + dx, dy, dz)
    b2 = gmsh.model.occ.addBox(r, 0.0, 0.0, dx, 2 * r + dy, dz)
    c1 = gmsh.model.occ.addCylinder(r, r, 0.0, 0.0, 0.0, dz, r)
    c2 = gmsh.model.occ.addCylinder(r, r + dy, 0.0, 0.0, 0.0, dz, r)
    c3 = gmsh.model.occ.addCylinder(r + dx, r, 0.0, 0.0, 0.0, dz, r)
    c4 = gmsh.model.occ.addCylinder(r + dx, r + dy, 0.0, 0.0, 0.0, dz, r)
    cmb1 = gmsh.model.occ.fuse([(3, b1)], [(3, b2), (3, c1), (3, c2), (3, c3), (3, c4)])
    cmb2 = gmsh.model.occ.addBox(r, r, 0.0, dx, dy, dz)
    cmb3 = gmsh.model.occ.cut([cmb1[0][0]], [(3, cmb2)])

    return cmb3


def rotor(H):
    # Inner rotor
    c0 = gmsh.model.occ.addCylinder(0.0, 0.0, 0.0, 0.0, 0.0, H, 38.1)
    b0 = gmsh.model.occ.addBox(-12.7, -55.0, 0.0, 25.4, 110.0, H)
    c1 = gmsh.model.occ.addCylinder(0.0, 0.0, 0.0, 0.0, 0.0, H, 51.05)
    cmb1 = gmsh.model.occ.intersect([(3, b0)], [(3, c1)])
    cmb2 = gmsh.model.occ.fuse([(3, c0)], [cmb1[0][0]])
    c2 = gmsh.model.occ.addCylinder(0.0, 0.0, 0.0, 0.0, 0.0, H, 25.4)
    r = gmsh.model.occ.cut([cmb2[0][0]], [(3, c2)])
    return r


def stator(H, r_champ):
    # Outer stator
    c3 = gmsh.model.occ.addCylinder(0.0, 0.0, 0.0, 0.0, 0.0, H, 209.0 / 2)
    c4 = gmsh.model.occ.addCylinder(0.0, 0.0, 0.0, 0.0, 0.0, H, 209.0 / 2 - 21.4)
    cmb4 = gmsh.model.occ.cut([(3, c3)], [(3, c4)])

    # Chamfer corners
    R = 209.0 / 2 - 21.4 - r_champ
    x = 27.8 / 2.0 + r_champ
    y = np.sqrt(R * R - x * x)
    x1 = 27.8 / 2
    y1 = 209.0 / 2 - 21.4

    champs = []
    for dir in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
        cA = gmsh.model.occ.addCylinder(dir[0] * x, dir[1] * y, 0, 0, 0, H, r_champ)
        cB = gmsh.model.occ.addCylinder(dir[0] * x1, dir[1] * y1, 0, 0, 0, H, y1 - y)
        c = gmsh.model.occ.cut([(3, cB)], [(3, cA)])
        champs += [c[0][0]]

    b1 = gmsh.model.occ.addBox(-27.8 / 2.0, -100.0, 0.0, 27.8, 200.0, H)
    c5 = gmsh.model.occ.addCylinder(0.0, 0.0, 0.0, 0.0, 0.0, H, 107.5 / 2.0)
    c = gmsh.model.occ.cut([(3, b1)], [(3, c5)])
    s = gmsh.model.occ.fuse([cmb4[0][0]], [c[0][0], c[0][1]] + champs)
    return s


parser = argparse.ArgumentParser(description="Generate the TEAM24 mesh")
parser.add_argument("--size", type=float, default=5.0, help="characteristic mesh size [mm]")
parser.add_argument("--refine", type=int, default=0, help="number of uniform refinements")
parser.add_argument("--output", default="team24", help="output file name (without extension)")
args = parser.parse_args()

gmsh.initialize()
gmsh.model.add("TEAM24")

H = 25.4

rotor1 = rotor(H)
gmsh.model.occ.rotate([rotor1[0][0]], 0, 0, 0, 0, 0, 1, -22.0 * np.pi / 180.0)

r_champ = 5.0
stator1 = stator(H, r_champ)

# Coils
coil1 = coil(24.0, 34.0, 31.0, 17.0)
gmsh.model.occ.rotate([coil1[0][0]], 0, 0, 0, 1, 0, 0, np.pi / 2.0)
D = 63 - 17.0 / 2.0
gmsh.model.occ.translate([coil1[0][0]], -41.0, -D, -27.0)

coil2 = coil(24.0, 34.0, 31.0, 17.0)
gmsh.model.occ.rotate([coil2[0][0]], 0, 0, 0, 1, 0, 0, -np.pi / 2.0)
gmsh.model.occ.translate([coil2[0][0]], -41.0, D, -27.0 + 79.0)

M = 35.0
outer_box = gmsh.model.occ.addBox(-120, -120, -M, 240, 240, H + 2 * M)

parts = gmsh.model.occ.fragment([(3, outer_box)], [rotor1[0][0], stator1[0][0], coil1[0][0], coil2[0][0]])

gmsh.model.occ.synchronize()
gmsh.model.occ.removeAllDuplicates()
gmsh.model.occ.synchronize()

# Order volumes as rotor, stator, coil1, coil2, air using the fragment map
# (newer gmsh does not return the fragmented volumes in input order)
volumes = [parts[1][i][0] for i in range(1, 5)]
volumes += [v for v in parts[0] if v not in volumes]

names = "rotor", "stator", "coil1", "coil2", "air"
grp = 0
for i, (w, name) in enumerate(zip(volumes, names)):

    # Get the 2D entities in the BoundingBox of the 3D entity
    # NB some of these are repeated, but the first tag to claim
    # an entity will take priority
    bb = gmsh.model.getBoundingBox(*w)
    ent2 = gmsh.model.getEntitiesInBoundingBox(*bb, dim=2)
    idx = [v for (d, v) in ent2]
    nr1 = []
    nr4 = []
    for part in ent2:
        cm = gmsh.model.occ.getCenterOfMass(*part)[1]
        if np.isclose(cm, bb[1]):
            nr1 += [part[1]]
        elif np.isclose(cm, bb[4]):
            nr4 += [part[1]]
    if "coil" in name:
        grp += 1
        gmsh.model.addPhysicalGroup(2, nr1, grp)
        gmsh.model.setPhysicalName(2, grp, "nr1_" + name)
        grp += 1
        gmsh.model.addPhysicalGroup(2, nr4, grp)
        gmsh.model.setPhysicalName(2, grp, "nr4_" + name)

    grp += 1
    gmsh.model.addPhysicalGroup(2, idx, grp)
    gmsh.model.setPhysicalName(2, grp, "surf_" + name)

    # Label 3D entities
    grp += 1
    gmsh.model.addPhysicalGroup(3, [w[1]], grp)
    gmsh.model.setPhysicalName(3, grp, name)

gmsh.model.mesh.setSize(gmsh.model.getEntities(0), args.size)

gmsh.model.mesh.generate(3)

# Uniform refinement (new nodes are placed on the CAD geometry)
for _ in range(args.refine):
    gmsh.model.mesh.refine()

gmsh.write(f"{args.output}.msh")

mesh_data = model_to_mesh(gmsh.model, MPI.COMM_WORLD, 0, gdim=3)
mesh = mesh_data.mesh
ct = mesh_data.cell_tags
ft = mesh_data.facet_tags
ct.name = "Cell tags"
ft.name = "Facet tags"

# Facet-to-cell connectivity is needed to write facet tags
mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)

num_cells = mesh.topology.index_map(mesh.topology.dim).size_global
if mesh.comm.rank == 0:
    print(f"{args.output}: {num_cells} cells (size {args.size} mm, {args.refine} refinements)")

with XDMFFile(MPI.COMM_WORLD, f"{args.output}.xdmf", "w") as xdmf:
    xdmf.write_mesh(mesh)
    xdmf.write_meshtags(ft, mesh.geometry)
    xdmf.write_meshtags(ct, mesh.geometry)

gmsh.finalize()