import numpy as np
import gmsh
from dolfinx.io import XDMFFile
from dolfinx.io.gmsh import read_from_msh
from mpi4py import MPI


def coil_horseshoe(r, dx, dy, dz, open_end="top"):
    """
    Builds a closed rectangular coil frame then cuts one face open
    to produce a horseshoe / C-shape.

    Parameters
    ----------
    r        : corner fillet radius
    dx, dy   : inner cavity width and depth
    dz       : coil thickness (height along z before any rotation)
    open_end : which face to open up.
               "top"    → cuts away the +y face  (default)
               "bottom" → cuts away the -y face
               "left"   → cuts away the -x face
               "right"  → cuts away the +x face
    """
    # ── 1. Build the closed frame (your original logic) ──────────────────
    b1 = gmsh.model.occ.addBox(0.0, r, 0.0, 2 * r + dx, dy, dz)
    b2 = gmsh.model.occ.addBox(r, 0.0, 0.0, dx, 2 * r + dy, dz)
    c1 = gmsh.model.occ.addCylinder(r, r, 0.0, 0, 0, dz, r)
    c2 = gmsh.model.occ.addCylinder(r, r + dy, 0.0, 0, 0, dz, r)
    c3 = gmsh.model.occ.addCylinder(r + dx, r, 0.0, 0, 0, dz, r)
    c4 = gmsh.model.occ.addCylinder(r + dx, r + dy, 0.0, 0, 0, dz, r)

    outer, _ = gmsh.model.occ.fuse(
        [(3, b1)], [(3, b2), (3, c1), (3, c2), (3, c3), (3, c4)]
    )

    inner_box = gmsh.model.occ.addBox(r, r, 0.0, dx, dy, dz)
    frame, _ = gmsh.model.occ.cut([outer[0]], [(3, inner_box)])

    # ── 2. Define a slab that slices off one arm of the frame ─────────────
    # Total bounding box of the closed frame runs from:
    #   x: 0  → 2*r + dx
    #   y: 0  → 2*r + dy
    #   z: 0  → dz
    # We build an oversized slab aligned with the chosen open face.
    pad = 1.0  # small oversize so the cut is clean
    total_x = 2 * r + dx
    total_y = 2 * r + dy

    if open_end == "top":
        # Remove the +y arm  (y from r+dy to total_y)
        slab = gmsh.model.occ.addBox(
            -pad, r + dy, -pad, total_x + 2 * pad, r + pad, dz + 2 * pad
        )
    elif open_end == "bottom":
        # Remove the -y arm  (y from 0 to r)
        slab = gmsh.model.occ.addBox(
            -pad, -pad, -pad, total_x + 2 * pad, r + pad, dz + 2 * pad
        )
    elif open_end == "right":
        # Remove the +x arm  (x from r+dx to total_x)
        slab = gmsh.model.occ.addBox(
            r + dx, -pad, -pad, r + pad, total_y + 2 * pad, dz + 2 * pad
        )
    elif open_end == "left":
        # Remove the -x arm  (x from 0 to r)
        slab = gmsh.model.occ.addBox(
            -pad, -pad, -pad, r + pad, total_y + 2 * pad, dz + 2 * pad
        )
    else:
        raise ValueError(
            f"open_end must be 'top','bottom','left','right'; got {open_end!r}"
        )

    horseshoe, _ = gmsh.model.occ.cut([frame[0]], [(3, slab)])
    return horseshoe[0]


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

domain_tags = {
    "air": 1,
    "coil1": 2,
    "coil2": 3,
    "rotor": 4,
    "stator": 5,
}

boundary_tags = {
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
}


gmsh.initialize()
gmsh.model.add("team24_horseshoe")


H = 25.4
M = 150.0 # Height of the outer air box
W = 150.0 # Half width of outer air box
D = 63 - 17.0 / 2.0

air_z0 = -M
air_dz = H + 2 * M
z_top = air_z0 + air_dz

r_coil = 24.0
coil_z_offset = 27.0  # the magnitude of the z translation

height = H + M + coil_z_offset - r_coil

coil1 = coil_horseshoe(24.0, 34.0, height, 17.0, open_end="top")
gmsh.model.occ.rotate([(3, coil1[1])], 0, 0, 0, 1, 0, 0, np.pi / 2.0)
gmsh.model.occ.translate([(3, coil1[1])], -41.0, -D, -27.0)

height_2_offset = height - 31.0

coil2 = coil_horseshoe(24.0, 34.0, 31.0 + height_2_offset, 17.0, open_end="bottom")
gmsh.model.occ.rotate([(3, coil2[1])], 0, 0, 0, 1, 0, 0, -np.pi / 2.0)
gmsh.model.occ.translate([(3, coil2[1])], -41.0, D, -27.0 + 79.0 + height_2_offset)

rotor1 = rotor(H)
gmsh.model.occ.rotate([rotor1[0][0]], 0, 0, 0, 0, 0, 1, -22.0 * np.pi / 180.0)

r_champ = 5.0
stator1 = stator(H, r_champ)

outer_box = gmsh.model.occ.addBox(-W, -W, -M, 2 * W, 2 * W, H + 2 * M)

parts, parts_map = gmsh.model.occ.fragment(
    [(3, outer_box)],
    [stator1[0][0], rotor1[0][0], coil1, coil2],
)

gmsh.model.occ.synchronize()
gmsh.model.occ.removeAllDuplicates()

# # Iterate through each tuple in the list, and generate tags starting from 1
# cell_entities = gmsh.model.getEntities(3)
# for i, entity in enumerate(cell_entities, start=1):
#     tag = i
#     gmsh.model.addPhysicalGroup(3, [entity[1]], tag=tag)

air_entities    = [e[1] for e in parts_map[0]][0]  # outer_box remainder — may be >1 piece!
stator_entities = [e[1] for e in parts_map[1]]
rotor_entities  = [e[1] for e in parts_map[2]]
coil1_entities  = [e[1] for e in parts_map[3]]
coil2_entities  = [e[1] for e in parts_map[4]]

gmsh.model.addPhysicalGroup(3, [air_entities],    tag=domain_tags["air"])
gmsh.model.addPhysicalGroup(3, stator_entities, tag=domain_tags["stator"])
gmsh.model.addPhysicalGroup(3, rotor_entities,  tag=domain_tags["rotor"])
gmsh.model.addPhysicalGroup(3, coil1_entities,  tag=domain_tags["coil1"])
gmsh.model.addPhysicalGroup(3, coil2_entities,  tag=domain_tags["coil2"])

# # Iterate over each facet entity
# facet_entities = gmsh.model.getEntities(2)
# for entity in facet_entities:
#     dim, tag = entity  # Extract the dimension and tag of the entity
#     gmsh.model.addPhysicalGroup(dim, [tag], tag=tag)  # Use the tag as the physical group tag

boundary_surfaces_stator = gmsh.model.getBoundary([(3, 4)], oriented=False, combined=False)
stator_tags = []
for i, j in boundary_surfaces_stator:
    stator_tags.append(j)

boundary_surfaces_rotor = gmsh.model.getBoundary([(3, 3)], oriented=False, combined=False)
rotor_tags = []
for i, j in boundary_surfaces_rotor:
    rotor_tags.append(j)

gmsh.model.addPhysicalGroup(2, [4], tag= boundary_tags["coil1_out"])
gmsh.model.addPhysicalGroup(2, [10], tag= boundary_tags["coil1_in"])
gmsh.model.addPhysicalGroup(2, [14], tag= boundary_tags["coil2_in"])
gmsh.model.addPhysicalGroup(2, [22], tag= boundary_tags["coil2_out"])
gmsh.model.addPhysicalGroup(2, stator_tags, tag= boundary_tags["stator_outer"])
gmsh.model.addPhysicalGroup(2, rotor_tags, tag= boundary_tags["rotor_outer"])
gmsh.model.addPhysicalGroup(2, [57], tag= boundary_tags["symmetry"])
gmsh.model.addPhysicalGroup(2, [56, 58, 59], tag= boundary_tags["outer"])


gmsh.model.mesh.setSize(gmsh.model.getEntities(0), 10)  # global coarse

def get_points(vol_tags):
    pts = gmsh.model.getBoundary([(3, v) for v in vol_tags], oriented=False, recursive=True)
    return [(0, tag) for _, tag in pts]

size = 6
gmsh.model.mesh.setSize(get_points(coil1_entities),  size)
gmsh.model.mesh.setSize(get_points(coil2_entities),  size)
gmsh.model.mesh.setSize(get_points(stator_entities), size)
gmsh.model.mesh.setSize(get_points(rotor_entities),  size)

gmsh.option.setNumber("Mesh.ScalingFactor", 1e-3) # Scale from mm to m

gmsh.model.mesh.generate(3)
gmsh.model.mesh.optimize("Netgen")

gmsh.write("team24_horseshoe.msh")
gmsh.finalize()

meshes = read_from_msh("team24_horseshoe.msh", MPI.COMM_WORLD, 0)
mesh, cell_markers, facet_markers = meshes[0], meshes[1], meshes[2]
cell_markers.name = "Cell_markers"
facet_markers.name = "Facet_markers"

with XDMFFile(mesh.comm, "team24_horseshoe.xdmf", "w") as xdmf:
    xdmf.write_mesh(mesh)
    xdmf.write_meshtags(cell_markers, mesh.geometry)
    xdmf.write_meshtags(facet_markers, mesh.geometry)

print("Done. Mesh written to team24_horseshoe.xdmf")

