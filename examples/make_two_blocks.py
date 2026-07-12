"""Generate the assets for `two_blocks.yaml`: a two-body tet mesh and a selector.

Run from the repo root:

    .venv/bin/python examples/make_two_blocks.py

Writes `examples/two_blocks.msh` (binary Gmsh 2.2, the format `mesh_io` reads) and
`examples/two_blocks_table.stl` (a box enclosing the lower block, which the Dirichlet selector
picks out as the "table").

The tets use the **Kuhn (Freudenthal) 6-tet split**, not the more common 5-tet one. The 5-tet
split only conforms if adjacent cells alternate on a checkerboard: apply one pattern everywhere
and neighbouring cells disagree about which way to cut each shared face, so interior faces never
pair up and get reported as *surface* faces. Self-collision then finds phantom contacts deep
inside the solid at rest, pushing the body apart from within. Kuhn routes every tet through the
same main diagonal and so conforms unconditionally -- and `assemble_problem` will reject a mesh
that does not, so this is not a matter of taste.
"""

from pathlib import Path
import struct

import numpy as np

SPACING = 0.25
CELLS = 2  # cells per axis, so 3 vertices per axis per block
GAP = 4.0e-3  # the upper block starts just clear of the lower one


def _block(origin):
    """Return (positions, tets) for one axis-aligned block of Kuhn-split cells."""
    n = CELLS + 1
    grid = np.stack(np.meshgrid(*(np.arange(n),) * 3, indexing="ij"), -1)
    positions = grid.reshape(-1, 3).astype(np.float64) * SPACING + np.asarray(origin)

    index = lambda i, j, k: (i * n + j) * n + k
    tets = []
    for i in range(CELLS):
        for j in range(CELLS):
            for k in range(CELLS):
                corner = [
                    index(i + a, j + b, k + c)
                    for a in (0, 1)
                    for b in (0, 1)
                    for c in (0, 1)
                ]
                v000, v001, v010, v011, v100, v101, v110, v111 = corner
                tets += [
                    [v000, v100, v110, v111],
                    [v000, v101, v100, v111],
                    [v000, v110, v010, v111],
                    [v000, v010, v011, v111],
                    [v000, v001, v101, v111],
                    [v000, v011, v001, v111],
                ]
    return positions, np.asarray(tets, dtype=np.int64)


def _write_gmsh22_binary(path, positions, tets):
    """Write the binary Gmsh 2.2 subset that `parse_gmsh22_binary_tets` reads."""
    out = bytearray()
    out += b"$MeshFormat\n2.2 1 8\n"
    out += struct.pack("<i", 1)  # endianness marker
    out += b"\n$EndMeshFormat\n"

    out += b"$Nodes\n%d\n" % len(positions)
    for tag, (x, y, z) in enumerate(positions, start=1):
        out += struct.pack("<i", tag) + struct.pack("<ddd", float(x), float(y), float(z))
    out += b"\n$EndNodes\n"

    out += b"$Elements\n%d\n" % len(tets)
    # One header for the whole homogeneous block: (type=4 tetra, count, num_tags=2).
    out += struct.pack("<iii", 4, len(tets), 2)
    for element, tet in enumerate(tets, start=1):
        out += struct.pack("<iii", element, 1, 1)  # tag, physical, geometrical
        out += struct.pack("<iiii", *(int(v) + 1 for v in tet))
    # No separator newline here: the reader loops until it *finds* `$EndElements`, so any byte
    # between the last record and the marker is parsed as another element header.
    out += b"$EndElements\n"

    Path(path).write_bytes(bytes(out))


def _write_box_stl(path, lower, upper):
    """Write an axis-aligned box as binary STL: the selector volume."""
    x0, y0, z0 = lower
    x1, y1, z1 = upper
    corners = [
        (x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
        (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1),
    ]
    faces = [
        (0, 3, 2), (0, 2, 1),  # bottom
        (4, 5, 6), (4, 6, 7),  # top
        (0, 1, 5), (0, 5, 4),
        (1, 2, 6), (1, 6, 5),
        (2, 3, 7), (2, 7, 6),
        (3, 0, 4), (3, 4, 7),
    ]

    out = bytearray(b"\0" * 80)
    out += struct.pack("<I", len(faces))
    for a, b, c in faces:
        pa, pb, pc = (np.asarray(corners[i]) for i in (a, b, c))
        normal = np.cross(pb - pa, pc - pa)
        length = np.linalg.norm(normal)
        normal = normal / length if length else normal
        out += struct.pack("<3f", *normal.astype(np.float32))
        for point in (pa, pb, pc):
            out += struct.pack("<3f", *point.astype(np.float32))
        out += struct.pack("<H", 0)
    Path(path).write_bytes(bytes(out))


def main():
    here = Path(__file__).parent
    height = SPACING * CELLS

    lower_positions, lower_tets = _block((0.0, 0.0, 0.0))
    upper_positions, upper_tets = _block((0.0, 0.0, height + GAP))

    positions = np.concatenate([lower_positions, upper_positions])
    tets = np.concatenate([lower_tets, upper_tets + len(lower_positions)])
    _write_gmsh22_binary(here / "two_blocks.msh", positions, tets)

    # A box that contains the lower block and nothing of the upper one.
    margin = 0.1 * SPACING
    _write_box_stl(
        here / "two_blocks_table.stl",
        (-margin, -margin, -margin),
        (height + margin, height + margin, height + 0.5 * GAP),
    )

    print(f"two_blocks.msh: {len(positions)} vertices, {len(tets)} tets")
    print("two_blocks_table.stl: selector box around the lower block")


if __name__ == "__main__":
    main()
