"""Mesh import helpers."""

from pathlib import Path
import struct

import jax.numpy as jnp


def parse_gmsh22_binary_tets(path: str | Path, dtype=jnp.float32):
    """Parse a binary Gmsh 2.2 mesh and return positions and zero-based tets."""
    data = Path(path).read_bytes()

    nodes_start = data.find(b"$Nodes\n")
    if nodes_start < 0:
        raise ValueError("Missing $Nodes section in Gmsh mesh")
    nodes_start += len(b"$Nodes\n")
    nodes_count_end = data.find(b"\n", nodes_start)
    num_nodes = int(data[nodes_start:nodes_count_end])
    pos = nodes_count_end + 1

    node_tags = []
    node_positions = []
    for _ in range(num_nodes):
        tag = struct.unpack("<i", data[pos : pos + 4])[0]
        x, y, z = struct.unpack("<ddd", data[pos + 4 : pos + 28])
        pos += 28
        node_tags.append(tag)
        node_positions.append((x, y, z))

    tag_to_index = {tag: idx for idx, tag in enumerate(node_tags)}

    elements_start = data.find(b"$Elements\n")
    if elements_start < 0:
        raise ValueError("Missing $Elements section in Gmsh mesh")
    elements_start += len(b"$Elements\n")
    elements_count_end = data.find(b"\n", elements_start)
    _num_elements = int(data[elements_start:elements_count_end])
    pos = elements_count_end + 1
    elements_end = data.find(b"$EndElements\n", pos)

    tets = []
    nodes_per_element = {1: 2, 2: 3, 4: 4, 15: 1}
    while pos < elements_end:
        element_type, count, num_tags = struct.unpack("<iii", data[pos : pos + 12])
        pos += 12
        if element_type not in nodes_per_element:
            raise ValueError(f"Unsupported Gmsh element type {element_type}")

        nodes_per = nodes_per_element[element_type]
        record_width = 4 * (1 + num_tags + nodes_per)
        for _ in range(count):
            fields = struct.unpack(
                "<" + "i" * (1 + num_tags + nodes_per),
                data[pos : pos + record_width],
            )
            pos += record_width
            if element_type == 4:
                node_ids = fields[1 + num_tags :]
                tets.append([tag_to_index[node_id] for node_id in node_ids])

    return jnp.array(node_positions, dtype=dtype), jnp.array(tets, dtype=jnp.int32)
