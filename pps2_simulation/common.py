from typing import Any

def draw_slab(model: Any, width: float, length: float, thickness: float, z_offset: float = 0.0) -> int:
    """
    Creates a pure 3D rectangular slab geometry without assigning physical tags.

    Args:
        model (Any): The active gmsh.model module.
        width (float): X dimension (e.g., in meters).
        length (float): Y dimension (e.g., in meters).
        thickness (float): Z dimension (e.g., in meters).
        z_offset (float): Z-axis starting position (e.g., in meters).

    Returns:
        int: The OpenCASCADE integer tag of the created box.
    """
    # 1. Create the box geometry (queued in OCC)
    box_tag = model.occ.addBox(-width/2, -length/2, z_offset, width, length, thickness)

    return box_tag