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


def draw_active_passive_slab(model: Any, width: float, length: float, total_thickness: float, active_thickness: float, z_offset: float = 0.0, flip: bool = False) -> tuple[int, int]:
    """
    Creates a 3D rectangular slab split into two horizontal layers (active and passive).

    Args:
        model (Any): The active gmsh.model module.
        width (float): X dimension (e.g., in meters).
        length (float): Y dimension (e.g., in meters).
        total_thickness (float): Total Z dimension of the slab (e.g., in meters).
        active_thickness (float): Z dimension of the active power-dissipating layer.
        z_offset (float): Z-axis starting position (e.g., in meters).
        flip (bool): If False, active layer is on top. If True, active layer is on the bottom.

    Returns:
        tuple[int, int]: The OpenCASCADE integer tags for the (active_tag, passive_tag).
    """
    passive_thickness = total_thickness - active_thickness

    if not flip:
        # Standard: Passive on bottom, Active on top
        passive_tag = model.occ.addBox(-width/2, -length/2, z_offset, width, length, passive_thickness)
        active_tag = model.occ.addBox(-width/2, -length/2, z_offset + passive_thickness, width, length, active_thickness)
    else:
        # Flipped: Active on bottom, Passive on top
        active_tag = model.occ.addBox(-width/2, -length/2, z_offset, width, length, active_thickness)
        passive_tag = model.occ.addBox(-width/2, -length/2, z_offset + active_thickness, width, length, passive_thickness)

    return active_tag, passive_tag