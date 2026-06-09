import gmsh
import math
from typing import Any
from .common import draw_active_passive_slab

def create_detailed_silicon_slab(model: Any, width: float, length: float, total_thickness: float, active_thickness: float, z_offset: float = 0.0, flip: bool = False, name: str = "DetailedSiSlab", tag_start: int = 1) -> dict:
    """
    Creates a 3D rectangular slab split into active/passive layers and tags volumes and faces.

    Args:
        model (Any): The active gmsh.model module.
        width (float): X dimension in meters.
        length (float): Y dimension in meters.
        total_thickness (float): Total Z dimension in meters.
        active_thickness (float): Z dimension of the active layer in meters.
        z_offset (float): Z-axis starting position.
        flip (bool): False = active on top, True = active on bottom.
        name (str): Prefix for naming physical groups.
        tag_start (int): Starting integer for physical group IDs.

    Returns:
        dict: Topological string names mapped to their integer tags.
    """
    # 1. Create the raw geometry from the common library
    active_tag, passive_tag = draw_active_passive_slab(model, width, length, total_thickness, active_thickness, z_offset, flip)

    # 2. FRAGMENT: This is critical for multi-layer FEM! It forces OpenCASCADE to
    # compute the intersection and share nodes at the boundary.
    model.occ.fragment([(3, active_tag)], [(3, passive_tag)])
    model.occ.synchronize()

    # Calculate the exact Z-coordinate of the internal interface
    if flip:
        z_internal = z_offset + active_thickness
    else:
        z_internal = z_offset + (total_thickness - active_thickness)

    # 3. Identify Volumes (since fragmenting can sometimes reassign tags, we search by coordinates)
    volumes = model.getEntities(dim=3)
    vol_active = None
    vol_passive = None

    for dim, tag in volumes:
        bbox = model.getBoundingBox(dim, tag)
        z_center = (bbox[2] + bbox[5]) / 2.0

        if flip:
            if z_center < z_internal: vol_active = tag
            else: vol_passive = tag
        else:
            if z_center > z_internal: vol_active = tag
            else: vol_passive = tag

    # 4. Extract faces and their Z-centers
    surfaces = model.getEntities(dim=2)
    face_data = []

    for dim, tag in surfaces:
        bbox = model.getBoundingBox(dim, tag)
        z_center = (bbox[2] + bbox[5]) / 2.0
        face_data.append((tag, z_center))

    min_z = min(f[1] for f in face_data)
    max_z = max(f[1] for f in face_data)

    face_bottom = []
    face_top = []
    faces_side = []

    # 5. Categorize using Python's robust floating-point comparison
    for tag, z_center in face_data:
        if math.isclose(z_center, min_z, rel_tol=1e-9, abs_tol=1e-12):
            face_bottom.append(tag)
        elif math.isclose(z_center, max_z, rel_tol=1e-9, abs_tol=1e-12):
            face_top.append(tag)
        elif math.isclose(z_center, z_internal, rel_tol=1e-9, abs_tol=1e-12):
            # This is the internal interface between the active/passive layers.
            # We explicitly ignore it because we only want external boundaries for convection.
            pass
        else:
            faces_side.append(tag)

    # 6. Create Physical Groups
    tags = {
        "occ_active": vol_active,
        "occ_passive": vol_passive,
        "vol_active": tag_start,
        "vol_passive": tag_start + 1,
        "face_bottom": tag_start + 2,
        "face_top": tag_start + 3,
        "faces_side": tag_start + 4
    }

    model.addPhysicalGroup(3, [vol_active], tags["vol_active"], name=f"{name}_ActiveVolume")
    model.addPhysicalGroup(3, [vol_passive], tags["vol_passive"], name=f"{name}_PassiveVolume")
    model.addPhysicalGroup(2, face_bottom, tags["face_bottom"], name=f"{name}_Bottom")
    model.addPhysicalGroup(2, face_top, tags["face_top"], name=f"{name}_Top")
    model.addPhysicalGroup(2, faces_side, tags["faces_side"], name=f"{name}_Sides")

    return tags