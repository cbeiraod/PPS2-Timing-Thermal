import gmsh
import math
from typing import Any
from .common import draw_slab

def create_silicon_slab(model: Any, width: float, length: float, thickness: float, z_offset: float = 0.0, name: str = "SiSlab", tag_start: int = 1) -> dict:
    """
    Creates a 3D rectangular slab and tags its faces specifically for the simple standalone ASIC simulation.

    Args:
        model (Any): The active gmsh.model module.
        width (float): X dimension (e.g., in meters).
        length (float): Y dimension (e.g., in meters).
        thickness (float): Z dimension (e.g., in meters).
        z_offset (float): Z-axis starting position (e.g., in meters).
        name (str): Prefix for naming physical groups. Defaults to "SiSlab".
        tag_start (int): Starting integer for physical group IDs (ensures uniqueness).

    Returns:
        dict: A dictionary mapping topological string names ("occ_tag", "volume",
              "face_bottom", "face_top", "faces_side") to their assigned integer tags.
    """
    # 1. Create the base geometry using the common library
    box_tag = draw_slab(model, width, length, thickness, z_offset)

    # 2. Synchronize the geometry NOW so we can search its faces
    model.occ.synchronize()

    # 3. Extract faces and their Z-centers
    surfaces = model.getEntities(dim=2)
    face_data = []

    for dim, tag in surfaces:
        bbox = model.getBoundingBox(dim, tag)
        # Calculate the geometric center of the face in the Z-axis: (zmin + zmax) / 2
        z_center = (bbox[2] + bbox[5]) / 2.0
        face_data.append((tag, z_center))

    # Find the absolute extreme Z values of the generated shape
    min_z = min(f[1] for f in face_data)
    max_z = max(f[1] for f in face_data)

    face_bottom = []
    face_top = []
    faces_side = []

    # 3. Categorize using Python's robust floating-point comparison
    # rel_tol=1e-9 handles scaling (works for kilometers or nanometers)
    # abs_tol=1e-12 handles values infinitesimally close to pure zero
    for tag, z_center in face_data:
        if math.isclose(z_center, min_z, rel_tol=1e-9, abs_tol=1e-12):
            face_bottom.append(tag)
        elif math.isclose(z_center, max_z, rel_tol=1e-9, abs_tol=1e-12):
            face_top.append(tag)
        else:
            faces_side.append(tag)

    # 4. Create Physical Groups
    tags = {
        "occ_tag": box_tag,
        "volume": tag_start,
        "face_bottom": tag_start + 1,
        "face_top": tag_start + 2,
        "faces_side": tag_start + 3
    }

    model.addPhysicalGroup(3, [box_tag], tags["volume"], name=f"{name}_Volume")
    model.addPhysicalGroup(2, face_bottom, tags["face_bottom"], name=f"{name}_Bottom")
    model.addPhysicalGroup(2, face_top, tags["face_top"], name=f"{name}_Top")
    model.addPhysicalGroup(2, faces_side, tags["faces_side"], name=f"{name}_Sides")

    return tags
