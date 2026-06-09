import gmsh
import math

def create_silicon_slab(model, width, length, thickness, z_offset=0.0, name="SiSlab", tag_start=1):
    # 1. Create the box geometry
    box_tag = model.occ.addBox(-width/2, -length/2, z_offset, width, length, thickness)
    model.occ.synchronize()

    # 2. Extract faces and their Z-centers
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
