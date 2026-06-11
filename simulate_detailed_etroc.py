import argparse
import os
import sys
import numpy as np

import gmsh
from pps2_simulation import create_detailed_silicon_slab

def run_simulation(args, thickness, thickness_idx):
    """Generates the mesh and solves the detailed heat equation for a given thickness."""
    print(f"\n--- Running Detailed simulation for ASIC thickness: {thickness*1000:.2f} mm ---")

    # --- 1. MESHING (GMSH) ---
    gmsh.initialize()
    gmsh.model.add(f"DetailedETROC_{thickness_idx}")

    width_m = args.width * 1e-3
    length_m = args.length * 1e-3
    total_thickness_m = thickness * 1e-3
    active_thickness_m = args.active_thickness * 1e-3

    if active_thickness_m >= total_thickness_m:
        print("Error: Active thickness must be less than total thickness.")
        sys.exit(1)

    # Generate the 2-layer topological slab
    tags = create_detailed_silicon_slab(
        gmsh.model, width_m, length_m, total_thickness_m, active_thickness_m,
        z_offset=0.0, flip=args.flip, name="DetailedETROC", tag_start=1
    )

    if args.adaptive_mesh:
        print("Using adaptive meshing (GMSH Fields) to optimize element count...")
        gmsh.model.mesh.field.add("Box", 1)

        # Inside the active layer box, use the tiny mesh size
        gmsh.model.mesh.field.setNumber(1, "VIn", active_thickness_m)
        # Outside the box (the passive bulk), allow a much larger mesh size
        gmsh.model.mesh.field.setNumber(1, "VOut", total_thickness_m / 2.0)

        # Define the box coordinates to tightly wrap the active layer
        # We pad X and Y slightly to avoid precision issues at the boundaries
        gmsh.model.mesh.field.setNumber(1, "XMin", -width_m/2 * 1.05)
        gmsh.model.mesh.field.setNumber(1, "XMax", width_m/2 * 1.05)
        gmsh.model.mesh.field.setNumber(1, "YMin", -length_m/2 * 1.05)
        gmsh.model.mesh.field.setNumber(1, "YMax", length_m/2 * 1.05)

        if args.flip:
            gmsh.model.mesh.field.setNumber(1, "ZMin", -1e-6)
            gmsh.model.mesh.field.setNumber(1, "ZMax", active_thickness_m + 1e-6)
        else:
            gmsh.model.mesh.field.setNumber(1, "ZMin", total_thickness_m - active_thickness_m - 1e-6)
            gmsh.model.mesh.field.setNumber(1, "ZMax", total_thickness_m + 1e-6)

        gmsh.model.mesh.field.setAsBackgroundMesh(1)

        # Disable default size extensions so the field strictly dictates sizing
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        gmsh.option.setNumber("Mesh.MeshSizeMax", total_thickness_m / 2.0)
    else:
        # Force GMSH to use a uniformly tiny mesh size globally
        gmsh.option.setNumber("Mesh.MeshSizeMax", active_thickness_m)

    gmsh.model.mesh.generate(3)

    if args.mesh_only:
        os.makedirs("output", exist_ok=True)
        mesh_filename = f"output/DetailedETROC_{thickness*1000:.1f}mm.vtk"
        gmsh.write(mesh_filename)
        print(f"Mesh generation successful. Saved mesh to {mesh_filename}.")
        gmsh.finalize()
        return

    # --- 2. IMPORT TO FENICSX ---
    try:
        from dolfinx import fem, io
        from dolfinx.fem.petsc import LinearProblem
        import ufl
        from mpi4py import MPI
        import petsc4py.PETSc as PETSc

        try:
            import dolfinx.io.gmsh as gmshio
        except ImportError:
            from dolfinx.io import gmshio

    except ImportError as e:
        print(f"\nError loading FEniCSx dependencies: {e}")
        gmsh.finalize()
        sys.exit(1)

    mesh_output = gmshio.model_to_mesh(gmsh.model, MPI.COMM_WORLD, 0, gdim=3)

    if isinstance(mesh_output, tuple):
        domain = mesh_output[0]
        cell_markers = mesh_output[1]
        facet_markers = mesh_output[2]
    else:
        domain = mesh_output.mesh
        cell_markers = mesh_output.cell_tags
        facet_markers = mesh_output.facet_tags

    gmsh.finalize()

    # --- 3. PHYSICS SETUP ---
    V = fem.functionspace(domain, ("Lagrange", 1))
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    k_silicon = 130.0 # W/mK

    # Calculate volumetric heat generation specifically for the active layer
    active_volume = width_m * length_m * active_thickness_m
    Q_volumetric = args.power / active_volume # W/m^3
    h_conv = 10.0 # W/m^2K

    ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_markers)
    dx = ufl.Measure("dx", domain=domain, subdomain_data=cell_markers)

    # --- 4. BOUNDARY CONDITIONS ---
    heatsink_facets = facet_markers.find(tags["face_bottom"])
    heatsink_dofs = fem.locate_dofs_topological(V, domain.topology.dim - 1, heatsink_facets)

    try:
        from dolfinx import default_scalar_type
        scalar_type = default_scalar_type
    except ImportError:
        scalar_type = PETSc.ScalarType

    bc_heatsink = fem.dirichletbc(scalar_type(args.heatsink_temp), heatsink_dofs, V)
    bcs = [bc_heatsink]

    # --- 5. VARIATIONAL FORMULATION ---
    # Heat conduction applies globally across the entire silicon slab (both layers)
    a = k_silicon * ufl.dot(ufl.grad(u), ufl.grad(v)) * dx

    # Heat generation ONLY happens in the cells tagged as the active volume
    L = Q_volumetric * v * dx(tags["vol_active"])

    if args.convection:
        print(f"Applying convection to top and sides (T_air = {args.air_temp}°C)")
        exposed_surfaces = (tags["face_top"], tags["faces_side"])
        for surf_tag in exposed_surfaces:
            a += h_conv * u * v * ds(surf_tag)
            L += h_conv * args.air_temp * v * ds(surf_tag)

    # --- 6. SOLVE ---
    print("Solving linear system...")
    solver_options = {"ksp_type": "cg", "pc_type": "jacobi", "ksp_rtol": 1e-6}

    try:
        problem = LinearProblem(a, L, bcs=bcs, petsc_options=solver_options, petsc_options_prefix="thermal_")
    except TypeError:
        problem = LinearProblem(a, L, bcs=bcs, petsc_options=solver_options)

    uh = problem.solve()
    uh.name = "Temperature"

    max_temp = domain.comm.allreduce(np.max(uh.x.array), op=MPI.MAX)
    print(f"Max Temperature: {max_temp:.2f} °C")

    # --- 7. EXPORT TO PARAVIEW ---
    os.makedirs("output", exist_ok=True)
    filename = f"output/DetailedETROC_{thickness*1000:.1f}mm.xdmf"
    if args.convection:
        filename = f"output/DetailedETROC_{thickness*1000:.1f}mm_convection.xdmf"
    with io.XDMFFile(domain.comm, filename, "w") as xdmf:
        xdmf.write_mesh(domain)
        xdmf.write_function(uh)
    print(f"Saved results to {filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run detailed 2-layer ASIC thermal simulations.")
    parser.add_argument("--width", type=float, default=21.0, help="ASIC width in mm")
    parser.add_argument("--length", type=float, default=23.0, help="ASIC length in mm")
    parser.add_argument("--thickness", type=float, nargs="+", default=[0.2, 0.4, 0.8], help="Total ASIC thickness in mm")

    # New detailed arguments
    parser.add_argument("--active-thickness", type=float, default=0.05, help="Thickness of active power layer in mm")
    parser.add_argument("--flip", action="store_true", help="Put active layer on the bottom (touching heatsink) instead of top")

    parser.add_argument("--power", type=float, default=1.0, help="Total power in Watts")
    parser.add_argument("--heatsink-temp", type=float, default=20.0, help="Heatsink temperature in C")
    parser.add_argument("--convection", action="store_true", help="Enable natural convection")
    parser.add_argument("--air-temp", type=float, default=20.0, help="Ambient air temperature in C")
    parser.add_argument("--mesh-only", action="store_true", help="Generate meshes only")

    parser.add_argument("--adaptive-mesh", action="store_true", help="Enable GMSH Fields for adaptive mesh sizing")

    args = parser.parse_args()

    for idx, t in enumerate(args.thickness):
        run_simulation(args, t, idx)