import vtk

def load_vtk_unstructured_grid(filename):
    """
    Loads a VTK unstructured grid file and returns its data object.

    Args:
        filename (str): Path to the VTK file.

    Returns:
        vtkUnstructuredGrid: Loaded unstructured grid.
    """
    reader = vtk.vtkUnstructuredGridReader()
    reader.SetFileName(filename)
    reader.Update()
    return reader.GetOutput()

def convert_to_polydata(unstructured_grid):
    """
    Converts a VTK unstructured grid to a surface mesh (vtkPolyData).

    Args:
        unstructured_grid (vtkUnstructuredGrid): The input volume mesh.

    Returns:
        vtkPolyData: The surface (geometry) mesh.
    """
    surface_filter = vtk.vtkGeometryFilter()
    surface_filter.SetInputData(unstructured_grid)
    surface_filter.Update()
    return surface_filter.GetOutput()

def cap_bowl(polydata):
    """
    Caps the open boundary of a bowl-like polydata surface mesh to create a closed surface.
    Essential for generating a watertight mesh for volume computations.

    Args:
        polydata (vtkPolyData): Open surface mesh.

    Returns:
        vtkPolyData: Closed surface mesh.
    """
    # Extract open edges (the boundary of the 'bowl')
    boundary_edges = vtk.vtkFeatureEdges()
    boundary_edges.SetInputData(polydata)
    boundary_edges.BoundaryEdgesOn()
    boundary_edges.FeatureEdgesOff()
    boundary_edges.NonManifoldEdgesOff()
    boundary_edges.ManifoldEdgesOff()
    boundary_edges.Update()

    # Convert edges to points and lines
    boundary_polydata = vtk.vtkPolyData()
    boundary_polydata.SetPoints(boundary_edges.GetOutput().GetPoints())
    boundary_polydata.SetLines(boundary_edges.GetOutput().GetLines())

    # Triangulate the boundary to form a cap
    delaunay = vtk.vtkDelaunay2D()
    delaunay.SetInputData(boundary_polydata)
    delaunay.Update()

    # Combine the cap with the original mesh to make closed surface
    append_filter = vtk.vtkAppendPolyData()
    append_filter.AddInputData(polydata)
    append_filter.AddInputData(delaunay.GetOutput())
    append_filter.Update()

    # Clean to remove duplicate points and fix topology
    clean_filter = vtk.vtkCleanPolyData()
    clean_filter.SetInputData(append_filter.GetOutput())
    clean_filter.Update()

    return clean_filter.GetOutput()

def create_solid_mesh(closed_surface):
    """
    Generates a solid 3D mesh from a closed surface using Delaunay tetrahedralization.

    Args:
        closed_surface (vtkPolyData): Closed surface mesh.

    Returns:
        vtkPolyData: Surface mesh of the generated solid.
    """
    # Delaunay3D will fill the interior to create a solid mesh
    delaunay3D = vtk.vtkDelaunay3D()
    delaunay3D.SetInputData(closed_surface)
    delaunay3D.Update()

    # Extracts the surface of the resulting tetrahedral mesh
    surface_filter = vtk.vtkDataSetSurfaceFilter()
    surface_filter.SetInputConnection(delaunay3D.GetOutputPort())
    surface_filter.Update()

    return surface_filter.GetOutput()

def calculate_volume(polydata):
    """
    Calculates the volume enclosed by a polydata surface mesh.

    Args:
        polydata (vtkPolyData): A closed surface mesh.

    Returns:
        float: Calculated volume.
    """
    mass_props = vtk.vtkMassProperties()
    mass_props.SetInputData(polydata)
    volume = mass_props.GetVolume()
    return volume

def cal_bp_volume(vtk_file_path):
    """
    Calculates the blood pool volume and muscle volume from a VTK mesh file.
    - Loads the mesh
    - Extracts & caps the open surface
    - Creates a closed solid mesh
    - Computes volumes for heart muscle and total volume of muscle + blood pool
    - The blood pool is the difference between the total and muscle volumes.

    Args:
        vtk_file_path (str): Path to the .vtk file (unstructured grid).

    Returns:
        tuple: (blood_pool_volume, muscle_volume)
    """
    # Load VTK unstructured grid
    unstructured_grid = load_vtk_unstructured_grid(vtk_file_path)

    # Get surface mesh as polydata
    surface_mesh = convert_to_polydata(unstructured_grid)

    # Cap the open surface to create a closed surface
    closed_mesh = cap_bowl(surface_mesh)

    # Create a solid mesh from the closed surface
    solid_mesh = create_solid_mesh(closed_mesh)

    #  Compute total (solid) volume (blood pool + muscle)
    total_volume = calculate_volume(solid_mesh)

    # Compute muscle volume (original surface mesh)
    musc_volume = calculate_volume(surface_mesh)

    # Compute blood pool volume as difference
    bp_volume = total_volume - musc_volume

    return bp_volume, musc_volume