
import bpy

def no_padding():
    # Clear existing objects in Blender
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

    # Settings for the checkerboard
    rows = 7
    cols = 10
    cube_width = 0.01  # Width in meters (10mm)
    cube_height = 0.001  # Initial height in meters (1mm), adjustable later
    base_thickness = 0.004  # Base layer thickness in meters (4mm)

    # Calculate the offset to center the board
    offset_x = (cols * cube_width) / 2
    offset_y = (rows * cube_width) / 2

    # Create the checkerboard
    for i in range(rows):
        for j in range(cols):
            # Position calculation to center the grid
            pos_x = (j * cube_width) - offset_x + cube_width / 2
            pos_y = (i * cube_width) - offset_y + cube_width / 2
            
            # Create a cube for the checkerboard
            bpy.ops.mesh.primitive_cube_add(size=cube_width, enter_editmode=False, location=(pos_x, pos_y, base_thickness + cube_height / 2))
            cube = bpy.context.object
            cube.scale.z = cube_height / cube_width  # Set initial Z scale for 1mm height
            

    # Create the base layer
    bpy.ops.mesh.primitive_cube_add(size=cube_width, enter_editmode=False, location=(0, 0, base_thickness / 2))
    base = bpy.context.object
    base.scale = (cols, rows, base_thickness/ cube_width)


    # Update the scene to apply scaling
    bpy.context.view_layer.update()


def padding():
        
    # Clear existing objects in Blender
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

    # Settings for the checkerboard and paddings
    rows = 7
    cols = 10
    cube_size = 0.008  # Cube edge length in meters (8mm)
    cube_height = 0.001  # Cube height in meters (1mm)
    base_thickness = 0.004  # Base layer thickness in meters (4mm)
    padding = 0.002  # Padding around each cube in meters (2mm)
    total_cube_size = cube_size + 2 * padding  # Total space occupied by a cube including padding

    # Calculate the offset to center the board
    offset_x = (cols * total_cube_size) / 2
    offset_y = (rows * total_cube_size) / 2

    # Create the base layer
    bpy.ops.mesh.primitive_cube_add(size=1, enter_editmode=False, location=(0, 0, base_thickness / 2))
    base = bpy.context.object
    base.scale = (cols * total_cube_size, rows * total_cube_size, base_thickness)


    # Create padding mesh
    bpy.ops.mesh.primitive_cube_add(size=1, enter_editmode=False, location=(0, 0, base_thickness + cube_height / 2))
    padding_mesh = bpy.context.object
    padding_mesh.scale = (cols * total_cube_size, rows * total_cube_size, cube_height)  # Scale to create the top layer


    # Create the checkerboard cubes and cut holes in the padding mesh
    for i in range(rows):
        for j in range(cols):
            pos_x = (j * total_cube_size) - offset_x + total_cube_size / 2
            pos_y = (i * total_cube_size) - offset_y + total_cube_size / 2

            # Create a cube for the checkerboard
            bpy.ops.mesh.primitive_cube_add(size=cube_size, enter_editmode=False, location=(pos_x, pos_y, base_thickness + cube_height / 2))
            cube = bpy.context.object
            cube.scale.z = cube_height/cube_size  # Set initial Z scale for 1mm height

            # Use boolean modifier to create a hole in the padding mesh
            mod = padding_mesh.modifiers.new(name=f"Cutout_{i}_{j}", type='BOOLEAN')
            mod.operation = 'DIFFERENCE'
            mod.object = cube
            bpy.context.view_layer.objects.active = padding_mesh
            bpy.ops.object.modifier_apply(modifier=f"Cutout_{i}_{j}")

    # Update the scene to apply changes
    bpy.context.view_layer.update()


def padding_new():
    # Clear existing objects in Blender
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

    # Settings for the checkerboard and paddings
    rows = 7
    cols = 10
    cube_size = 0.008  # Cube edge length in meters (8mm)
    cube_height = 0.0005  # Cube height in meters (0.5mm)
    base_thickness = 0.004  # Base layer thickness in meters (4mm)
    padding = 0.002  # Padding around each cube in meters (2mm)
    total_cube_size = cube_size + padding  # Total space occupied by a cube including padding

    # Calculate the offset to center the board
    total_width = cols * cube_size + (cols + 1) * padding
    total_height = rows * cube_size + (rows + 1) * padding
    offset_x = total_width / 2
    offset_y = total_height / 2

    # Create the base layer
    bpy.ops.mesh.primitive_cube_add(size=1, enter_editmode=False, location=(0, 0, base_thickness / 2))
    base = bpy.context.object
    base.scale = (total_width, total_height, base_thickness)
    base.name = 'base'


    # Create padding mesh
    bpy.ops.mesh.primitive_cube_add(size=1, enter_editmode=False, location=(0, 0, base_thickness + cube_height / 2))
    padding_mesh = bpy.context.object
    padding_mesh.scale = (total_width, total_height, cube_height)  # Scale to create the top layer
    padding_mesh.name = "padding"

    # Create the checkerboard cubes and cut holes in the padding mesh
    for i in range(rows):
        for j in range(cols):
            pos_x = (j * total_cube_size) - offset_x + cube_size / 2 + padding
            pos_y = (i * total_cube_size) - offset_y + cube_size / 2 + padding

            # Create a cube for the checkerboard
            bpy.ops.mesh.primitive_cube_add(size=cube_size, enter_editmode=False, location=(pos_x, pos_y, base_thickness + cube_height / 2))
            cube = bpy.context.object
            cube.scale.z = cube_height/cube_size  # Set initial Z scale for 1mm height

            # Use boolean modifier to create a hole in the padding mesh
            mod = padding_mesh.modifiers.new(name=f"Cutout_{i}_{j}", type='BOOLEAN')
            mod.operation = 'DIFFERENCE'
            mod.object = cube
            bpy.context.view_layer.objects.active = padding_mesh
            bpy.ops.object.modifier_apply(modifier=f"Cutout_{i}_{j}")

    # Update the scene to apply changes
    bpy.context.view_layer.update()



def export_STL():
    import os

    # Define the directory to save the STL files
    output_directory = '/path/to/your/output/directory'

    # Check if the output directory exists, if not create it
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Deselect all objects initially
    bpy.ops.object.select_all(action='DESELECT')

    # Loop through all objects in the scene
    for obj in bpy.data.objects:
        # Select the object
        obj.select_set(True)
        
        # Ensure we only export mesh type objects
        if obj.type == 'MESH':
            # Specify the path and name for the STL file
            file_path = os.path.join(output_directory, f"{obj.name}.stl")
            
            # Export the object as an STL
            bpy.ops.export_mesh.stl(filepath=file_path, use_selection=True)
        
        # Deselect the object ready for the next iteration
        obj.select_set(False)
