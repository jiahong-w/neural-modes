import trimesh

def write_obj(vertices, faces, path):
    '''Output OBJ mesh file
    
    Args:
        vertices (tensor): [V, 3]
        faces (tensor): [F, 3]
        path (str): output file path
    Shape:
        V: number of vertices
        F: number of faces
    '''
    mesh = trimesh.Trimesh(vertices=vertices.numpy(), faces=faces.numpy(), process=False)
    mesh.export(path, 'obj')
