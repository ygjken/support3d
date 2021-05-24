import open3d as o3d
from PIL import Image
import numpy as np


def rotated_view_2_gif(data_path, output_file):
    """This the function to make a rotating .gif from .ply

    Args:
        data_path (str): path to .ply file
        output_file (str): path to .gif file
    """
    pcd = o3d.io.read_point_cloud(data_path)
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=900, height=900, visible=True)
    vis.add_geometry(pcd)

    ims = []

    for i in range(170):
        ctr = vis.get_view_control()
        ro = vis.get_render_option()
        ro.point_size = 1
        ctr.rotate(10.0, 0.0)  # contorl the camera rotation
        vis.poll_events()
        vis.update_renderer()
        im = np.asarray(vis.capture_screen_float_buffer(False))*255
        im = np.uint8(im)
        ims.append(Image.fromarray(im))

    ims[0].save(output_file, save_all=True, append_images=ims[1:])
    vis.destroy_window()
