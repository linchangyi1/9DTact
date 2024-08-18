import numpy as np
import open3d
from open3d import *


class Visualizer:
    def __init__(self, points):
        self.vis = open3d.visualization.Visualizer()
        self.vis.create_window(window_name='9DTact-Shape_Reconstruction', width=2000, height=1600)

        # load, color, relocate and add sensor objs
        black_base = open3d.io.read_triangle_mesh(
            "sensor_obj/black_base.obj")
        white_shell = open3d.io.read_triangle_mesh(
            "sensor_obj/white_shell.obj")
        black_contact = open3d.io.read_triangle_mesh(
            "sensor_obj/black_contact.obj")
        black_base.compute_vertex_normals()
        white_shell.compute_vertex_normals()
        black_contact.compute_vertex_normals()
        black_base.paint_uniform_color([0, 0, 0])
        white_shell.paint_uniform_color([1, 1, 1])
        black_contact.paint_uniform_color([0, 0, 0])
        white_shell.translate([14, -10.5, 2.5])
        black_contact.translate([14, -10.5, 0])
        black_base.translate([14, -10.5, 22])

        self.vis.add_geometry(black_base)
        self.vis.add_geometry(white_shell)
        self.vis.add_geometry(black_contact)

        self.pcd = open3d.geometry.PointCloud()
        self.pcd.points = open3d.utility.Vector3dVector(points)
        self.vis.add_geometry(self.pcd)

        self.colors = np.zeros([points.shape[0], 3])

        self.ctr = self.vis.get_view_control()
        self.ctr.change_field_of_view(-25)
        print("fov", self.ctr.get_field_of_view())
        self.ctr.convert_to_pinhole_camera_parameters()
        self.ctr.set_zoom(0.7)
        self.ctr.rotate(0, -40)  # mouse drag in x-axis, y-axis
        self.ctr.set_front([0.4, 0.8, -1])
        self.ctr.set_lookat([20, 5, -5])
        self.ctr.set_up([0, 0, -1])
        self.vis.update_renderer()

    def update(self, points, gradients):
        dx, dy = gradients
        np_colors = dx + dy
        if abs(np_colors.max()) > 0:
            np_colors = (np_colors - np_colors.min()) / (np_colors.max() - np_colors.min()) * 0.6 + 0.2
        np_colors = np.ndarray.flatten(np_colors)

        # set the non-contact areas as black
        np_colors[points[:, 2] <= 0.08] = 0
        for _ in range(3):
            self.colors[:, _] = np_colors
        self.pcd.points = open3d.utility.Vector3dVector(points)
        self.pcd.colors = open3d.utility.Vector3dVector(self.colors)
        self.vis.update_geometry(self.pcd)
        self.vis.poll_events()
        self.vis.update_renderer()
        return self.pcd
