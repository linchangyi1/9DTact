import numpy as np
import open3d
from open3d import *
from math import sqrt

final_force = False
reference_frame = True


def get_rotation_matrix_from_2_vectors(v2, v1=None):
    if v1 is None:
        v1 = [0, 0, -1]
    v2 = v2 / np.linalg.norm(v2)
    v1 = v1 / np.linalg.norm(v1)
    if np.allclose(v1, v2):
        return np.eye(3)
    if np.allclose(v1, -v2):
        return -np.eye(3)

    origin = v1
    target = np.array([-1, -1, 1]) * v2
    target[0], target[1] = target[1], target[0]
    v = np.cross(origin, target)
    c = np.dot(origin, target)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix


def draw_vector(x, y, z, site, cylinder_radius=0.5, cone_radius=0.7, cone_height=1):
    x_axis = open3d.geometry.TriangleMesh.create_arrow(
        cylinder_radius=cylinder_radius, cone_radius=cone_radius, cylinder_height=abs(x), cone_height=cone_height)
    y_axis = open3d.geometry.TriangleMesh.create_arrow(
        cylinder_radius=cylinder_radius, cone_radius=cone_radius, cylinder_height=abs(y), cone_height=cone_height)
    z_axis = open3d.geometry.TriangleMesh.create_arrow(
        cylinder_radius=cylinder_radius, cone_radius=cone_radius, cylinder_height=abs(z), cone_height=cone_height)
    xyz = open3d.geometry.TriangleMesh.create_arrow(
        cylinder_radius=cylinder_radius, cone_radius=cone_radius, cylinder_height=sqrt(x ** 2 + y ** 2 + z ** 2),
        cone_height=cone_height)

    if x > 0:
        x_axis.rotate(open3d.geometry.get_rotation_matrix_from_xyz(
            [0, np.pi / 2, 0]), [0, 0, 0])
    else:
        x_axis.rotate(open3d.geometry.get_rotation_matrix_from_xyz(
            [0, -np.pi / 2, 0]), [0, 0, 0])

    if y > 0:
        y_axis.rotate(open3d.geometry.get_rotation_matrix_from_xyz(
            [np.pi / 2, 0, 0]), [0, 0, 0])
    else:
        y_axis.rotate(open3d.geometry.get_rotation_matrix_from_xyz(
            [-np.pi / 2, 0, 0]), [0, 0, 0])

    if z > 0:
        z_axis.rotate(open3d.geometry.get_rotation_matrix_from_xyz(
            [0, 0, np.pi]), [0, 0, 0])
    else:
        z_axis.rotate(open3d.geometry.get_rotation_matrix_from_xyz(
            [0, np.pi, 0]), [0, 0, 0])

    xyz.rotate(get_rotation_matrix_from_2_vectors(
        np.array([1, 1, 1])), [0, 0, 0])

    x_axis.paint_uniform_color([1, 0, 0])
    y_axis.paint_uniform_color([0, 1, 0])
    z_axis.paint_uniform_color([0, 0, 1])
    xyz.paint_uniform_color([0, 0, 0])

    # axis = x_axis + y_axis + z_axis
    x_axis.translate(site)
    y_axis.translate(site)
    z_axis.translate(site)
    xyz.translate(site)

    return x_axis, y_axis, z_axis, xyz


def draw_coordinate_frame(x, y, z, site=None, cylinder_radius=0.05, cone_radius=0.01, cone_height=1):
    if site is None:
        site = [0, 0, 0]
    cx, cy, cz, none = draw_vector(
        x, y, z, site, cylinder_radius, cone_radius, cone_height)

    return cx + cy + cz


class Visualizer:
    def __init__(self, points):
        self.vis = open3d.visualization.Visualizer()
        self.vis.create_window(window_name='9DTact', width=2000, height=1600)

        # load, color, relocate and add sensor objs
        black_base = open3d.io.read_triangle_mesh(
            "../shape_reconstruction/sensor_obj/black_base.obj")
        white_shell = open3d.io.read_triangle_mesh(
            "../shape_reconstruction/sensor_obj/white_shell.obj")
        black_contact = open3d.io.read_triangle_mesh(
            "../shape_reconstruction/sensor_obj/black_contact.obj")
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

        self.force_origin = [45, -30, -2]
        self.torque_origin = [53, -10.5, 21]

        if reference_frame:
            self.force_axis = draw_coordinate_frame(
                10, 10, 10, self.force_origin)
            self.force_axis_1 = draw_coordinate_frame(
                -10, -10, -10, self.force_origin)
            self.torque_axis = draw_coordinate_frame(
                10, 10, 10, self.torque_origin)
            self.torque_axis_1 = draw_coordinate_frame(
                -10, -10, -10, self.torque_origin)
            self.vis.add_geometry(self.force_axis)
            self.vis.add_geometry(self.force_axis_1)
            self.vis.add_geometry(self.torque_axis)
            self.vis.add_geometry(self.torque_axis_1)

        self.force_x, self.force_y, self.force_z, self.force_xyz = draw_vector(
            10, 10, 10, self.force_origin)
        self.torque_x, self.torque_y, self.torque_z, self.torque_xyz = draw_vector(
            10, 10, 10, self.torque_origin)

        self.last_wrench = np.ones(6)
        self.last_force = np.ones(3)
        self.last_torque = np.ones(3)

        self.vis.add_geometry(self.force_x)
        self.vis.add_geometry(self.force_y)
        self.vis.add_geometry(self.force_z)
        self.vis.add_geometry(self.torque_x)
        self.vis.add_geometry(self.torque_y)
        self.vis.add_geometry(self.torque_z)

        if final_force:
            self.vis.add_geometry(self.force_xyz)
            self.vis.add_geometry(self.torque_xyz)

        self.pcd = open3d.geometry.PointCloud()
        self.pcd.points = open3d.utility.Vector3dVector(points)
        self.vis.add_geometry(self.pcd)

        self.colors = np.zeros([points.shape[0], 3])

        self.ctr = self.vis.get_view_control()
        self.ctr.change_field_of_view(-20)
        print("fov", self.ctr.get_field_of_view())
        # self.ctr.convert_to_pinhole_camera_parameters()
        self.ctr.set_zoom(0.5)
        self.ctr.rotate(0, -40)  # mouse drag in x-axis, y-axis
        self.ctr.set_front([0.3, 0.8, -1])
        self.ctr.set_lookat([36, 0, -10])
        self.ctr.set_up([0, 0, -1])
        self.vis.update_renderer()

        self.recall = np.array([-0.5, -0.5, -1, -0.5, -0.5, -0.5])
        self.scale = np.array([8, 8, -4, 10, 10, 20])

    def update_wrench(self, wrench):
        wrench = (wrench + self.recall) * self.scale
        force = wrench[:3].copy()
        torque = wrench[3:].copy()

        last_rotation = get_rotation_matrix_from_2_vectors(
            self.last_force)
        rotation = get_rotation_matrix_from_2_vectors(force)
        rotation = np.matmul(rotation, np.linalg.inv(last_rotation))
        if final_force:
            self.force_xyz.rotate(rotation, center=self.force_origin)
            self.force_xyz.scale(np.linalg.norm(
                force) / np.linalg.norm(self.last_force), center=self.force_origin)
        self.last_force = force

        last_rotation = get_rotation_matrix_from_2_vectors(
            self.last_torque)
        rotation = get_rotation_matrix_from_2_vectors(torque)
        rotation = np.matmul(rotation, np.linalg.inv(last_rotation))
        if final_force:
            self.torque_xyz.rotate(rotation, center=self.torque_origin)
            self.torque_xyz.scale(np.linalg.norm(
                torque) / np.linalg.norm(self.last_torque), center=self.torque_origin)
        self.last_torque = torque

        wrench[wrench == 0] = 0.00001
        wrench /= self.last_wrench
        self.last_wrench = self.last_wrench * wrench

        self.force_x.scale(wrench[0], center=self.force_origin)
        self.force_y.scale(wrench[1], center=self.force_origin)
        self.force_z.scale(wrench[2], center=self.force_origin)
        self.torque_x.scale(wrench[3], center=self.torque_origin)
        self.torque_y.scale(wrench[4], center=self.torque_origin)
        self.torque_z.scale(wrench[5], center=self.torque_origin)

        self.vis.update_geometry(self.force_x)
        self.vis.update_geometry(self.force_y)
        self.vis.update_geometry(self.force_z)
        self.vis.update_geometry(self.torque_x)
        self.vis.update_geometry(self.torque_y)
        self.vis.update_geometry(self.torque_z)

        if final_force:
            self.vis.update_geometry(self.force_xyz)
            self.vis.update_geometry(self.torque_xyz)

    def update(self, points, gradients, wrench):
        self.update_wrench(wrench)

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

