import numpy as np
import open3d
from open3d import *
from math import sqrt
import rospy
from geometry_msgs.msg import WrenchStamped

final_force = False
reference_frame = False


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
            [np.pi, 0, 0]), [0, 0, 0])
    # if y > 0:
    #     y_axis.rotate(open3d.geometry.get_rotation_matrix_from_xyz(
    #         [np.pi / 0.85, 0.45, 0.45]), [0, 0, 0])
    # else:
    #     y_axis.rotate(open3d.geometry.get_rotation_matrix_from_xyz(
    #         [np.pi / 2, 0, 0]), [0, 0, 0])

    if z > 0:
        z_axis.rotate(open3d.geometry.get_rotation_matrix_from_xyz(
            [np.pi / 2, 0, 0]), [0, 0, 0])
    else:
        z_axis.rotate(open3d.geometry.get_rotation_matrix_from_xyz(
            [-np.pi / 2, 0, 0]), [0, 0, 0])

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
    def __init__(self):
        self.vis = open3d.visualization.Visualizer()
        self.vis.create_window(window_name='6D_Force_Estimation', width=2000, height=1200)

        self.force_origin = [-15, 0, 0]
        self.torque_origin = [15, 0, 0]

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

        self.ctr = self.vis.get_view_control()
        self.ctr.change_field_of_view(-20)
        # print("fov", self.ctr.get_field_of_view())
        # self.ctr.convert_to_pinhole_camera_parameters()
        self.ctr.set_zoom(0.4)
        self.ctr.rotate(-80, 120)  # mouse drag in x-axis, y-axis
        self.ctr.set_lookat([0, 0, 5])
        self.vis.update_renderer()
        self.wrench = np.array([.5, .5, 1.0, .5, .5, .5])
        self.recall = np.array([-0.5, -0.5, -1, -0.5, -0.5, -0.5])
        self.scale = np.array([8, 8, -4, 10, 10, 20])

    def update_force(self, wrench):
        self.wrench = wrench
        visualized_wrench = (wrench + self.recall) * self.scale
        force = visualized_wrench[:3].copy()
        torque = visualized_wrench[3:].copy()

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

        visualized_wrench[visualized_wrench == 0] = 0.00001
        visualized_wrench /= self.last_wrench
        self.last_wrench = self.last_wrench * visualized_wrench

        self.force_x.scale(visualized_wrench[0], center=self.force_origin)
        self.force_y.scale(visualized_wrench[1], center=self.force_origin)
        self.force_z.scale(visualized_wrench[2], center=self.force_origin)
        self.torque_x.scale(visualized_wrench[3], center=self.torque_origin)
        self.torque_y.scale(visualized_wrench[4], center=self.torque_origin)
        self.torque_z.scale(visualized_wrench[5], center=self.torque_origin)

        self.vis.update_geometry(self.force_x)
        self.vis.update_geometry(self.force_y)
        self.vis.update_geometry(self.force_z)
        self.vis.update_geometry(self.torque_x)
        self.vis.update_geometry(self.torque_y)
        self.vis.update_geometry(self.torque_z)

        if final_force:
            self.vis.update_geometry(self.force_xyz)
            self.vis.update_geometry(self.torque_xyz)

    def ros_callback(self, msg):
        self.wrench = np.array([msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z,
                                msg.wrench.torque.x, msg.wrench.torque.y, msg.wrench.torque.z])


if __name__ == '__main__':
    visualizer = Visualizer()
    rospy.init_node('force_visualization')
    predicted_force_sub = rospy.Subscriber('/predicted_wrench', WrenchStamped, visualizer.ros_callback)

    while not rospy.is_shutdown():
        if not visualizer.vis.poll_events():
            break
        else:
            visualizer.update_force(visualizer.wrench)
