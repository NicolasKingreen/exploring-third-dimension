import pygame as pg
import numpy as np

# author: https://www.youtube.com/watch?v=D96wb46mjIQ


SCREEN_W, SCREEN_H = 800, 600
FOV_V = np.pi / 4
FOV_H = FOV_V * SCREEN_W / SCREEN_H


def main():
    pg.init()
    screen = pg.display.set_mode((SCREEN_W, SCREEN_H))
    running = True
    clock = pg.time.Clock()
    surf = pg.surface.Surface((SCREEN_W, SCREEN_H))

    # points = np.array([[1, 1, 1, 1, 1], [4, 2, 0, 1, 1], [1, .5, 3, 1, 1]])
    # triangles = np.asarray([[0, 1, 2]])
    points, triangles = read_obj("teapot.obj")

    z_order = np.zeros(len(triangles))

    camera = np.asarray([13, 0.5, 2, 3.3, 0])

    while running:
        surf.fill([50, 127, 200])
        print(clock.get_fps())

        for event in pg.event.get():
            if event.type == pg.QUIT: running = False
            if event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE: running = False

        project_points(points, camera)
        sort_triangles(points, triangles, camera, z_order)

        for index in np.argsort(z_order):
            triangle = [points[triangles[index][0]][3:], points[triangles[index][1]][3:], points[triangles[index][2]][3:]]

            color = np.abs(points[triangles[index][0]][:3])*45 + 25

            pg.draw.polygon(surf, color, triangle)

            screen.blit(surf, (0, 0)); pg.display.update()


def project_points(points, camera):

    for point in points:
        h_angle_camera_point = np.arctan((point[2]-camera[2])/(point[0]-camera[0] + 1e-16))

        if abs(camera[0] + np.cos(h_angle_camera_point) - point[0]) > abs(camera[0]-point[0]):
            h_angle_camera_point = (h_angle_camera_point - np.pi)%(2*np.pi)

        h_angle = (h_angle_camera_point - camera[3])%(2*np.pi)

        if h_angle > np.pi: h_angle = h_angle - 2*np.pi

        point[3] = SCREEN_W*h_angle/FOV_H + SCREEN_W/2

        distance = np.sqrt((point[0]-camera[0])**2 + (point[1]-camera[1])**2+(point[2]-camera[2])**2)

        v_angle_camera_point = np.arcsin((camera[1]-point[1])/distance)

        v_angle = (v_angle_camera_point - camera[4])%(2*np.pi)

        if v_angle > np.pi: v_angle = v_angle - 2*np.pi

        point[4] = SCREEN_H * v_angle / FOV_V + SCREEN_H / 2


def sort_triangles(points, triangles, camera, z_order):
    for i in range(len(triangles)):
        triangle = triangles[i]

        camera_ray = points[triangle[0]][:3] - camera[:3]
        dist_to_cam = np.sqrt(camera_ray[0]**2 + camera_ray[1]**2 + camera_ray[2]**2)

        z_order[i] = -dist_to_cam


def read_obj(file_name):
    vertices = []
    triangles = []
    f = open(file_name)
    for line in f:
        if line[:2] == "v ":
            index1 = line.find(" ") + 1
            index2 = line.find(" ", index1 + 1)
            index3 = line.find(" ", index2 + 1)

            vertex = [float(line[index1:index2]), float(line[index2:index3]), float(line[index3:-1]), 1, 1]
            vertices.append(vertex)
        elif line[0] == "f":
            index1 = line.find(" ") + 1
            index2 = line.find(" ", index1 + 1)
            index3 = line.find(" ", index2 + 1)

            triangles.append([int(line[index1:index2]) - 1, int(line[index2:index3]) -1, int(line[index3:-1]) - 1])

    f.close()

    return np.asarray(vertices), np.asarray(triangles)


if __name__ == '__main__':
    main()
    pg.quit()
