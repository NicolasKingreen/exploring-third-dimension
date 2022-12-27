import pygame as pg
import numpy as np
from numba import njit

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
    shade = np.zeros(len(triangles))

    while running:
        pg.mouse.set_pos(SCREEN_W/2, SCREEN_H/2)

        elapsed_time = clock.tick() / 1000

        surf.fill([50, 127, 200])
        light_dir = np.asarray([np.sin(pg.time.get_ticks()/1000), 1, 1])
        light_dir = light_dir / np.linalg.norm(light_dir)
        print(int(clock.get_fps()))

        for event in pg.event.get():
            if event.type == pg.QUIT: running = False
            if event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE: running = False

        project_points(points, camera)
        sort_triangles(points, triangles, camera, z_order, light_dir,shade)

        for index in np.argsort(z_order):

            if z_order[index] == 9999: break

            triangle = [points[triangles[index][0]][3:], points[triangles[index][1]][3:], points[triangles[index][2]][3:]]

            color = np.abs(points[triangles[index][0]][:3])*45 + 25

            pg.draw.polygon(surf, color, triangle)

        screen.blit(surf, (0, 0)); pg.display.update()
        pg.display.set_caption(str(round(1/(elapsed_time + 1e-16), 1)) + " " + str(camera))
        movement(camera, elapsed_time)


def movement(camera, elapsed_time):

    if pg.mouse.get_focused():
        p_mouse = pg.mouse.get_pos()
        camera[3] = (camera[3] + 10*elapsed_time*np.clip((p_mouse[0]-SCREEN_W/2)/SCREEN_W, -0.2, .2))%(2*np.pi)
        camera[4] = camera[4] + 10 * elapsed_time * np.clip((p_mouse[1]-SCREEN_H/2)/SCREEN_H, -0.2, .2)
        camera[4] = np.clip(camera[4], -.3, .3)

    pressed_keys = pg.key.get_pressed()

    if pressed_keys[ord('e')]: camera[1] += elapsed_time
    elif pressed_keys[ord('q')]: camera[1] -= elapsed_time

    if (pressed_keys[ord('w')] or pressed_keys[ord('s')]) and (pressed_keys[ord('a')] or pressed_keys[ord('d')]):
        elapsed_time *= 0.707

    if pressed_keys[pg.K_UP] or pressed_keys[ord('w')]:
        camera[0] += elapsed_time * np.cos(camera[3])
        camera[2] += elapsed_time * np.sin(camera[3])

    elif pressed_keys[pg.K_DOWN] or pressed_keys[ord('s')]:
        camera[0] -= elapsed_time * np.cos(camera[3])
        camera[2] -= elapsed_time * np.sin(camera[3])

    if pressed_keys[pg.K_LEFT] or pressed_keys[ord('a')]:
        camera[0] += elapsed_time * np.sin(camera[3])
        camera[2] -= elapsed_time * np.cos(camera[3])

    elif pressed_keys[pg.K_RIGHT] or pressed_keys[ord('d')]:
        camera[0] -= elapsed_time * np.sin(camera[3])
        camera[2] += elapsed_time * np.cos(camera[3])


@njit()
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


@njit()
def sort_triangles(points, triangles, camera, z_order, light_dir, shade):
    for i in range(len(triangles)):
        triangle = triangles[i]

        vet1 = points[triangle[1]][:3] - points[triangle[0]][:3]
        vet2 = points[triangle[2]][:3] - points[triangle[0]][:3]

        normal = np.cross(vet1, vet2)
        normal = normal / np.sqrt(normal[0]**2 + normal[1]**2 + normal[2]**2)

        camera_ray = points[triangle[0]][:3] - camera[:3]
        dist_to_cam = np.sqrt(camera_ray[0]**2 + camera_ray[1]**2 + camera_ray[2]**2)
        camera_ray = camera_ray / dist_to_cam

        if dot_3d(normal, camera_ray) < 0:
            z_order[i] = -dist_to_cam
            shade[i] = 0.5 * dot_3d(light_dir, normal) + 0.5
        else:
            z_order[i] = 9999


@njit()
def dot_3d(arr1, arr2):
    return arr1[0]*arr2[0] + arr1[1]*arr2[1] + arr1[2]*arr2[2]


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
