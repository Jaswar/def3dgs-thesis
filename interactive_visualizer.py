import torch
from scene import Scene, DeformModel
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from utils.pose_utils import pose_spherical, render_wander_path
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import imageio
import numpy as np
import time
from utils.image_utils import psnr
import cv2 as cv
import pygame as pg
import pygame_gui as pgui


def get_new_camera_extrinsic(T, R, zoom, rot_x, rot_y, pan_x, pan_y):
    w2c = np.eye(4)
    w2c[:3, :3] = R
    w2c[:3, 3] = T
    c2w = np.linalg.inv(w2c)

    rot_yaw = np.eye(4)
    rot_yaw[:3, :3] = np.array([
        [np.cos(rot_x), 0, np.sin(rot_x)],
        [0, 1, 0],
        [-np.sin(rot_x), 0, np.cos(rot_x)]
    ])

    rot_pitch = np.eye(4)
    rot_pitch[:3, :3] = np.array([
        [1, 0, 0],
        [0, np.cos(rot_y), -np.sin(rot_y)],
        [0, np.sin(rot_y), np.cos(rot_y)]
    ])

    rotation = rot_yaw @ rot_pitch
    c2w = rotation @ c2w
    c2w_t = c2w[:3, 3]
    c2w_R = c2w[:3, :3]
    c2w_t = c2w_t + zoom * c2w_R @ np.array([0, 0, 1])
    c2w_t = c2w_t + pan_x * c2w_R @ np.array([1, 0, 0])
    c2w_t = c2w_t + pan_y * c2w_R @ np.array([0, 1, 0])
    c2w = np.eye(4)
    c2w[:3, :3] = c2w_R
    c2w[:3, 3] = c2w_t
    w2c = np.linalg.inv(c2w)
    return w2c[:3, 3], w2c[:3, :3]


class Visualizer(object):

    def __init__(self, model_path, is_6dof, name, iteration, views, gaussians, pipeline, background, deform):
        self.render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
        self.gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
        self.depth_path = os.path.join(model_path, name, "ours_{}".format(iteration), "depth")
        self.views = views
        self.gaussians = gaussians
        self.pipeline = pipeline
        self.background = background
        self.deform = deform
        self.is_6dof = is_6dof
        self.current_idx = 0

        self.mouse_x, self.mouse_y = 0, 0
        self.zoom = 0
        self.rot_x, self.rot_y = 0, 0
        self.pan_x, self.pan_y = 0, 0

        pg.init()
        display = pg.display.Info()
        self.width = display.current_w
        self.height = int(self.views[0].image_height / self.views[0].image_width * self.width)
        self.screen = pg.display.set_mode((self.width, self.height))
        self.manager = pgui.UIManager((self.width, self.height))
        self.clock = pg.time.Clock()

        self.current_frame_label = pgui.elements.ui_label.UILabel(relative_rect=pg.Rect((0, self.height - 30), (200, 30)), text=f'Current frame: 0/{len(self.views)}', manager=self.manager)

        self.follow_camera = False
        self.follow_camera_button = pgui.elements.ui_button.UIButton(relative_rect=pg.Rect((0, 0), (200, 30)), text='Follow camera', manager=self.manager)
        self.stop_following_button = pgui.elements.ui_button.UIButton(relative_rect=pg.Rect((0, 30), (200, 30)), text='Stop following', manager=self.manager)
        self.stop_following_button.disable()

        self.mouse_wheel_pressed = False
        self.right_button_pressed = False
        self.prev_mouse_x, self.prev_mouse_y = 0, 0


    def run(self):
        running = True
        base_view = self.views[0]
        while running:
            time_delta = self.clock.tick() / 1000.0
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    running = False
                elif event.type == pg.MOUSEWHEEL:
                    self.zoom += event.y
                elif event.type == pg.MOUSEBUTTONDOWN:
                    if event.button == 2:  # scroll wheel press
                        self.mouse_wheel_pressed = True
                        self.prev_mouse_x = event.pos[0]
                        self.prev_mouse_y = event.pos[1]
                    elif event.button == 3:
                        self.right_button_pressed = True
                        self.prev_mouse_x = event.pos[0]
                        self.prev_mouse_y = event.pos[1]
                elif event.type == pg.MOUSEBUTTONUP:
                    if event.button == 2:
                        self.mouse_wheel_pressed = False
                    elif event.button == 3:
                        self.right_button_pressed = False
                elif event.type == pg.MOUSEMOTION:
                    if self.mouse_wheel_pressed:
                        self.rot_x -= (event.pos[0] - self.prev_mouse_x) / (self.width / 2) * np.pi / 8
                        self.rot_y += (event.pos[1] - self.prev_mouse_y) / (self.height / 2) * np.pi / 8
                        self.prev_mouse_x = event.pos[0]
                        self.prev_mouse_y = event.pos[1]
                    elif self.right_button_pressed:
                        self.pan_x -= (event.pos[0] - self.prev_mouse_x) / (self.width / 2)
                        self.pan_y -= (event.pos[1] - self.prev_mouse_y) / (self.height / 2)
                        self.prev_mouse_x = event.pos[0]
                        self.prev_mouse_y = event.pos[1]
                elif event.type == pgui.UI_BUTTON_PRESSED:
                    if event.ui_element == self.follow_camera_button:
                        self.follow_camera = True
                        self.stop_following_button.enable()
                        self.follow_camera_button.disable()
                    elif event.ui_element == self.stop_following_button:
                        self.follow_camera = False
                        self.stop_following_button.disable()
                        self.follow_camera_button.enable()
                self.manager.process_events(event)

            self.current_frame_label.set_text(f'Current frame: {self.current_idx + 1}/{len(self.views)}')
            self.manager.update(time_delta)

            keys = pg.key.get_pressed()
            if keys[pg.K_RIGHT]:
                self.current_idx = (self.current_idx + 1) % len(self.views)
            if keys[pg.K_LEFT]:
                self.current_idx = (self.current_idx - 1) % len(self.views)

            if self.follow_camera:
                self.zoom = 0
                self.rot_x, self.rot_y = 0, 0
                self.pan_x, self.pan_y = 0, 0
                base_view = self.views[self.current_idx]
                view = base_view
                new_t, new_r = base_view.T, base_view.R
            else:
                view = base_view
                new_t, new_r = get_new_camera_extrinsic(view.T, view.R, self.zoom, self.rot_x, self.rot_y, self.pan_x, self.pan_y)
            view.reset_extrinsic(new_r, new_t)

            fid = self.views[self.current_idx].fid
            xyz = self.gaussians.get_xyz
            time_input = fid.unsqueeze(0).expand(xyz.shape[0], -1)
            d_xyz, d_rotation, d_scaling = self.deform.step(xyz.detach(), time_input)
            results = render(view, self.gaussians, self.pipeline, self.background, d_xyz, d_rotation, d_scaling, self.is_6dof)

            rendering = results["render"]
            rendering = rendering.cpu().numpy().transpose(1, 2, 0)
            rendering = np.clip(rendering, 0, 1)
            rendering = (rendering * 255).astype(np.uint8)
            rendering = cv.resize(rendering, (self.width, self.height))
            rendering = np.transpose(rendering, (1, 0, 2))

            self.screen.blit(pg.surfarray.make_surface(rendering), (0, 0))
            self.manager.draw_ui(self.screen)

            pg.display.flip()
        pg.quit() 



def render_sets(dataset: ModelParams, iteration: int, pipeline: PipelineParams, skip_train: bool, skip_test: bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        deform = DeformModel(dataset.is_blender, dataset.is_6dof)
        deform.load_weights(dataset.model_path)

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        visualizer = Visualizer(dataset.model_path, dataset.is_6dof, 'test', scene.loaded_iter, scene.getTestCameras(),
                                gaussians, pipeline, background, deform)
        visualizer.run()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)
