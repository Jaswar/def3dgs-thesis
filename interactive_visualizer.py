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


def get_new_camera_extrinsic(T, R, zoom):
    w2c = np.eye(4)
    w2c[:3, :3] = R
    w2c[:3, 3] = T
    c2w = np.linalg.inv(w2c)
    c2w_t = c2w[:3, 3]
    c2w_R = c2w[:3, :3]
    c2w_t = c2w_t + zoom * c2w_R @ np.array([0, 0, 1])
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
        pg.init()
        self.width = self.views[0].image_width * 5
        self.height = self.views[0].image_height * 5
        self.screen = pg.display.set_mode((self.width, self.height))

    def run(self):
        running = True
        while running:
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    running = False
                if event.type == pg.MOUSEWHEEL:
                    self.zoom += event.y
            
            keys = pg.key.get_pressed()
            if keys[pg.K_RIGHT]:
                self.current_idx = (self.current_idx + 1) % len(self.views)
            if keys[pg.K_LEFT]:
                self.current_idx = (self.current_idx - 1) % len(self.views)

            self.screen.fill("purple")
            view = self.views[0]
            new_t, new_r = get_new_camera_extrinsic(view.T, view.R, self.zoom)
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
