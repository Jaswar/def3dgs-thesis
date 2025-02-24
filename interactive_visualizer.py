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
        cv.namedWindow('visualization')
        cv.setMouseCallback('visualization', self.mouse_callback)

    def mouse_callback(self, event, x, y, flags, param):
        self.mouse_x, self.mouse_y = x, y
        if event == cv.EVENT_MOUSEWHEEL:
            self.zoom += -1 if flags > 0 else 1

    def run(self):
        while True:
            view = self.views[0]
            new_t = view.T + self.zoom * view.R @ np.array([0, 0, -1])
            view.reset_extrinsic(view.R, new_t)
            print(new_t, self.zoom)

            key = cv.waitKey(1) & 0xFF
            if key == 83:
                self.current_idx += 1
                self.current_idx = min(self.current_idx, len(self.views) - 1)
            elif key == 81:
                self.current_idx -= 1
                self.current_idx = max(self.current_idx, 0)
            elif key == ord('q'):
                break
            
            fid = self.views[self.current_idx].fid
            xyz = self.gaussians.get_xyz
            time_input = fid.unsqueeze(0).expand(xyz.shape[0], -1)
            d_xyz, d_rotation, d_scaling = self.deform.step(xyz.detach(), time_input)
            results = render(view, self.gaussians, self.pipeline, self.background, d_xyz, d_rotation, d_scaling, self.is_6dof)
            rendering = torch.clip(results["render"].permute(1, 2, 0), 0, 1).cpu().numpy()
            rendering = cv.cvtColor((rendering * 255).astype(np.uint8), cv.COLOR_RGB2BGR)
            rendering = cv.resize(rendering, (rendering.shape[1] * 5, rendering.shape[0] * 5))
            cv.imshow('visualization', rendering)

        cv.destroyAllWindows()



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
