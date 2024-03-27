"""
precompute the df_h and df_o, convert it into binary bits of occupancies, which will be used later to compute segmentation labels for any points in space.
Usage: python scripts/compute_occ.py -s ProciGen_path/Date04_Subxx_monitor_synzv2-01

Author: Xianghui Xie
Date: March 27, 2024
Cite: Template Free Reconstruction of Human-object Interaction with Procedural Interaction Generation
"""
import os, sys
import time

import numpy as np
import trimesh

sys.path.append(os.getcwd())
from glob import glob
import os.path as osp
from tqdm import tqdm
import igl

from behave.kinect_transform import KinectTransform
from behave.seq_utils import SeqInfo

def create_grid_points(bound=1.0, res=128):
    x_ = np.linspace(-bound, bound, res)
    y_ = np.linspace(-bound, bound, res)
    z_ = np.linspace(-bound, bound, res)

    x, y, z = np.meshgrid(x_, y_, z_)
    # print(x.shape, y.shape) # (res, res, res)
    pts = np.concatenate([y.reshape(-1, 1), x.reshape(-1, 1), z.reshape(-1, 1)], axis=-1)
    return pts

def main(args):
    seq = args.seq_folder

    if 'Subxx' in seq:
        # ProciGen data
        frames = sorted(glob(osp.join(seq, '[0-9]*')))
        smpl_name, obj_name = 'fit01', 'fit01'
    else:
        # behave data
        frames = sorted(glob(osp.join(seq, 't*')))
        smpl_name, obj_name = 'fit02', 'fit01'

    seq_info = SeqInfo(seq)
    obj = seq_info.get_obj_name(False)
    start, end = args.start, args.end if args.end is not None else len(frames)

    kin_transform = KinectTransform(seq, no_intrinsic=True)
    grid_points = create_grid_points(args.bound, args.res)
    num_samples = 8096
    for frame in tqdm(frames[start:end]):
        if not osp.isfile(osp.join(frame, 'k1.color.jpg')):
            print(frame)
            continue
        done = True
        kids = range(6) if "Date09" in seq else range(4)
        for kid in kids:
            outfile = osp.join(frame, f'k{kid}.grid_df_res{args.res}_b{args.bound}.npz')
            if not osp.isfile(outfile):
                done = False
                break
        if done and not args.redo:
            print(frame, 'done, skipped')
            continue

        time_start = time.time()
        obj_file = osp.join(frame, f'{obj}/{obj_name}/{obj}_fit.ply')
        if not osp.isfile(obj_file):
            print(obj_file, 'does not exist!')
            continue
        mobj = trimesh.load_mesh(obj_file, processing=False)
        smpl_file = osp.join(frame, f'person/{smpl_name}/person_fit.ply')
        if not osp.isfile(smpl_file):
            print(smpl_file, 'does not exist!')
            continue
        msmpl = trimesh.load_mesh(smpl_file, processing=False)
        time_end = time.time()
        print(f"Time to load one mesh: {time_end - time_start:.4f}")
        sample_obj = mobj.sample(num_samples)
        sample_smpl = msmpl.sample(num_samples)
        samples = np.concatenate([sample_obj, sample_smpl], 0)

        for kid in kids:
            outfile = osp.join(frame, f'k{kid}.grid_df_res{args.res}_b{args.bound}.npz')
            if osp.isfile(outfile) and not args.redo:
                continue

            verts_obj = kin_transform.world2local(np.array(mobj.vertices), kid)
            verts_smpl = kin_transform.world2local(np.array(msmpl.vertices), kid)

            # normalize the points
            samples_local = kin_transform.world2local(samples, kid)
            cent = np.mean(samples_local, 0)
            radius = np.sqrt(np.max(np.sum((samples_local - cent) ** 2, -1)))

            # compute over the normalized meshes
            time_start = time.time()
            df_o = np.abs(igl.signed_distance(grid_points, (verts_obj - cent)/(2*radius), mobj.faces)[0])
            df_h = np.abs(igl.signed_distance(grid_points, (verts_smpl - cent)/(2*radius), msmpl.faces)[0])
            time_end = time.time()
            print(f"Time to compute df: {time_end - time_start:.4f}")

            # save results as compressed occupancy
            binary = (df_h < df_o).astype(np.uint8)  # 1-human, 0-object
            compressed_occ = np.packbits(binary)
            np.savez(outfile, compressed_occ=compressed_occ)
            print("saved to", outfile)

    print('all done')

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('-s', '--seq_folder')
    parser.add_argument('-fs', '--start', default=0, type=int)
    parser.add_argument('-fe', '--end', default=None, type=int)
    parser.add_argument('-r', '--res', type=int, default=128)
    parser.add_argument('-b', '--bound', type=float, default=0.5)
    parser.add_argument('-redo', default=False, action='store_true')

    args = parser.parse_args()

    import traceback
    try:
        main(args)
    except Exception as e:
        print(e)
        log = traceback.format_exc()
        print(log)