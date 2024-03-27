"""
compute separate numbers for human, object, and combined
"""
import sys, os
sys.path.append(os.getcwd())
import json
import os.path as osp
from glob import glob
import multiprocessing as mp

import numpy as  np
import pickle as pkl
import trimesh
from tqdm import tqdm

from eval.eval_single import SingleShapeEvaluator



class SeparateEvaluator(SingleShapeEvaluator):
    def eval_files(self, files_gt, files_pred, th_list):
        ""
        files_gt, files_pred = self.check_files(files_gt, files_pred)
        dist_chamf = []
        fscores_comb, fscores_hum, fscores_obj = [], [], []
        for fgt, fpred in zip(tqdm(files_gt), files_pred):
            assert osp.basename(fgt) == osp.basename(fpred)

            pc = trimesh.load_mesh(fpred, process=False)
            mask_hum = pc.colors[:, 2] > 0.5
            pc_hum, pc_obj = np.array(pc.vertices[mask_hum]), np.array(pc.vertices[~mask_hum])
            gt = trimesh.load_mesh(fgt, process=False)
            L = len(gt.vertices)
            gt_hum, gt_obj = np.array(gt.vertices[:L//2]), np.array(gt.vertices[L//2:])

            # evaluate human + object
            scores, cd = self.compute_fscores(np.concatenate([gt_hum, gt_obj], 0),
                                              np.concatenate([pc_hum, pc_obj], 0), th_list)
            dist_chamf.append(cd)
            fscores_comb.append(scores[0])

            # evaluate human/object separately
            scores, _ = self.compute_fscores(gt_hum, pc_hum, th_list)
            fscores_hum.append(scores[0])
            scores, _ = self.compute_fscores(gt_obj, pc_obj, th_list)
            fscores_obj.append(scores[0])

        if len(fscores_comb) == 0:
            return None, None, None, None
        fscores = np.stack(fscores_comb, 0)
        fscores_hum = np.stack(fscores_hum, 0)
        fscores_obj = np.stack(fscores_obj, 0)
        return dist_chamf, fscores, fscores_hum, fscores_obj


    def format_errors(self, files_pred, ret_values):
        "different meaning for the errors returned by eval_files"
        dist_chamf, fscores, fscores_hum, fscores_obj = ret_values
        ret = {
            "files": files_pred,
            'chamf': dist_chamf,
            'fscores_comb': fscores,
            'fscores_hum': fscores_hum,
            'fscores_obj': fscores_obj
        }
        return ret

    def print_summary(self, errors_all, th_list, category_name='All'):
        ""
        dist_chamf, fscores_comb = errors_all['chamf'], errors_all['fscores_comb']
        fscores_hum = errors_all['fscores_hum']
        fscores_obj = errors_all['fscores_obj']
        res = f'{category_name} {len(fscores_comb)} images: '
        names = ['hum', 'obj', 'H+O']
        for fscores, name in zip([fscores_hum, fscores_obj, fscores_comb], names):
            mean_fscores = np.mean(fscores, 0)
            for s, t in zip(mean_fscores, th_list):
                if t ==0.01:
                    ss = f"{name}_F-score@{t}m={s:.4f}  "
                    res += ss

        mean_cd = np.mean(dist_chamf)
        res += f"CD={mean_cd:.4f}"
        print(res)
        return mean_cd, mean_fscores


def main(args):
    evaluator = SeparateEvaluator()
    evaluator.evaluate(args)
    print("All done")


if __name__ == '__main__':
    parser = SeparateEvaluator.get_parser()
    args = parser.parse_args()

    main(args)