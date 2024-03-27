"""
evaluate results for CO3Dv2/shape reconstructions
"""
import os, sys

import numpy as np

sys.path.append(os.getcwd())
from tqdm import tqdm
import open3d
import os.path as osp
from glob import glob
import json

from eval.chamfer_distance import chamfer_distance

class ShapeEvaluator:
    def __init__(self):
        CUBE_SIDE_LEN = 1.0
        self.threshold_list = [CUBE_SIDE_LEN/200, CUBE_SIDE_LEN/100,
                      CUBE_SIDE_LEN/50, CUBE_SIDE_LEN/20,
                      # CUBE_SIDE_LEN/10, CUBE_SIDE_LEN/5
                               ]
        self.outpath= 'results'


    def evaluate(self, args):
        files_pred = self.get_pred_files(args)
        files_gt = self.get_gt_files(args)
        assert len(files_gt) == len(files_pred)

        th_list = [args.th] if args.th is not None else self.threshold_list

        dist_chamf, fscores, recalls, precs = self.eval_files(files_gt, files_pred, th_list)

        res = ''
        mean_fscores = np.mean(fscores, 0)
        for s, t in zip(mean_fscores, th_list):
            ss = f"F-score@{t}m={s:.4f}  "
            res += ss
        mean_cd = np.mean(dist_chamf)
        res += f"CD={mean_cd:.4f}"
        print(res)

        # also save a copy of the results for convenience
        ss = args.pr_path.split(os.sep)
        outfile = osp.join(self.outpath, f'{ss[-5]}_{ss[-4]}_{ss[-3]}_{ss[-2]}_{self.get_timestamp()}.json')
        results = {
            "CD":mean_cd,
            "F-scores": mean_fscores.tolist(),
            "threshold":th_list,
            "pr_path":args.pr_path,
            'gt_path':args.gt_path
        }
        json.dump(results, open(outfile, 'w'), indent=2)
        print("Results saved to", outfile)

    def eval_files(self, files_gt, files_pred, th_list):
        """
        evaluate list of files
        Parameters
        ----------
        files_gt
        files_pred
        th_list

        Returns
        -------

        """
        files_gt, files_pred = self.check_files(files_gt, files_pred)
        dist_chamf = []
        precs, recalls, fscores = [], [], []
        for fgt, fpred in zip(tqdm(files_gt), files_pred):
            assert osp.basename(fgt) == osp.basename(fpred)
            gt = open3d.io.read_point_cloud(fgt)
            pr = open3d.io.read_point_cloud(fpred)

            # normalize points
            pts_gt, pts_pr = np.array(gt.points), np.array(pr.points)
            # pts_gt = pts_gt[:len(pts_gt)//2]
            cent = np.mean(pts_gt, 0)
            scale = np.sqrt(np.max(np.sum((pts_gt - cent) ** 2, -1)))
            pts_gt = (pts_gt - cent) / (2*scale)

            pts_pr = (pts_pr - cent) / (2*scale)

            scores, cd = self.compute_fscores(pts_gt, pts_pr, th_list)
            fscores.append(scores[0])
            precs.append(scores[1])
            recalls.append(scores[2])
            dist_chamf.append(cd)

        if len(fscores) == 0:
            return None, None, None, None
        fscores = np.stack(fscores, 0)
        recalls = np.stack(recalls, 0)
        precs = np.stack(precs, 0)
        return dist_chamf, fscores, recalls, precs


    def check_files(self, files_gt, files_pred):
        "find common files"
        if len(files_pred) != len(files_gt):
            print('unequal number of files')
        fnames_pred = [osp.basename(x) for x in files_pred]
        fnames_gt = [osp.basename(x) for x in files_gt]
        common_files = []
        for fname in fnames_gt:
            if fname in fnames_pred:
                common_files.append(fname)
        ppre = osp.dirname(files_pred[0])
        pgt = osp.dirname(files_gt[0])
        files_pred = [osp.join(ppre, x) for x in common_files]
        files_gt = [osp.join(pgt, x) for x in common_files]
        return files_gt, files_pred

    def compute_fscores(self, gt, pred, th_list):
        """

        Parameters
        ----------
        gt (N, 3)  points
        pred (M, 3) points
        th_list thresholds list L

        Returns 3xL, fscore, precision, recall and a scalar value of chamfer
        -------
        this is slightly different from the calculate_fscores by the what3d paper
        """
        chamf, d1, d2 = chamfer_distance(gt, pred, ret_intermediate=True) #

        scores = np.zeros((3, len(th_list)))
        for i, th in enumerate(th_list):
            recall = float(sum(d < th for d in d2)) / float(len(d2))
            precision = float(sum(d < th for d in d1)) / float(len(d1))

            if recall + precision > 0:
                fscore = 2 * recall * precision / (recall + precision)
            else:
                fscore = 0
            scores[:, i] = np.array([fscore, precision, recall])
        return scores, chamf

    def get_timestamp(self):
        from datetime import datetime
        now = datetime.now()
        time_str = f'{now.year}-{now.month:02d}-{now.day:02d}T{now.hour:02d}-{now.minute:02d}-{now.second:02d}'
        return time_str

    def get_gt_files(self, args):
        files_gt = sorted(glob(args.gt_path + "/*.ply"))
        return files_gt

    def get_pred_files(self, args):
        files_pred = sorted(glob(args.pr_path + "/*.ply"))
        return files_pred

    @staticmethod
    def get_parser():
        from argparse import ArgumentParser
        parser = ArgumentParser()
        parser.add_argument('-pr', '--pr_path')
        parser.add_argument('-gt', '--gt_path')
        parser.add_argument('--out_path')
        parser.add_argument('-th', type=float)

        parser.add_argument('-ct', '--cont_thres', default=0.05, type=float, help='radius to find contact region')
        return parser

def main(args):
    evaluator = ShapeEvaluator()
    evaluator.evaluate(args)
    print("All done")


if __name__ == '__main__':
    parser = ShapeEvaluator.get_parser()
    args = parser.parse_args()

    main(args)