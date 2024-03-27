"""
evaluate human and object as a single shape/pc
"""
import sys, os
sys.path.append(os.getcwd())
import json
import os.path as osp
from glob import glob
import multiprocessing as mp

import numpy as np
import pickle as pkl

from eval.eval_shape import ShapeEvaluator


class SingleShapeEvaluator(ShapeEvaluator):
    def __init__(self):
        super(SingleShapeEvaluator, self).__init__()
        # multi-threading
        # data pipe for multi-threading communication
        self.manager = mp.Manager()
        self.errors_dict = self.manager.dict()
        self.args = None
        self.init_others()

    def init_others(self):
        pass

    def evaluate(self, args):
        self.args = args
        seqs = json.load(open(args.split))['seqs']
        jobs = []
        print("evaluating on {} seqs".format(len(seqs)))
        th_list = [args.th] if args.th is not None else self.threshold_list
        batch_size = 100
        for seq in seqs:
            gt_files = self.glob_files(args.gt_path, seq)
            pred_files = self.glob_files(args.pr_path, seq)
            # further split if too many files
            if len(gt_files) > batch_size:
                for i in range(0, len(gt_files), batch_size):
                    p = mp.Process(target=self.eval_seq, args=(gt_files[i:i+batch_size], pred_files[i:i+batch_size], th_list, seq+"_"+str(i)))
                    p.start()
                    jobs.append(p)
            else:
                if len(pred_files) == 0:
                    print("No files found in folder", args.pr_path, seq) # do not evaluate this seq
                    continue
                p = mp.Process(target=self.eval_seq, args=(gt_files, pred_files, th_list, seq))
                p.start()
                jobs.append(p)
        for job in jobs:
            job.join()

        self.collect_results(args, th_list)

    def glob_files(self, folder, seq):
        gt_files = sorted(glob(folder + f"/{seq}/*.ply"))
        return gt_files

    def extract_objname(self, seq_name):
        "chairs are classified in the same category, same for tables"
        ss = seq_name.split("_")
        if len(ss) < 3:
            name = seq_name
        else:
            name = ss[2]
        icap_obj_names = {
            'obj01': 'icap-suitcase',  # aligned
            'obj02': 'skateboard',  # aligned
            'obj03': 'ball',  # aligned
            'obj04': 'umbrella',  # not related objects
            'obj05': 'tennis-racket',  # no related objects
            'obj06': 'toolbox',  # aligned
            'obj07': 'icap-chair',  # chair, aligned
            'obj08': 'bottle',  # aligned
            'obj09': 'cup',  # aligned
            'obj10': 'icap-chair'  # stool, aligned
        }
        if 'chair' in name:
            return 'chair'
        elif 'table' in name:
            return 'table'
        elif 'toolbox' in name:
            return 'toolbox'
        elif 'box' in name:
            return 'box'
        elif 'ball' in name:
            return 'ball'
        elif name in icap_obj_names.keys():
            return icap_obj_names[name]
        return name

    def collect_results(self, args, th_list):
        ""
        files_all = []
        errors_all = {}
        errors_obj_specific = {} # split errors based on object
        for seq, errors in self.errors_dict.items():
            obj = self.extract_objname(seq)
            if obj not in errors_obj_specific:
                errors_obj_specific[obj] = {}
            for k, err in errors.items():
                if k not in errors_all:
                    errors_all[k] = []
                if isinstance(err, list):
                    errors_all[k].extend(err)
                else:
                    errors_all[k].append(err)
                # add to object specific as well
                if k not in errors_obj_specific[obj]:
                    errors_obj_specific[obj][k] = []
                if isinstance(err, list):
                    errors_obj_specific[obj][k].extend(err)
                else:
                    errors_obj_specific[obj][k].append(err)

        prefix = osp.basename(args.split).replace('*', '').replace('.json', '_')
        for k in errors_all.keys():
            if isinstance(errors_all[k][0], np.ndarray):
                errors_all[k] = np.concatenate(errors_all[k], 0)
        self.print_summary(errors_all, th_list)

        # print summary for each object
        for obj in sorted(errors_obj_specific.keys()):
            errors = errors_obj_specific[obj]
            # first concate all errors
            for k in errors.keys():
                if isinstance(errors[k][0], np.ndarray):
                    errors[k] = np.concatenate(errors[k], 0)
            # then print
            self.print_summary(errors, th_list, category_name=obj)

        mean_errors = {}
        for k, v in errors_all.items():
            if k not in ['chamf', 'files']:
                avg = np.mean(v, 0)
                mean_errors[k] = avg.tolist()

        mean_errors_obj_specific = {}
        for obj, errors in errors_obj_specific.items():
            mean_errors_obj_specific[obj] = {}
            for k, v in errors.items():
                if k not in ['chamf', 'files']:
                    avg = np.mean(v, 0)
                    mean_errors_obj_specific[obj][k] = avg.tolist()

        # also save a copy of the results for convenience
        outfile = self.get_outfile(args, prefix)
        results = {
            "CD": np.mean(errors_all['chamf']),
            **mean_errors,
            # "F-scores": np.mean(errors_all['fscores'], 0).tolist(),
            "threshold": th_list,
            "num_images":len(errors_all['files']),
            "pr_path": args.pr_path,
            'gt_path': args.gt_path,
            "object_specific": mean_errors_obj_specific,

        }
        json.dump(results, open(outfile, 'w'), indent=2)

        # also save raw errors
        raw_data = errors_all
        pkl.dump(raw_data, open(osp.join(self.outpath, 'raw', osp.basename(outfile.replace('.json', '.pkl'))), 'wb'))
        print("Results saved to", outfile)

    def get_outfile(self, args, prefix):
        ss = args.pr_path.split(os.sep)
        outfile = osp.join(self.outpath, f'{prefix}_{ss[-5]}_{ss[-4]}_{ss[-3]}_{ss[-2]}_{args.id}_{self.get_timestamp()}.json')
        return outfile

    def print_summary(self, errors_all, th_list, category_name='All'):
        dist_chamf, fscores = errors_all['chamf'], errors_all['fscores']
        res = f'{category_name} {len(fscores)} images: '
        mean_fscores = np.mean(fscores, 0)
        for s, t in zip(mean_fscores, th_list):
            ss = f"F-score@{t}m={s:.4f}  "
            res += ss
        mean_cd = np.mean(dist_chamf)
        res += f"CD={mean_cd:.4f}"
        print(res)
        return mean_cd, mean_fscores

    def eval_seq(self, files_gt, files_pred, th_list, seq_name):
        try:
            ret_values = self.eval_files(files_gt, files_pred, th_list)
        except Exception as e:
            print(seq_name, 'failed due to', e)
            return
        if ret_values[0] is None:
            print(f'{seq_name} no results, done')
            return
        ret = self.format_errors(files_pred, ret_values)
        self.errors_dict[seq_name] = ret
        print(f'{seq_name} done.')

    def format_errors(self, files_pred, ret_values):
        dist_chamf, fscores, recalls, precs = ret_values
        ret = {
            "files": files_pred,
            'chamf': dist_chamf,
            'fscores': fscores,
            'recalls': recalls,
            'precisions': precs
        }
        return ret

    @staticmethod
    def get_parser():
        parser = SingleShapeEvaluator.get_parser()
        parser.add_argument('-split', default='configs/splits/behave-test.json')
        parser.add_argument('-i', '--id', help='additional information/identification')
        return parser

def main(args):
    evaluator = SingleShapeEvaluator()
    evaluator.evaluate(args)
    print("All done")


if __name__ == '__main__':
    parser = SingleShapeEvaluator.get_parser()
    args = parser.parse_args()

    main(args)