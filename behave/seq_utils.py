"""
utils to get sequence information
Cite: BEHAVE: Dataset and Method for Tracking Human Object Interaction
"""
import os, sys
sys.path.append(os.getcwd())
import json
from os.path import join, basename, dirname, isdir


class SeqInfo:
    "a simple class to handle information of a sequence"
    def __init__(self, seq_path):
        self.info = self.get_seq_info_data(seq_path)

    def get_obj_name(self, convert=False):
        if convert: # for using detectron
            if 'chair' in self.info['cat']:
                return 'chair'
            if 'ball' in self.info['cat']:
                return 'sports ball'
        return self.info['cat']

    def get_gender(self):
        return self.info['gender']

    def get_config(self):
        return self.info['config']

    def get_intrinsic(self):
        return self.info['intrinsic']

    def get_empty_dir(self):
        return self.info['empty']

    def beta_init(self):
        return self.info['beta']

    def kinect_count(self):
        if 'kinects' in self.info:
            return len(self.info['kinects'])
        else:
            return 3

    @property
    def kids(self):
        # count = self.kinect_count()
        # return [i for i in range(count)]
        return self.info['kinects']

    def get_seq_info_data(self, seq):
        info_file = join(seq, 'info.json')
        data = json.load(open(info_file))
        # all paths are relative to the sequence path
        path_names = ['config', 'empty', 'intrinsic']
        for name in path_names:
            if data[name] is not None:
                path = join(seq, data[name])
                if not isdir(path):
                    path = data[name] # abs path
                    if "Date0" in seq or "ICap" in seq:  # check config path for behave and intercap dataset
                        assert isdir(path), f'given path {path} does not exist!'
                data[name] = path
        return data


def save_seq_info(seq_folder, config, intrinsic, cat,
                  gender, empty, beta,
                  kids=[0, 1, 2, 3]):
    # from behave.utils import load_kinect_poses
    outfile = join(seq_folder, 'info.json')
    info = {
        "config":config,
        "intrinsic":intrinsic,
        'cat':cat,
        'gender':gender,
        'empty':empty,
        'kinects':kids,
        'beta':beta
    }
    with open(outfile, 'w', encoding='utf-8') as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    print("{} saved.".format(outfile))
    print("{}: {}, {}, {}, {}, {}".format(seq_folder, config, intrinsic, cat, beta, gender))


def get_config_idx(name):
    "get config index?"
    ss = name.split('_')
    subject = ss[1][3:]
    motion = int(ss[2][3:]) # XH: set to object index and let's see

    if (subject in ['01', '02', '03'] and motion in [1, 2, 3, 4, 5, 6, 7]) or (
            subject in ['04'] and motion in [1, 2, 3, 4]):
        return 0

    if (subject in ['08', '09', '10']) or (subject in ['01', '02', '04', '05', '06', '07'] and motion in [8, 9, 10]):
        return 1

    if (subject in ['03'] and motion in [8, 9, 10]) or (
            subject in ['05', '06', '07'] and motion in [1, 2, 3, 4, 5, 6, 7]):
        return 2

    return 0  # default config

def save_intercap_info():
    "save seq info for intercap seqs"
    from glob import glob
    seqs = sorted(glob("/scratch/inf0/user/xxie/behave/InterCap_*/"))
    print(f'in total {len(seqs)} sequences') # in total 213 sequences
    kids = [0, 1, 2, 3, 4, 5]
    gender_map = {
        "sub01":"male",
        "sub02":"male",
        "sub03":"female",
        "sub04":"male",
        "sub05":"male",
        "sub06":"female",
        "sub07":"female",
        "sub08":"female",
        "sub09":"female",
        "sub10":"male"
    }
    config = "/BS/xxie-4/work/rawvideo/Intercap"
    intrinsic = "/BS/xxie-4/work/rawvideo/Intercap/intrinsics/"

    for seq in seqs:
        seq_name = basename(seq[:-1])
        ss = seq_name.split('_')
        sub = ss[1]
        obj = ss[2]
        outfile = join(seq, 'info.json')

        config_idx = get_config_idx(seq_name)

        print(seq_name, config_idx)
        continue

        assert config_idx in [0, 1, 2], f'no config index for seq {seq_name}'
        seq_config = f"{config}/config{config_idx:02d}"
        info = {
            "config": seq_config,
            "intrinsic": intrinsic,
            'cat': obj,
            'gender': gender_map[sub],
            'empty': None,
            'kinects': kids,
            'beta': None
        }
        json.dump(info, open(outfile, 'w'), indent=2)
        print(f'{outfile} saved')
    print('all done')

"""
example: 
"""
if __name__ == '__main__':
    save_intercap_info()

