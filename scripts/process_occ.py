"""
move occupancy files into the same path as the sequence files
e.g.: Date04_Subxx_monitor_synzv2-01/00000-125611504306885.k0.grid_df_res128_b0.5.npz -> procigen_root/Date04_Subxx_monitor_synzv2-01/00000-125611504306885/k0.grid_df_res128_b0.5.npz

Author: Xianghui Xie
Date: March 27, 2024
Cite: Template Free Reconstruction of Human-object Interaction with Procedural Interaction Generation
"""
import sys, os
sys.path.append(os.getcwd())
from tqdm import tqdm
import os.path as osp
from glob import glob


def main(args):
    occ_path, procigen_path = args.occ_path, args.procigen_path
    seqs = sorted(glob(occ_path + "/*"))
    for seq in tqdm(seqs):
        files = sorted(glob(seq + "/*.npz"))
        seq_name = osp.basename(seq)
        for file in tqdm(files, desc=f'Processing {seq_name}'):
            fname = osp.basename(file)
            frame = fname.split('.')[0]
            newfile = osp.join(procigen_path, seq_name, frame, fname.replace(frame+".", ''))
            if osp.isfile(newfile):
                continue
            os.system(f'cp {file} {newfile}')
            break
        break
    print('All done')



if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('-o', '--occ_path', help='path to unpacked occupancy files')
    parser.add_argument('-p', '--procigen_path', help='root path to procigen sequences')

    args = parser.parse_args()

    main(args)