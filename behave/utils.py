"""
loads calibrations
Cite: BEHAVE: Dataset and Method for Tracking Human Object Interaction
"""
import os, sys
sys.path.append("/")
import json
from os.path import join, basename, dirname
import numpy as np
import os.path as osp
from psbody.mesh import Mesh


def rotate_yaxis(R, t):
    "rotate the transformation matrix around z-axis by 180 degree ==>> let y-axis point up"
    transform = np.eye(4)
    transform[:3, :3] = R
    transform[:3, 3] = t
    global_trans = np.eye(4)
    global_trans[0, 0] = global_trans[1, 1] = -1  # rotate around z-axis by 180
    rotated = np.matmul(global_trans, transform)
    return rotated[:3, :3], rotated[:3, 3]


def load_intrinsics(intrinsic_folder, kids):
    """
    kids: list of kinect id that should be loaded
    """
    from behave.kinect_calib import KinectCalib
    intrinsic_calibs = [json.load(open(join(intrinsic_folder, f"{x}/calibration.json"))) for x in kids]
    pc_tables = [np.load(join(intrinsic_folder, f"{x}/pointcloud_table.npy")) for x in kids]
    kinects = [KinectCalib(cal, pc) for cal, pc in zip(intrinsic_calibs, pc_tables)]

    return kinects


def load_kinect_poses(config_folder, kids):
    pose_calibs = [json.load(open(join(config_folder, f"{x}/config.json"))) for x in kids]
    rotations = [np.array(pose_calibs[x]['rotation']).reshape((3, 3)) for x in kids]
    translations = [np.array(pose_calibs[x]['translation']) for x in kids]
    return rotations, translations


def load_kinects(intrinsic_folder, config_folder, kids):
    from behave.kinect_calib import KinectCalib
    intrinsic_calibs = [json.load(open(join(intrinsic_folder, f"{x}/calibration.json"))) for x in kids]
    pc_tables = [np.load(join(intrinsic_folder, f"{x}/pointcloud_table.npy")) for x in kids]
    pose_files = [join(config_folder, f"{x}/config.json") for x in kids]
    kinects = [KinectCalib(cal, pc) for cal, pc in zip(intrinsic_calibs, pc_tables)]
    return kinects


def load_kinect_poses_back(config_folder, kids, rotate=False):
    """
    backward transform
    rotate: kinect y-axis pointing down, if rotate, then return a transform that make y-axis pointing up
    """
    rotations, translations = load_kinect_poses(config_folder, kids)
    rotations_back = []
    translations_back = []
    for r, t in zip(rotations, translations):
        trans = np.eye(4)
        trans[:3, :3] = r
        trans[:3, 3] = t

        trans_back = np.linalg.inv(trans) # now the y-axis point down

        r_back = trans_back[:3, :3]
        t_back = trans_back[:3, 3]
        if rotate:
            r_back, t_back = rotate_yaxis(r_back, t_back)

        rotations_back.append(r_back)
        translations_back.append(t_back)
    return rotations_back, translations_back


def availabe_kindata(input_video, kinect_count=3):
    # all available kinect videos in this folder, return the list of kinect id, and str representation
    fname_split = os.path.basename(input_video).split('.')
    idx = int(fname_split[1])
    kids = []
    comb = ''
    for k in range(kinect_count):
        file = input_video.replace(f'.{idx}.', f'.{k}.')
        if os.path.exists(file):
            kids.append(k)
            comb = comb + str(k)
        else:
            print("Warning: {} does not exist in this folder!".format(file))
    return kids, comb

OBJ_NAMES=['backpack', 'basketball', 'boxlarge', 'boxlong', 'boxmedium',
           'boxsmall', 'boxtiny', 'chairblack', 'chairwood', 'keyboard',
           'monitor', 'plasticcontainer', 'stool', 'suitcase', 'tablesmall',
           'tablesquare', 'toolbox', 'trashbin', 'yogaball', 'yogamat']

# path to the simplified mesh used for registration
_mesh_template = {
    "backpack":"backpack/backpack_f1000.ply",
    'basketball':"basketball/basketball_f1000.ply",
    'boxlarge':"boxlarge/boxlarge_f1000.ply",
    'boxtiny':"boxtiny/boxtiny_f1000.ply",
    'boxlong':"boxlong/boxlong_f1000.ply",
    'boxsmall':"boxsmall/boxsmall_f1000.ply",
    'boxmedium':"boxmedium/boxmedium_f1000.ply",
    'chairblack': "chairblack/chairblack_f2500.ply",
    'chairwood': "chairwood/chairwood_f2500.ply",
    'monitor': "monitor/monitor_closed_f1000.ply",
    'keyboard':"keyboard/keyboard_f1000.ply",
    'plasticcontainer':"plasticcontainer/plasticcontainer_f1000.ply",
    'stool':"stool/stool_f1000.ply",
    'tablesquare':"tablesquare/tablesquare_f2000.ply",
    'toolbox':"toolbox/toolbox_f1000.ply",
    "suitcase":"suitcase/suitcase_f1000.ply",
    'tablesmall':"tablesmall/tablesmall_f1000.ply",
    'yogamat': "yogamat/yogamat_f1000.ply",
    'yogaball':"yogaball/yogaball_f1000.ply",
    'trashbin':"trashbin/trashbin_f1000.ply",
}

_watertight_templates = {'backpack': 'backpack/backpack_f1000_cent_watertight_fused.ply',
                         'basketball': 'basketball/basketball_f1000_cent_watertight.ply',
                         'boxlarge': 'boxlarge/boxlarge_f1000_cent_watertight_fused.ply',
                         'boxtiny': 'boxtiny/boxtiny_f1000_cent_watertight.ply',
                         'boxlong': 'boxlong/boxlong_f1000_cent_watertight_fused.ply',
                         'boxsmall': 'boxsmall/boxsmall_f1000_cent_watertight.ply',
                         'boxmedium': 'boxmedium/boxmedium_f1000_cent_watertight.ply',
                         'chairblack': 'chairblack/chairblack_f2500_cent_watertight.ply',
                         'chairwood': 'chairwood/chairwood_f2500_cent_watertight.ply',
                         'monitor': 'monitor/monitor_closed_f1000_cent_watertight.ply',
                         'keyboard': 'keyboard/keyboard_f1000_cent_watertight_fused.ply',
                         'plasticcontainer': 'plasticcontainer/plasticcontainer_f1000_cent_watertight.ply',
                         'stool': 'stool/stool_f1000_cent_watertight_fused.ply',
                         'tablesquare': 'tablesquare/tablesquare_f2000_cent_watertight.ply',
                         'toolbox': 'toolbox/toolbox_f1000_cent_watertight.ply',
                         'suitcase': 'suitcase/suitcase_f1000_cent_watertight.ply',
                         'tablesmall': 'tablesmall/tablesmall_f1000_cent_watertight_fused.ply',
                         'yogamat': 'yogamat/yogamat_f1000_cent_watertight_fused.ply',
                         'yogaball': 'yogaball/yogaball_f1000_cent_watertight.ply',
                         'trashbin': 'trashbin/trashbin_f1000_cent_watertight.ply'}

# InterCap mesh templates
ICAP_TEMP_PATH = '/BS/xxie-2/work/InterCap/obj_track/objs/'
_icap_template = {
    'obj01': '/BS/xxie-2/work/InterCap/obj_track/objs/01.ply',
    'obj02': '/BS/xxie-2/work/InterCap/obj_track/objs/02.ply',
    'obj03': '/BS/xxie-2/work/InterCap/obj_track/objs/03.ply',
    'obj04': '/BS/xxie-2/work/InterCap/obj_track/objs/04.ply',
    'obj05': '/BS/xxie-2/work/InterCap/obj_track/objs/05.ply',
    'obj06': '/BS/xxie-2/work/InterCap/obj_track/objs/06.ply',
    'obj07': '/BS/xxie-2/work/InterCap/obj_track/objs/07.ply',
    'obj08': '/BS/xxie-2/work/InterCap/obj_track/objs/08.ply',
    'obj09': '/BS/xxie-2/work/InterCap/obj_track/objs/09.ply',
    'obj10': '/BS/xxie-2/work/InterCap/obj_track/objs/10.ply'
}

# path to original full-reso scan reconstructions
orig_scan = {
        "backpack": "/BS/xxie2020/work/objects/backpack/backpack_closed.ply",
        'basketball': "/BS/xxie2020/work/objects/basketball/basketball_closed.ply",
        'boxlarge': "/BS/xxie2020/work/objects/box_large/box_large_closed_centered.ply",
        'boxtiny': "/BS/xxie2020/work/objects/box_tiny/box_tiny_closed.ply",
        'boxlong': "/BS/xxie2020/work/objects/box_long/box_long_close.ply",
        'boxsmall': "/BS/xxie2020/work/objects/box_small/boxsmall_closed.ply",
        'boxmedium': "/BS/xxie2020/work/objects/box_medium/box_medium_closed.ply",
        'chairblack': "/BS/xxie2020/work/objects/chair_black/chair_black.ply",
        'chairwood': "/BS/xxie2020/work/objects/chair_wood/chair_wood_clean.ply",
        'monitor': "/BS/xxie2020/work/objects/monitor/monitor_closed_centered.ply",
        'keyboard': "/BS/xxie2020/work/objects/keyboard/keyboard_closed_centered.ply",
        'plasticcontainer': "/BS/xxie2020/work/objects/plastic_container/container_closed_centered.ply",
        "stool":"/BS/xxie2020/work/objects/stool/stool_clean_centered.ply",
        'tablesquare': "/BS/xxie2020/work/objects/table_square/table_square_closed.ply",
        'toolbox': "/BS/xxie2020/work/objects/toolbox/toolbox_closed_centered.ply",
        "suitcase": "/BS/xxie2020/work/objects/suitcase_small/suitcase_closed_centered.ply",
        'tablesmall': "/BS/xxie2020/work/objects/table_small/table_small_closed_aligned.ply",
        'yogamat': "/BS/xxie2020/work/objects/yoga_mat/yogamat_closed.ply",
        'yogaball': "/BS/xxie2020/work/objects/yoga_ball/yoga_ball_closed.ply",
        'trashbin': "/BS/xxie2020/work/objects/trash_bin/trashbin_closed.ply"
    }


def get_template_path(behave_path, obj_name):
    if obj_name in _mesh_template.keys():
        path = osp.join(behave_path, _mesh_template[obj_name])
    else:
        path = _icap_template[obj_name]
    if not osp.isfile(path):
        print(path, 'does not exist, please check input parameters!')
        raise ValueError()
    return path

def load_scan_centered(scan_path, cent=True):
    """load a scan and centered it around origin"""
    scan = Mesh()
    # print(scan_path)
    scan.load_from_file(scan_path)
    if cent:
        center = np.mean(scan.v, axis=0)
        verts_centerd = scan.v - center
        scan.v = verts_centerd

    return scan


def load_template(obj_name, cent=True, high_reso=False):
    "load object template mesh given object name"
    temp_path = get_template_path("/BS/xxie-5/static00/behave_release/objects", obj_name)
    if high_reso:
        assert obj_name in _mesh_template.keys(), f'does not support high reso template for {obj_name}'
        lowreso_temp = Mesh(filename=temp_path)
        # highreso_path = orig_scan[obj_name].replace('.ply', '_f12000.ply')
        highreso_path = orig_scan[obj_name].replace('.ply', '_f1500.ply')
        highreso_temp = Mesh(filename=highreso_path)
        highreso_temp.v = highreso_temp.v - np.mean(lowreso_temp.v, 0)
        return highreso_temp
    return load_scan_centered(temp_path, cent)


def load_template_watertight(obj_name, cent=True, high_reso=False):
    "load watertight object template mesh given object name"
    # temp_path = get_template_path("/BS/xxie-5/static00/behave_release/objects",
    #                               obj_name).replace('.ply', '_cent_watertight.ply')
    temp_path = osp.join("/BS/xxie-5/static00/behave_release/objects", _watertight_templates[obj_name])
    return load_scan_centered(temp_path, False)


def load_configs_all():
    "all behave configures"
    BEHAVE_PATH = "/BS/xxie-4/static00/behave-fps30"
    ICAP_PATH = "/scratch/inf0/user/xxie/behave"
    date_seqs = {
        "Date01": BEHAVE_PATH + "/Date01_Sub01_backpack_back",
        "Date02": BEHAVE_PATH + "/Date02_Sub02_backpack_back",
        "Date03": BEHAVE_PATH + "/Date03_Sub03_backpack_back",
        "Date04": BEHAVE_PATH + "/Date04_Sub05_backpack",
        "Date05": BEHAVE_PATH + "/Date05_Sub05_backpack",
        "Date06": BEHAVE_PATH + "/Date06_Sub07_backpack_back",
        "Date07": BEHAVE_PATH + "/Date07_Sub04_backpack_back",
        "ICapS01": ICAP_PATH + "/ICapS01_sub01_obj01_Seg_0",
        "ICapS02": ICAP_PATH + "/ICapS02_sub01_obj08_Seg_0",
        "ICapS03": ICAP_PATH + "/ICapS03_sub07_obj05_Seg_0",
    }
    from .kinect_transform import KinectTransform
    kin_transforms = {
        k: KinectTransform(v, no_intrinsic=True) for k, v in date_seqs.items()
    }
    return kin_transforms
