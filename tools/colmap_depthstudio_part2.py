import tqdm, os, json, pycolmap
import numpy as np, shutil
import codecs, csv
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from pathlib import Path
from hloc import (
    extract_features,
    match_features,
    reconstruction,
    visualization,
    pairs_from_exhaustive,
)
from hloc.visualization import plot_images, read_image
from hloc.utils import viz_3d
from pyproj import Transformer
from scipy.spatial.transform import Rotation as R

import pyproj
def lla2ecef(lon, lat, alt):
    ecef = pyproj.Proj(proj='geocent', ellps='WGS84', datum='WGS84')
    lla = pyproj.Proj(proj='latlong', ellps='WGS84', datum='WGS84')
    x, y, z = pyproj.transform(lla, ecef, lon, lat, alt)
    return x, y, z

# WGS84转UTM
def WGS842UTM(gusaa_x, gusass_y):
    crs = pyproj.CRS.from_epsg(4326) #4326
    crs_cs = pyproj.CRS.from_epsg(32649)
    transformer = Transformer.from_crs(crs, crs_cs)
    lon, lat = transformer.transform(gusaa_x, gusass_y)
    return lon, lat

def inverse_pose(pose):
    tmp = np.zeros((4, 4))
    R = pose[:3, :3]
    T = pose[:3, 3]
    tmp[:3, :3] = np.linalg.inv(R)
    tmp[:3, 3] = -np.linalg.inv(R) @ T#.reshape(3, 1)
    tmp[3, 3] = 1
    return tmp

def filtering_pairs(sfm_pairs, frames=3):
    pairs = [x for x in open(sfm_pairs, \
        'r').readlines() \
        if int(x.split(' ')[0].split('.')[0].split('_')[-1]) - frames \
        <= int(x.split(' ')[-1].split('.')[0].split('_')[-1]) \
        <= int(x.split(' ')[0].split('.')[0].split('_')[-1]) + frames]
    
    sfm_pairs = open(sfm_pairs, 'w')
    for line in pairs:
        sfm_pairs.write(line)

    sfm_pairs.close()

cam = 'back'
cam_desci = dict(
    back='camera_40',
    left_back='camera_20',
    right_back='camera_10',
    front_narrow='camera_60',
    front_wide='camera_50',
    left_front='camera_30',
    right_front='camera_00')
data_root = '/media/flechazo/A1/depthstudio/GAC015/Clips'

if os.path.exists('parted.txt'):
    data_foler = [data.strip() for data in open('parted.txt', 'r').readlines()]
else:
    data_foler = [x for x in sorted([os.path.join(data_root, x) \
                    for x in os.listdir(data_root)]) if not os.path.exists(os.path.join(x, 'colmap'))]#[:1]

    parted = open('parted.txt', 'w')
    for data in data_foler:
        parted.write(data+'\n')

    parted.close()

part = 2
split_len = 4
if part == split_len:
    idxes = np.arange((part-1)*len(data_foler)//4, len(data_foler))
else:
    idxes = np.arange((part-1)*len(data_foler)//4, part*len(data_foler)//4,)

data_foler = [data_foler[idx] for idx in idxes]
data_bar = tqdm.tqdm(data_foler, total=len(data_foler))
for idx, folder in enumerate(data_bar): 
    data_bar.set_description(folder.split('/')[-1])
    images = Path(os.path.join(folder, 'image_undistortion/back'))
    outputs = Path(os.path.join(folder, 'colmap/back'))
    mask_folder = Path(os.path.join(folder, 'masks/back/foreground'))
    # colmap feature_extractor \
    #     --ImageReader.camera_model OPENCV \
    #     --ImageReader.single_camera_per_folder 1 \
    #     --database_path database.db \
    #     --image_path ./images/r_camera \
    #     --ImageReader.camera_params 2007.594319262585,2004.2646103455222,956.84513125447199,483.01833870807548,-0.4032521225946331,0.19465613741314503,0.004862449669118171,-0.0005263959011969029 \

    #     --ImageReader.mask_path ./masks/objects
        
    if os.path.exists(outputs): shutil.rmtree(outputs)
    sfm_pairs = outputs / "pairs-sfm.txt"
    loc_pairs = outputs / "pairs-loc.txt"
    sfm_dir = outputs / "sfm"
    features = outputs / "features.h5"
    matches = outputs / "matches.h5"

    ## obtaining ego poses
    ego_poses = dict()
    with codecs.open(os.path.join(os.path.join(folder, 'ego_raw'), 'ins_traj.csv'), encoding='utf-8-sig') as f:
        for row in csv.DictReader(f, skipinitialspace=True):
            key = row['sensor_stamp']
            ego_poses[key] = dict(latitude=row['position_gps.x'], \
                                longitude=row['position_gps.y'], \
                                altitude=row['position_gps.z'], \
                                roll=row['roll'], \
                                pitch=row['pitch'], \
                                azimuth=row['yaw'])

    sensor_infos = json.load(open(os.path.join(folder, 'sensor_info.json')))
    sensor_stamp = [e[str(i+1)][cam_desci[cam]]['time_stamp'] for i, e in enumerate(sensor_infos)]

    egopose, min_lat, min_lon, min_alt = dict(), [], [], []
    for i, stamp in enumerate(sensor_stamp):
        if str(stamp) in ego_poses:
            egopose[i] = ego_poses[str(stamp)]
        else:
            cost = None
            key_matched = None
            for j, key in enumerate(ego_poses):
                if j == 0:
                    cost = abs(int(key) - int(stamp))
                else:
                    cost_ = abs(int(key) - int(stamp))
                    if cost_ < cost:
                        cost = cost_
                        key_matched = key

            egopose[i] = ego_poses[key_matched]
            min_lat.append(float(ego_poses[key_matched]['latitude']))
            min_lon.append(float(ego_poses[key_matched]['longitude']))
            min_alt.append(float(ego_poses[key_matched]['altitude']))

    os.makedirs(sfm_dir, exist_ok=True)
    ref_gps_txt = open(os.path.join(sfm_dir, 'ref_gps.txt'), 'w')
    for imdir in sorted(os.listdir(images)):
        idx = imdir.split('.')[0].split('_')[-1]
        
        latitude, longitude, altitude, roll, pitch, azimuth = latitude, longitude, z, roll, pitch, azimuth = \
            [float(egopose[int(idx)][x]) for x in ['latitude', 'longitude', 'altitude', 'roll', 'pitch', 'azimuth']]
        
        # x, y = WGS842UTM(latitude, longitude)
        x, y, z = lla2ecef(longitude, latitude, altitude)
        euler = [roll, -pitch, 90-azimuth]
        # euler_rotation = [0, 0, -90]
        r_quat = R.from_euler('xyz', euler, degrees=True).as_quat()

        ego_pose = np.eye(4)
        ego_pose[:3, :3] = R.from_quat(r_quat).as_matrix() #qvec2rotmat(r_quat)
        ego_pose[:3, 3] = np.asarray([x, y, z])

        # image_undistortion/back
        cam_ = '-'.join(cam.split('_')[:])
        camera2car = json.load(open(os.path.join(folder, 'calib_extract', \
                                f'calib_camera_{cam}_to_car.json')))[f'camera-{cam_}-to-car']
        camera2car = np.asarray(camera2car['param']['sensor_calib']['data'])

        lidartop2car = json.load(open(os.path.join(folder, 'calib_extract', \
                                 'calib_lidar_top_to_car.json')))['lidar-top-to-car']
        lidartop2car = np.asarray(lidartop2car['param']['sensor_calib']['data'])

        ins2lidartop = json.load(open(os.path.join(folder, 'calib_extract', \
                                 'calib_gnss_to_lidar_top.json')))['gnss-to-lidar-top']
        ins2lidartop = np.asarray(ins2lidartop['param']['sensor_calib']['data'])
        cam_pose = ego_pose @ inverse_pose(ins2lidartop) @ inverse_pose(lidartop2car) @ camera2car

        x, y, z = cam_pose[0, -1], cam_pose[1, -1], cam_pose[2, -1]#lla2ecef(longitude, latitude, altitude)
        ref_gps_txt.write(' '.join([imdir, str(x), str(y), str(z)])+'\n')

    ref_gps_txt.close()

    # '''
    min_ecef_x, min_ecef_y, min_ecef_z = [], [], []
    min_ecef_txt = open(os.path.join(sfm_dir, 'min_ecef.txt'), 'w')
    ref_infos = open(os.path.join(sfm_dir, 'ref_gps.txt'), 'r').readlines()
    for info in ref_infos:
        x, y, z = [float(x) for x in info.strip().split()[1:]][:]

        min_ecef_x.append(x)
        min_ecef_y.append(y)
        min_ecef_z.append(z)
    
    min_ecef_x, min_ecef_y, min_ecef_z = min(min_ecef_x), min(min_ecef_y), min(min_ecef_z)
    min_ecef_txt.write(' '.join([str(min_ecef_x), str(min_ecef_y), str(min_ecef_z)])+'\n')
    min_ecef_txt.close()

    thre_dis = 0.5
    delete_images = []
    ref_used_infos = []
    for i, info in enumerate(ref_infos):
        info_ = [float(x) for x in info.strip().split()[1:]]
        if i==0:
            ref_used_infos.append(info)
        else:
            pre_info = [float(x) for x in ref_used_infos[-1].strip().split()[1:]]
            dist = np.sqrt((info_[0]-pre_info[0])**2+(info_[1]-pre_info[1])**2+(info_[2]-pre_info[2])**2)
            if dist > thre_dis: 
                ref_used_infos.append(info)
            else:
                delete_images.append(info)

    ref_infos = ref_used_infos
    for info in delete_images:
        name = info.strip().split()[0]
        imdir = os.path.join(images, name)
        cmd = f'rm -rf {imdir}'
        os.system(cmd)

    ref_gps_txt = open(os.path.join(sfm_dir, 'ref_gps.txt'), 'w')
    for info in ref_infos:
        info = info.strip().split()
        ref_gps_txt.write(' '.join([info[0], \
            str(float(info[1])-min_ecef_x), \
            str(float(info[2])-min_ecef_y), \
            str(float(info[3])-min_ecef_z)])+'\n')

    ref_gps_txt.close()
    # '''

    intrinsics = dict()
    for cam in ['back', 'left_back', 'right_back']:
        cam_ = '-'.join(cam.split('_'))
        infos = json.load(open(os.path.join(folder, \
                                f'calib_extract/calib_camera_{cam}_to_car.json')))
        intrinsics[cam] = np.asarray(infos[f'camera-{cam_}']['param']['cam_matrix']['data'])
    
    feature_conf = extract_features.confs["d2net-ss"]
    feature_conf['preprocessing']['resize_max'] = 3840
    matcher_conf = match_features.confs["NN-superpoint"]

    references = [p.relative_to(images).as_posix() for p in (images).iterdir()]#[:10]
    extract_features.main(
        feature_conf, images, image_list=references, feature_path=features, masking=True)
    pairs_from_exhaustive.main(sfm_pairs, image_list=references)
    filtering_pairs(sfm_pairs, frames=10)

    # if os.path.exists(matches): os.remove(matches)
    match_features.main(matcher_conf, sfm_pairs, features=features, matches=matches)
    reconstruction.main(
        sfm_dir, 
        images, 
        sfm_pairs, 
        features, matches,  
        image_list=references,
        camera_mode='SINGLE',
        cameras=['_20_', '_40_', '_10_'],
        image_options={cam: {'camera_model': 'PINHOLE',
                    'camera_params': "{}, {}, {}, {}".format(intrinsics[cam][0,0], \
                        intrinsics[cam][1, 1], intrinsics[cam][0,2], intrinsics[cam][1,2]),} \
                        for cam in ['back', 'left_back', 'right_back']},
        mapper_options=dict(ba_refine_focal_length=False, 
                            ba_refine_principal_point=False,
                            ba_refine_extra_params=True,
                            ba_global_max_refinements=50,
                            ba_local_max_refinements=50))