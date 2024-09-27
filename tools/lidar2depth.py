import os, open3d as o3d
import matplotlib.pyplot as plt
import numpy as np, json, cv2
import multiprocessing

from functools import partial
from tqdm import tqdm
from glob import glob

def lidar2cam(pts_3d_lidar, L2C):
    n = pts_3d_lidar.shape[0]
    pts_3d_hom = np.hstack((pts_3d_lidar, np.ones((n,1))))
    pts_3d_cam_ref = np.dot(pts_3d_hom, np.transpose(L2C))
    # pts_3d_cam_rec = np.transpose(np.dot(self.R0, np.transpose(pts_3d_cam_ref)))
    return pts_3d_cam_ref

def rect2Img(rect_pts, img_width, img_height, P):
    n = rect_pts.shape[0]
    points_hom = np.hstack((rect_pts, np.ones((n,1))))
    points_2d = np.dot(points_hom, np.transpose(P)) # nx3
    points_2d[:,0] /= points_2d[:,2]
    points_2d[:,1] /= points_2d[:,2]
    
    mask = (points_2d[:,0] >= 0) & (points_2d[:,0] <= img_width) & (points_2d[:,1] >= 0) & (points_2d[:,1] <= img_height)
    mask = mask & (rect_pts[:,2] > 2)
    return points_2d[mask,0:2], mask

def dense_map(Pts, n, m, grid):
    ng = 2 * grid + 1
    
    mX = np.zeros((m,n)) + np.float32("inf")
    mY = np.zeros((m,n)) + np.float32("inf")
    mD = np.zeros((m,n))
    mX[np.int32(Pts[1]),np.int32(Pts[0])] = Pts[0] - np.round(Pts[0])
    mY[np.int32(Pts[1]),np.int32(Pts[0])] = Pts[1] - np.round(Pts[1])
    mD[np.int32(Pts[1]),np.int32(Pts[0])] = Pts[2]
    
    KmX = np.zeros((ng, ng, m - ng, n - ng))
    KmY = np.zeros((ng, ng, m - ng, n - ng))
    KmD = np.zeros((ng, ng, m - ng, n - ng))
    
    for i in range(ng):
        for j in range(ng):
            KmX[i,j] = mX[i : (m - ng + i), j : (n - ng + j)] - grid - 1 +i
            KmY[i,j] = mY[i : (m - ng + i), j : (n - ng + j)] - grid - 1 +i
            KmD[i,j] = mD[i : (m - ng + i), j : (n - ng + j)]
    S = np.zeros_like(KmD[0,0])
    Y = np.zeros_like(KmD[0,0])
    
    for i in range(ng):
        for j in range(ng):
            s = 1/np.sqrt(KmX[i,j] * KmX[i,j] + KmY[i,j] * KmY[i,j])
            Y = Y + s * KmD[i,j]
            S = S + s
    
    S[S == 0] = 1
    out = np.zeros((m,n))
    out[grid + 1 : -grid, grid + 1 : -grid] = Y/S
    return out

data_root = './depthstudio/GAC015/Clips'
sections = glob(os.path.join(data_root, 'Sec005*'))

cameras = ['back',]
image_list = []
for cam in cameras:
    for sec in sections:
        image_list += sorted(glob(os.path.join(sec, f'image_undistortion/{cam}/*.jpg')), \
            key=lambda x: int(x.split('_')[-1].split('.')[0]))

def funs_lidar2depth(imdir):
    cam = imdir.split('/')[-2]
    sec = imdir.replace('/'.join(imdir.split('/')[-3:]), '')
    
    depth_folder = os.path.join(sec, f'depths/{cam}')
    if not os.path.exists(depth_folder): os.makedirs(depth_folder, exist_ok=True)
    
    save_name_jpg = os.path.join(depth_folder, imdir.split('/')[-1])
    save_name_pfm = os.path.join(depth_folder, imdir.split('/')[-1].replace('.jpg', '.pfm'))
    if os.path.exists(save_name_jpg) and os.path.exists(save_name_pfm): return None
    
    
    imidx = imdir.split('_')[-1].split('.')[0]
    img = cv2.imread(imdir)
    lidar = glob(os.path.join(sec, f'output_top_lidar/*_640_{imidx}.pcd'))
    assert len(lidar) == 1 and os.path.exists(lidar[0])
    
    points = o3d.io.read_point_cloud(lidar[0])
    points = np.asarray(points.points)
    
    lidar2cam_infos = np.asarray(json.load(open(os.path.join(sec, \
        'calib_extract/calib_lidar_top_to_cam.json')))[f'{cam}']).reshape(4, 4)[:3, :]
    
    # lidar_rect = lidar2cam(points, lidar2cam_infos)
    # functions of lidar2cam
    pts_3d_lidar = points
    L2C = lidar2cam_infos
    n = pts_3d_lidar.shape[0]
    pts_3d_hom = np.hstack((pts_3d_lidar, np.ones((n,1))))
    pts_3d_cam_ref = np.dot(pts_3d_hom, np.transpose(L2C))
    lidar_rect = pts_3d_cam_ref

    P_cam = np.zeros((3, 4), dtype=lidar2cam_infos.dtype)
    P_cam[:3, :3] = np.asarray(json.load(open(os.path.join(sec, \
        f'calib_extract/calib_camera_{cam}_to_car.json')))[f'camera-{cam}']['param']['cam_matrix']['data']).reshape(3, 3)
    
    # lidarOnImage, mask = rect2Img(lidar_rect, img.shape[1], img.shape[0], P_cam)
    # functions of rect2Img
    P = P_cam
    rect_pts = lidar_rect
    img_width, img_height = img.shape[1], img.shape[0]
    
    n = rect_pts.shape[0]
    points_hom = np.hstack((rect_pts, np.ones((n,1))))
    points_2d = np.dot(points_hom, np.transpose(P)) # nx3
    points_2d[:,0] /= points_2d[:,2]
    points_2d[:,1] /= points_2d[:,2]
    
    mask = (points_2d[:,0] >= 0) & (points_2d[:,0] <= img_width) & (points_2d[:,1] >= 0) & (points_2d[:,1] <= img_height)
    mask = mask & (rect_pts[:,2] > 2)
    lidarOnImage, mask = points_2d[mask,0:2], mask

    lidarOnImage = np.concatenate((lidarOnImage, lidar_rect[mask,2].reshape(-1,1)), 1)

    # out = dense_map(lidarOnImage.T, img.shape[1], img.shape[0], 1)
    # functions of dense_map
    # dense_map(Pts, n, m, grid):
    Pts, n, m, grid = lidarOnImage.T, img.shape[1], img.shape[0], 1
    ng = 2 * grid + 1
    
    mX = np.zeros((m,n)) + np.float32("inf")
    mY = np.zeros((m,n)) + np.float32("inf")
    mD = np.zeros((m,n))
    mX[np.int32(Pts[1]),np.int32(Pts[0])] = Pts[0] - np.round(Pts[0])
    mY[np.int32(Pts[1]),np.int32(Pts[0])] = Pts[1] - np.round(Pts[1])
    mD[np.int32(Pts[1]),np.int32(Pts[0])] = Pts[2]
    
    KmX = np.zeros((ng, ng, m - ng, n - ng))
    KmY = np.zeros((ng, ng, m - ng, n - ng))
    KmD = np.zeros((ng, ng, m - ng, n - ng))
    
    for i in range(ng):
        for j in range(ng):
            KmX[i,j] = mX[i : (m - ng + i), j : (n - ng + j)] - grid - 1 +i
            KmY[i,j] = mY[i : (m - ng + i), j : (n - ng + j)] - grid - 1 +i
            KmD[i,j] = mD[i : (m - ng + i), j : (n - ng + j)]
    S = np.zeros_like(KmD[0,0])
    Y = np.zeros_like(KmD[0,0])
    
    for i in range(ng):
        for j in range(ng):
            s = 1/np.sqrt(KmX[i,j] * KmX[i,j] + KmY[i,j] * KmY[i,j])
            Y = Y + s * KmD[i,j]
            S = S + s
    
    S[S == 0] = 1
    out = np.zeros((m,n))
    out[grid + 1 : -grid, grid + 1 : -grid] = Y/S

    # value_min = min(out.reshape(-1))
    # value_max = max(out.reshape(-1))
    
    out = np.asarray(out, dtype=np.float32)
    # value_min_a = min(out.reshape(-1))
    # value_max_a = max(out.reshape(-1))
    
    plt.figure(figsize=(20,40))
    plt.imsave(save_name_jpg, out)
    plt.close()
    
    cv2.imwrite(save_name_pfm, out)
    return None
    
with multiprocessing.Pool(processes = 32) as pool:
    partial_func = partial(funs_lidar2depth)
    results = list(tqdm(pool.imap(partial_func, image_list), total=len(image_list), desc="process lidar to sparse depth maps ..."))
    

"""
data_bar = tqdm(sections, total=len(sections))
for i, sec in enumerate(data_bar):
    data_bar.set_description('Tackling with Section: {} ...'.format(sec.split('/')[-1]))
    for cam in cameras:
        imgs = sorted(glob(os.path.join(sec, f'image_undistortion/{cam}/*.jpg')), \
            key=lambda x: int(x.split('_')[-1].split('.')[0]))
        
        depth_folder = os.path.join(sec, f'depths/{cam}')
        if not os.path.exists(depth_folder): os.makedirs(depth_folder, exist_ok=True)
        for imdir in imgs:
            save_name_jpg = os.path.join(depth_folder, imdir.split('/')[-1])
            save_name_pfm = os.path.join(depth_folder, imdir.split('/')[-1].replace('.jpg', '.pfm'))
            if os.path.exists(save_name_jpg) and os.path.exists(save_name_pfm):
                continue
            
            imidx = imdir.split('_')[-1].split('.')[0]
            img = cv2.imread(imdir)
            lidar = glob(os.path.join(sec, f'output_top_lidar/*_640_{imidx}.pcd'))
            assert len(lidar) == 1 and os.path.exists(lidar[0])
            
            points = o3d.io.read_point_cloud(lidar[0])
            points = np.asarray(points.points)
            
            lidar2cam_infos = np.asarray(json.load(open(os.path.join(sec, \
                'calib_extract/calib_lidar_top_to_cam.json')))[f'{cam}']).reshape(4, 4)[:3, :]
            
            # lidar_rect = lidar2cam(points, lidar2cam_infos)
            # functions of lidar2cam
            pts_3d_lidar = points
            L2C = lidar2cam_infos
            n = pts_3d_lidar.shape[0]
            pts_3d_hom = np.hstack((pts_3d_lidar, np.ones((n,1))))
            pts_3d_cam_ref = np.dot(pts_3d_hom, np.transpose(L2C))
            lidar_rect = pts_3d_cam_ref
    
            P_cam = np.zeros((3, 4), dtype=lidar2cam_infos.dtype)
            P_cam[:3, :3] = np.asarray(json.load(open(os.path.join(sec, \
                f'calib_extract/calib_camera_{cam}_to_car.json')))[f'camera-{cam}']['param']['cam_matrix']['data']).reshape(3, 3)
            
            # lidarOnImage, mask = rect2Img(lidar_rect, img.shape[1], img.shape[0], P_cam)
            # functions of rect2Img
            P = P_cam
            rect_pts = lidar_rect
            img_width, img_height = img.shape[1], img.shape[0]
            
            n = rect_pts.shape[0]
            points_hom = np.hstack((rect_pts, np.ones((n,1))))
            points_2d = np.dot(points_hom, np.transpose(P)) # nx3
            points_2d[:,0] /= points_2d[:,2]
            points_2d[:,1] /= points_2d[:,2]
            
            mask = (points_2d[:,0] >= 0) & (points_2d[:,0] <= img_width) & (points_2d[:,1] >= 0) & (points_2d[:,1] <= img_height)
            mask = mask & (rect_pts[:,2] > 2)
            lidarOnImage, mask = points_2d[mask,0:2], mask

            lidarOnImage = np.concatenate((lidarOnImage, lidar_rect[mask,2].reshape(-1,1)), 1)

            # out = dense_map(lidarOnImage.T, img.shape[1], img.shape[0], 1)
            # functions of dense_map
            # dense_map(Pts, n, m, grid):
            Pts, n, m, grid = lidarOnImage.T, img.shape[1], img.shape[0], 1
            ng = 2 * grid + 1
            
            mX = np.zeros((m,n)) + np.float32("inf")
            mY = np.zeros((m,n)) + np.float32("inf")
            mD = np.zeros((m,n))
            mX[np.int32(Pts[1]),np.int32(Pts[0])] = Pts[0] - np.round(Pts[0])
            mY[np.int32(Pts[1]),np.int32(Pts[0])] = Pts[1] - np.round(Pts[1])
            mD[np.int32(Pts[1]),np.int32(Pts[0])] = Pts[2]
            
            KmX = np.zeros((ng, ng, m - ng, n - ng))
            KmY = np.zeros((ng, ng, m - ng, n - ng))
            KmD = np.zeros((ng, ng, m - ng, n - ng))
            
            for i in range(ng):
                for j in range(ng):
                    KmX[i,j] = mX[i : (m - ng + i), j : (n - ng + j)] - grid - 1 +i
                    KmY[i,j] = mY[i : (m - ng + i), j : (n - ng + j)] - grid - 1 +i
                    KmD[i,j] = mD[i : (m - ng + i), j : (n - ng + j)]
            S = np.zeros_like(KmD[0,0])
            Y = np.zeros_like(KmD[0,0])
            
            for i in range(ng):
                for j in range(ng):
                    s = 1/np.sqrt(KmX[i,j] * KmX[i,j] + KmY[i,j] * KmY[i,j])
                    Y = Y + s * KmD[i,j]
                    S = S + s
            
            S[S == 0] = 1
            out = np.zeros((m,n))
            out[grid + 1 : -grid, grid + 1 : -grid] = Y/S
    
            # value_min = min(out.reshape(-1))
            # value_max = max(out.reshape(-1))
            
            out = np.asarray(out, dtype=np.float32)
            # value_min_a = min(out.reshape(-1))
            # value_max_a = max(out.reshape(-1))
            
            plt.figure(figsize=(20,40))
            plt.imsave(save_name_jpg, out)
            plt.close()
            
            cv2.imwrite(save_name_pfm, out)
"""