import os
import multiprocessing
from functools import partial
from glob import glob
from tqdm import tqdm

save_folder = '/media/flechazo/d8ffe5af-8df8-4c7d-9ece-abd98e5eb238/UrbanX'
data_root = './depthstudio/GAC015/Clips'
sections = glob(os.path.join(data_root, 'Sec005*'))
# data_bar = tqdm(sections, total=len(sections))
# for idx, sec in enumerate(data_bar):
#     clip_folder = os.path.join(save_folder, sec.split('/')[-1])
#     if not os.path.exists(clip_folder): os.makedirs(clip_folder, exist_ok=True)

#     old_mvs = os.path.join(sec, 'mvs/')
#     move_cmd = f'cp -r {old_mvs} {clip_folder}'
#     os.system(move_cmd)

def move2urbanx(sec):
    clip_folder = os.path.join(save_folder, sec.split('/')[-1])
    if not os.path.exists(clip_folder): os.makedirs(clip_folder, exist_ok=True)

    old_mvs = os.path.join(sec, 'mvs/')
    move_cmd = f'cp -r {old_mvs} {clip_folder}'
    os.system(move_cmd)
    return None

with multiprocessing.Pool(processes = 6) as pool:
    partial_func = partial(move2urbanx)
    results = list(tqdm(pool.imap(partial_func, sections), total=len(sections), desc="moving data to urbanx ..."))