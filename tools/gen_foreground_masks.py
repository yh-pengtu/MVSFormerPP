import os, cv2, json
from labelme import utils
from tqdm import tqdm
from glob import glob

data_root = '/media/flechazo/A1/depthstudio/GAC015/Clips'
data_foler = sorted([os.path.join(data_root, x) \
                for x in os.listdir(data_root) if '.zip' not in x])

mask_left_back = data_root.replace(data_root.split('/')[-1], 'CarMask/left_back.json')
mask_right_back = data_root.replace(data_root.split('/')[-1], 'CarMask/right_back.json')
assert os.path.exists(mask_left_back) and os.path.exists(mask_right_back)

data1 = json.load(open(mask_left_back))
data2 = json.load(open(mask_right_back))

img = utils.image.img_b64_to_arr(data1['imageData'])
mask_left_back, _ = utils.shape.labelme_shapes_to_label(img.shape, data1['shapes'])

img = utils.image.img_b64_to_arr(data2['imageData'])
mask_right_back, _ = utils.shape.labelme_shapes_to_label(img.shape, data2['shapes'])

# cv2.imwrite('mask_left_back.jpg', mask_left_back*255.0)
# cv2.imwrite('mask_right_back.jpg', mask_right_back*255.0)

data_stream = tqdm(data_foler, total=len(data_foler))
for idx, folder in enumerate(data_stream):
    mask_sky_groups = glob(os.path.join(folder, 'masks/right_back/sky/*'))

    foregrounddir = os.path.join(folder, 'masks/right_back/foreground/')
    if not os.path.exists(foregrounddir): os.makedirs(foregrounddir, exist_ok=True)

    for _, skydir in enumerate(mask_sky_groups):
        objectdir = skydir.replace('masks/right_back/sky/', \
                'masks/right_back/object/')#.replace(skydir.split('.')[-1], '*'))
        objectdir = glob(objectdir.split('.')[0]+'*')

        assert len(objectdir)==1 and os.path.exists(objectdir[0])

        if 'left_back' in skydir:
            mask_car = mask_left_back
        else:
            mask_car = mask_right_back
        
        mask_sky = cv2.imread(skydir, -1)
        mask_object = cv2.imread(objectdir[0], -1)
        mask_foreground = (1-((mask_sky>0)+(mask_object>0)+(mask_car>0))) * 255

        cv2.imwrite(os.path.join(foregrounddir, objectdir[0].split('/')[-1]), mask_foreground)

    data_stream.set_description(folder.split('/')[-1])