import os, random
from glob import glob

data_root = './depthstudio/GAC015/Clips'
sections = glob(os.path.join(data_root, 'Sec005*'))
random.shuffle(sections)

ratio_train = 0.85
train_list = sections[:int(ratio_train*len(sections))]
valid_list = sections[int(ratio_train*len(sections)):]

train_txt = open(os.path.join(data_root.replace(data_root.split('/')[-1], ''), 'trainval.txt'), 'w')
test_txt = open(os.path.join(data_root.replace(data_root.split('/')[-1], ''), 'test.txt'), 'w')
for sec in train_list:
    sec = '/'.join(sec.split('/')[-2:]) + '\n'
    train_txt.write(sec)

train_txt.close()
for sec in valid_list:
    sec = '/'.join(sec.split('/')[-2:]) + '\n'
    test_txt.write(sec)
    
test_txt.close()

min_length = 300
folder_path = ''
for sec in sections:
    im_len = os.listdir(os.path.join(sec, 'image_undistortion/back'))
    if len(im_len) < 120:
        cmd = f'rm -rf {sec}'
        os.system(cmd)
            
    if len(im_len) < min_length: 
        print(len(im_len))
        min_length = len(im_len)
        folder_path = sec

a = 1