from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
import sys
import os
import shutil

pylab.rcParams['figure.figsize'] = (8.0, 10.0)

dataDir='.'
dataType='val2014'
annFile='{}/instances_{}.json'.format(dataDir,dataType)


# initialize COCO api for instance annotations
coco=COCO(annFile)
# display COCO categories and supercategories

cats = coco.loadCats(coco.getCatIds())
cats =[cat['name'] for cat in cats]
print(cats)
print(len(cats))



# plt.axis('off')
# plt.imshow(I)
# plt.show()

predicted_dir_path = './predicted'
ground_truth_dir_path = '../../mAP/ground-truth'
# if os.path.exists(predicted_dir_path):
#     shutil.rmtree(predicted_dir_path)
if os.path.exists(ground_truth_dir_path):
    shutil.rmtree(ground_truth_dir_path)
# os.mkdir(predicted_dir_path)
os.mkdir(ground_truth_dir_path)


test_images_file = './5k.txt'  #for coco
with open(test_images_file, 'r') as f:
    txt = f.readlines()
    test_images = [line.strip() for line in txt]




for idx, input_image_path in enumerate(test_images):
    # print(idx)
    filename = os.path.split(input_image_path)[1]# 'COCO_val2014_000000290979.jpg'
    filename = os.path.splitext(filename)[0]  # 'COCO_val2014_000000290979.jpg'
    image_id =  int(filename.split('_')[2])
    ground_truth_path = os.path.join(ground_truth_dir_path, str(idx) + '.txt')

    # img = coco.loadImgs([image_id])[0]
    # I = io.imread(img['coco_url'])

    annIds = coco.getAnnIds(imgIds=[image_id]);
    anns = coco.loadAnns(annIds)

    with open(ground_truth_path, 'w') as f:
        for i in range(len(anns)):
            category_id = anns[i]["category_id"]
            category = coco.loadCats(category_id)[0]["name"]
            category = ''.join(category.split())

            print("{}: {}".format(category_id, category))

            bbox = anns[i]['bbox']
            x, y,w,h = bbox

            box = [x , y , x+w , y+h]
            box = list(map(round, box))

            xmin, ymin, xmax, ymax = list(map(str, box))

            bbox_mess = ' '.join([category, xmin, ymin, xmax, ymax]) + '\n'
            f.write(bbox_mess)
