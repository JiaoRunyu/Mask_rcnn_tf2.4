# Mask-rcnn mask all image file successful!!!!!!!!!!!!!!!!!!!!!!!!!!!
import os
import sys

# Root directory of the project
import skimage.io

ROOT_DIR = os.path.abspath("../")

#Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import  mrcnn.model as modellib
from mrcnn import visualize
# Import COCO config
sys.path.append(os.path.join(ROOT_DIR, "samples/coco/")) #To find local version
import coco

#Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR,"logs")

#Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR,"mask_rcnn_coco.h5")
#Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

#Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR,"images")

#2______________________________
class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()
# 3______________________________
# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)
#4_______________________________
# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']
# _________________________________________自定义数据集处理——————————————————————————————————————————
import os
import glob
import cv2
import numpy as np

# INPUT_DIR = '/home/vad/VAD/DATA/KITTI_Raw/'
# OUTPUT_DIR = '/home/vad/VAD/struct2depth/train/seg/'

INPUT_DIR = '/home/vad/VAD/Utile/ORB_SLAM2/ORB_SLAM2-master/data/KITTI/'
OUTPUT_DIR = '/home/vad/VAD/Utile/ORB_SLAM2/ORB_SLAM2-master/data/seg/'

for d in glob.glob(INPUT_DIR + '/*/'):
    date = d.split('/')[-2]

    for d2 in glob.glob(d + '*/'):
        seqname = d2.split('/')[-2]
        print('Processing sequence',seqname)
        for subfolder in ['image_02/data']:
            seqname = d2.split('/')[-2] + subfolder.replace('image','').replace('/data','')
            # if not os.path.exists(OUTPUT_DIR + seqname):
            #     os.mkdir(OUTPUT_DIR + seqname)
            folder = d2 + subfolder
            print("folder= ",folder)
            files = glob.glob(folder + '/*.png')
            files.sort()
            img_dir = files
            save_folder1 = d2 + 'image_02_new'
            save_folder = os.path.join(OUTPUT_DIR,save_folder1.split('/',6)[6])
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
# MASK-------------------------------------------------------------------
            data_dir = folder
            if not data_dir.endswith('/'):
                data_dir = data_dir + '/'
                img_dir = os.listdir(data_dir)
                img_dir.sort()

            print(img_dir)
            for i, gambar_img in enumerate(img_dir):
                raw = os.path.splitext(gambar_img)[0].split(".")[0] \
                      + os.path.splitext(gambar_img)[1]
                print(raw)
                img = skimage.io.imread(os.path.join(data_dir,raw))

                results = model.detect([img],verbose=1)

                r = results[0]

                detected = r['class_ids'].size

                print(detected)

                for i in range(detected):

                    if r['class_ids'][i] == 3:
                        continue
                    elif r['class_ids'][i] == 6:
                        continue
                    elif r['class_ids'][i] == 8:
                        continue
                    else:
                        r['class_ids'][i] = 0
                        r['rois'][i] = 0
                        r['masks'][i] = 0
                        r['scores'][i] = 0

                print("raw = ",raw)
                raw_name = raw.split('.')
                a = visualize.save_image(img, raw_name[0], r['rois'], r['masks'], r['class_ids'], r['scores'],class_names,
                                 save_dir=save_folder, mode=3)

                # 如果图中无实例则返回纯黑图片，否则将彩色mask转成灰度mask图
                if a == 'None':
                    black = np.zeros(img.shape).astype(np.uint8)
                    black = cv2.cvtColor(black, cv2.COLOR_BGR2GRAY)
                    print("he")
                    cv2.imwrite(os.path.join(save_folder, raw_name[0] + '.png'), black)
                else:
                    print("aa = ", raw)
                    black = cv2.imread(os.path.join(save_folder,raw))
                    black = cv2.cvtColor(black, cv2.COLOR_BGR2GRAY)
                    cv2.imwrite(os.path.join(save_folder, raw_name[0] + '.png'), black)


