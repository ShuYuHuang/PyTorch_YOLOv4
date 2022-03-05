import os
import glob
import shutil
import xml.etree.ElementTree as ET

def getImagesInDir(dir_path):
    image_list = []
    for filename in glob.glob(dir_path + '/*.jpg'):
        image_list.append(filename)

    return image_list

def convert(size, box):
    dw = 1./(size[0])
    dh = 1./(size[1])
    x = (box[0] + box[1])/2.0 - 1
    y = (box[2] + box[3])/2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)

def convert_annotation(img_path, ann_dir, output_image_path, output_label_path):
    basename = os.path.basename(img_path)
    basename_no_ext = os.path.splitext(basename)[0]
    
    #copy image
    shutil.copy(img_path, os.path.join(output_image_path, basename))

    in_file = open(ann_dir + '/' + basename_no_ext + '.xml')
    out_file = open(output_label_path + basename_no_ext + '.txt', 'w')
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult)==1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
        bb = convert((w,h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')
        
classes = ['cat', 'dog']
train_test_split_rate = 0.2

img_dir = 'datasets/datasets/JPEGImages/'
ann_dir = 'datasets/datasets/Annotations/'
image_paths = getImagesInDir(img_dir)
random.seed(2022)
random.shuffle(image_paths)

train_image_path = 'datasets/pet/train/images/'
train_label_path = 'datasets/pet/train/labels/'
valid_image_path = 'datasets/pet/valid/images/'
valid_label_path = 'datasets/pet/valid/labels/'

if not os.path.exists(train_image_path):
    os.makedirs(train_image_path)
if not os.path.exists(train_label_path):
    os.makedirs(train_label_path)
if not os.path.exists(valid_image_path):
    os.makedirs(valid_image_path)
if not os.path.exists(valid_label_path):
    os.makedirs(valid_label_path)

train_test_split = len(image_paths)*train_test_split_rate

for i, img_path in enumerate(image_paths):
    if i >= train_test_split:
        # train
        convert_annotation(img_path, ann_dir, train_image_path, train_label_path)
    else:
        # valid
        convert_annotation(img_path, ann_dir, valid_image_path, valid_label_path)