import os
import cv2
import json
import shutil
import random
import xml.etree.ElementTree as ET

def get_annotations(xml_path, class_names):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    annotations = []
    for object in root.findall('object'):
        cls = object.find('name').text
        obj = object.find('bndbox')
        xmin = obj.find('xmin').text
        ymin = obj.find('ymin').text
        xmax = obj.find('xmax').text
        ymax = obj.find('ymax').text
        annotations.append((int(xmin), int(ymin), int(xmax)-int(xmin), int(ymax)-int(ymin), int(class_names.index(cls))+1))
    return annotations

data_path = 'datasets/JPEGImages/'
label_path = 'datasets/Annotations/'
output_dir = 'pet/'
jsons_path = os.path.join(output_dir, 'annotations/')
train_imgs_path = os.path.join(output_dir, 'train2017/')
val_imgs_path = os.path.join(output_dir, 'val2017/')
if not os.path.exists(output_dir):
    os.makedirs(jsons_path)
    os.makedirs(train_imgs_path)
    os.makedirs(val_imgs_path)
    
class_names = ['cat', 'dog']

data = {}
data['info'] = {
    'description': '',
    'url': '',
    'version': '1.0',
    'year': 2022,
    'contributor': 'Jacky',
    'date_created': ''}
data['categories'] = [{'supercategory': 'deterioration', 'id': i+1, 'name': cls} for i, cls in enumerate(class_names)]
data['licenses'] = []

train_data = data.copy()
val_data = data.copy()

imgs_list = sorted([file for file in os.listdir(data_path) if file.split('.')[-1] == 'jpg'])
random.seed(2022)
random.shuffle(imgs_list)

img_id = 1
ann_id = 1
train_test_split = 0.2
train_imgs = []
val_imgs = []
train_anns = []
val_anns = []
for img in imgs_list:
    img_path = os.path.join(data_path, img)
    w, h, _ = cv2.imread(img_path).shape
    xml_path = os.path.join(label_path, img.split('.')[0]+'.xml')
    if img_id/len(imgs_list) > train_test_split:
        out_img_path = os.path.join(train_imgs_path, img)
        shutil.copyfile(img_path, out_img_path)
        train_imgs.append({'id':img_id,
                           'width':w, 
                           'height':h,
                           'license':1,
                           'file_name':img})
        annotations = get_annotations(xml_path, class_names)
        for ann in annotations:
            # annotation_json
            train_anns.append({'id': ann_id,
                               'image_id': img_id,
                               'category_id': ann[-1],
                               'segmentation': [],
                               'bbox': [ann[0], ann[1], ann[2], ann[3]],
                               'area': ann[2]*ann[3],
                               'iscrowd': 0})
            ann_id+=1
    else:
        out_img_path = os.path.join(val_imgs_path, img)
        shutil.copyfile(img_path, out_img_path)   
        val_imgs.append({'id':img_id,
                         'width':w, 
                         'height':h,
                         'license':1,
                         'file_name':img})
        annotations = get_annotations(xml_path, class_names)
        for ann in annotations:
            # annotation_json
            val_anns.append({'id': ann_id,
                             'image_id': img_id,
                             'category_id': ann[-1],
                             'segmentation': [],
                             'bbox': [ann[0], ann[1], ann[2], ann[3]],
                             'area': ann[2]*ann[3],
                             'iscrowd': 0})
            ann_id+=1
    img_id+=1
    
train_data['images'] = train_imgs
train_data['annotations'] = train_anns
val_data['images'] = val_imgs
val_data['annotations'] = val_anns

with open(os.path.join(jsons_path, 'instances_train2017.json'), 'w') as json_file:
    json.dump(train_data, json_file)
    
with open(os.path.join(jsons_path, 'instances_val2017.json'), 'w') as json_file:
    json.dump(val_data, json_file)