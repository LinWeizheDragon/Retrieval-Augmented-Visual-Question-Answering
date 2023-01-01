# This file is modified from tools/mini_tsv/tsv_demo.py

import os
import os.path as op
import json
import cv2
import base64
from tqdm import tqdm

from maskrcnn_benchmark.structures.tsv_file_ops import tsv_reader, tsv_writer
from maskrcnn_benchmark.structures.tsv_file_ops import generate_linelist_file
from maskrcnn_benchmark.structures.tsv_file_ops import generate_hw_file
from maskrcnn_benchmark.structures.tsv_file import TSVFile
from maskrcnn_benchmark.data.datasets.utils.image_ops import img_from_base64

from vqa_tools import VQA

dummy = False

qs_valid_file = '../../data/ok-vqa/OpenEnded_mscoco_val2014_questions.json'
qs_train_file = '../../data/ok-vqa/OpenEnded_mscoco_train2014_questions.json'
annotation_valid_file = '../../data/ok-vqa/mscoco_val2014_annotations.json'
annotation_train_file = '../../data/ok-vqa/mscoco_train2014_annotations.json'

for mode in ['train', 'valid']:
    if mode == 'train':
        data_path = "../../data/ok-vqa/train2014"
    else:
        data_path = "../../data/ok-vqa/val2014"

    # where the data should be stored
    tsv_file = "datasets/okvqa/{}.tsv".format(mode)
    label_file = "datasets/okvqa/{}.label.tsv".format(mode)
    hw_file = "datasets/okvqa/{}.hw.tsv".format(mode)
    linelist_file = "datasets/okvqa/{}.linelist.tsv".format(mode)

    rows = []
    rows_label = []
    rows_hw = []

    if mode == 'train':
        vqa_helper = VQA(annotation_train_file, qs_train_file)
    else:
        vqa_helper = VQA(annotation_valid_file, qs_valid_file)

    vqa_helper.createIndex()
    vqa_helper.info()

    img_list = []

    for imgId in vqa_helper.imgToQA.keys():
        dataSubType = vqa_helper.dataSubType
        imgFilename = 'COCO_' + dataSubType + '_'+ str(imgId).zfill(12) + '.jpg'
        img_path = os.path.join(data_path, imgFilename)
        img_list.append((imgId, img_path))
        if dummy:
            if len(img_list) > 20:
                break

    print('total imgs related to this set:', len(img_list))

    for imgId, img_p in tqdm(img_list):
        img_key = img_p.split('.')[0].split('_')[-1]
        img_path = img_p
        img = cv2.imread(img_path)
        img_encoded_str = base64.b64encode(cv2.imencode('.jpg', img)[1])
        
        related_qs = vqa_helper.getQuesIds(imgIds=[imgId])
        related_qas = vqa_helper.loadQA(ids=related_qs)
        # print(related_qas)
        
        # vqa_helper.showQA(related_qas)
        
        # Here is just a toy example of labels.
        # The real labels can be generated from the annotation files
        # given by each dataset. The label is a list of dictionary 
        # where each box with at least "rect" (xyxy mode) and "class"
        # fields. It can have any other fields given by the dataset.
        labels = []
        labels.append({"rect": [1, 1, 30, 40], "class": "Dog", 'ok-vqa': related_qas})
        # labels.append({"rect": [2, 3, 100, 100], "class": "Cat"})
        # labels += related_qas

        row = [img_key, img_encoded_str]
        rows.append(row)

        row_label = [img_key, json.dumps(labels)]
        rows_label.append(row_label)

        height = img.shape[0]
        width = img.shape[1]
        row_hw = [img_key, json.dumps([{"height":height, "width":width}])]
        rows_hw.append(row_hw)

    tsv_writer(rows, tsv_file)
    tsv_writer(rows_label, label_file)
    tsv_writer(rows_hw, hw_file)

    # generate linelist file
    generate_linelist_file(label_file, save_file=linelist_file)


# # To access a tsv file:
# # 1) Use tsv_reader to read dataset in given order
# rows = tsv_reader("tools/mini_tsv/data/train.tsv")
# rows_label = tsv_reader("tools/mini_tsv/data/train.label.tsv")
# for row, row_label in zip(rows, rows_label):
#     img_key = row[0]
#     labels = json.loads(row_label[1])
#     img = img_from_base64(row[1])

# # 2) use TSVFile to access dataset at any given row.
# tsv = TSVFile("tools/mini_tsv/data/train.tsv")
# row = tsv.seek(1) # to access the second row 
# img_key = row[0]
# img = img_from_base64(row[1])



