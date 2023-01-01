import io
import os
import json
from utils.vqa_tools import VQA
from tqdm import tqdm
from google.cloud import vision


def detect_text(path):
    """Detects text in the file."""
    
    client = vision.ImageAnnotatorClient()

    with io.open(path, 'rb') as image_file:
        content = image_file.read()

    image = vision.Image(content=content)

    response = client.text_detection(image=image)
    # return response 

    texts = response.text_annotations
    full_text_annotation = response.full_text_annotation

    # print('Texts:')

    result = {
        'img_path': path,
        'filtered_text_annotations': [],
        # 'text_annotations': texts,
        # 'full_text_annotation': full_text_annotation,
    }

    for index, text in enumerate(texts):
        # print('\n"{}"'.format(text.description))
        result['filtered_text_annotations'].append({
            'description': text.description,
            'vertices': [(vertex.x, vertex.y) for vertex in text.bounding_poly.vertices],
            # 'language':[{
            #     'language_code': detected_language.language_code,
            #     'confidence': detected_language.confidence,
            # } for detected_language in full_text_annotation.pages[0].blocks[index].property.detected_languages]
        })
    
    return result




dummy = False

mode = 'valid'
for mode in ['train', 'valid']:
    # To generate a tsv file:
    if mode == 'train':
        data_path = "../data/ok-vqa/train2014"
    else:
        data_path = "../data/ok-vqa/val2014"

    qs_valid_file = '../data/ok-vqa/OpenEnded_mscoco_val2014_questions.json'
    qs_train_file = '../data/ok-vqa/OpenEnded_mscoco_train2014_questions.json'
    annotation_valid_file = '../data/ok-vqa/mscoco_val2014_annotations.json'
    annotation_train_file = '../data/ok-vqa/mscoco_train2014_annotations.json'

    print('start loading files!')
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
            if len(img_list) > 2:
                break

    print('total imgs related to this set:', len(img_list))

    data_save_path = '../data/ok-vqa/pre-extracted_features/OCR/'+mode
    os.makedirs(data_save_path, exist_ok=True)

    for imgId, img_p in tqdm(img_list):
        img_key = img_p.split('.')[0].split('_')[-1]
        img_path = os.path.join(data_path, img_p)
        
        to_save_data_path = os.path.join(data_save_path, img_key+'_ocr.json')
        if os.path.exists(to_save_data_path):
            print('skipping file', to_save_data_path)
        else:
            res = detect_text(img_path)
            res.update({
                'img_key': img_key,
            })
            with open(to_save_data_path, 'w') as f:
                json.dump(res, f)
            print('file saved', to_save_data_path)