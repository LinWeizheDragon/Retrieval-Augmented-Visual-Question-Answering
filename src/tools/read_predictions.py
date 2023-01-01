import csv
import json

tsv_file = open("data/ok-vqa/pre-extracted_features/vinvl_output/vinvl_large_okvqa_trainset/inference/vinvl_large/predictions.tsv")
read_tsv = csv.reader(tsv_file, delimiter="\t")
csv.field_size_limit(100000000)
print(csv.field_size_limit())
for row in read_tsv:
    image_key, prediction = row
    prediction = json.loads(prediction)
    print(prediction.keys())
    for obj in prediction['objects']:
        print(obj['rect'])
        print(obj['class'])
        print(obj['conf'])
    input()