import os
import json
import csv
import pandas as pd

def namelist():
    out = {"img": [], "label": []}
    out_path = "data/{}.json".format("imgs_NamesLbls")
    in_path = open('data/train.csv','r')
    fieldNames = ('img','label')
    reader = csv.DictReader(in_path,fieldNames)
    for row in reader:
        row['img'] = os.path.join('data/train_images/',row['img'])
        out['img'].append(row['img'])
        out['label'].append(row['label'])
    json.dump(out,open(out_path,'w'),indent=4,ensure_ascii=False)

# def pairsList():
#     out = {"img1": [], "img2": [], "img1_lbl": [], "img2_lbl": []}
#     out_path = "JSONFiles/{}.json".format("imgs_pairsGen")

#     with open("JSONFiles/imgs_NamesLbls.json") as f:
#         imgs_names = json.load(f)  #'img':[], 'label':[]

#     n = len(imgs_names["img"])
#     o = len(imgs_names["label"])

#     # combs_list = list(range(n))
#     # combs = list(combinations(combs_list,2)) #tuple of number combination with range 1-50, binomialCoeff of 1275

#     for i in range(0, n - 1):
#         for j in range(i + 1, n):
#             # if j<i:
#             #     j = i
#             out["img1"].append(imgs_names["img"][i])
#             out["img2"].append(imgs_names["img"][j])
#             out["img1_lbl"].append(imgs_names["label"][i])
#             out["img2_lbl"].append(imgs_names["label"][j])
#             # j = j+1

#     json.dump(out, open(out_path, "w"), indent=4, ensure_ascii=False)


if __name__ == "__main__":
    namelist()
    # pairsList()
    # pass
