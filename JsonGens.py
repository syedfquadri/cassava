import os
import json


def namelist():
    root_dirs = "data"
    files = os.listdir(root_dirs)  # A B C
    # files = os.listdir(classes)
    out = {"img": [], "label": []}
    out_path = "JSONFiles/{}.json".format("imgs_NamesLbls")

    for file in files:
        class_dir = os.path.join(root_dirs, file)
        label_name = str(file)
        images = os.listdir(class_dir)
        for img in images:
            image = os.path.join(class_dir, img)
            out["img"].append(image)
            out["label"].append(label_name)

    json.dump(out, open(out_path, "w"), indent=4, ensure_ascii=False)


def pairsList():
    out = {"img1": [], "img2": [], "img1_lbl": [], "img2_lbl": []}
    out_path = "JSONFiles/{}.json".format("imgs_pairsGen")

    with open("JSONFiles/imgs_NamesLbls.json") as f:
        imgs_names = json.load(f)  #'img':[], 'label':[]

    n = len(imgs_names["img"])
    o = len(imgs_names["label"])

    # combs_list = list(range(n))
    # combs = list(combinations(combs_list,2)) #tuple of number combination with range 1-50, binomialCoeff of 1275

    for i in range(0, n - 1):
        for j in range(i + 1, n):
            # if j<i:
            #     j = i
            out["img1"].append(imgs_names["img"][i])
            out["img2"].append(imgs_names["img"][j])
            out["img1_lbl"].append(imgs_names["label"][i])
            out["img2_lbl"].append(imgs_names["label"][j])
            # j = j+1

    json.dump(out, open(out_path, "w"), indent=4, ensure_ascii=False)


if __name__ == "__main__":
    namelist()
    pairsList()
    # pass
