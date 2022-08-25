import random

# load labels, first manually split sceneCategories.txt into sceneCategories_tr.txt (first 20210 lines) and sceneCategories_val.txt (rest 2000 lines), which we have provided in ./ade/ADEChallengeData2016/
scene_categories_tr = './ade/ADEChallengeData2016/sceneCategories_tr.txt'
scene_categories_val = './ade/ADEChallengeData2016/sceneCategories_val.txt'


img_path_list_tr = []
img_label_list_tr = []
img_path_list_val = []
img_label_list_val = []

with open(scene_categories_tr, 'r') as rf:
    for line in rf.readlines():
        img_name, img_label = line.strip().split()
        img_path_list_tr.append('./ade/ADEChallengeData2016/images/training/' + img_name + '.jpg')
        img_label_list_tr.append(img_label)

with open(scene_categories_val, 'r') as rf:
    for line in rf.readlines():
        img_name, img_label = line.strip().split()
        img_path_list_val.append('./ade/ADEChallengeData2016/images/validation/' + img_name + '.jpg')
        img_label_list_val.append(img_label)



img_label_index_list_tr = []
img_label_index_list_val = []
picked_img_path_list_val = []

# make label dictionary
labelName_labelIndex = {}
label_index = 0
label_name = []
for i in range(len(img_label_list_tr)):
    if img_label_list_tr[i] in labelName_labelIndex:
        img_label_index_list_tr.append(labelName_labelIndex[img_label_list_tr[i]])
    else:
        labelName_labelIndex[img_label_list_tr[i]] = label_index
        label_index = label_index + 1
        img_label_index_list_tr.append(labelName_labelIndex[img_label_list_tr[i]])
        label_name.append(img_label_list_tr[i])

for i in range(len(img_label_list_val)):
    if img_label_list_val[i] in labelName_labelIndex:
        img_label_index_list_val.append(labelName_labelIndex[img_label_list_val[i]])
        picked_img_path_list_val.append(img_path_list_val[i])

# picked 10 % from training set to be as validation set

combined = list(zip(img_path_list_tr, img_label_index_list_tr))
random.shuffle(combined)
img_path_list_tr[:], img_label_index_list_tr[:] = zip(*combined)
img_path_list_tr_val = img_path_list_tr[round(len(img_label_list_tr)*0.9):]
img_label_index_list_tr_val = img_label_index_list_tr[round(len(img_label_list_tr)*0.9):]
img_path_list_tr = img_path_list_tr[:round(len(img_label_list_tr)*0.9)]
img_label_index_list_tr = img_label_index_list_tr[:round(len(img_label_list_tr)*0.9)]


# save list
ADE_gt_tr = './ade/ADEChallengeData2016/ADE_gt_tr.txt'
num_tr = 0
fl = open(ADE_gt_tr, 'w')
for i in range(len(img_path_list_tr)):
    example_info = img_path_list_tr[i] + " " + str(img_label_index_list_tr[i]) + " " + str(num_tr)
    fl.write(example_info)
    fl.write('\n')
    num_tr = num_tr + 1
fl.close()

ADE_gt_tr_val = './ade/ADEChallengeData2016/ADE_gt_tr_val.txt'
num_tr_val = 0
fl = open(ADE_gt_tr_val, 'w')
for i in range(len(img_path_list_tr_val)):
    example_info = img_path_list_tr_val[i] + " " + str(img_label_index_list_tr_val[i]) + " " + str(num_tr_val)
    fl.write(example_info)
    fl.write('\n')
    num_tr_val = num_tr_val + 1
fl.close()


ADE_gt_val = './ade/ADEChallengeData2016/ADE_gt_val.txt'
num_val = 0
fl = open(ADE_gt_val, 'w')
for i in range(len(picked_img_path_list_val)):
    example_info = picked_img_path_list_val[i] + " " + str(img_label_index_list_val[i]) + " " + str(num_val)
    fl.write(example_info)
    fl.write('\n')
    num_val = num_val + 1
fl.close()


ADE_label_name = './ade/ADEChallengeData2016/ADE_label_name.txt'
num_name = 0
fl = open(ADE_label_name, 'w')
for i in range(len(label_name)):
    example_info = str(num_name) + " " + label_name[i]
    fl.write(example_info)
    fl.write('\n')
    num_name = num_name + 1
fl.close()