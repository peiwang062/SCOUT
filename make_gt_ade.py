import numpy as np
from scipy import misc

def softmax(x):
    ex = np.exp(x)
    sum_ex = np.sum(np.exp(x))
    return ex/sum_ex

ADE_gt_tr = './ade/ADE_gt_tr.txt'
img_path_list_tr = []
img_label_list_tr = []
with open(ADE_gt_tr, 'r') as rf:
    for line in rf.readlines():
        img_name, img_label, img_index = line.strip().split()
        img_name = img_name[:45] + "annotations" + img_name[51:-3] + "png"
        img_path_list_tr.append(img_name)
        img_label_list_tr.append(int(img_label))

Class_num = 1040

# for all_cls_attributes_info
# all_cls_attributes_info = np.zeros((Class_num, 150))
# img_label_list_tr = np.array(img_label_list_tr)
# for i_cls in range(Class_num):
#     print(i_cls)
#     idx = np.where(img_label_list_tr == i_cls)[0]
#     cur_img_path_list_tr = [img_path_list_tr[i] for i in idx]
#     cls_attributes_info = np.zeros((len(cur_img_path_list_tr), 150))
#
#     if len(cur_img_path_list_tr) > 0:
#         for i_img in range(len(cur_img_path_list_tr)):
#             image = misc.imread(cur_img_path_list_tr[i_img])
#             image = image.flatten().tolist()
#             objectList = list(set(image))
#             objectList.sort()
#             objectList = np.array(objectList)
#             objectList = objectList[objectList != 0]
#             cls_attributes_info[i_img, objectList-1] = 1
#         cls_attributes_info = np.sum(cls_attributes_info, axis=0) / len(cur_img_path_list_tr)
#         # cls_attributes_info = softmax(cls_attributes_info)
#         all_cls_attributes_info[i_cls, :] = cls_attributes_info
#
# np.save('./ade/all_cls_attributes_info.npy', all_cls_attributes_info)



all_cls_attributes_info = np.load('./ade/all_cls_attributes_info.npy')


def compute_ClsDissimilarity_underObject(class_attributes):
    ClsDissimilarity_underObject = np.zeros((1040, 1040))
    for i in range(1040):
        for j in range(1040):
            if i != j:
                if class_attributes[i] > 0.0 and class_attributes[j] == 0:
                    ClsDissimilarity_underObject[i, j] = class_attributes[i]  # > 0 means this object i class has but j class doesn't have
    return ClsDissimilarity_underObject

Object_num = 150

# # all_cls_attributes_info[all_cls_attributes_info < 0.3] = 0
# dissimilarityMatrix_cls_part = np.zeros((Class_num, Class_num, Object_num))
# for i_obj in range(Object_num):
#     print(i_obj)
#     dissimilarityMatrix_cls_part[:, :, i_obj] = compute_ClsDissimilarity_underObject(all_cls_attributes_info[:, i_obj])
#
# np.save('./ade/dissimilarityMatrix_cls_part.npy', dissimilarityMatrix_cls_part)


dissimilarityMatrix_cls_part = np.load('./ade/dissimilarityMatrix_cls_part.npy')

dissimilarityMatrix_cls_part_copy = np.copy(dissimilarityMatrix_cls_part)
dissimilarityMatrix_cls_part_copy[dissimilarityMatrix_cls_part_copy > 0.0] = 1
dissimilarityMatrix_cls_part_copy = dissimilarityMatrix_cls_part_copy.astype(int)
print(np.sum(dissimilarityMatrix_cls_part_copy) / (Class_num*Class_num*Object_num))
dis_extracted_attributes = np.zeros((Class_num, Class_num), dtype=object)
for i in range(Class_num):
    for j in range(Class_num):
        if i != j:
            object_idx = np.argwhere(dissimilarityMatrix_cls_part_copy[i, j, :] == 1).flatten()
            object_idx = object_idx + 1
            object_idx = object_idx.tolist()
            dis_extracted_attributes[i, j] = object_idx

np.save('./ade/dis_extracted_attributes.npy', dis_extracted_attributes)


#dissimilarityMatrix_cls_part_copy = np.copy(dissimilarityMatrix_cls_part)
#dissimilarityMatrix_cls_part_copy[dissimilarityMatrix_cls_part_copy > 0.5] = 1
#dissimilarityMatrix_cls_part_copy = dissimilarityMatrix_cls_part_copy.astype(int)
#print(np.sum(dissimilarityMatrix_cls_part_copy) / (Class_num*Class_num*Object_num))
#dis_extracted_attributes = np.zeros((Class_num, Class_num), dtype=object)
#for i in range(Class_num):
#    for j in range(Class_num):
#        if i != j:
#            object_idx = np.argwhere(dissimilarityMatrix_cls_part_copy[i, j, :] == 1).flatten()
#            object_idx = object_idx + 1
#            object_idx = object_idx.tolist()
#            dis_extracted_attributes[i, j] = object_idx
#
#np.save('./ade/dis_extracted_attributes_05.npy', dis_extracted_attributes)

# # threshold and remain most similar parts between each class pair
# threshold = np.sort(similarityMatrix_cls_part.flatten())[int(0.99*Class_num*Class_num*Object_num)]
# similarityMatrix_cls_part_copy = np.copy(similarityMatrix_cls_part)
# similarityMatrix_cls_part_copy[similarityMatrix_cls_part >= threshold] = 1
# similarityMatrix_cls_part_copy[similarityMatrix_cls_part < threshold] = 0
# similarityMatrix_cls_part_copy = similarityMatrix_cls_part_copy.astype(int)
#
# com_extracted_attributes = np.zeros((Class_num, Class_num), dtype=object)
# for i in range(Class_num):
#     for j in range(i+1, Class_num):
#         object_idx = np.argwhere(similarityMatrix_cls_part_copy[i, j, :] == 1).flatten()
#         object_idx = object_idx + 1
#         object_idx = object_idx.tolist()
#         com_extracted_attributes[i, j] = object_idx
#         com_extracted_attributes[j, i] = object_idx
# np.save('./ade/com_extracted_attributes_001.npy', com_extracted_attributes)
