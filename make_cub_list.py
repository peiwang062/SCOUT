import random

image_list_path = './cub200/CUB_200_2011/images.txt'
class_list_path = './cub200/CUB_200_2011/image_class_labels.txt'
tr_te_split_list_path = './cub200/CUB_200_2011/train_test_split.txt'


imlist = []
with open(image_list_path, 'r') as rf:
    for line in rf.readlines():
        imindex, impath = line.strip().split()
        imlist.append(impath)

classlist = []
with open(class_list_path, 'r') as rf:
    for line in rf.readlines():
        imindex, label = line.strip().split()
        classlist.append(str(int(label)-1))

tr_te_list = []
with open(tr_te_split_list_path, 'r') as rf:
    for line in rf.readlines():
        imindex, tr_te = line.strip().split()
        tr_te_list.append(tr_te)


save_train_list = './cub200/CUB_200_2011/CUB200_gt_tr.txt'
save_val_list = './cub200/CUB_200_2011/CUB200_gt_val.txt'
save_test_list = './cub200/CUB_200_2011/CUB200_gt_te.txt'


raw_train_list = []
for i in range(len(imlist)):
    if tr_te_list[i] == '1':
        raw_train_list.append(imlist[i] + " " + classlist[i])

random.shuffle(raw_train_list)
train_list = raw_train_list[:round(0.9*len(raw_train_list))]
val_list = raw_train_list[round(0.9*len(raw_train_list)):]

num_tr = 0
fl = open(save_train_list, 'w')
for i in range(len(train_list)):
    example_info = './cub200/CUB_200_2011/images/' + train_list[i].split()[0] + " " + train_list[i].split()[1] + " " + str(num_tr)
    fl.write(example_info)
    fl.write("\n")
    num_tr = num_tr + 1
fl.close()


num_val = 0
fl = open(save_val_list, 'w')
for i in range(len(val_list)):
    example_info = './cub200/CUB_200_2011/images/' + val_list[i].split()[0] + " " + val_list[i].split()[1] + " " + str(num_val)
    fl.write(example_info)
    fl.write("\n")
    num_val = num_val + 1
fl.close()


num_te = 0
fl = open(save_test_list, 'w')
for i in range(len(imlist)):
    if tr_te_list[i] == '0':
        example_info = './cub200/CUB_200_2011/images/' + imlist[i] + " " + classlist[i] + " " + str(num_te)
        fl.write(example_info)
        fl.write("\n")
        num_te = num_te + 1

fl.close()
