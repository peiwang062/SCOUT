import numpy as np
from scipy._lib._util import _asarray_validated
from scipy.stats import entropy


def compute_ClsSimilarity_underPart(class_attributes, part_type):
    ClsSimilarity_underPart = np.zeros((200, 200))
    for i_p in range(part_type):
        part_attributes = softmax(class_attributes[i_p], axis=1)
        for i in range(200):
            for j in range(i+1, 200):
                ClsSimilarity_underPart[i, j] = ClsSimilarity_underPart[i, j] + Dominik2003IT(part_attributes[i], part_attributes[j])
                ClsSimilarity_underPart[j, i] = ClsSimilarity_underPart[i, j]
    return ClsSimilarity_underPart / part_type


def Dominik2003IT(distribution1, distribution2):
    term1 = distribution1 * np.log((2.0 * distribution1) / (distribution1 + distribution2))
    term2 = distribution2 * np.log((2.0 * distribution2) / (distribution1 + distribution2))
    return np.sum(term1 + term2)

def logsumexp(a, axis=None, b=None, keepdims=False, return_sign=False):

    a = _asarray_validated(a, check_finite=False)
    if b is not None:
        a, b = np.broadcast_arrays(a, b)
        if np.any(b == 0):
            a = a + 0.  # promote to at least float
            a[b == 0] = -np.inf

    a_max = np.amax(a, axis=axis, keepdims=True)

    if a_max.ndim > 0:
        a_max[~np.isfinite(a_max)] = 0
    elif not np.isfinite(a_max):
        a_max = 0

    if b is not None:
        b = np.asarray(b)
        tmp = b * np.exp(a - a_max)
    else:
        tmp = np.exp(a - a_max)

    # suppress warnings about log of zero
    with np.errstate(divide='ignore'):
        s = np.sum(tmp, axis=axis, keepdims=keepdims)
        if return_sign:
            sgn = np.sign(s)
            s *= sgn  # /= makes more sense but we need zero -> zero
        out = np.log(s)

    if not keepdims:
        a_max = np.squeeze(a_max, axis=axis)
    out += a_max

    if return_sign:
        return out, sgn
    else:
        return out


def softmax(x, axis=None):
    # compute in log space for numerical stability
    return np.exp(x - logsumexp(x, axis=axis, keepdims=True))


flist = './cub200/CUB_200_2011/attributes/class_attribute_labels_continuous.txt'
part_num = 15
class_num = 200
attributes = np.zeros((200, 312))

class_index = 0
with open(flist, 'r') as rf:
    for line in rf.readlines():
        attributes_per_class = line.strip().split()
        attributes[class_index, :] = np.float32(attributes_per_class)
        class_index = class_index + 1


similarityMatrix_cls_part = np.zeros((class_num, class_num, part_num))
for i_part in range(part_num):
    if i_part == 0:
        similarityMatrix_cls_part[:, :, i_part] = compute_ClsSimilarity_underPart([attributes[:, 59 - 1:73], attributes[:, 237 - 1: 240]], 2)
    if i_part == 1:
        similarityMatrix_cls_part[:, :, i_part] = compute_ClsSimilarity_underPart([attributes[:, 1 - 1:9], attributes[:, 150 - 1: 152], attributes[:, 279 - 1:293]], 3)
    if i_part == 2:
        similarityMatrix_cls_part[:, :, i_part] = compute_ClsSimilarity_underPart([attributes[:, 198 - 1:212], attributes[:, 245 - 1: 248]], 2)
    if i_part == 3:
        similarityMatrix_cls_part[:, :, i_part] = compute_ClsSimilarity_underPart([attributes[:, 55 - 1:58], attributes[:, 106 - 1: 120]], 2)
    if i_part == 4:
        similarityMatrix_cls_part[:, :, i_part] = compute_ClsSimilarity_underPart([attributes[:, 95 - 1:105], attributes[:, 153 - 1: 167], attributes[:, 294 - 1: 308]], 3)
    if i_part == 5:
        similarityMatrix_cls_part[:, :, i_part] = compute_ClsSimilarity_underPart([attributes[:, 95 - 1:105], attributes[:, 153 - 1: 167]], 2)
    if i_part == 6:
        similarityMatrix_cls_part[:, :, i_part] = compute_ClsSimilarity_underPart([attributes[:, 136 - 1:149], attributes[:, 95 - 1: 105]], 2)
    if i_part == 7:
        similarityMatrix_cls_part[:, :, i_part] = compute_ClsSimilarity_underPart([attributes[:, 264 - 1:278]], 1)
    if i_part == 8:
        similarityMatrix_cls_part[:, :, i_part] = compute_ClsSimilarity_underPart([attributes[:, 10 - 1:24], attributes[:, 213 - 1: 217], attributes[:, 309 - 1: 312]], 3)
    if i_part == 9:
        similarityMatrix_cls_part[:, :, i_part] = compute_ClsSimilarity_underPart([attributes[:, 183 - 1:197]], 1)
    if i_part == 10:
        similarityMatrix_cls_part[:, :, i_part] = compute_ClsSimilarity_underPart([attributes[:, 136 - 1:149], attributes[:, 95 - 1: 105]], 2)
    if i_part == 11:
        similarityMatrix_cls_part[:, :, i_part] = compute_ClsSimilarity_underPart([attributes[:, 264 - 1:278]], 1)
    if i_part == 12:
        similarityMatrix_cls_part[:, :, i_part] = compute_ClsSimilarity_underPart([attributes[:, 10 - 1:24], attributes[:, 213 - 1: 217], attributes[:, 309 - 1: 312]], 3)
    if i_part == 13:
        similarityMatrix_cls_part[:, :, i_part] = compute_ClsSimilarity_underPart([attributes[:, 74 - 1:79], attributes[:, 80 - 1: 94], attributes[:, 168 - 1: 182], attributes[:, 241 - 1: 244]], 4)
    if i_part == 14:
        similarityMatrix_cls_part[:, :, i_part] = compute_ClsSimilarity_underPart([attributes[:, 95 - 1:105], attributes[:, 121 - 1: 135]], 2)

np.save('./cub200/similarityMatrix_cls_part.npy', similarityMatrix_cls_part)
similarityMatrix_cls_part = np.load('./cub200/similarityMatrix_cls_part.npy')

np.save('./cub200/Dominik2003IT_similarityMatrix_cls_part.npy', similarityMatrix_cls_part)
similarityMatrix_cls_part = np.load('./cub200/Dominik2003IT_similarityMatrix_cls_part.npy')


# threshold and remain most unsimilar and distinguising parts between each class pair
threshold = np.sort(similarityMatrix_cls_part.flatten())[int(0.2*class_num*class_num*part_num)]  # the less, the more similar
DistinctMatrix_cls_part_copy = np.copy(similarityMatrix_cls_part)
DistinctMatrix_cls_part_copy[similarityMatrix_cls_part <= threshold] = 0
DistinctMatrix_cls_part_copy[similarityMatrix_cls_part > threshold] = 1
DistinctMatrix_cls_part_copy = DistinctMatrix_cls_part_copy.astype(int)

distinct_extracted_attributes = np.zeros((class_num, class_num), dtype=object)

num_tmp = 0

for i in range(class_num):
    for j in range(i+1, class_num):
        part_idx = np.argwhere(DistinctMatrix_cls_part_copy[i, j, :] == 1).flatten()
        part_idx = part_idx.tolist()
        if len(part_idx) == 0:
            print(i,j)
            num_tmp = num_tmp + 1
        distinct_extracted_attributes[i, j] = part_idx
        distinct_extracted_attributes[j, i] = part_idx
np.save('./cub200/Dominik2003IT_dis_extracted_attributes_02.npy', distinct_extracted_attributes)
print(num_tmp)

