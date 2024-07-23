# -*- coding: utf-8 -*-
"""
 @time: 2024/1/16 10:38
 @desc:
"""

from utils.file_tools import file_scanf2
import random

classes = {"n02106662": 0,
           "n02124075": 1,
           "n02281787": 2,
           "n02389026": 3,
           "n02492035": 4,
           "n02504458": 5,
           "n02510455": 6,
           "n02607072": 7,
           "n02690373": 8,
           "n02906734": 9,
           "n02951358": 10,
           "n02992529": 11,
           "n03063599": 12,
           "n03100240": 13,
           "n03180011": 14,
           "n03272010": 15,
           "n03272562": 16,
           "n03297495": 17,
           "n03376595": 18,
           "n03445777": 19,
           "n03452741": 20,
           "n03584829": 21,
           "n03590841": 22,
           "n03709823": 23,
           "n03773504": 24,
           "n03775071": 25,
           "n03792782": 26,
           "n03792972": 27,
           "n03877472": 28,
           "n03888257": 29,
           "n03982430": 30,
           "n04044716": 31,
           "n04069434": 32,
           "n04086273": 33,
           "n04120489": 34,
           "n04555897": 35,
           "n07753592": 36,
           "n07873807": 37,
           "n11939491": 38,
           "n13054560": 39}


class LabelReader(object):
    def __init__(self):
        pass

    def read(self, file_path, dic):
        with open(file_path) as f:
            lines = f.readlines()
            for line in lines:
                class_id = line.split('_')[0]
                try:
                    if isinstance(dic[class_id], list):
                        dic[class_id].append(line.strip())
                except KeyError:
                    dic[class_id] = []
                    dic[class_id].append(line.strip())
        return dic


def get_session(dic, idx=0):
    this_session = []
    class_names = list(dic.keys())
    random.shuffle(class_names)
    for c in class_names:
        images_cls = dic[c][idx*25:(idx+1)*25]
        this_session.append(c+'  '+str(classes[c]))
        this_session.extend(images_cls)
    return this_session


if __name__ == '__main__':
    set_idx = 1
    label_reader = LabelReader()
    txt_filenames = file_scanf2(path='/data0/hossam/1-EEG_repeate/1-Dataset/CVPR2021-02785/design',
                                contains=['run'], endswith='.txt')
    dic = {}
    for f in txt_filenames:
        dic = label_reader.read(file_path=f, dic=dic)

    for s in range(40):
        session = get_session(dic, idx=s)

        with open('image_idx_set_' + str(s) + '.txt', 'w') as f:
            session = [e + '\n' for e in session]
            f.writelines(session)

    # for c in classes.keys():
    #     with open('image_idx_as_'+c+'.txt', 'w') as f:
    #         lines = dic[c]
    #         class_idx = classes[c]
    #         lines = [e + ' ' + str(class_idx) + '\n' for e in lines]
    #         f.writelines(lines)
