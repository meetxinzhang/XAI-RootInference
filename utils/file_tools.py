# -*- coding: utf-8 -*-
"""
 @time: 2024/1/16 13:07
 @desc:
"""

import glob
import os
import platform


def file_scanf2(path, contains, endswith, sub_ratio=1):
    files = glob.glob(path + '/*')
    input_files = []
    for f in files[:int(len(files) * sub_ratio)]:
        if platform.system().lower() == 'windows':
            f.replace('\\', '/')
        if not any([c in f.split('/')[-1] for c in contains]):
            continue
        if not f.endswith(endswith):
            continue
        input_files.append(f)
    return input_files

