# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 19:23:23 2018

@author: user
"""

import os
import shutil

def create_dictionary(path,count):
    lang = {}
    for root, dirs, files in os.walk(path):
        for file in files:
          rs = root.split("\\")
          l = rs[1]
          if not l in lang.keys():
              lang[l] = count
              count+=1
#              print count
    return lang
      
#      print l
#      shutil.copy2(path_file,"Omniglot Dataset/test")
def create_new_folder(path,count):
    lang = create_dictionary(path,count)
    print str(len(lang))
    for root, dirs, files in os.walk(path):
        for file in files:
            rs = root.split("\\")
            pre = lang.get(rs[1])
            new_file_name = str(pre)+"."+file
#            print new_file_name
      
            shutil.copy2(os.path.join(root,file),dest_folder)
            dst_file = os.path.join(dest_folder, file)
            new_dst_file_name = os.path.join(dest_folder, new_file_name)
            os.rename(dst_file, new_dst_file_name)

#lang = {}
#count=0
dest_folder = 'knn_dataset22/'
os.makedirs(dest_folder)
background_path = "Omniglot Dataset/images_background"
evaluation_path = "Omniglot Dataset/images_evaluation"

create_new_folder(background_path,0)

create_new_folder(evaluation_path,30)
