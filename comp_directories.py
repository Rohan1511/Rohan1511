# -*- coding: utf-8 -*-
"""
Created on Mon Jun 21 00:57:34 2021

@author: SHAGUN
"""
############## Comparing 2 directories and delete extra files #################

# There is some problem with the dataset. We found that one of the folder
# has 13 more images than others and hence this code.   


from pathlib import Path
dir1 = r'G:\\My Drive\\_Train_data3'
dir2 = r'G:\\My Drive\\_Train_labels'

def cmp_file_lists(dir1, dir2):
    dir1_filenames = set(f.name for f in Path(dir1).rglob('*'))
    dir2_filenames = set(f.name for f in Path(dir2).rglob('*'))
    files_in_dir1_but_not_dir2 = dir1_filenames - dir2_filenames 
    files_in_dir2_but_not_dir1 = dir2_filenames - dir1_filenames 
    return files_in_dir1_but_not_dir2, files_in_dir2_but_not_dir1

dir1_filenames, dir2_filenames = cmp_file_lists(dir1, dir2)


# Showing all the 