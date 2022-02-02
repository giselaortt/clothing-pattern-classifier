#Script to split the data into train and test sets

import re
import os
from sklearn.model_selection import train_test_split


def split_files( source_dir_path, train_dir_path, test_dir_path, train_size = 0.7 ):

    os.system("mkdir "+train_dir_path)
    os.system("mkdir "+test_dir_path)
    class_names = os.listdir( source_dir_path )

    for class_name in class_names:

        train_class_dir_path = train_dir_path+"/"+class_name
        test_class_dir_path = test_dir_path+"/"+class_name
        source_class_dir_path = source_dir_path+"/"+class_name

        os.system("mkdir "+train_class_dir_path)
        os.system("mkdir "+test_class_dir_path)

        file_names = os.listdir( source_class_dir_path )
        train_files, test_files = train_test_split(file_names,train_size = train_size)

        for name in train_files:
            os.system('cp '+ os.path.join(source_class_dir_path, name) + ' ' + os.path.join(train_class_dir_path, name) )

        for name in test_files:
            os.system('cp '+ os.path.join(source_class_dir_path, name) + ' ' + os.path.join(test_class_dir_path, name) )

    return


if __name__ == '__main__':
    split_files("../database/FingerCamera", "../database/train_folder", "../database/test_folder", train_size = 0.7)

