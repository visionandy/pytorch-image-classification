import os
import glob
import random
import shutil

ratio=0.9

data_input_path='/home/andywang/class_abc_800FPhotos/'
data_output_path='/home/andywang/project/dataset/rock/v1/'
task_list=['task1','task2','task3']
folder_list=glob.glob(data_input_path+'*/') ## 1st folder hierarchy



        ###file copy
        
def file_copy_fun(source_file_list,destina_folder):
    
    for ori_file in source_file_list:
        image_name=ori_file.split('/')[-1]
        destina_image_name=destina_folder+image_name
        shutil.copyfile(ori_file,destina_image_name)

def folder_check(folder_create):
    if not os.path.exists(folder_create):
        os.makedirs(folder_create)


for i in range(len(task_list)):   
    input_folder_id=folder_list[1]  ## 1st folder hierarchy
    sub_folder_list=glob.glob(input_folder_id+'*/') 
    output_folder_id=data_output_path+task_list[i]+'/'
    if not os.path.exists(output_folder_id):
        os.makedirs(output_folder_id)

    train_task_path_i=output_folder_id+'train/'
    val_task_path_i=output_folder_id+'val/'

    folder_check(train_task_path_i)
    folder_check(val_task_path_i)


    for j in range(len(sub_folder_list)):  ## the 2nd folder hierarchy
        category_train_path=train_task_path_i+str(j)+'/'
        category_val_path=val_task_path_i+str(j)+'/'


        if not os.path.exists(category_train_path):
            os.makedirs(category_train_path)

        if not os.path.exists(category_val_path):
            os.makedirs(category_val_path)


        original_im_folder=sub_folder_list[j]
        all_img_list=glob.glob(original_im_folder+'*.jpg')
        random.shuffle(all_img_list)
        train_file_index=int(len(all_img_list)*ratio)
        train_im_list=all_img_list[:train_file_index]
        val_im_list=all_img_list[train_file_index:]
        file_copy_fun(train_im_list,category_train_path)
        file_copy_fun(val_im_list,category_val_path)









