import os

#This script changes the names of all the images in the subfolders

#pull root folder
root = os.getcwd()

#get list of subfolders
folder_list = os.listdir(root)
print(folder_list)

#for each folder, iteratively rename
for folder in folder_list:
    #add back slash between spaces
    path = root + "/" + folder 
    if (folder != "Label.py" and folder != ".DS_Store"):
        file_list = os.listdir(path)
        #print(file_list)
        i = 0
        for file in file_list:
            oldfile_path = path + "/" + file
            newfile_path = path + "/" + folder + " " + str(i) + ".jpg"
            os.rename(oldfile_path, newfile_path)
            i = i + 1 