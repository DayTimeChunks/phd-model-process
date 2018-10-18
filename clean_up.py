import os
import glob

dir_name = "/Users/DayTightChunks/Documents/PhD/Models/phd-model-process/Gen10/"
folders = os.listdir(dir_name)
print(folders)

f_num = 9
for item in folders[1:]:
    os.chdir(dir_name)
    # print(dir_name)

    # Remove input maps
    for name in glob.glob('*/*.map'):  # Remove map file endings
        # print(name)
        # print(os.path.join(dir_name, name))
        os.remove(os.path.join(dir_name, name))

    # Remove output maps
    for num in range(1, f_num+1):
        model_version = dir_name + item + "/" + str(num)
        # print(model_version)
        if os.path.exists(model_version):
            os.chdir(model_version)
        else:
            continue
        # print(os.getcwd())
        for name in glob.glob('*[0-9]'):  # Remove numeric file endings
            print(os.path.join(model_version, name))
            os.remove(os.path.join(model_version, name))

# import glob
# for name in glob.glob('dir/*.[0-9]*'):
#     print name
