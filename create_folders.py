import os
import glob

def get_files(file_path, file_type="/*.mp4"):
    path = file_path + file_type
    files = glob.glob(path)
    return files

def create_folder(file, file_type=".mp4"):
    path = file.replace(file_type, "")
    if not os.path.exists(path):
        os.makedirs(path)
    return path

def move_file(file, new_path):
    _, file_name = os.path.split(file)

    new_path = new_path +"/"+ file_name
    os.rename(file, new_path)

if __name__ == "__main__":
    path = ""
    files = get_files(path)
    print(files)
    for file in files:
        new_path = create_folder(file)
        move_file(file, new_path)






