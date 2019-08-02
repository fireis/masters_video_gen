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
if __name__ == "__main__":
    path = ""
    files = get_files(path)
    print(files)
    for file in files:
        create_folder(file)





