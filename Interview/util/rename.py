import os


def traverse_dir(directory):
    for root, dirs, files in os.walk(directory):
        if files:
            for file in files:
                filename = os.path.join(root, file)
                suffix = os.path.splitext(file)[1]
                new_name = os.path.join(IMAGE_DIR, str(traverse_dir.count)+suffix)
                os.popen('copy %s %s' % (filename, new_name))
                traverse_dir.count += 1
                print(traverse_dir.count)
                print(new_name)
        if dirs:
            for dir_ in dirs:
                sub_dir = os.path.join(root, dir_)
                traverse_dir(sub_dir)


IMAGE_DIR = '..\\images'
os.chdir(IMAGE_DIR)
traverse_dir.count = 1
traverse_dir(IMAGE_DIR)
