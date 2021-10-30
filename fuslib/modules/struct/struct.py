import os
import shutil

def struct(conf):

    print("Creation of the structure of the project...")

    # Structure folder
    if conf.remove_struct and os.path.exists(conf.work_dir):
        shutil.rmtree(conf.work_dir)
    if not os.path.exists(conf.work_dir):
        os.mkdir(conf.work_dir)

  
    # Data folder
    if not os.path.exists(conf.work_dir + os.sep + "data"):
        os.mkdir(conf.work_dir + os.sep + "data")

    # Models folder
    if not os.path.exists(conf.work_dir + os.sep + "models"):
        os.mkdir(conf.work_dir + os.sep + "models")


    print("Creation of the project done.\n")