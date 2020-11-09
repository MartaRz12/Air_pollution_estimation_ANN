
# -------------------------------------------------------------------------------------------------------
# Untar .tar.gz filr with image from Landsat
# -------------------------------------------------------------------------------------------------------
import tarfile
import sys

def file_untar(path, file_name):
    try:
        my_tar = tarfile.open(path + file_name)
        my_tar.extractall(path) # specify which folder to extract to
        my_tar.close()
    except:
        print('   -> my_tar failed: ', sys.exc_info()[0])
        return -1
    return 0