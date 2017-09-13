import os
from tfImport import *



def main():
    sess = tf.Session()
    li = os.listdir("temp")
    print(li)

    for f in li:
        if os.path.isdir(f):
            cdir = os.path.join('temp',f)
            cdirfs = os.listdir(cdir)
            print(cdirfs)

if __name__ == '__main__':
    main()