from tfImport import *
import utils

def main():
    picG = 'picG.zip'
    picF = 'picF.zip'
    files = ['picf%d.zip'%i for i in range(1,5)]
    files.extend(['picg1.zip','picg2.zip'])
    print(files)

    oldpath = FLAGS.buckets

    files = ['picf%d.zip'%i for i in range(1,4)]
    for f in files:
        fn = utils.pai_copy(f,oldpath)
        print(fn)
        utils.Unzip(fn)

    # fn = util.pai_copy(picG,FLAGS.buckets)
    # print(fn)
    # util.Unzip(fn)
    
    # fn = util.pai_copy(picF,FLAGS.buckets)
    # print(fn)
    # util.Unzip(fn)

    l = os.listdir(util.temppath)

    print(len(l))
    print(l)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--buckets', type=str, default='',
                        help='input data path')

    parser.add_argument('--checkpointDir', type=str, default='',
                        help='output model path')
    FLAGS, _ = parser.parse_known_args()

    
    main()