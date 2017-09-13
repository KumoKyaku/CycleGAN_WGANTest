from PIL import Image
import os
from const import * 
import random

HEIGHT = baseHeight*30
WIDTH =baseWidth*30

def crop(img):
    xs,ys= img.size
    if xs < WIDTH  or ys < HEIGHT:
        return None

    startx = random.randint(0,xs - WIDTH) 
    starty = random.randint(0,ys-HEIGHT)
    Image.Image.crop
    return img.crop((startx,starty,startx + WIDTH,starty + HEIGHT))

def makeTrainPic(path, outNum, outPath,startNum = 0):
    fileNames = os.listdir(path)
    lengh = len(fileNames)
    for i in range(startNum,startNum + outNum):
        fN = fileNames[i%lengh]
        img = Image.open(os.path.join(path,fN))
        img = crop(img)
        while(img == None):
            fileNames.remove(fN)
            lengh = len(fileNames)
            fN = fileNames[i%lengh]
            img = Image.open(os.path.join(path,fN))
            img = crop(img)
        img.save(os.path.join(outPath,"%d-"%i + fN))

def main():
    makeTrainPic('H:\\aaaa', 20000, 'picG',130000)
    #makeTrainPic('H:\\train2014',110000,'picF')

if __name__ == '__main__':
    main()