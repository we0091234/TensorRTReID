import os
import shutil

print("haha")

f = open("/home/cxl/tensorCaffe/caffeTensorRT/logo_0.800000.txt")
savePath = '/home/data/cxl/ReidData/result'
i = 0
while True:
    line = f.readline()
    if not line:
        break
    line = line.strip('\n')
    fileList = line.split(" ")
    for file in fileList:
        # print(file)
        if os.path.isfile(file):
            picName = file.split('/')[-1]
            newPicFolder = os.path.join(savePath,str(i))
            # print(newPicFolder)
            if not os.path.exists(newPicFolder):
                os.mkdir(newPicFolder)
            newPicPath = os.path.join(newPicFolder,picName)
            print(file,newPicPath,i)
            shutil.copy(file,newPicPath)
        # else:
        #     print(file)

    i=i+1