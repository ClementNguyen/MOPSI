import os
import subprocess
import time

dossier='/home/sasha/darknet'
os.chdir(dossier)
# os.system('./darknet detect cfg/yolov3-tiny.cfg yolov3-tiny.weights data/dog.jpg')

def detect(image):
    #utilisation de yolo-tiny v3
    result = subprocess.check_output(['./darknet','detect','cfg/yolov3-tiny.cfg','yolov3-tiny.weights','data/'+image])
    
    #analyse des résultats
    result=str(result).split('seconds.')[1]
    result=result.split("\\n")
    result.pop(0)
    result.pop(-1)
    names=[]
    percentages=[]
    for k in range(len(result)):
        names.append(result[k].split(': ')[0])
        percentages.append(result[k].split(': ')[1])
    print(names)
    return(names,percentages)

def similarite(image1,image2):
    time1=time.time()
    names1=detect(image1)[0]
    names2=detect(image2)[0]
    for i in range(len(names1)):
        for j in range(len(names2)):
            if names1[i]==names2[j] :
                print(time.time()-time1)
                return True
    print(time.time()-time1)
    return False
                
    

