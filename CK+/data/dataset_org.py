import glob
from shutil import copyfile

emotions = ["neutral", "anger", "contempt", "disgust", "fear", "happy", "sadness", "surprise"]
participants = glob.glob("CK+/Emotion//*") #{S005, S010, ...}
#Returns a list of all folders with participant numbers")
#participants = glob.glob("source_emotion//*")

for x in participants:
    part = x[-4:] #store current participant number exmp: S005 given file name without file's path
    for sessions in glob.glob("%s//*" %x): #Store list of sessions for current participant
        for files in glob.glob("%s//*" %sessions):
            current_session = sessions[-3:]
            #current_session = files
            file = open(files, 'r')

            emotion = int(float(file.readline())) #emotions are encoded as a float, readline as float, then convert to integer.
            
            #get path for last image in sequence, which contains the emotion
            sourcefile_emotion = glob.glob("CK+/cohn-kanade-images//%s//%s//*" %(part, current_session))[-1] 
            #print (sourcefile_emotion)
            #print (len(glob.glob("C:/CK+/cohn-kanade-images//%s//%s//*" %(part, current_session))))
           # print len(glob.glob("source_images//%s//%s//*" %(part, current_session)))
            sourcefile_neutral = glob.glob("CK+/cohn-kanade-images//%s//%s//*" %(part, current_session))[0] #do same for neutral image
            #print (sourcefile_neutral)
            
            #Generate path to put neutral image
            dest_neut = "CK+/sorted_set//neutral//%s" %sourcefile_neutral[-21:] 
            
            #Do same for emotion containing image #print dest_neut
            dest_emot = "CK+/sorted_set//%s//%s" %(emotions[emotion], sourcefile_emotion[-21:]) 

            copyfile(sourcefile_neutral, dest_neut) #Copy file
            copyfile(sourcefile_emotion, dest_emot) #Copy file
