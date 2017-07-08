import pandas
from datetime import datetime
import time
import numpy as np
from keras.preprocessing.text import one_hot


#the function returns a list of strings.
def load_dataset_SIAV():

    dataframe = pandas.read_csv("data/logSiav_last_anonimyzed.csv", header=0)
    dataframe = dataframe.replace(r's+', 'empty', regex=True)
    dataframe = dataframe.fillna(0)
    #print dataframe.dtypes
    #print dataframe.select_dtypes(['float64','int64'])



    dataset=dataframe.values
    ev_translation = {}
    n_ev = 97 #lowercase letters, just useful for printing
    for line in dataset:
        if not line[1] in ev_translation:
            ev_translation[line[1]]=chr(n_ev)
            n_ev+=1
            #print n_ev
        line[1]=ev_translation[line[1]]
    #print dataset[0]
    #dataset=dataset[:,:8]
    values = []
    for i in range(dataset.shape[1]):
        values.append(len(np.unique(dataset[:, i])) )#+1
    #print values
    #print np.unique(dataset[:, 5])
    elems_per_fold = int(values[0] / 3)

    print "elemns per fold",elems_per_fold
    datasetTR = dataset[dataset[:,0]<2*elems_per_fold]
    #test set
    datasetTS = dataset[dataset[:,0]>=2*elems_per_fold]
    #trick empty column siav log
    #datasetTR=datasetTR[:,:8]
    #datasetTS=datasetTS[:,:8]

    #print len(values)
    #print dataset[0]
    def generate_set(dataset):

        data=[]
        newdataset=[]
        temptarget=[]
        #analyze first dataset line
        caseID=dataset[0][0]
        event=dataset[0][1]
        n = 1

        newdataset.append(event)
        #print newdataset

        temptarget.append(datetime.fromtimestamp(time.mktime(time.strptime(dataset[0][2], "%Y-%m-%d %H:%M:%S"))))

        for line in dataset[1:,:]:
            #print line
            case=line[0]
            event = line[1]

            if case==caseID:
                #print "case", case
                #continues the current case

                newdataset.append(event)

                temptarget.append(datetime.fromtimestamp(time.mktime(time.strptime(line[2], "%Y-%m-%d %H:%M:%S"))))
                n+=1

                finishtime=datetime.fromtimestamp(time.mktime(time.strptime(line[2], "%Y-%m-%d %H:%M:%S")))
            else:
                caseID=case
                for i in xrange(1,len(newdataset)): # +1 not adding last case. target is 0, not interesting. era 1
                    data.append(newdataset[:i])
                    #print newdataset[:i]
                newdataset=[]
                newdataset.append(event)

                for i in range(n): # era n
                    temptarget[-(i+1)]=(finishtime-temptarget[-(i+1)]).total_seconds()
                temptarget.pop() #remove last element with zero target
                temptarget.append(datetime.fromtimestamp(time.mktime(time.strptime(line[2], "%Y-%m-%d %H:%M:%S"))))
                finishtime=datetime.fromtimestamp(time.mktime(time.strptime(line[2], "%Y-%m-%d %H:%M:%S")))


                n = 1

        #last case
        for i in xrange(1, len(newdataset) ): #+ 1 not adding last event, target is 0 in that case. era 1
            data.append(newdataset[:i])
            #print newdataset[:i]
        for i in range(n): # era n. rimosso esempio con singolo evento
            temptarget[-(i + 1)] = (finishtime - temptarget[-(i + 1)]).total_seconds()
            #print temptarget[-(i + 1)]
        temptarget.pop()  # remove last element with zero target

        #print temptarget
        print "Generated dataset with n_samples:", len(temptarget)
        assert(len(temptarget)== len(data))
        #print temptarget
        return data, temptarget
    return generate_set(datasetTR), generate_set(datasetTS)







(X_train, y_train),(X_test, y_test)= load_dataset_SIAV()

