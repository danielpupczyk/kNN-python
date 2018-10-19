import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform, cdist

class kNN:
    def __init__(self,k,nameOfLearningDataFile):									#constructor with number of neighbor and name of file with learningdata		
        self.k=k
        self.fullLearningData=np.array(pd.read_csv(nameOfLearningDataFile))			#load learning data
        self.learningData=self.fullLearningData[:,0:4]								#devide to data and labels
        self.labels=self.fullLearningData[:,4]
        
    def getData(self):
        print("Data:")
        print(self.learningData)
        
    def getLabels(self):
        print("Labels:")
        print(self.labels)
        
    def computeDistanse(self,a,b):													#computing euclidean distanse, parameters are two vectores
        return cdist(a,b, 'euclidean')[0][0]										#return distans, [][]- beocuse its dimensional array i tylko tak mi działało
        
    def predict(self,nameOfTestingDataFile):      
        self.neighbors=np.array([[10000]*self.k*2])									#empty array of k-neighbors ('*2' becouse value and index)
        #print(self.neighbors[0][0])
        testingData=np.array(pd.read_csv(nameOfTestingDataFile))    
        testingData=testingData[:,0:4]
        i=0                                                                      	#number of row in testing data
        for xa in testingData:
            j=0                                                                		#number of row in learning data
            for xb in self.learningData:
                #isInside=0                                                       	#is neighbor in array?
                for tmp in self.neighbors:
                    d=self.computeDistanse(np.array([xa]),np.array([xb]))
                    if d<self.neighbors[0][0]:										#znowu dałem [][] bo inaczej nie działa, chuj wie czemu
                        self.neighbors[0][0]=d                                     	#value of min neighbors  
                        self.neighbors[0][1]=j                                     	#index of min neighbors
                j+=1
            i+=1
            print(str(i)+".Przewidziana etykieta: "+str(self.labels[self.neighbors[0][1]]))				#print index and predic label
            self.neighbors[0]=10000													#reset val
                        
                    
                

                

kNN = kNN(1,"iris.data.learning")
kNN.predict("iris.data.test")
