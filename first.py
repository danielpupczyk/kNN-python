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
        self.neighbors=np.array([[10000.1]*self.k]*2)									#empty array of k-neighbors ('*2' becouse value and index)
        testingData=np.array(pd.read_csv(nameOfTestingDataFile))                    #load testing data
        testingData=testingData[:,0:4]                                              #delete labels
        i=0                                                                      	#number of row in testing data
        for xa in testingData:
            j=0                                                                		#number of row in learning data
            for xb in self.learningData:
                #isInside=0                                                       	#is neighbor in array?
                #for tmp in self.neighbors:
                    #d=self.computeDistanse(np.array([xa]),np.array([xb]))
                   # if d<self.neighbors[0][0]:										#znowu dałem [][] bo inaczej nie działa, chuj wie czemu
                        #self.neighbors[0][0]=d                                     	#value of min neighbors  
                        #self.neighbors[0][1]=j                                     	#index of min neighbors
                        
                d=self.computeDistanse(np.array([xa]),np.array([xb]))
                if d<self.neighbors[0][self.k-1]:                                         #jeli d jest mniejsze od ostatniego elementu w tablicy (najwiekszego)
                    self.neighbors[0][self.k-1]=d                                     	#to wkładamy tam dystans  
                    self.neighbors[1][self.k-1]=j                                         #oraz index pod jakim siedzi etykieta   
                    for tmp1 in range((self.k)-1,0,-1):                                         #jesli dodalismy to srotujemy bąblekowo
                       for tmp2 in range(tmp1):
                           if self.neighbors[0][tmp2]>self.neighbors[0][tmp2+1]:                                  #powronujemy dystanse
                               temp = np.array(self.neighbors[:,tmp2])                           #ale zamieniamy razem z etykietą
                               self.neighbors[:,tmp2] = self.neighbors[:,tmp2+1]
                               self.neighbors[:,tmp2+1] = temp 
                j+=1
            print(self.neighbors)
            i+=1
            #print(self.neighbors)				#print index and predic label
            self.neighbors=np.array([[10000.1]*self.k]*2)													#reset val
                        
                    
                

                

kNN = kNN(3,"iris.data.learning")
kNN.predict("iris.data.test")
a=np.array([[32, 10, 14, 10,20,30,1],[10,1,2,3,3,2,0]])
w=7
for tmp1 in range(w-1,0,-1): 
    for tmp2 in range(tmp1):
        if a[0][tmp2]>a[0][tmp2+1]:
            temp = np.array(a[:,tmp2])
            a[:,tmp2] = a[:,tmp2+1]
            a[:,tmp2+1] = temp
            
