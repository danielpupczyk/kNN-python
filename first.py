import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform, cdist

class kNN:
    def __init__(self,k,nameOfLearningDataFile):									#constructor with number of neighbor and name of file with learningdata		
        self.k=k
        self.fullLearningData=np.array(pd.read_csv(nameOfLearningDataFile,header=None))			#load learning data
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
        self.neighbors=np.array([[10000.1]*self.k]*2)								#empty array of k-neighbors ('*2' becouse value and index)
        testingData=np.array(pd.read_csv(nameOfTestingDataFile,header=None))        #load testing data, header, zeby wczytywało też pierwszy wiersz
        testingData=testingData[:,0:4]                                              #delete labels
        i=0                                                                      	#number of row in testing data
        for xa in testingData:                                                      #przechodzimy przez dane do testownia        
            j=0                                                                		#number of row in learning data
            for xb in self.learningData:                                            #przechodzimy przed dane do uczenia                        
                d=self.computeDistanse(np.array([xa]),np.array([xb]))               #obliczamy odległosc pomiędzy wektorem uczącym i testującym
                if d<self.neighbors[0][self.k-1]:                                   #jeli d jest mniejsze od ostatniego elementu w tablicy (najwiekszego)
                    self.neighbors[0][self.k-1]=d                                   #to wkładamy tam dystans  
                    self.neighbors[1][self.k-1]=j                                   #oraz index pod jakim siedzi etykieta   
                    for tmp1 in range((self.k)-1,0,-1):                             #jesli dodalismy to srotujemy bąblekowo
                       for tmp2 in range(tmp1):
                           if self.neighbors[0][tmp2]>self.neighbors[0][tmp2+1]:    #powronujemy tylko dystanse
                               temp = np.array(self.neighbors[:,tmp2])              #ale zamieniamy razem z indeksem przypisanej etykiety
                               self.neighbors[:,tmp2] = self.neighbors[:,tmp2+1]
                               self.neighbors[:,tmp2+1] = temp 
                j+=1
            print("Lp."+str(i+1))                                                   #numer danej testujacej
            no=1;
            for neighbor in self.neighbors[1]:                                      #and his neighbors
                print("Neighbor no. "+str(no)+": "+self.labels[int(neighbor)])
                no+=1
            i+=1
            self.neighbors=np.array([[10000.1]*self.k]*2)							#reset val
                        

kNN = kNN(5,"iris.data.learning")
kNN.predict("iris.data.test")
            
