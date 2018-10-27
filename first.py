import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform, cdist

class kNN(object):
    def __init__(self,k,nameOfLearningDataFile):									#constructor with number of neighbor and name of file with learningdata		
        self.k=k
        self.fullLearningData=np.array(pd.read_csv(nameOfLearningDataFile,header=None))			#load learning data
        self.learningData=self.fullLearningData[:,0:4]								#devide to data and labels
        self.labels=self.fullLearningData[:,4]
        self.uniqueLabels = set(self.labels)
        self.uniqueLabels = dict.fromkeys(self.uniqueLabels, 0)                      #to store unique labels values (without duplication)
        
    def getData(self):
        print("Data:")
        print(self.learningData)
        
    def getLabels(self):
        print("Labels:")
        print(self.labels)
        
    def computeDistanse(self,a,b):													#computing euclidean distanse, parameters are two vectores
        return cdist(a,b, 'euclidean')[0][0]										#return distans, [][]- beocuse its dimensional array i tylko tak mi działało
      
    def predict(self,nameOfTestingDataFile, resultType=1):
        predictedValues = []
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
                    self.neighbors[1][self.k-1]=j                                #oraz index pod jakim siedzi etykieta   
                    for tmp1 in range((self.k)-1,0,-1):                             #jesli dodalismy to srotujemy bąblekowo
                       for tmp2 in range(tmp1):
                           if self.neighbors[0][tmp2]>self.neighbors[0][tmp2+1]:    #powronujemy tylko dystanse
                               temp = np.array(self.neighbors[:,tmp2])              #ale zamieniamy razem z indeksem przypisanej etykiety
                               self.neighbors[:,tmp2] = self.neighbors[:,tmp2+1]
                               self.neighbors[:,tmp2+1] = temp 
                j+=1
            #print("Lp."+str(i+1))                                                   #numer danej testujacej
            for neighbor in self.neighbors[1]:                                      #and his neighbors
                self.uniqueLabels[self.labels[int(neighbor)]] += 1                  #increment number of occurs the same nearest neighbor
            maxOccurs = 0                                                           #maximum number of occurs
            maxOccursName = 0                                                       #label name for maximum occurs
            for key in self.uniqueLabels:                                           #search for label that occurs the most
                if self.uniqueLabels[key] > maxOccurs:
                    maxOccurs = self.uniqueLabels[key]
                    maxOccursLabel = key
            #print('Predicted value: ' + str(maxOccursLabel))                        #our predicted value
            predictedValues.append(maxOccursLabel)
            i+=1
            maxOccurs = 0                                                           #reset all values before next iteration
            maxOccursLabel = 0
            for key in self.uniqueLabels:
                self.uniqueLabels[key] = 0
            self.neighbors=np.array([[10000.1]*self.k]*2)							#reset val
        if resultType==2:
            print(predictedValues)
        elif resultType==1:
            return predictedValues

    def score(self, nameOfTestingDataFile, predictedLabels,resultType=1):
        testingData=np.array(pd.read_csv(nameOfTestingDataFile,header=None))        #load testing data, header, zeby wczytywało też pierwszy wiersz
        testingLabels = testingData[:,4]                                            #labels for testing data
        testingData=testingData[:,0:4]                                              #delete labels
        i = 0                                                                       #index dla przewidzianych labelek
        result = 0                                                                  #to będzie nasz współczynnik
        for label in testingLabels:
            if str(label) == str(predictedLabels[i]):                               #jesli poprawnie przewidziano, to dodajemy 1
                result += 1
            i+=1
        result = result/i                                                           #na koniec sumę poprawnie przewidzianych, dzielimy na ilosc wszystkich
        if resultType==2:
            print('')
            print('The ratio of correctly recognized labels: ' + str(result))
        elif resultType==1:
            return result
        

