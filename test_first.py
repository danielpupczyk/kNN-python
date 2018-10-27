import unittest
import first

class TestFirst(unittest.TestCase):
    
    def test_predict(self):
        obj = first.kNN(2,"iris.data.learning")
        result = obj.predict("iris.data.test")
        test_result = ['Iris-setosa', 'Iris-setosa', 
                       'Iris-setosa', 'Iris-versicolor', 
                       'Iris-versicolor', 'Iris-versicolor', 
                       'Iris-versicolor', 'Iris-versicolor', 
                       'Iris-versicolor', 'Iris-virginica', 
                       'Iris-virginica', 'Iris-virginica', 
                       'Iris-versicolor', 'Iris-virginica', 
                       'Iris-virginica']
        self.assertSequenceEqual(result, test_result)
        