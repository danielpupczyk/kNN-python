import unittest
import first

class TestFirst(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        print('setupClass')
        
    @classmethod
    def tearDownClass(cls):
        print('teardownClass')
    
    def setUp(self):
        print('setUp')
        self.obj1 = first.kNN(3,"iris.data.learning")
        self.obj2 = first.kNN(5,"iris.data.learning")
        self.obj3 = first.kNN(7,"iris.data.learning")
    
    def tearDown(self):
        print('tearDown\n')
    
    def test_predict(self):
        print('test_predict')
        result1 = self.obj1.predict("iris.data.test")
        test_result1 = ['Iris-setosa', 'Iris-setosa', 
                        'Iris-setosa', 'Iris-versicolor', 
                        'Iris-versicolor', 'Iris-versicolor', 
                        'Iris-versicolor', 'Iris-versicolor', 
                        'Iris-versicolor', 'Iris-virginica', 
                        'Iris-virginica', 'Iris-virginica', 
                        'Iris-virginica', 'Iris-virginica', 
                        'Iris-virginica']
        
        result2 = self.obj2.predict("iris.data.test")
        test_result2 = ['Iris-setosa', 'Iris-setosa', 
                        'Iris-setosa', 'Iris-versicolor', 
                        'Iris-versicolor', 'Iris-versicolor', 
                        'Iris-versicolor', 'Iris-versicolor', 
                        'Iris-versicolor', 'Iris-virginica', 
                        'Iris-virginica', 'Iris-virginica', 
                        'Iris-virginica', 'Iris-virginica', 
                        'Iris-virginica']
       
        result3 = self.obj3.predict("iris.data.test")
        test_result3 = ['Iris-setosa', 'Iris-setosa', 
                        'Iris-setosa', 'Iris-versicolor', 
                        'Iris-versicolor', 'Iris-versicolor', 
                        'Iris-versicolor', 'Iris-versicolor', 
                        'Iris-versicolor', 'Iris-virginica', 
                        'Iris-virginica', 'Iris-virginica', 
                        'Iris-virginica', 'Iris-virginica', 
                        'Iris-virginica']
        self.assertSequenceEqual(result1, test_result1)
        self.assertSequenceEqual(result2, test_result2)
        self.assertSequenceEqual(result3, test_result3)
        
    def test_score(self):
        print('test_score')
        pred1 = self.obj1.predict("iris.data.test")
        pred2 = self.obj2.predict("iris.data.test")
        pred3 = self.obj3.predict("iris.data.test")
        result1 = self.obj1.score("iris.data.test", pred1)
        result2 = self.obj2.score("iris.data.test", pred2)
        result3 = self.obj3.score("iris.data.test", pred3)
        self.assertEquals(0.9333333333333333, result1)
        self.assertEquals(0.9333333333333333, result2)
        self.assertEquals(0.9333333333333333, result3)
