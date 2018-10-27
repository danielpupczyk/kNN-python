import test_first
import first
import unittest

option = 0
while True:
    print('')
    print('-----------------MENU---------------------')
    print("1.Run method: Predict")
    print("2.Run method: Score")
    print("3.Run Unitest")
    print("Choose q for exit")
    option = input('>>Choose a number: ')
    if option=='1':
        k = input('>>>>Choose number of neighbors: ')
        print('')
        print('--------------------------------------')
        print('|               RESULTS              |')
        print('--------------------------------------')
        print('')
        kNN=first.kNN(int(k),"iris.data.learning")
        kNN.predict("iris.data.test",2)
    elif option=='2':
        k = input('>>>>Choose number of neighbors: ')
        print('')
        print('--------------------------------------')
        print('|               RESULTS              |')
        print('--------------------------------------')
        print('')
        kNN=first.kNN(int(k),"iris.data.learning")
        kNN.score("iris.data.test", kNN.predict("iris.data.test",1),2)
    elif option=='3':
        print('')
        print('--------------------------------------')
        print('|              UNITTEST              |')
        print('--------------------------------------')
        print('')
        suite = unittest.TestLoader().loadTestsFromModule(test_first)
        unittest.TextTestRunner(verbosity=2).run(suite)
    elif option=='q':
        break;
    else:
        print("Wrong values. Pleas try again.")
print("See you later")