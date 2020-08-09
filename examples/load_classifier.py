from nymph.module import Classifier

if __name__ == '__main__':
    raw_data = [['a', 1.0], ['b', 1.0]]
    classifier = Classifier()
    classifier.load('./saves')
    res = classifier.predict(raw_data)
    print(res)
