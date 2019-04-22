from sklearn.model_selection import train_test_split

def classifyDataset(dataset, classifier):
    clf = classifier
    train_x, test_x, train_y, test_y = train_test_split(dataset[0], dataset[1:])
    clf.fit(train_x, train_y)

    return clf.score(test_x, test_y)