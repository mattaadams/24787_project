import numpy as np
import time
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from xgboost import plot_importance
from sklearn import metrics


import torch
import torch.nn as nn
import torch.optim as optim


def load_data():
    nonbinary = np.array(pd.read_csv('botometer-feedback-2019-nonbinary_features.csv').values)[:, 1:]
    binary = np.array(pd.read_csv('botometer-feedback-2019-binary_features.csv').values)[:, 1:]
    labels = np.array(pd.read_csv('botometer-feedback-2019-labels.csv').values)[:, 1:]
    return binary, nonbinary, labels.squeeze()
    # return binary, labels.squeeze()

def logistic_regression(x_train, y_train):
    return LogisticRegression(penalty='l1').fit(x_train, y_train)


def random_forest(x_train, y_train):
    return RandomForestClassifier(n_estimators=200, random_state=0).fit(x_train, y_train)


def kNN(x_train, y_train):
    return KNeighborsClassifier(5).fit(x_train, y_train)


def neural_network(x_train, y_train, hidden_list):
    model = MLP(hidden_list)
    train_network(model, x_train, y_train)
    return model


def SVM(x_train, y_train):
    return LinearSVC().fit(x_train, y_train)


def RBF(x_train, y_train):
    return SVC(C=1.0, kernel='rbf').fit(x_train, y_train)

def XGB(x_train, y_train):
    return XGBClassifier(learning_rate=1e-4,
                         n_estimators=200,
                         scale_pos_weight=1,  # unbalanced dataset
                         random_state=0,
                        ).fit(x_train, y_train)


class MLP(nn.Module):
    def __init__(self, size_list):
        super(MLP, self).__init__()
        layers = []
        self.size_list = size_list
        for i in range(len(size_list) - 2):
            layers.append(nn.Linear(size_list[i], size_list[i+1]))
            if i < 2:
                layers.append(nn.BatchNorm1d(num_features=size_list[i + 1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(size_list[-2], size_list[-1]))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

    def __call__(self, x):
        return self.forward(x)

    def predict(self, x):
        out = self.forward(torch.from_numpy(x).float())
        _, pred = torch.max(out, 1)
        pred = pred.numpy()
        return pred

class KNNbinary:
    def __init__(self):
        self.x_train, self.y_train = None, None

    def fit(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train
        return self

    def predict(self, x_test):
        y_pred = np.zeros_like(y_test)
        for i in range(x_test.shape[0]):
            dist = np.zeros_like(self.y_train)
            for j in range(self.x_train.shape[0]):
                dist[j] = np.sum(self.x_train[j] != x_test[i])
            nearest = np.where(dist == np.min(dist))
            nearest = self.y_train[nearest]
            if np.sum(nearest == 1) > np.sum(nearest == 0):
                y_pred[i] = 1
            else:
                y_pred[i] = 0
        return y_pred


def train_network(model, x_train, y_train, n_epochs=20, batch_size=16):
    model.train()
    data_size, feature_dim = x_train.shape
    Train_loss = []
    Dev_loss = []
    Dev_acc = []
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    xtensor = torch.from_numpy(x_train).float()
    ytensor = torch.from_numpy(y_train).long()

    for i in range(n_epochs):
        start_time = time.time()
        print('epoch: {}'.format(i))
        running_loss = 0.0
        # random shuffle
        order = np.array(range(data_size))
        np.random.shuffle(order)
        xtensor = xtensor[order]
        ytensor = ytensor[order]
        for batch_idx in range(int(data_size / batch_size)):
            data = xtensor[batch_idx * batch_size: (batch_idx + 1) * batch_size]
            target = ytensor[batch_idx * batch_size: (batch_idx + 1) * batch_size]
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, target)
            running_loss += loss.item()
            loss.backward()
            optimizer.step()

        end_time = time.time()
        running_loss /= data_size / batch_size
        print('Training Loss: ', running_loss, 'Time: ', end_time - start_time, 's')


def test(model, x_test, y_test):
    batch_size, feature_dim = x_test.shape
    y_pred = model.predict(x_test)
    acc = np.sum(y_pred == y_test) / batch_size
    conf_mat = confusion_matrix(y_test, y_pred)
    print(conf_mat)
    print('accuracy: {}%'.format(acc * 100))


def test_tree(model, x_test, y_test, threshold=0.5):
    batch_size, feature_dim = x_test.shape
    predicted = model.predict_proba(x_test)
    y_pred = (predicted[:, 1] >= threshold).astype('int')
    # y_pred = model.predict(x_test)
    acc = np.sum(y_pred == y_test) / batch_size
    conf_mat = confusion_matrix(y_test, y_pred)
    print(conf_mat)
    print('accuracy: {}%'.format(acc * 100))


def part1(x_train, y_train, x_test, y_test):
    """
    For this part, we are testing several algorithms on the non-binary features.
    The conclusions are:
    1. Tree classifiers (Random Forest, XGBoost) has descent performance
    2. They still have a hard time detecting bots.
    """
    print('========================part1========================')
    # logistic regression
    logistic_regressor = logistic_regression(x_train, y_train)
    test(logistic_regressor, x_test, y_test)
    # kNN
    kNN_regressor = kNN(x_train, y_train)
    test(kNN_regressor, x_test, y_test)
    # SVM
    SVM_regressor = SVM(x_train, y_train)
    test(SVM_regressor, x_test, y_test)
    # SVM with RBF kernel
    RBF_regressor = RBF(x_train, y_train)
    test(RBF_regressor, x_test, y_test)
    # neural network
    hidden_list = [6, 10, 20, 10, 5, 2]
    mlp = neural_network(x_train, y_train, hidden_list)
    test(mlp, x_test, y_test)
    # random foresttrain
    random_forest_regressor = random_forest(x_train, y_train)
    test(random_forest_regressor, x_test, y_test)
    # XGBoost
    XGB_regressor = XGB(x_train, y_train)
    test(XGB_regressor, x_test, y_test)


def part2(x_train, y_train, x_test, y_test):
    """
    We argue that the imbalanced dataset is the main reason for bot detection failure.
    So we are using several tricks to solve that problem.
    """
    print('========================part2========================')
    # change classification threshold
    random_forest_regressor = random_forest(x_train, y_train)
    test_tree(random_forest_regressor, x_test, y_test, threshold=0.4)
    # dataset processing(drop human)
    human = np.where(y_train == 0)[0]
    bots = np.where(y_train == 1)[0]
    human_subset = np.random.choice(human, len(bots))
    x_train_new = np.row_stack([x_train[human_subset], x_train[bots]])
    y_train_new = np.row_stack([y_train[human_subset, np.newaxis], y_train[bots, np.newaxis]]).squeeze()
    random_forest_regressor = random_forest(x_train_new, y_train_new)
    test(random_forest_regressor, x_test, y_test)
    # dataset processing(add bots)
    human = np.where(y_train == 0)[0]
    bots = np.where(y_train == 1)[0]
    bots_synthesis = np.random.choice(bots, len(human) - len(bots))
    x_train_new = np.row_stack([x_train[human], x_train[bots_synthesis]])
    y_train_new = np.row_stack([y_train[human, np.newaxis], y_train[bots_synthesis, np.newaxis]]).squeeze()
    random_forest_regressor = random_forest(x_train_new, y_train_new)
    test(random_forest_regressor, x_test, y_test)


def part3(x_train, y_train, x_test, y_test):
    """
    Test binary classification algorithms
    The performance is not good
    One reason might be: we have 10 binary features, that only occupy 2^10 = 1024 possible states
    """
    print('========================part1========================')
    # KNN based on hammington distance
    knn = KNNbinary().fit(x_train, y_train)
    test(knn, x_test, y_test)


if __name__ == '__main__':
    x_binary, x, y = load_data()
    x = np.column_stack([x[:, :5], x[:, -1]]) # Because the 6th and 7th column are binary features
    x = (x - x.mean(axis=0)) / x.std(axis=0) # normalize features
    # split dataset
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)
    x_train_binary, x_test_binary, y_train_binary, y_test_binary = train_test_split(x_binary, y, test_size=0.25, random_state=0)
    print('percentage of bots in training data: {}%'.format(np.sum(y_train == 1) / y_train.shape[0]))
    print('percentage of bots in testing data: {}%'.format(np.sum(y_test == 1) / y_test.shape[0]))
    part1(x_train, y_train, x_test, y_test)
    part2(x_train, y_train, x_test, y_test)
    part3(x_train_binary, y_train_binary, x_test_binary, y_test_binary)
