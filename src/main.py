from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier


def main():
    dataset = load_dataset("dataset.csv")
    show_some_statistics(dataset)
    X, Y = preprocess(dataset)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=1)

    print("~~~~~~~~~~~~~~~~KNN~~~~~~~~~~~~~~~~")
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, Y_train)
    result = knn.predict(X_test)
    print_results(Y_test, result)

    print("~~~~~~~~~~~~~~~~Decision Tree~~~~~~~~~~~~~~~~")
    dt = DecisionTreeClassifier(max_depth=5)
    dt = dt.fit(X_train, Y_train)
    result = dt.predict(X_test)
    print_results(Y_test, result)

    print("~~~~~~~~~~~~~~~~Random Forest~~~~~~~~~~~~~~~~")
    weight = {0: 0.5, 1: 1}
    rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, oob_score=True, random_state=42, class_weight=weight)
    rf = rf.fit(X_train, Y_train)
    result = rf.predict(X_test)
    print_results(Y_test, result)

    print("~~~~~~~~~~~~~~~~Naive Bayes~~~~~~~~~~~~~~~~")
    nb = GaussianNB()
    nb.fit(X_train, Y_train)
    result = nb.predict(X_test)
    print_results(Y_test, result)


def load_dataset(file_name):
    dataset = pd.read_csv(f"../raw/{file_name}")
    dataset.drop('Unnamed: 0', inplace=True, axis=1)
    return dataset


def show_some_statistics(dataset):
    products = {}
    for i in range(len(dataset)):
        product = dataset.loc[i]['Product']
        products[product] = products.get(product, 0) + 1

    fig = plt.figure(1, [7, 5])
    ax = fig.add_axes([0, 0, 1, 1])
    x_products = []
    y_freqs = []
    for key, value in products.items():
        x_products.append(key)
        y_freqs.append(value)
    ax.bar(x_products, y_freqs)
    plt.setp(ax.get_xticklabels(), rotation='vertical')
    plt.show()


def preprocess(dataset):
    # dropping unuseful columns
    unused_column = ['SalesAgentEmailID', 'ContactEmailID', 'Customer', 'Agent']
    for column_name in unused_column:
        dataset.drop(column_name, inplace=True, axis=1)

    # filter records whose state is not specified
    dataset = dataset[dataset.Stage != "In Progress"]

    dataset['Stage'] = dataset['Stage'].replace(['Won'], int(1))
    dataset['Stage'] = dataset['Stage'].replace(['Lost'], int(0))

    # compute interval
    intervals = []
    for i in range(len(dataset)):
        created_date = dataset.iloc[i]['Created Date']
        close_date = dataset.iloc[i]['Close Date']
        interval = get_interval(created_date, close_date)
        intervals.append(interval)
    dataset['Interval'] = intervals
    dataset.drop(['Created Date', 'Close Date'], inplace=True, axis=1)

    X = dataset.drop(['Stage'], axis=1)
    Y = dataset['Stage']
    ct = ColumnTransformer(transformers=[
        ("encoder", OneHotEncoder(sparse=False), [X.columns.get_loc('Product')]),
        ("imputer", SimpleImputer(missing_values=np.nan, strategy="mean"), [X.columns.get_loc('Close_Value')])
    ], remainder="passthrough")
    X = np.array(ct.fit_transform(X))

    sc = StandardScaler()
    X[:, 7:9] = sc.fit_transform(X[:, 7:9])

    return X, Y


def get_interval(first_date, second_date):
    first_date_time = datetime.strptime(first_date, '%Y-%m-%d %H:%M:%S')
    second_date_time = datetime.strptime(second_date, '%Y-%m-%d %H:%M:%S')
    return (second_date_time - first_date_time).days


def print_results(Y_test, result):
    print(f"accuracy= {accuracy_score(Y_test, result)}")
    print(f"F1= {f1_score(Y_test, result)}")
    print(f"confusion matrix:\n{confusion_matrix(Y_test, result)}")


if __name__ == '__main__':
    main()
