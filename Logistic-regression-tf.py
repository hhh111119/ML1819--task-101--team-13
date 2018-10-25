from util.data import load_data

def main():
    X_train, X_test, y_train, y_test = load_data('iris.data')
    print(X_train)


main()