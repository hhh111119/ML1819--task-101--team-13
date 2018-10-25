def load_data(filename):
    with open(filename) as i:
        list = i.readlines()
    for l in list:
        print(l)

def main():
    load_data('iris.data')