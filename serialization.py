import pickle

def save(obj, path):
    with open(path, 'wb') as file:
        pickle.dump(obj, file)

def load(path):
    with open(path, 'rb') as file:  # 'rb' means reading in binary mode
        obj = pickle.load(file)  # Deserialize the object from the file
    return obj
