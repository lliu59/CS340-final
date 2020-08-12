def load_dataset(filename):
    with open(os.path.join('../..', 'data', filename), 'rb') as f:
        return pickle.load(f)