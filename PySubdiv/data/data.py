
class data_dictionary(object):
    """
    class to store data of the mesh as multiple numpy arrays

    """
    def __init__(self):
        self.data = {}

    def __setitem__(self, key, data):
        self.data[key] = data

    def __getitem__(self, key):
        return self.data[key]

    def __iter__(self):
        return iter(self.data)

    def delete(self, key):
        self.data.pop(key, None)

    def clear(self):
        self.data.clear()




