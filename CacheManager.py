import csv


class CacheManager:
    """
    Class handling a cache from an element (tuple, string) to integer (in order to save memory)
    """

    def __init__(self):
        self._cache = dict()  # element -> index (int)
        self._inverse_cache = dict()  # index (int) -> element
        self._max_ind = -1

    def get_element_index(self, element):
        """
        Get the element (string) index (an integer). If the element is not in cache, it is added and the corresponding
        index is returned
        :param element: the element to look up in cache (or to add to cache)
        :return: the element index
        """

        if element not in self._cache:
            self._max_ind += 1
            self._cache[element] = self._max_ind
            self._inverse_cache[self._max_ind] = element

        return self._cache[element]

    def get_element_from_index(self, index):
        """
        Return the element associated with the given index. If index is not in cache, raise KeyError
        :param index: the index to look up in cache
        :return: the element associated with the index
        """

        if index not in self._inverse_cache:
            raise KeyError("Given index {} not in CacheManager".format(index))

        return self._inverse_cache[index]

    def is_element_in_cache(self, element):
        """
        Check if element is in cache
        :param element: element to look up in cache
        :return: True if the element is in cache, False otherwise
        """

        return element in self._cache

    def get_size(self):
        """
        Return the number of elements in cache
        :return: the number of elements in cache
        """

        return len(self._cache)

    def get_max_ind(self):
        """
        Return current max index for cache indices
        :return: current max index for cache indices
        """
        return self._max_ind

    def delete_from_index(self, index):
        """
        Remove an element from cache by given its associated index
        :param index: cache index of the element to remove
        :return: None
        """

        if index in self._inverse_cache:
            element = self._inverse_cache[index]
            del self._inverse_cache[index]
            del self._cache[element]

    def delete_from_element(self, element):
        """
        Remove an element from cache by giving the element
        :param element: the element to remove
        :return: None
        """

        if element in self._cache:
            index = self._cache[element]
            del self._cache[element]
            del self._inverse_cache[index]

    def save_to_csv(self, file):
        """
        Save CacheManager to the given CSV file
        :param file: path of the file used to save CacheManager
        :return: None
        """

        with open(file, 'w') as csvfile:
            writer = csv.writer(csvfile)

            for k, i in self._cache.items():
                writer.writerow([k, i])

    def load_from_csv(self, file):
        """
        Load CacheManager from the given CSV file
        :param file: path to the file used to load CacheManager
        :return: None
        """

        with open(file, 'r') as csvfile:
            reader = csv.reader(csvfile)

            for row in reader:
                self._cache[row[0]] = int(row[1])
                self._inverse_cache[int(row[1])] = row[0]

        if len(self._inverse_cache) > 0:
            self._max_ind = max(self._inverse_cache.keys())
        else:
            self._max_ind = -1

    def __str__(self):
        cache_str = "-- CacheManager --\n"
        for uri, i in self._cache.items():
            cache_str += uri + " <=> " + str(i) + "\n"
        cache_str += "--------------------"
        return cache_str
