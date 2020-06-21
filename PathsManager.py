class PathsManager:
    """
    Class handling a cache from a path (tuple of integers) to integer (in order to save memory)
    """

    class CacheNode:
        """
        Represent a node in the Cache graph representation
        """

        def __init__(self, p_index, p_element):
            """
            Constructor of a cache node
            :param p_index: the cache index of the path represented by the current cache node and its ancestors.
            If -1, it means that no path index is associated with path represented.
            :param p_element: the element (integer) in the path represented by the current cache node
            """

            self.p_index = p_index
            self.p_element = p_element
            self.previous = None
            self.next = dict()

    def __init__(self):
        self._cache = PathsManager.CacheNode(-1, -1)
        self._inverse_cache = dict()  # index (int) -> CacheNode
        self._max_ind = -1

    def get_element_index(self, element):
        """
        Get the path index (an integer). If the path is not in cache, it is added and the corresponding
        index is returned
        :param element: the paths to look up in cache (or to add to cache)
        :return: the element index
        """

        current_n_cache = self._cache
        for i, p_element in enumerate(element):
            if p_element not in current_n_cache.next:
                new_n_cache = PathsManager.CacheNode(-1, p_element)
                current_n_cache.next[p_element] = new_n_cache
                new_n_cache.previous = current_n_cache

            current_n_cache = current_n_cache.next[p_element]

        if current_n_cache.p_index == -1:
            self._max_ind += 1
            current_n_cache.p_index = self._max_ind
            self._inverse_cache[self._max_ind] = current_n_cache

        return current_n_cache.p_index

    def get_element_from_index(self, index):
        """
        Return the path associated with the given index. If index is not in cache, raise KeyError
        :param index: the index to look up in cache
        :return: the path associated with the index
        """

        if index not in self._inverse_cache:
            raise KeyError("Given index {} not in PathsManager".format(index))

        current_n_cache = self._inverse_cache[index]
        path = []

        while current_n_cache.previous is not None:
            path.insert(0, current_n_cache.p_element)
            current_n_cache = current_n_cache.previous

        return tuple(path)

    def is_element_in_cache(self, element):
        """
        Check if element (path) is in cache
        :param element: path to look up in cache
        :return: True if the element is in cache, False otherwise
        """

        current_n_cache = self._cache
        for i, p_element in enumerate(element):
            if p_element in current_n_cache.next:
                current_n_cache = current_n_cache.next[p_element]

                if i == len(element) - 1 and current_n_cache.p_index != -1:
                    return True

            else:
                return False

        return False

    def delete_from_index(self, index):
        """
        Remove an element from cache by given its associated index
        :param index: cache index of the element to remove
        :return: None
        """

        if index in self._inverse_cache:
            current_n_cache = self._inverse_cache[index]
            current_n_cache.p_index = -1

            while current_n_cache.previous is not None and len(current_n_cache.next) == 0 and \
                    current_n_cache.p_index == -1:
                del current_n_cache.previous.next[current_n_cache.p_element]
                current_n_cache = current_n_cache.previous

            del self._inverse_cache[index]
