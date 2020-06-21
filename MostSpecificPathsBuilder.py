import queue

from CacheManager import CacheManager


class MostSpecificPathsBuilder:
    """
    Class keeping most specific paths after each insertion
    Paths MUST have same length and their elements (nodes) should be in the partial ordering given at building
    """

    def __init__(self, paths_manager, n_partial_ordering):
        """
        Constructor
        :param paths_manager: manager for paths
        :param n_partial_ordering: partial ordering between nodes involved in paths
        """

        self.paths_manager = paths_manager
        self.n_partial_ordering = n_partial_ordering

        self.predicates_to_most_specific_index = CacheManager()
        self.most_specific_indices = dict()

    def insert(self, generalized_p_index):
        """
        Insert new generalized path into the structure and keep most specific paths
        :param generalized_p_index: the index of the generalized path to insert
        :return: None
        """

        p = self.paths_manager.get_element_from_index(generalized_p_index)

        # Work with the most specific index corresponding to predicates used in path
        predicates = [p[i] for i in range(0, len(p), 2)]
        predicates = tuple(predicates)
        predicates_index = self.predicates_to_most_specific_index.get_element_index(predicates)

        if predicates_index not in self.most_specific_indices:
            if len(p) == 2:
                self.most_specific_indices[predicates_index] = set()
            else:
                self.most_specific_indices[predicates_index] = dict()

        # Check for more specific paths inside index
        q = queue.LifoQueue()  # Lifo to ensure early exit once a specific path is found
        q.put((self.most_specific_indices[predicates_index], 1))

        found_more_specific = False
        while not q.empty() and not found_more_specific:
            most_specific_index, index_in_p = q.get()

            for el in most_specific_index:
                if el == p[index_in_p] or el in self.n_partial_ordering[p[index_in_p]]["descendants"]:
                    if index_in_p == len(p) - 1:
                        found_more_specific = True

                    else:
                        q.put((most_specific_index[el], index_in_p + 2))

        # If no path is more specific, find/delete less specific paths and insert new path
        if not found_more_specific:
            # Find/delete less specific
            q = queue.Queue()
            q.put((self.most_specific_indices[predicates_index], 1, []))

            while not q.empty():
                most_specific_index, index_in_p, to_remove = q.get()

                if index_in_p < len(p):
                    for el in most_specific_index:
                        if el == p[index_in_p] or el in self.n_partial_ordering[p[index_in_p]]["ancestors"]:
                            expanded_to_remove = list(to_remove)
                            expanded_to_remove.append((most_specific_index, el))

                            if index_in_p != len(p) - 1:
                                q.put((
                                    most_specific_index[el],
                                    index_in_p + 2,
                                    expanded_to_remove
                                ))

                            else:
                                q.put((
                                    None,
                                    index_in_p + 2,
                                    expanded_to_remove
                                ))

                else:
                    for i in range(len(to_remove) - 1, -1, -1):
                        index, el_to_remove = to_remove[i]

                        if i == len(to_remove) - 1:
                            index -= {el_to_remove}

                        elif len(index[el_to_remove]) == 0:
                            del index[el_to_remove]

            # Insert
            most_specific_index = self.most_specific_indices[predicates_index]

            for index_in_p in range(1, len(p), 2):

                if p[index_in_p] not in most_specific_index:
                    if index_in_p < len(p) - 3:
                        most_specific_index[p[index_in_p]] = dict()
                        most_specific_index = most_specific_index[p[index_in_p]]

                    elif index_in_p == len(p) - 3:
                        most_specific_index[p[index_in_p]] = set()
                        most_specific_index = most_specific_index[p[index_in_p]]

                    else:
                        most_specific_index.add(p[index_in_p])

                else:
                    if index_in_p != len(p) - 1:
                        most_specific_index = most_specific_index[p[index_in_p]]

    def is_most_specific(self, generalized_p_index):
        """
        Check if generalized path is one of the most specific previously inserted in the structure
        :param generalized_p_index: the index of the generalized path to test
        :return: True if it is one of the most specific paths previously inserted in the structure, False otherwise
        """

        p = self.paths_manager.get_element_from_index(generalized_p_index)

        # Work with the most specific index corresponding to predicates used in path
        predicates = [p[i] for i in range(0, len(p), 2)]
        predicates = tuple(predicates)
        predicates_index = self.predicates_to_most_specific_index.get_element_index(predicates)

        if predicates_index not in self.most_specific_indices:
            return False

        most_specific_index = self.most_specific_indices[predicates_index]

        for index_in_p in range(1, len(p), 2):
            if p[index_in_p] in most_specific_index:
                if index_in_p == len(p) - 1:
                    return True

                else:
                    most_specific_index = most_specific_index[p[index_in_p]]

            else:
                return False
