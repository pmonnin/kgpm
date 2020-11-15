import tqdm

__author__ = "Pierre Monnin"


def add_thing_class_type_subclassof(canonical_graph, cache_managers):
    """
    Ensure owl:Thing, owl:Class, rdf:type and rdfs:subClassOf are in canonical graph
    :param canonical_graph: canonical graph
    :param cache_managers: canonical graph cache managers
    :return: the canonical graph
    """

    top_rdf_index = cache_managers["rdf_nodes"].get_element_index("http://www.w3.org/2002/07/owl#Thing")
    if top_rdf_index not in canonical_graph["rdf_to_canonical_index"]:
        canonical_graph["rdf_to_canonical_index"][top_rdf_index] = len(canonical_graph["canonical_to_rdf_index"])
        canonical_graph["canonical_to_rdf_index"].append({top_rdf_index})

    type_p_index = cache_managers["predicates"].get_element_index("http://www.w3.org/1999/02/22-rdf-syntax-ns#type")
    if type_p_index not in canonical_graph["adjacency"]:
        canonical_graph["adjacency"][type_p_index] = dict()
        canonical_graph["inv_adjacency"][type_p_index] = dict()

    subclass_p_index = cache_managers["predicates"].get_element_index("http://www.w3.org/2000/01/rdf-schema#subClassOf")
    if subclass_p_index not in canonical_graph["adjacency"]:
        canonical_graph["adjacency"][subclass_p_index] = dict()
        canonical_graph["inv_adjacency"][subclass_p_index] = dict()

    return canonical_graph


def get_nodes_degree(canonical_graph, predicates_cache_manager, predicates_blacklist, undirected=False):
    """
    Compute degree for each node in canonical graph (only considering predicates that are not blacklisted)
    :param canonical_graph: canonical graph
    :param predicates_cache_manager: cache manager for predicates
    :param predicates_blacklist: blacklist for predicates
    :param undirected: if the graph should be considered directed or undirected
    :return: dict associating each canonical node index with its degree
    """

    nodes_degree = dict()

    # We ensure every canonical node has a degree
    for n_canonical_index in range(0, len(canonical_graph["canonical_to_rdf_index"])):
        nodes_degree[n_canonical_index] = 0

    for p_index in tqdm.tqdm(canonical_graph["adjacency"], desc="degrees"):
        if not is_blacklisted(predicates_cache_manager.get_element_from_index(p_index), predicates_blacklist):
            for n1 in canonical_graph["adjacency"][p_index]:
                nodes_degree[n1] += len(canonical_graph["adjacency"][p_index][n1])

    if undirected:
        for p_index in tqdm.tqdm(canonical_graph["inv_adjacency"], desc="inv adj. degrees"):
            if not is_blacklisted(predicates_cache_manager.get_element_from_index(p_index), predicates_blacklist):
                for n1 in canonical_graph["inv_adjacency"][p_index]:
                    nodes_degree[n1] += len(canonical_graph["inv_adjacency"][p_index][n1])

    return nodes_degree


def is_blacklisted(uri, blacklist):
    """
    Check if a URI is blacklisted.

    A URI is considered blacklisted if it fully matches or starts with a URI inside blacklist
    :param uri: the URI to check
    :param blacklist: blacklist of URIs
    :return: True if the URI is blacklisted w.r.t. the given blacklist, False otherwise
    """

    for b_uri in blacklist:
        if uri.startswith(b_uri):
            return True

    return False


def get_nodes_ancestors(canonical_graph, cache_managers):
    """
    Compute ancestors for all classes in canonical graph. All classes (except owl:Thing) have at least owl:Thing as
    their ancestor. Nodes that are not classes are not in the returned dict
    :param canonical_graph: canonical graph
    :param cache_managers: cache managers
    :return: dict associating each canonical node index representing a class with the set of indices of its ancestors
    """

    # Get rdf:type and rdfs:subClassOf predicate index
    type_p_index = cache_managers["predicates"].get_element_index("http://www.w3.org/1999/02/22-rdf-syntax-ns#type")
    subclass_p_index = cache_managers["predicates"].get_element_index("http://www.w3.org/2000/01/rdf-schema#subClassOf")

    # Get owl:Thing canonical index
    top_rdf_index = cache_managers["rdf_nodes"].get_element_index("http://www.w3.org/2002/07/owl#Thing")
    top_canonical_index = canonical_graph["rdf_to_canonical_index"][top_rdf_index]

    ancestors = dict()
    for current_n in tqdm.tqdm(range(0, len(canonical_graph["canonical_to_rdf_index"])), desc="ancestors"):

        # If n is subclass of another node, or superclass of another node or type of another node: it can have ancestors
        if current_n in canonical_graph["adjacency"][subclass_p_index] or \
                current_n in canonical_graph["inv_adjacency"][subclass_p_index] or \
                current_n in canonical_graph["inv_adjacency"][type_p_index]:

            # Add the transitive closure of rdfs:subClassOf links
            ancestors[current_n] = set()
            if current_n in canonical_graph["adjacency"][subclass_p_index]:
                ancestors[current_n] |= set(canonical_graph["adjacency"][subclass_p_index][current_n])

            diff = set(ancestors[current_n])
            while len(diff) != 0:
                temp = set(ancestors[current_n])

                for n in diff:
                    if n in canonical_graph["adjacency"][subclass_p_index]:
                        temp |= set(canonical_graph["adjacency"][subclass_p_index][n])

                diff = temp - ancestors[current_n]
                ancestors[current_n] = temp

            # Always add T as ancestor except for T itself
            if current_n != top_canonical_index:
                ancestors[current_n] |= {top_canonical_index}

    return ancestors


def get_nodes_full_type(canonical_graph, cache_managers):
    """
    Compute full type (rdf:type + transitive closure of rdfs:subClassOf) for nodes that are not classes, i.e.,
    that do not appear in the ancestors field of canonical graph. Therefore, nodes that are classes do not appear
    in the returned dict.
    :param canonical_graph: canonical graph
    :param cache_managers: cache managers
    :return: dict associating each canonical node index representing an individual with the set of indices of
    its types (rdf:type and transitive closure of rdfs:subClassOf)
    """

    # Get rdf:type predicate index
    type_p_index = cache_managers["predicates"].get_element_index("http://www.w3.org/1999/02/22-rdf-syntax-ns#type")

    # Get owl:Thing canonical index
    top_rdf_index = cache_managers["rdf_nodes"].get_element_index("http://www.w3.org/2002/07/owl#Thing")
    top_canonical_index = canonical_graph["rdf_to_canonical_index"][top_rdf_index]

    # Compute full type for all nodes in canonical graph
    nodes_full_type = dict()

    for current_n in tqdm.tqdm(range(0, len(canonical_graph["canonical_to_rdf_index"])), desc="full types"):
        if current_n not in canonical_graph["nodes_ancestors"]:
            # At least owl:Thing is in full type
            nodes_full_type[current_n] = {top_canonical_index}

            if current_n in canonical_graph["adjacency"][type_p_index]:
                nodes_full_type[current_n] |= set(canonical_graph["adjacency"][type_p_index][current_n])

                for type_index in canonical_graph["adjacency"][type_p_index][current_n]:
                    nodes_full_type[current_n] |= canonical_graph["nodes_ancestors"][type_index]

    return nodes_full_type


def is_node_blacklisted(n_index, canonical_graph, cache_managers, blacklists):
    """
    Test if node is blacklisted w.r.t. its full type and the types expansion blacklist
    :param n_index: canonical index of the node to test
    :param canonical_graph: canonical graph
    :param cache_managers: cache managers
    :param blacklists: blacklists
    :return: True if n_index instantiate a blacklisted type (in types expansion blacklist), False otherwise
    """

    if n_index in canonical_graph["nodes_full_type"]:
        for t_index in canonical_graph["nodes_full_type"][n_index]:
            t_uris = canonical_index_to_uris(t_index, canonical_graph["canonical_to_rdf_index"],
                                             cache_managers["rdf_nodes"])

            if any(is_blacklisted(uri, blacklists["types_expansion"]) for uri in t_uris):
                return True

    return False


def canonical_index_to_uris(n_canonical_index, canonical_to_rdf_index, rdf_nodes_cache_manager):
    """
    Return the list of URIs associated with a canonical node index
    :param n_canonical_index: canonical node index to transform into a list of URIs
    :param canonical_to_rdf_index: mapping from canonical indices to RDF indices
    :param rdf_nodes_cache_manager: cache manager for RDF nodes
    :return: list of URIs associated with a canonical node index
    """

    return [rdf_nodes_cache_manager.get_element_from_index(rdf_index)
            for rdf_index in canonical_to_rdf_index[n_canonical_index]]
