import argparse
import datetime
import enum

import bitarray
import numpy
import scipy.sparse
import tqdm

from CacheManager import CacheManager
from MostSpecificPathsBuilder import MostSpecificPathsBuilder
from PathsManager import PathsManager
from graph_utils import add_thing_class_type_subclassof, get_nodes_degree, is_blacklisted, get_nodes_ancestors, \
    get_nodes_full_type, is_node_blacklisted, canonical_index_to_uris
from load_utils import load_configuration_parameters, get_logger, get_blacklists, load_graph, load_cache_managers, \
    load_dataset


class MeaningfulStrategies(enum.Flag):
    """
    Binary flags to define the meaningful strategies to use
    """
    NO_CHECK = enum.auto()  # Keep features w/o checking for meaningfulness
    P = enum.auto()  # Keep features having at least 1 pathway involved
    G = enum.auto()  # Keep features having at least 1 gene or 1 GO class involved
    M = enum.auto()  # Keep features having at least 1 MeSH class involved
    PG = enum.auto()  # Keep features having at least 1 pathway, or 1 gene, or 1 GO class involved
    PGM = enum.auto()  # Keep features having at least 1 pathway, or 1 gene, or 1 GO class or 1 MeSH class involved


def print_debug_degrees(nodes_degree, canonical_to_rdf_index, rdf_nodes_cache_manager, logger):
    logger.debug("---- CANONICAL NODES DEGREE BEGIN -----")

    for n_canonical_index in nodes_degree:
        logger.debug("({}): {}".format(
            canonical_index_to_uris(n_canonical_index, canonical_to_rdf_index, rdf_nodes_cache_manager),
            nodes_degree[n_canonical_index]
        ))

    logger.debug("----- CANONICAL NODES DEGREE END ------")


def print_debug_ancestors(nodes_ancestors, canonical_to_rdf_index, rdf_nodes_cache_manager, logger):
    logger.debug("---- CANONICAL NODES ANCESTORS BEGIN -----")

    for n_canonical_index in nodes_ancestors:
        ancestors_uris = [canonical_index_to_uris(ancestor_index, canonical_to_rdf_index, rdf_nodes_cache_manager)
                          for ancestor_index in nodes_ancestors[n_canonical_index]]

        logger.debug("({}): {}".format(
            canonical_index_to_uris(n_canonical_index, canonical_to_rdf_index, rdf_nodes_cache_manager),
            ancestors_uris
        ))

    logger.debug("----- CANONICAL NODES ANCESTORS END ------")


def print_debug_full_types(nodes_full_type, canonical_to_rdf_index, rdf_nodes_cache_manager, logger):
    logger.debug("---- CANONICAL NODES FULL TYPE BEGIN -----")

    for n_canonical_index in nodes_full_type:
        types_uris = [canonical_index_to_uris(type_index, canonical_to_rdf_index, rdf_nodes_cache_manager)
                      for type_index in nodes_full_type[n_canonical_index]]

        logger.debug("({}): {}".format(
            canonical_index_to_uris(n_canonical_index, canonical_to_rdf_index, rdf_nodes_cache_manager),
            types_uris
        ))

    logger.debug("----- CANONICAL NODES FULL TYPE END ------")


def get_node_neighborhood(n_index, k, d, undirected, canonical_graph, cache_managers, blacklists):
    """
    Compute the k-hops neighborhood of the given canonical node
    :param n_index: canonical index of seed node whose neighborhood is returned (except itself)
    :param k: number of hops of neighborhood to consider
    :param d: max degree. Nodes having a greater degree are not expanded
    :param canonical_graph: canonical graph
    :param cache_managers: cache managers
    :param undirected: consider the graph directed or undirected
    :param blacklists: blacklist for predicates and nodes types (not considered when expanding neighborhood)
    :return: set of k-neighboring canonical node indices of the given n_index
    """

    neighborhood = {n_index}
    diff = set()
    if canonical_graph["nodes_degree"][n_index] <= d:
        diff = {n_index}

    i = 1
    while i <= k and len(diff) != 0:
        temp = set(neighborhood)

        for p_index in canonical_graph["adjacency"]:
            if not is_blacklisted(cache_managers["predicates"].get_element_from_index(p_index),
                                  blacklists["path_predicates"]):
                for n in diff:
                    if n in canonical_graph["adjacency"][p_index]:
                        for n2 in canonical_graph["adjacency"][p_index][n]:
                            if not is_node_blacklisted(n2, canonical_graph, cache_managers, blacklists):
                                temp.add(n2)

        if undirected:
            for p_index in canonical_graph["inv_adjacency"]:
                if not is_blacklisted(cache_managers["predicates"].get_element_from_index(p_index),
                                      blacklists["path_predicates"]):
                    for n in diff:
                        if n in canonical_graph["inv_adjacency"][p_index]:
                            for n2 in canonical_graph["inv_adjacency"][p_index][n]:
                                if not is_node_blacklisted(n2, canonical_graph, cache_managers, blacklists):
                                    temp.add(n2)

        diff = {n for n in (temp - neighborhood) if canonical_graph["nodes_degree"][n] <= d}
        neighborhood = set(temp)
        i += 1

    return neighborhood - {n_index}


def print_debug_features(features, canonical_graph, cache_managers, sorted_seed_uris, logger):
    logger.debug("---- FEATURES BEGIN -----")

    for ft, support_vector in features.items():
        if isinstance(ft, int):  # Feature is a neighbor
            n_uris = canonical_index_to_uris(ft, canonical_graph["canonical_to_rdf_index"], cache_managers["rdf_nodes"])

            logger.debug("({}): {}".format(
                n_uris,
                [seed_uri for seed_index, seed_uri in enumerate(sorted_seed_uris) if support_vector[seed_index]]
            ))

        else:  # Feature is a path
            path = []
            for i, index in enumerate(ft):
                if i % 2 == 0:
                    path.append(cache_managers["predicates"].get_element_from_index(index))

                else:
                    path.append(canonical_index_to_uris(index, canonical_graph["canonical_to_rdf_index"],
                                                        cache_managers["rdf_nodes"]))

            logger.debug("({}): {}".format(
                path,
                [seed_uri for seed_index, seed_uri in enumerate(sorted_seed_uris) if support_vector[seed_index]]
            ))

    logger.debug("----- FEATURES END ------")


def get_node_type(n_canonical_index, t, canonical_graph, cache_managers, types_blacklist):
    """
    Return the type (along t hierarchical hops) of n_canonical_index considering rdf:type and rdfs:subClassOf links.
    owl:Thing is added in the returned type if t >= 0.
    :param n_canonical_index: node whose type is wanted
    :param t: number of hops in the type hierarchy to consider
    :param canonical_graph: canonical graph
    :param cache_managers: cache managers
    :param types_blacklist: blacklist of types
    :return: types of the given node w.r.t. parameter t
    """

    node_types = set()

    # Get rdf:type and rdfs:subClassOf predicate index
    type_p_index = cache_managers["predicates"].get_element_index("http://www.w3.org/1999/02/22-rdf-syntax-ns#type")
    subclass_p_index = cache_managers["predicates"].get_element_index("http://www.w3.org/2000/01/rdf-schema#subClassOf")

    if t >= 1 and n_canonical_index in canonical_graph["adjacency"][type_p_index]:
        node_types |= set(canonical_graph["adjacency"][type_p_index][n_canonical_index])
        diff = set(node_types)

        i = 2
        while i <= t and len(diff) != 0:
            temp = set(node_types)

            for t_index in diff:
                if t_index in canonical_graph["adjacency"][subclass_p_index]:
                    temp |= set(canonical_graph["adjacency"][subclass_p_index][t_index])

            diff = temp - node_types
            node_types = set(temp)

            i += 1

    # Remove blacklisted types
    node_types = {
        t_index for t_index in node_types
        if all(
            not is_blacklisted(uri, types_blacklist)
            for uri in canonical_index_to_uris(t_index, canonical_graph["canonical_to_rdf_index"],
                                               cache_managers["rdf_nodes"])
        )
    }

    if t >= 0:
        # Add owl:Thing canonical index
        top_rdf_index = cache_managers["rdf_nodes"].get_element_index("http://www.w3.org/2002/07/owl#Thing")
        top_canonical_index = canonical_graph["rdf_to_canonical_index"][top_rdf_index]
        node_types.add(top_canonical_index)

    return node_types


def print_debug_frequent_types(frequent_types, canonical_to_rdf_index, rdf_nodes_cache_manager, logger):
    logger.debug("---- FREQUENT TYPES BEGIN -----")

    for t_index in frequent_types:
        logger.debug("({}): {}".format(
            canonical_index_to_uris(t_index, canonical_to_rdf_index, rdf_nodes_cache_manager),
            frequent_types[t_index].count(1)
        ))

    logger.debug("----- FREQUENT TYPES END ------")


def get_authorized_neighborhoods(sorted_seed_uris, seed_uri_to_canonical_index, k, d, undirected, canonical_graph,
                                 cache_managers, blacklists):
    """
    Return the authorized neighborhood for each seed node at level k
    Authorized neighborhood = neighborhood up to k - neighborhood up to k-1
    :param sorted_seed_uris: seed nodes URIs
    :param seed_uri_to_canonical_index: mapping from seed nodes URIs to canonical indices
    :param k: current max neighborhing hops
    :param d: max degree of nodes to be considered for expansion
    :param undirected: consider graph as undirected
    :param canonical_graph: canonical graph
    :param cache_managers: cache managers
    :param blacklists: blacklists of predicates and nodes types not to consider during expansion
    :return: dict: seed index in support vector -> its authorized neighborhood (set of canonical nodes indices)
    """

    seed_neighborhoods = dict()

    for seed_index, seed_uri in enumerate(tqdm.tqdm(sorted_seed_uris, desc="authorized neighborhood")):
        if seed_uri in seed_uri_to_canonical_index:
            seed_canonical_index = seed_uri_to_canonical_index[seed_uri]

            seed_neighborhoods[seed_index] = get_node_neighborhood(seed_canonical_index, k, d, undirected,
                                                                   canonical_graph, cache_managers,
                                                                   blacklists) - \
                                             get_node_neighborhood(seed_canonical_index, k - 1, d, undirected,
                                                                   canonical_graph, cache_managers,
                                                                   blacklists)

    return seed_neighborhoods


def print_debug_authorized_neighborhoods(authorized_neighborhoods, canonical_graph, cache_managers, sorted_seed_uris,
                                         logger):

    logger.debug("---- AUTHORIZED NEIGHBORHOODS BEGIN -----")

    for seed_index, neighborhood in authorized_neighborhoods.items():
        neighborhood_uris = [
            canonical_index_to_uris(n_index, canonical_graph["canonical_to_rdf_index"], cache_managers["rdf_nodes"])
            for n_index in authorized_neighborhoods[seed_index]
        ]

        logger.debug("{}: {}".format(sorted_seed_uris[seed_index], neighborhood_uris))

    logger.debug("----- AUTHORIZED NEIGHBORHOODS END ------")


def get_node_neighborhood_path_expansion(n_index, d, undirected, authorized_neighborhood, canonical_graph,
                                         cache_managers, blacklists):
    """
    Return a set of (pred_index, n_index) representing the neighborhood path expansion of the given node, w.r.t.
    the authorized neighborhood, the blacklist of predicates and max degree d
    :param n_index: canonical index of node to expand
    :param d: max degree of a node to be expanded
    :param undirected: if the canonical graph is considered undirected
    :param authorized_neighborhood: authorized node neighborhood
    :param canonical_graph: canonical graph
    :param cache_managers: cache managers
    :param blacklists: blacklist of predicates and nodes types not to consider during expansion
    :return: a set of (pred_index, n_index) representing the neighborhood path expansion of the given node
    """

    retval = set()

    if canonical_graph["nodes_degree"][n_index] <= d:
        for p_index in canonical_graph["adjacency"]:
            if not is_blacklisted(cache_managers["predicates"].get_element_from_index(p_index),
                                  blacklists["path_predicates"]):
                if n_index in canonical_graph["adjacency"][p_index]:
                    for n2_index in canonical_graph["adjacency"][p_index][n_index]:
                        if n2_index in authorized_neighborhood and not is_node_blacklisted(n2_index, canonical_graph,
                                                                                           cache_managers, blacklists):
                            retval.add((p_index, n2_index))

        if undirected:
            for p_index in canonical_graph["inv_adjacency"]:
                if not is_blacklisted(cache_managers["predicates"].get_element_from_index(p_index),
                                      blacklists["path_predicates"]):
                    if n_index in canonical_graph["inv_adjacency"][p_index]:
                        for n2_index in canonical_graph["inv_adjacency"][p_index][n_index]:
                            if n2_index in authorized_neighborhood and not is_node_blacklisted(n2_index,
                                                                                               canonical_graph,
                                                                                               cache_managers,
                                                                                               blacklists):
                                retval.add((p_index, n2_index))

    return retval


def print_debug_paths(paths, canonical_graph, cache_managers, sorted_seed_uris, paths_manager, logger):
    logger.debug("---- PATHS BEGIN -----")

    for p_index, support_vector in paths.items():
        p = paths_manager.get_element_from_index(p_index)

        path = []
        for i, index in enumerate(p):
            if i % 2 == 0:
                path.append(cache_managers["predicates"].get_element_from_index(index))
            else:
                path.append(canonical_index_to_uris(index, canonical_graph["canonical_to_rdf_index"],
                                                    cache_managers["rdf_nodes"]))

        logger.debug("({}): {}".format(
            path,
            [seed_uri for seed_index, seed_uri in enumerate(sorted_seed_uris) if support_vector[seed_index]]
        ))

    logger.debug("----- PATHS END ------")


def exists_shorter_path(p, support_vector, path_features):
    """
    Check if a shorter path from p exists in path_features with the same support vector
    :param p: path to check
    :param support_vector: the support vector associated with p
    :param path_features: the existing paths features
    :return: the shorter path found or None if none was found
    """

    short_p = p[:-2]

    while len(short_p) > 0:
        if short_p in path_features and path_features[short_p] == support_vector:
            return short_p

        short_p = short_p[:-2]

    return None


def get_paths(sorted_seed_uris, seed_uri_to_canonical_index, canonical_graph, cache_managers, blacklists,
              frequent_types, n_partial_ordering, k, t, d, limit_min, limit_max, undirected, statistics, logger,
              debug=False):

    # Paths to return
    path_features = dict()  # ABox/ATBox paths (tuples canonical nodes/predicates indices) -> bitarray (support vector)
    paths_manager = CacheManager()

    if k < 1:
        return path_features

    # k >= 1, explore neighborhood
    pbar = tqdm.tqdm(total=k, desc="paths k")

    # Expansion list
    abox_paths_to_expand = dict()  # ABox path index -> bitarray (support vector)

    # Path features dependency tree
    generalized_paths_origins = dict()  # ATBox path index -> set(ABox paths indices)
    generalized_paths_generated = dict()  # ABox path index -> set(ATBox paths indices)

    # Variables used during expansion
    new_abox_paths = dict()
    new_generalized_paths = dict()

    # Expansion initialization
    # A path (seed_canonical_index, ) is set to True for each seed node
    # The first element will then be removed in the expansion code to allow generalization
    for seed_index, seed_uri in enumerate(sorted_seed_uris):
        if seed_uri in seed_uri_to_canonical_index:
            seed_canonical_index = seed_uri_to_canonical_index[seed_uri]
            seed_path_index = paths_manager.get_element_index((seed_canonical_index,))

            if seed_path_index not in abox_paths_to_expand:
                abox_paths_to_expand[seed_path_index] = bitarray.bitarray([False] * len(sorted_seed_uris))
            abox_paths_to_expand[seed_path_index][seed_index] = True

    # Expand while paths to expand exist and farthest neighborhood is not reached
    i = 1
    while i <= k and len(abox_paths_to_expand) != 0:
        logger.info("k = {}; #ABox paths to expand = {}".format(i, len(abox_paths_to_expand)))

        # Compute seed nodes authorized neighborhood
        logger.info("Compute authorized neighborhood")
        seed_neighborhoods = get_authorized_neighborhoods(sorted_seed_uris, seed_uri_to_canonical_index, i, d,
                                                          undirected, canonical_graph, cache_managers,
                                                          blacklists)

        if debug:
            print_debug_authorized_neighborhoods(seed_neighborhoods, canonical_graph, cache_managers,
                                                 sorted_seed_uris, logger)
            logger.debug("PAUSED -- PRESS ENTER TO CONTINUE...")
            input()

        # Expand ABox paths
        logger.info("Expand ABox paths")
        new_abox_paths = dict()  # ABox path index -> bitarray (support vector)

        for p_index in tqdm.tqdm(abox_paths_to_expand, desc="ABox path expansion"):
            p = paths_manager.get_element_from_index(p_index)

            # Authorized neighborhood to expand p at level i is is union of neighborhoods at level i of ..
            authorized_neighborhood = set()
            for seed_index in range(0, len(sorted_seed_uris)):
                # ... seed nodes associated with p
                if abox_paths_to_expand[p_index][seed_index]:
                    authorized_neighborhood |= seed_neighborhoods[seed_index]

                # ... seed nodes associated with a generalized path generated from p
                elif p_index in generalized_paths_generated:
                    if any(new_generalized_paths[generalized_p_index][seed_index]
                           for generalized_p_index in generalized_paths_generated[p_index]):
                        authorized_neighborhood |= seed_neighborhoods[seed_index]

            for pred_index, n_index in get_node_neighborhood_path_expansion(p[-1], d, undirected,
                                                                            authorized_neighborhood, canonical_graph,
                                                                            cache_managers, blacklists):
                if i == 1:
                    expanded_p = (pred_index, n_index)
                else:
                    expanded_p = p + (pred_index, n_index)

                # Add expanded ABox path
                expanded_p_index = paths_manager.get_element_index(expanded_p)
                if expanded_p_index not in new_abox_paths:
                    new_abox_paths[expanded_p_index] = bitarray.bitarray([False] * len(sorted_seed_uris))
                new_abox_paths[expanded_p_index] |= abox_paths_to_expand[p_index]

        if debug:
            logger.debug("NEW ABOX PATHS")
            print_debug_paths(new_abox_paths, canonical_graph, cache_managers, sorted_seed_uris, paths_manager, logger)
            logger.debug("PAUSED -- PRESS ENTER TO CONTINUE...")
            input()

        # Generate mixed ABox/TBox paths
        logger.info("Generate generalized mixed ABox/TBox paths")
        new_generalized_paths = dict()

        for p_index in tqdm.tqdm(new_abox_paths, desc="paths generalization"):
            p = paths_manager.get_element_from_index(p_index)
            last_n_types = {
                t_index for
                t_index in get_node_type(p[-1], t, canonical_graph, cache_managers, blacklists["types"])
                if t_index in frequent_types
            }

            # Generalize ABox path with types of last node
            for t_index in last_n_types:
                generalized_p = p[:-1] + (t_index,)
                generalized_p_index = paths_manager.get_element_index(generalized_p)

                # Add new generalized path
                if generalized_p_index not in new_generalized_paths:
                    new_generalized_paths[generalized_p_index] = bitarray.bitarray([False] * len(sorted_seed_uris))
                new_generalized_paths[generalized_p_index] |= new_abox_paths[p_index]

                # Add generalized path in path features dependency tree
                if p_index not in generalized_paths_generated:
                    generalized_paths_generated[p_index] = set()
                generalized_paths_generated[p_index].add(generalized_p_index)

                if generalized_p_index not in generalized_paths_origins:
                    generalized_paths_origins[generalized_p_index] = set()
                generalized_paths_origins[generalized_p_index].add(p_index)

            # Expand shorter ABox path generalizations with last node and types of last node
            if len(p) >= 4 and paths_manager.get_element_index(p[:-2]) in generalized_paths_generated:
                last_elements = last_n_types | {p[-1]}

                for generalized_p_index in generalized_paths_generated[paths_manager.get_element_index(p[:-2])]:
                    generalized_p = paths_manager.get_element_from_index(generalized_p_index)

                    for el in last_elements:
                        expanded_generalized_p = generalized_p + (p[-2], el)
                        expanded_generalized_p_index = paths_manager.get_element_index(expanded_generalized_p)

                        # Add expanded generalized path
                        if expanded_generalized_p_index not in new_generalized_paths:
                            new_generalized_paths[expanded_generalized_p_index] = bitarray.bitarray(
                                [False] * len(sorted_seed_uris)
                            )
                        new_generalized_paths[expanded_generalized_p_index] |= new_abox_paths[p_index]

                        # Add expanded generalized path in path features dependency tree
                        if p_index not in generalized_paths_generated:
                            generalized_paths_generated[p_index] = set()
                        generalized_paths_generated[p_index].add(expanded_generalized_p_index)

                        if expanded_generalized_p_index not in generalized_paths_origins:
                            generalized_paths_origins[expanded_generalized_p_index] = set()
                        generalized_paths_origins[expanded_generalized_p_index].add(p_index)

        if debug:
            logger.debug("NEW MIXED ABOX/TBOX PATHS")
            print_debug_paths(new_generalized_paths, canonical_graph, cache_managers, sorted_seed_uris, paths_manager,
                              logger)
            logger.debug("PAUSED -- PRESS ENTER TO CONTINUE...")
            input()

        # Remove generalized paths with 1 origin (the origin ABox path will be considered as it is more specific)
        logger.info("Remove generalized mixed ABox/TBox paths with one origin")

        new_generalized_paths_keys = set(new_generalized_paths.keys())
        for generalized_p_index in tqdm.tqdm(new_generalized_paths_keys, desc="Mixed paths - 1 origin"):
            if len(generalized_paths_origins[generalized_p_index]) == 1:
                origin_p_index = next(iter(generalized_paths_origins[generalized_p_index]))

                del new_generalized_paths[generalized_p_index]
                del generalized_paths_origins[generalized_p_index]
                generalized_paths_generated[origin_p_index] -= {generalized_p_index}
                paths_manager.delete_from_index(generalized_p_index)

        if debug:
            logger.debug("MIXED ABOX/TBOX PATHS - 1 ORIGIN")
            print_debug_paths(new_generalized_paths, canonical_graph, cache_managers, sorted_seed_uris, paths_manager,
                              logger)
            logger.debug("PAUSED -- PRESS ENTER TO CONTINUE...")
            input()

        # When paths have identical support, keep most specific
        logger.info("Keep most specific paths when having identical support vectors")

        support_to_generalized_paths = dict()
        for generalized_p_index in tqdm.tqdm(new_generalized_paths, desc="inv(gen. paths / support dict)"):
            if new_generalized_paths[generalized_p_index].tobytes() not in support_to_generalized_paths:
                support_to_generalized_paths[new_generalized_paths[generalized_p_index].tobytes()] = set()

            support_to_generalized_paths[new_generalized_paths[generalized_p_index].tobytes()].add(generalized_p_index)

        for support_vector in tqdm.tqdm(support_to_generalized_paths, desc="support vector"):
            # If several generalized paths have the same support vector
            if len(support_to_generalized_paths[support_vector]) != 1:
                # => Keep most specific paths
                most_specific_paths = MostSpecificPathsBuilder(paths_manager, n_partial_ordering)

                for generalized_p_index in tqdm.tqdm(support_to_generalized_paths[support_vector], desc="p. insert"):
                    most_specific_paths.insert(generalized_p_index)

                for generalized_p_index in tqdm.tqdm(support_to_generalized_paths[support_vector], desc="p. selection"):
                    if not most_specific_paths.is_most_specific(generalized_p_index):
                        del new_generalized_paths[generalized_p_index]

                        for origin_p_index in generalized_paths_origins[generalized_p_index]:
                            generalized_paths_generated[origin_p_index] -= {generalized_p_index}

                        del generalized_paths_origins[generalized_p_index]
                        paths_manager.delete_from_index(generalized_p_index)

        support_to_generalized_paths = dict()

        if debug:
            logger.debug("MOST SPECIFIC MIXED ABOX/TBOX PATHS")
            print_debug_paths(new_generalized_paths, canonical_graph, cache_managers, sorted_seed_uris, paths_manager,
                              logger)
            logger.debug("PAUSED -- PRESS ENTER TO CONTINUE...")
            input()

        # Select paths as features and for expansion
        logger.info("Select paths as features and for expansion")
        abox_paths_to_expand = dict()

        for generalized_p_index in tqdm.tqdm(new_generalized_paths, desc="generalized paths"):
            generalized_p = paths_manager.get_element_from_index(generalized_p_index)

            if new_generalized_paths[generalized_p_index].count(1) >= limit_min:
                # Add its origin ABox paths for expansion
                for origin_p_index in generalized_paths_origins[generalized_p_index]:
                    abox_paths_to_expand[origin_p_index] = new_abox_paths[origin_p_index]

                # If support is <= limit_max
                if new_generalized_paths[generalized_p_index].count(1) <= limit_max:
                    # Try to find a shorter path with same support
                    short_p = exists_shorter_path(generalized_p, new_generalized_paths[generalized_p_index],
                                                  path_features)

                    # If no shorter path exists or shorter path ends with a class and current path ends with an entity
                    if short_p is None or (short_p[-1] in canonical_graph["nodes_ancestors"]
                                           and generalized_p[-1] not in canonical_graph["nodes_ancestors"]):
                        # If no origin ABox path with same support vector
                        if not any(new_generalized_paths[generalized_p_index] == new_abox_paths[origin_p_index]
                                   for origin_p_index in generalized_paths_origins[generalized_p_index]):
                            # We can add generalized path to returned features
                            path_features[generalized_p] = new_generalized_paths[generalized_p_index]

                            # If shorter path exists and ends with a class, remove it from features
                            if short_p is not None and short_p[-1] in canonical_graph["nodes_ancestors"]:
                                del path_features[short_p]

            else:
                for origin_p_index in generalized_paths_origins[generalized_p_index]:
                    generalized_paths_generated[origin_p_index] -= {generalized_p_index}

                del generalized_paths_origins[generalized_p_index]
                paths_manager.delete_from_index(generalized_p_index)

        for p_index in tqdm.tqdm(new_abox_paths, desc="abox paths"):
            p = paths_manager.get_element_from_index(p_index)

            if new_abox_paths[p_index].count(1) >= limit_min:
                # Add for expansion
                abox_paths_to_expand[p_index] = new_abox_paths[p_index]

                # If support is <= limit_max
                if new_abox_paths[p_index].count(1) <= limit_max:
                    # If no shorter paths with same support vector in features
                    if exists_shorter_path(p, new_abox_paths[p_index], path_features) is None:
                        path_features[p] = new_abox_paths[p_index]

            # If ABox path was not chosen for expansion from generalized or ABox path, it can be deleted
            if p_index not in abox_paths_to_expand:
                for generalized_p_index in generalized_paths_generated[p_index]:
                    generalized_paths_origins[generalized_p_index] -= {p_index}

                del generalized_paths_generated[p_index]
                paths_manager.delete_from_index(p_index)

        i += 1
        pbar.update(1)

    # In case of early-stopping, progress bar is set to 100%
    if i <= k:
        pbar.update(k - i + 1)

    logger.info("Memory cleaning")
    del abox_paths_to_expand
    del generalized_paths_origins
    del generalized_paths_generated

    statistics["paths_generated"] = paths_manager.get_max_ind()
    del paths_manager
    pbar.close()

    return path_features


def get_nodes_partial_ordering(neighborhood, frequent_types, canonical_graph):
    """
    Return the partial ordering between nodes in neighborhood and frequent types. Types ancestors are restricted to
    their frequent ancestors. Types descendants are restricted to their frequent descendants and instances in the
    neighborhood. Ancestors of nodes in neighborhood are their frequent types. Nodes in neighborhood have no descendants
    :param neighborhood: nodes in the considered neighborhood
    :param frequent_types: frequent types computed over the neighborhood
    :param canonical_graph: canonical graph
    :return: dict associating a node index or a type index to a dict with two keys (``descendants'' and ``ancestors'')
    """

    n_partial_ordering = dict()

    # For nodes: init = no descendants and ancestors = types in their full types and frequent
    for n_index in tqdm.tqdm(neighborhood, desc="partial ordering / init nodes"):
        n_partial_ordering[n_index] = dict()

        n_partial_ordering[n_index]["ancestors"] = set()
        n_partial_ordering[n_index]["descendants"] = set()

        if n_index in canonical_graph["nodes_full_type"]:
            n_partial_ordering[n_index]["ancestors"] = frequent_types & canonical_graph["nodes_full_type"][n_index]

    # For types: init = no descendants and ancestors = frequent ancestors
    for t_index in tqdm.tqdm(frequent_types, desc="partial ordering / init frequent types"):
        n_partial_ordering[t_index] = dict()

        n_partial_ordering[t_index]["ancestors"] = set()
        n_partial_ordering[t_index]["descendants"] = set()

        if t_index in canonical_graph["nodes_ancestors"]:
            n_partial_ordering[t_index]["ancestors"] = frequent_types & canonical_graph["nodes_ancestors"][t_index]

    # Complete descendants based on the ancestors relation
    for n_index in tqdm.tqdm(n_partial_ordering, desc="partial ordering / completion"):
        for ancestor in n_partial_ordering[n_index]["ancestors"]:
            n_partial_ordering[ancestor]["descendants"].add(n_index)

    return n_partial_ordering


def print_debug_partial_ordering(partial_ordering, canonical_to_rdf_index, rdf_nodes_cache_manager, logger):
    logger.debug("---- PARTIAL ORDERING BEGIN -----")

    for n_index in partial_ordering:
        logger.debug("{}: ancestors({}); descendants({})".format(
            canonical_index_to_uris(n_index, canonical_to_rdf_index, rdf_nodes_cache_manager),
            [
                canonical_index_to_uris(a, canonical_to_rdf_index, rdf_nodes_cache_manager)
                for a in partial_ordering[n_index]["ancestors"]
            ],
            [
                canonical_index_to_uris(d, canonical_to_rdf_index, rdf_nodes_cache_manager)
                for d in partial_ordering[n_index]["descendants"]
            ]
        ))

    logger.debug("----- PARTIAL ORDERING END ------")


def save_features(features, sorted_seed_uris, seed_classes, canonical_graph, cache_managers, dataset_name, k, t, d,
                  undirected, limit_min, limit_max, meaningful_features, logger, output_dir):

    # Save features labels
    logger.info("Save features labels")
    with open(output_dir + "{}_colname_k{}_t{}_d{}_undirected{}_min{}_max{}_meaningful{}.txt".format(
            dataset_name, k, t, d, undirected, limit_min, limit_max, meaningful_features), "w") as col:

        for i, ft in enumerate(tqdm.tqdm(features)):
            if isinstance(ft, int):  # ft is a node
                ft_rdf_index = next(iter(canonical_graph["canonical_to_rdf_index"][ft]))
                col.write(cache_managers["rdf_nodes"].get_element_from_index(ft_rdf_index)
                          + ",{}\n".format(features[ft].count(1)))

            else:  # ft is a path
                p = ""
                for j, n in enumerate(ft):
                    if j % 2 == 0:
                        p += cache_managers["predicates"].get_element_from_index(n) + "|"
                    else:
                        n = next(iter(canonical_graph["canonical_to_rdf_index"][n]))
                        p += cache_managers["rdf_nodes"].get_element_from_index(n) + "|"

                col.write(p[:-1] + ",{}\n".format(features[ft].count(1)))

    # Generate matrices
    logger.info("Generate matrices")
    lil_matrix = scipy.sparse.lil_matrix((len(sorted_seed_uris), len(features)), dtype=numpy.bool)
    np_classes = numpy.zeros(len(sorted_seed_uris), dtype=numpy.bool)

    for seed_index, seed_uri in enumerate(tqdm.tqdm(sorted_seed_uris)):
        if seed_classes[seed_uri] == "1":
            np_classes[seed_index] = True

        for ft_index, ft in enumerate(features):
            if features[ft][seed_index]:
                lil_matrix[seed_index, ft_index] = True

    # Saving matrices
    logger.info("Save matrices")
    numpy.save(output_dir + "{}_class_k{}_t{}_d{}_undirected{}_min{}_max{}_meaningful{}".format(
        dataset_name, k, t, d, undirected, limit_min, limit_max, meaningful_features), np_classes)
    scipy.sparse.save_npz(output_dir + "{}_sparse_k{}_t{}_d{}_undirected{}_min{}_max{}_meaningful{}".format(
        dataset_name, k, t, d, undirected, limit_min, limit_max, meaningful_features), lil_matrix.tocoo())


def is_meaningful(feature, check_pathways, check_genetic_factors, check_mesh_classes, canonical_graph, cache_managers):
    # feature is a node
    if isinstance(feature, int):
        uris = canonical_index_to_uris(feature, canonical_graph["canonical_to_rdf_index"], cache_managers["rdf_nodes"])

        # Is this a kegg:Pathway?
        if check_pathways:
            pathway_class_uri = "http://bio2rdf.org/kegg_vocabulary:Pathway"
            pathway_rdf_index = cache_managers["rdf_nodes"].get_element_index(pathway_class_uri)

            if pathway_rdf_index in canonical_graph["rdf_to_canonical_index"]:
                pathway_canonical_index = canonical_graph["rdf_to_canonical_index"][pathway_rdf_index]

                if feature in canonical_graph["nodes_full_type"] and \
                        pathway_canonical_index in canonical_graph["nodes_full_type"][feature]:
                    return True

        # Is this a pgxo:GeneticFactor?
        if check_genetic_factors:
            genetic_factor_class_uri = "http://pgxo.loria.fr/GeneticFactor"
            genetic_factor_rdf_index = cache_managers["rdf_nodes"].get_element_index(genetic_factor_class_uri)

            if genetic_factor_rdf_index in canonical_graph["rdf_to_canonical_index"]:
                genetic_factor_canonical_index = canonical_graph["rdf_to_canonical_index"][genetic_factor_rdf_index]

                if feature in canonical_graph["nodes_full_type"] and \
                        genetic_factor_canonical_index in canonical_graph["nodes_full_type"][feature]:
                    return True

            # Is this a GO class?
            if any(uri.lower().startswith("http://bio2rdf.org/go:") or
                   uri.lower().startswith("http://identifiers.org/go") for uri in uris):
                return True

        # Is this a MeSH class?
        if check_mesh_classes:
            if any("mesh" in uri.lower() for uri in uris):
                return True

    # feature is a path, check for at least one node in it to be meaningful
    else:
        for index, el in enumerate(feature):
            if index % 2 == 1:
                if is_meaningful(el, check_pathways, check_genetic_factors, check_mesh_classes, canonical_graph,
                                 cache_managers):
                    return True

    return False


def get_features_repartition(features):
    """
    Count number of neighbors and number of paths in the given features
    :param features: dict associating features (int or tuple of ints) to their vector support
    :return: the number of neighbors and the number of paths in the given features
    """

    neighbors_number = 0
    paths_number = 0

    for f in features:
        if isinstance(f, int):
            neighbors_number += 1
        else:
            paths_number += 1

    return neighbors_number, paths_number


def select_meaningful_features(features, check_pathways, check_genetic_factors, check_mesh_classes,  canonical_graph,
                               cache_managers):
    features_keys = set(features.keys())

    for f in tqdm.tqdm(features_keys, desc="meaningful features"):
        if not is_meaningful(f, check_pathways, check_genetic_factors, check_mesh_classes, canonical_graph,
                             cache_managers):
            del features[f]

    return features


def compute_features(canonical_graph, cache_managers, blacklists, dataset_name, classes, k, t, d, undirected,
                     limit_min, limit_max, meaningful_strategies, output_dir, logger, debug=False):

    features = dict()
    statistics = dict()

    # From sorted seed URIs to canonical indices
    sorted_seed_uris = sorted(classes)

    logger.info("Compute canonical indices associated with seed URIs")
    seed_uri_to_canonical_index = dict()
    for seed_uri in tqdm.tqdm(sorted_seed_uris):
        rdf_index = cache_managers["rdf_nodes"].get_element_index(seed_uri)

        if rdf_index in canonical_graph["rdf_to_canonical_index"]:
            seed_uri_to_canonical_index[seed_uri] = canonical_graph["rdf_to_canonical_index"][rdf_index]

    # Compute neighborhood features
    logger.info("Compute neighborhood features")
    for seed_index, seed_uri in enumerate(tqdm.tqdm(sorted_seed_uris, desc="neighbors")):
        if seed_uri in seed_uri_to_canonical_index:
            seed_neighborhood = get_node_neighborhood(
                seed_uri_to_canonical_index[seed_uri],
                k,
                d,
                undirected,
                canonical_graph,
                cache_managers,
                blacklists
            )

            for n_canonical_index in seed_neighborhood:
                if n_canonical_index not in features:
                    features[n_canonical_index] = bitarray.bitarray([False] * len(classes))

                features[n_canonical_index][seed_index] = True

    # Compute frequent types on whole neighborhood (used to avoid generating too many generalized paths)
    logger.info("Compute frequent types")
    frequent_types = dict()

    # We only have node indices in features for now so we can use them to have the global neighborhood
    for n_index in tqdm.tqdm(features, desc="frequent types"):
        for t_index in get_node_type(n_index, t, canonical_graph, cache_managers, blacklists["types"]):
            if t_index not in frequent_types:
                frequent_types[t_index] = bitarray.bitarray(features[n_index])
            else:
                frequent_types[t_index] |= features[n_index]

    if debug:
        print_debug_frequent_types(frequent_types, canonical_graph["canonical_to_rdf_index"],
                                   cache_managers["rdf_nodes"], logger)
        logger.debug("PAUSED -- PRESS ENTER TO CONTINUE...")
        input()

    statistics["neighbors_wo_limits"] = len(features)
    statistics["types_wo_limits"] = len(frequent_types)

    frequent_types = {t_index for t_index in frequent_types if frequent_types[t_index].count(1) >= limit_min}

    # Compute partial ordering for neighboring nodes and frequent types (will be used to keep most specific paths)
    # We can do this now because we still have the whole considered neighborhood in features and frequent types
    # Only these nodes and these types will appear in paths
    logger.info("Compute partial ordering for neighboring nodes and frequent types")
    n_partial_ordering = get_nodes_partial_ordering(features.keys(), frequent_types, canonical_graph)

    if debug:
        print_debug_partial_ordering(n_partial_ordering, canonical_graph["canonical_to_rdf_index"],
                                     cache_managers["rdf_nodes"], logger)
        logger.debug("PAUSED -- PRESS ENTER TO CONTINUE...")
        input()

    # Now that frequent types and nodes partial ordering are known, clean neighborhood
    # Drop neighbors not respecting limit_min / limit_max
    features = {
        ft: support_vector for ft, support_vector in features.items()
        if limit_min <= support_vector.count(1) <= limit_max
    }

    statistics["neighbors_w_limits"] = len(features)
    statistics["types_w_limits"] = len(frequent_types)

    if debug:
        print_debug_features(features, canonical_graph, cache_managers, sorted_seed_uris, logger)
        logger.debug("PAUSED -- PRESS ENTER TO CONTINUE...")
        input()

    # Compute paths and check limit_min / limit_max before adding to features
    statistics["paths_kept"] = 0
    for p, support_vector in get_paths(sorted_seed_uris, seed_uri_to_canonical_index, canonical_graph, cache_managers,
                                       blacklists, frequent_types, n_partial_ordering, k, t, d, limit_min, limit_max,
                                       undirected, statistics, logger, debug).items():
        if limit_min <= support_vector.count(1) <= limit_max:
            features[p] = support_vector
            statistics["paths_kept"] += 1

    if debug:
        print_debug_features(features, canonical_graph, cache_managers, sorted_seed_uris, logger)
        logger.debug("PAUSED -- PRESS ENTER TO CONTINUE...")
        input()

    # Save features and statistics
    logger.info("Save features and statistics")

    fp_statistics = open(output_dir + "{}_features_statistics_k{}_t{}_d{}_undirected{}_min{}_max{}.md".format(
            dataset_name, k, t, d, undirected, limit_min, limit_max), "w")

    fp_statistics.write("# Features statistics\n")
    fp_statistics.write("Generated on {}\n\n".format(datetime.datetime.now()))

    fp_statistics.write("## Total neighbors, types and paths reached / generated\n")
    fp_statistics.write("* Number of neighbors: {}\n".format(statistics["neighbors_wo_limits"]))
    fp_statistics.write("* Number of types: {}\n".format(statistics["types_wo_limits"]))
    fp_statistics.write("* Number of paths: {}\n\n".format(statistics["paths_generated"]))

    if meaningful_strategies & MeaningfulStrategies.NO_CHECK:
        logger.info("Save features w/o meaningfulness")
        save_features(features, sorted_seed_uris, classes, canonical_graph, cache_managers, dataset_name, k, t, d,
                      undirected, limit_min, limit_max, False, logger, output_dir)

        fp_statistics.write("## Features w/o meaningfulness\n")
        fp_statistics.write("* Number of neighbors: {}\n".format(statistics["neighbors_w_limits"]))
        fp_statistics.write("* Number of types: {}\n".format(statistics["types_w_limits"]))
        fp_statistics.write("* Number of paths: {}\n\n".format(statistics["paths_kept"]))

    if meaningful_strategies & MeaningfulStrategies.P:
        logger.info("Keep features w/ Pathways")
        m_features = select_meaningful_features(dict(features), True, False, False, canonical_graph, cache_managers)

        logger.info("Save features w/ Pathways")
        save_features(m_features, sorted_seed_uris, classes, canonical_graph, cache_managers, dataset_name, k, t, d,
                      undirected, limit_min, limit_max, "P", logger, output_dir)

        neighbors_number, paths_number = get_features_repartition(m_features)
        fp_statistics.write("## Features w/ Pathways\n")
        fp_statistics.write("* Number of neighbors: {}\n".format(neighbors_number))
        fp_statistics.write("* Number of paths: {}\n\n".format(paths_number))

    if meaningful_strategies & MeaningfulStrategies.G:
        logger.info("Keep features w/ GeneticFactors/GO classes")
        m_features = select_meaningful_features(dict(features), False, True, False, canonical_graph, cache_managers)

        logger.info("Save features w/ GeneticFactors/GO classes")
        save_features(m_features, sorted_seed_uris, classes, canonical_graph, cache_managers, dataset_name, k, t, d,
                      undirected, limit_min, limit_max, "G", logger, output_dir)

        neighbors_number, paths_number = get_features_repartition(m_features)
        fp_statistics.write("## Features w/ GeneticFactors/GO classes\n")
        fp_statistics.write("* Number of neighbors: {}\n".format(neighbors_number))
        fp_statistics.write("* Number of paths: {}\n\n".format(paths_number))

    if meaningful_strategies & MeaningfulStrategies.M:
        logger.info("Keep features w/ MeSH classes")
        m_features = select_meaningful_features(dict(features), False, False, True, canonical_graph, cache_managers)

        logger.info("Save features w/ MeSH classes")
        save_features(m_features, sorted_seed_uris, classes, canonical_graph, cache_managers, dataset_name, k, t, d,
                      undirected, limit_min, limit_max, "M", logger, output_dir)

        neighbors_number, paths_number = get_features_repartition(m_features)
        fp_statistics.write("## Features w/ MeSH classes\n")
        fp_statistics.write("* Number of neighbors: {}\n".format(neighbors_number))
        fp_statistics.write("* Number of paths: {}\n\n".format(paths_number))

    if meaningful_strategies & MeaningfulStrategies.PG:
        logger.info("Keep features w/ Pathways + GeneticFactors/GO classes")
        m_features = select_meaningful_features(dict(features), True, True, False, canonical_graph, cache_managers)

        logger.info("Save features w/ Pathways + GeneticFactors/GO classes")
        save_features(m_features, sorted_seed_uris, classes, canonical_graph, cache_managers, dataset_name, k, t, d,
                      undirected, limit_min, limit_max, "PG", logger, output_dir)

        neighbors_number, paths_number = get_features_repartition(m_features)
        fp_statistics.write("## Features w/ Pathways + GeneticFactors/GO classes\n")
        fp_statistics.write("* Number of neighbors: {}\n".format(neighbors_number))
        fp_statistics.write("* Number of paths: {}\n\n".format(paths_number))

    if meaningful_strategies & MeaningfulStrategies.PGM:
        logger.info("Keep features w/ Pathways + GeneticFactors/GO classes + MeSH classes")
        m_features = select_meaningful_features(dict(features), True, True, True, canonical_graph, cache_managers)

        logger.info("Save features w/ Pathways + GeneticFactors/GO classes + MeSH classes")
        save_features(m_features, sorted_seed_uris, classes, canonical_graph, cache_managers, dataset_name, k, t, d,
                      undirected, limit_min, limit_max, "PGM", logger, output_dir)

        neighbors_number, paths_number = get_features_repartition(m_features)
        fp_statistics.write("## Features w/ Pathways + GeneticFactors/GO classes + MeSH classes\n")
        fp_statistics.write("* Number of neighbors: {}\n".format(neighbors_number))
        fp_statistics.write("* Number of paths: {}\n\n".format(paths_number))

    fp_statistics.close()


def main():
    # Parsing command line parameters and necessary configuration
    parser = argparse.ArgumentParser()
    parser.add_argument("--configuration", help="JSON file configuring the program (triplestore, ...)",
                        required=True, dest="conf_file_path")
    parser.add_argument("--graph", dest="graph_dir", help="Base directory for input graph files", required=True)
    parser.add_argument("--dataset-csv", dest="dataset_file", help="CSV file with the seed nodes URIs", required=True)
    parser.add_argument("--dataset-name", dest="dataset_name", help="Name of the data set", required=True)
    parser.add_argument("--output", dest="output_dir", help="Base directory for output files", required=True)
    parser.add_argument("--lmin", dest="limit_min", help="Minimum support size to consider features", required=True,
                        type=int)
    parser.add_argument("--lmax", dest="limit_max", help="Maximum support size to consider features", required=True,
                        type=int)
    parser.add_argument("--kmin", dest="k_min", help="Minimum for range of max hops to consider (included)",
                        default=1, type=int)
    parser.add_argument("--kmax", dest="k_max", help="Maximum for range of max hops to consider (included)",
                        default=1, type=int)
    parser.add_argument("--tmin", dest="t_min", help="Minimum for range of hops in type hierarchies for generalization",
                        default=1, type=int)
    parser.add_argument("--tmax", dest="t_max", help="Maximum for range of hops in type hierarchies for generalization",
                        default=1, type=int)
    parser.add_argument("-d", dest="d", help="Max degree (included) to allow expansion of nodes", default=500,
                        type=int)
    parser.add_argument("--undirected", dest="undirected", help="If the graph is considered as undirected",
                        default=False, action="store_true")
    parser.add_argument("--meaningful", dest="meaningful", help="Meaningful strategies to consider",
                        default="no_check", choices=["all", "p", "g", "m", "pg", "pgm", "no_check"])
    parser.add_argument("--debug", dest="debug", help="Enable debug output", action="store_true")
    args = parser.parse_args()

    # Logging parameters
    logger = get_logger(args.debug)

    # Configuration parameters
    logger.info("Load configuration")
    configuration_parameters = load_configuration_parameters(args.conf_file_path)

    # Configure meaningful strategies
    meaningful_strategies = 0b000000
    if args.meaningful == "all":
        meaningful_strategies = MeaningfulStrategies.NO_CHECK | MeaningfulStrategies.P | MeaningfulStrategies.G | \
                                MeaningfulStrategies.M | MeaningfulStrategies.PG | MeaningfulStrategies.PGM
    elif args.meaningful == "no_check":
        meaningful_strategies = MeaningfulStrategies.NO_CHECK
    elif args.meaningful == "p":
        meaningful_strategies = MeaningfulStrategies.P
    elif args.meaningful == "g":
        meaningful_strategies = MeaningfulStrategies.G
    elif args.meaningful == "m":
        meaningful_strategies = MeaningfulStrategies.M
    elif args.meaningful == "pg":
        meaningful_strategies = MeaningfulStrategies.PG
    else:  # PGM
        meaningful_strategies = MeaningfulStrategies.PGM

    # Blacklists
    blacklists = get_blacklists(configuration_parameters)

    # Load graph
    logger.info("Load graph from " + args.graph_dir)
    canonical_graph = load_graph(args.graph_dir)

    # Load cache managers
    logger.info("Load cache managers from " + args.graph_dir)
    cache_managers = load_cache_managers(args.graph_dir)

    # Load dataset
    logger.info("Load dataset from " + args.dataset_file)
    classes = load_dataset(args.dataset_file)

    # Configuring output
    output_dir = args.output_dir
    if output_dir[-1] != "/":
        output_dir += "/"
    logger.info("Output configured in " + output_dir)

    # Ensure owl:Thing, owl:Class, rdf:type and rdfs:subClassOf are in canonical graph
    canonical_graph = add_thing_class_type_subclassof(canonical_graph, cache_managers)

    # Compute node degree
    logger.info("Compute degree for nodes")
    canonical_graph["nodes_degree"] = get_nodes_degree(
        canonical_graph,
        cache_managers["predicates"],
        blacklists["path_predicates"],
        args.undirected
    )

    if args.debug:
        print_debug_degrees(
            canonical_graph["nodes_degree"],
            canonical_graph["canonical_to_rdf_index"],
            cache_managers["rdf_nodes"],
            logger
        )
        logger.debug("PAUSED -- PRESS ENTER TO CONTINUE...")
        input()

    # Compute ancestors
    logger.info("Compute ancestors of classes (transitive closure of rdfs:subClassOf)")
    canonical_graph["nodes_ancestors"] = get_nodes_ancestors(canonical_graph, cache_managers)

    if args.debug:
        print_debug_ancestors(
            canonical_graph["nodes_ancestors"],
            canonical_graph["canonical_to_rdf_index"],
            cache_managers["rdf_nodes"],
            logger
        )
        logger.debug("PAUSED -- PRESS ENTER TO CONTINUE...")
        input()

    # Compute full type
    logger.info("Compute full type of nodes (transitive closure of rdf:type and rdfs:subClassOf)")
    canonical_graph["nodes_full_type"] = get_nodes_full_type(canonical_graph, cache_managers)

    if args.debug:
        print_debug_full_types(
            canonical_graph["nodes_full_type"],
            canonical_graph["canonical_to_rdf_index"],
            cache_managers["rdf_nodes"],
            logger
        )
        logger.debug("PAUSED -- PRESS ENTER TO CONTINUE...")
        input()

    # Computing matrix
    for k in tqdm.tqdm(range(args.k_min, args.k_max + 1), desc="current k"):
        for t in tqdm.tqdm(range(args.t_min, args.t_max + 1), desc="current t"):
            logger.info("Computing matrix k={}, t={}".format(k, t))

            compute_features(
                canonical_graph,
                cache_managers,
                blacklists,
                args.dataset_name,
                classes,
                k,
                t,
                args.d,
                args.undirected,
                args.limit_min,
                args.limit_max,
                meaningful_strategies,
                output_dir,
                logger,
                args.debug
            )


if __name__ == '__main__':
    main()
